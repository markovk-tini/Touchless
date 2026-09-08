"""Tests for ProjectProfile substrate (Phase 6 B5)."""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.project_profile import (  # noqa: E402
    ProjectProfile, ProjectProfileStore, _git_branch,
    current_profile, find_project_root, render_for_planner,
    reset_global_store, touch,
)
import hgr.live_api.project_profile as pp  # noqa: E402


def _isolated_store(tmp_path: Path) -> ProjectProfileStore:
    """Helper: build a store + install it as the singleton."""
    store = ProjectProfileStore(db_path=tmp_path / "p.db")
    pp._store = store
    return store


def setup_function():
    reset_global_store()


# ---- store crud -------------------------------------------------------

def test_store_upsert_then_get(tmp_path):
    store = ProjectProfileStore(db_path=tmp_path / "p.db")
    p = ProjectProfile(root=str(tmp_path), name="proj-1",
                       summary="hello")
    store.upsert(p)
    got = store.get(str(tmp_path))
    assert got is not None
    assert got.name == "proj-1"
    assert got.summary == "hello"


def test_store_get_missing_returns_none(tmp_path):
    store = ProjectProfileStore(db_path=tmp_path / "p.db")
    assert store.get("does-not-exist") is None


def test_store_upsert_idempotent(tmp_path):
    store = ProjectProfileStore(db_path=tmp_path / "p.db")
    p = ProjectProfile(root=str(tmp_path), name="proj")
    store.upsert(p)
    p.name = "renamed"
    store.upsert(p)
    got = store.get(str(tmp_path))
    assert got.name == "renamed"


def test_store_recent_ordering(tmp_path):
    store = ProjectProfileStore(db_path=tmp_path / "p.db")
    now = time.time()
    a = ProjectProfile(root="/a", name="A",
                       last_seen_at=now - 10)
    b = ProjectProfile(root="/b", name="B",
                       last_seen_at=now)
    store.upsert(a)
    store.upsert(b)
    recent = store.recent(limit=10)
    assert [r.name for r in recent] == ["B", "A"]


def test_store_delete(tmp_path):
    store = ProjectProfileStore(db_path=tmp_path / "p.db")
    store.upsert(ProjectProfile(root="/x", name="X"))
    assert store.delete("/x") is True
    assert store.get("/x") is None


def test_store_delete_missing_returns_false(tmp_path):
    store = ProjectProfileStore(db_path=tmp_path / "p.db")
    assert store.delete("/nope") is False


def test_store_all(tmp_path):
    store = ProjectProfileStore(db_path=tmp_path / "p.db")
    store.upsert(ProjectProfile(root="/a", name="A",
                                 last_seen_at=1.0))
    store.upsert(ProjectProfile(root="/b", name="B",
                                 last_seen_at=2.0))
    assert len(store.all()) == 2


def test_store_persists_recent_files(tmp_path):
    store = ProjectProfileStore(db_path=tmp_path / "p.db")
    p = ProjectProfile(root="/x", recent_files=["a.py", "b.py"])
    store.upsert(p)
    got = store.get("/x")
    assert got.recent_files == ["a.py", "b.py"]


def test_store_persists_activity(tmp_path):
    store = ProjectProfileStore(db_path=tmp_path / "p.db")
    p = ProjectProfile(
        root="/x",
        recent_activity=[{"kind": "ran tests", "ts": 1.0}])
    store.upsert(p)
    got = store.get("/x")
    assert len(got.recent_activity) == 1
    assert got.recent_activity[0]["kind"] == "ran tests"


# ---- find_project_root -----------------------------------------------

def test_find_project_root_with_git_marker(tmp_path):
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == tmp_path


def test_find_project_root_with_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]")
    assert find_project_root(tmp_path) == tmp_path


def test_find_project_root_no_marker(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    assert find_project_root(nested) is None


def test_find_project_root_from_file(tmp_path):
    (tmp_path / ".git").mkdir()
    f = tmp_path / "src" / "main.py"
    f.parent.mkdir()
    f.write_text("x")
    assert find_project_root(f) == tmp_path


# ---- git branch ------------------------------------------------------

def test_git_branch_reads_head_ref(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text(
        "ref: refs/heads/main\n")
    assert _git_branch(tmp_path) == "main"


def test_git_branch_handles_detached(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("abc1234deadbeef\n")
    assert _git_branch(tmp_path) == "abc1234d"


def test_git_branch_missing_returns_empty(tmp_path):
    assert _git_branch(tmp_path) == ""


# ---- touch facade ----------------------------------------------------

def test_touch_creates_profile():
    d = Path(tempfile.mkdtemp())
    _isolated_store(d)
    p = touch(d, name="My Project", summary="testing")
    assert p.name == "My Project"
    assert p.summary == "testing"
    assert p.last_seen_at > 0


def test_touch_adds_recent_file():
    d = Path(tempfile.mkdtemp())
    _isolated_store(d)
    touch(d, file_hint="foo.py")
    touch(d, file_hint="bar.py")
    p = touch(d, file_hint="foo.py")
    # foo.py was MRU'd to the front; bar.py stays in list once.
    assert p.recent_files[0] == "foo.py"
    assert "bar.py" in p.recent_files
    assert p.recent_files.count("foo.py") == 1


def test_touch_records_branch_when_git_present(tmp_path):
    _isolated_store(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/dev")
    p = touch(tmp_path)
    assert p.branch == "dev"


def test_touch_adds_activity_entry():
    d = Path(tempfile.mkdtemp())
    _isolated_store(d)
    p = touch(d, activity={"kind": "ran tests",
                            "detail": "all green"})
    assert p.recent_activity[0]["kind"] == "ran tests"
    assert p.recent_activity[0]["detail"] == "all green"
    assert "ts" in p.recent_activity[0]


def test_touch_caps_recent_files_at_24():
    d = Path(tempfile.mkdtemp())
    _isolated_store(d)
    for i in range(30):
        touch(d, file_hint=f"f{i}.py")
    p = current_profile(d)
    assert len(p.recent_files) == 24
    # Most recently touched is at the front.
    assert p.recent_files[0] == "f29.py"


# ---- current_profile -------------------------------------------------

def test_current_profile_resolves_from_path(tmp_path):
    _isolated_store(tmp_path)
    (tmp_path / ".git").mkdir()
    touch(tmp_path, name="P")
    nested = tmp_path / "src"
    nested.mkdir()
    p = current_profile(nested)
    assert p is not None
    assert p.name == "P"


def test_current_profile_falls_back_to_most_recent(tmp_path):
    _isolated_store(tmp_path)
    touch(tmp_path, name="last-active")
    assert current_profile(None).name == "last-active"


def test_current_profile_returns_none_when_empty(tmp_path):
    _isolated_store(tmp_path)
    assert current_profile(None) is None


# ---- render ----------------------------------------------------------

def test_render_for_planner_full_profile():
    p = ProjectProfile(
        root="/repo", name="My Repo", summary="working on auth",
        branch="feat/auth",
        recent_files=["auth.py", "test_auth.py"],
        recent_activity=[{"kind": "ran tests",
                          "detail": "12 passed"}])
    out = render_for_planner(p)
    assert "My Repo" in out
    assert "working on auth" in out
    assert "feat/auth" in out
    assert "auth.py" in out
    assert "12 passed" in out


def test_render_for_planner_minimal_profile():
    p = ProjectProfile(root="/r")
    out = render_for_planner(p)
    assert "/r" in out


# ---- dataclass + short_label ----------------------------------------

def test_short_label_uses_name():
    assert (ProjectProfile(root="/x", name="proj").short_label()
            == "proj")


def test_short_label_falls_back_to_dirname():
    assert (ProjectProfile(root="/some/repo").short_label()
            == "repo")


def test_project_profile_defaults():
    p = ProjectProfile(root="/x")
    assert p.recent_files == []
    assert p.recent_activity == []
    assert p.created_at > 0
