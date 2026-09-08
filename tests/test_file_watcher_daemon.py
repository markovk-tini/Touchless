"""Tests for FileSystemWatcher (Phase 3 wiring)."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.file_watcher_daemon import (  # noqa: E402
    FileSystemWatcher, _diff_snapshots,
    _extract_roots_from_rules, _first_meta_pos, _snapshot_root,
)
from hgr.live_api.file_watcher_rules import (  # noqa: E402
    FsEventKind, Rule, RulesEngine,
)


# ---- glob-prefix extraction --------------------------------------------

def test_first_meta_pos_finds_star():
    assert _first_meta_pos("foo/*.pdf") == 4
    assert _first_meta_pos("foo/bar.pdf") == -1
    assert _first_meta_pos("**/x") == 0


def test_extract_roots_strips_glob_suffix():
    with tempfile.TemporaryDirectory() as tmp:
        sub = (Path(tmp) / "Downloads").resolve()
        sub.mkdir()
        rule = Rule(id="r", name="r",
                    path_glob=f"{sub}/**/*.pdf", action="x")
        roots = _extract_roots_from_rules([rule])
        assert len(roots) == 1
        # Both sides resolved to long form on Windows.
        assert roots[0].resolve() == sub.resolve()


def test_extract_roots_dedupes():
    with tempfile.TemporaryDirectory() as tmp:
        sub = Path(tmp) / "Same"
        sub.mkdir()
        rules = [
            Rule(id="r1", name="r1",
                 path_glob=f"{sub}/*.pdf", action="x"),
            Rule(id="r2", name="r2",
                 path_glob=f"{sub}/*.png", action="y"),
        ]
        roots = _extract_roots_from_rules(rules)
        assert len(roots) == 1


def test_extract_roots_skips_nonexistent():
    rule = Rule(id="r", name="r",
                path_glob="/path/does/not/exist/*.pdf", action="x")
    assert _extract_roots_from_rules([rule]) == []


# ---- snapshot + diff ---------------------------------------------------

def test_snapshot_root_captures_files():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.txt").write_text("hello")
        (Path(tmp) / "b.txt").write_text("world")
        snap = _snapshot_root(Path(tmp))
        assert len(snap) == 2
        # Each entry is (mtime, size).
        for path, (mtime, size) in snap.items():
            assert mtime > 0
            assert size > 0


def test_diff_detects_created():
    prev = {"/a": (1.0, 10)}
    curr = {"/a": (1.0, 10), "/b": (2.0, 20)}
    events = _diff_snapshots(prev, curr)
    kinds = [e.kind for e in events]
    assert FsEventKind.CREATED in kinds
    assert FsEventKind.DELETED not in kinds


def test_diff_detects_deleted():
    prev = {"/a": (1.0, 10), "/b": (2.0, 20)}
    curr = {"/a": (1.0, 10)}
    events = _diff_snapshots(prev, curr)
    kinds = [e.kind for e in events]
    assert FsEventKind.DELETED in kinds


def test_diff_detects_modified():
    prev = {"/a": (1.0, 10)}
    curr = {"/a": (1.5, 15)}
    events = _diff_snapshots(prev, curr)
    assert events[0].kind == FsEventKind.MODIFIED


def test_diff_clean_when_no_change():
    snap = {"/a": (1.0, 10), "/b": (2.0, 20)}
    events = _diff_snapshots(snap, snap)
    assert events == []


# ---- watcher tick end-to-end -------------------------------------------

def _fresh_engine():
    d = Path(tempfile.mkdtemp())
    return RulesEngine(db_path=d / "rules.db")


def test_first_tick_baselines_no_events_fire():
    with tempfile.TemporaryDirectory() as tmp:
        # File ALREADY exists before the watcher starts.
        (Path(tmp) / "preexisting.txt").write_text("old")
        eng = _fresh_engine()
        eng.add_rule(name="r", path_glob=f"{tmp}/*.txt",
                     action="a")
        calls = []
        watcher = FileSystemWatcher(
            engine=eng,
            dispatcher=lambda action, args: calls.append((action, args)))
        watcher.tick()  # baseline pass
        assert calls == []


def test_second_tick_fires_for_new_file():
    with tempfile.TemporaryDirectory() as tmp:
        # Resolve once so Windows short-form vs long-form path
        # collisions don't break the glob match in the engine.
        tmp_resolved = str(Path(tmp).resolve())
        eng = _fresh_engine()
        eng.add_rule(name="r", path_glob=f"{tmp_resolved}/*.txt",
                     action="notion_create",
                     action_args={"title": "{filename}"})
        calls = []
        watcher = FileSystemWatcher(
            engine=eng,
            dispatcher=lambda action, args:
                calls.append((action, args)) or {"ok": True})
        watcher.tick()  # baseline
        time.sleep(0.05)  # ensure mtime differs
        (Path(tmp_resolved) / "fresh.txt").write_text("new")
        watcher.tick()  # detect the create
        assert len(calls) == 1
        assert calls[0][0] == "notion_create"
        assert calls[0][1]["title"] == "fresh.txt"


def test_tick_handles_empty_rules():
    eng = _fresh_engine()
    watcher = FileSystemWatcher(engine=eng, dispatcher=lambda *a: None)
    watcher.tick()  # should not raise


def test_tick_does_not_dispatch_when_glob_misses():
    with tempfile.TemporaryDirectory() as tmp:
        eng = _fresh_engine()
        # Rule only matches .pdf — we'll create a .txt.
        eng.add_rule(name="r", path_glob=f"{tmp}/*.pdf", action="x")
        calls = []
        watcher = FileSystemWatcher(
            engine=eng, dispatcher=lambda a, args: calls.append((a, args)))
        watcher.tick()  # baseline
        time.sleep(0.05)
        (Path(tmp) / "fresh.txt").write_text("new")
        watcher.tick()
        assert calls == []


def test_reset_clears_snapshots():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_resolved = str(Path(tmp).resolve())
        (Path(tmp_resolved) / "a.txt").write_text("existing")
        eng = _fresh_engine()
        eng.add_rule(name="r", path_glob=f"{tmp_resolved}/*.txt",
                     action="x")
        watcher = FileSystemWatcher(
            engine=eng, dispatcher=lambda *a: None)
        watcher.tick()  # baseline
        # After baseline, exactly one snapshot keyed by the resolved root.
        keys = list(watcher._snapshots.keys())
        assert len(keys) == 1
        watcher.reset()
        assert watcher._snapshots == {}
        assert watcher._first_tick is True
