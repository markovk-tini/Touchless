"""Tests for file_watcher_rules (Phase 3 B4)."""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.file_watcher_rules import (  # noqa: E402
    FsEvent, FsEventKind, Rule, RulesEngine,
    _glob_to_regex, _path_matches, _resolve_placeholders,
)


def _fresh() -> RulesEngine:
    d = Path(tempfile.mkdtemp())
    return RulesEngine(db_path=d / "rules.db")


def _evt(path: str, *, kind: FsEventKind = FsEventKind.CREATED,
         size: int = 0) -> FsEvent:
    return FsEvent(kind=kind, path=path, size_bytes=size,
                   ts=time.time())


# ---- CRUD --------------------------------------------------------------

def test_add_then_list_rule():
    eng = _fresh()
    r = eng.add_rule(name="invoices",
                     path_glob="C:/Downloads/**/*.pdf",
                     action="notion_create_page",
                     action_args={"title": "{filename}"})
    rules = eng.rules()
    assert len(rules) == 1
    assert rules[0].id == r.id
    assert rules[0].action_args["title"] == "{filename}"


def test_remove_rule():
    eng = _fresh()
    r = eng.add_rule(name="x", path_glob="*.png", action="ocr")
    assert eng.remove_rule(r.id) is True
    assert eng.remove_rule(r.id) is False  # already removed


def test_set_enabled_toggle():
    eng = _fresh()
    r = eng.add_rule(name="x", path_glob="*.png", action="ocr")
    assert eng.set_enabled(r.id, False) is True
    enabled_only = eng.rules(include_disabled=False)
    assert all(rule.id != r.id for rule in enabled_only)


def test_wipe_clears_all():
    eng = _fresh()
    eng.add_rule(name="a", path_glob="*.a", action="x")
    eng.add_rule(name="b", path_glob="*.b", action="y")
    n = eng.wipe()
    assert n == 2
    assert eng.rules() == []


# ---- glob matching -----------------------------------------------------

def test_basename_only_glob():
    rule = Rule(id="x", name="x", path_glob="*.pdf", action="x")
    e = _evt("C:/users/k/Documents/report.PDF")
    assert _path_matches(rule, e) is True


def test_recursive_glob_matches_at_any_depth():
    rule = Rule(id="x", name="x",
                path_glob="C:/Downloads/**/*.pdf", action="x")
    e1 = _evt("C:/Downloads/inv.pdf")
    e2 = _evt("C:/Downloads/2026/Q1/inv.pdf")
    e3 = _evt("C:/Downloads/inv.txt")  # wrong ext
    e4 = _evt("D:/elsewhere/inv.pdf")  # wrong prefix
    assert _path_matches(rule, e1) is True
    assert _path_matches(rule, e2) is True
    assert _path_matches(rule, e3) is False
    assert _path_matches(rule, e4) is False


def test_single_star_does_not_cross_slashes():
    pat = _glob_to_regex("C:/Downloads/*.pdf")
    assert pat.match("c:/downloads/x.pdf") is not None
    assert pat.match("c:/downloads/sub/x.pdf") is None


def test_double_star_crosses_slashes():
    pat = _glob_to_regex("**/*.pdf")
    assert pat.match("a/b/c.pdf") is not None
    assert pat.match("a.pdf") is not None


# ---- match conditions --------------------------------------------------

def test_match_with_kind_filter():
    eng = _fresh()
    r = eng.add_rule(name="x", path_glob="*.txt", action="a",
                     kinds=[FsEventKind.CREATED])
    matched = eng.match(_evt("/x.txt", kind=FsEventKind.MODIFIED))
    assert matched == []
    matched = eng.match(_evt("/x.txt", kind=FsEventKind.CREATED))
    assert len(matched) == 1


def test_match_with_min_size():
    eng = _fresh()
    eng.add_rule(name="x", path_glob="*.bin", action="a",
                 min_size_bytes=1024)
    assert eng.match(_evt("/x.bin", size=500)) == []
    # Add unique evt time to avoid rate-limit collision between the
    # two match() calls.
    e = FsEvent(kind=FsEventKind.CREATED, path="/x.bin",
                size_bytes=2048, ts=time.time() + 10)
    assert len(eng.match(e)) == 1


def test_match_with_max_size():
    eng = _fresh()
    eng.add_rule(name="x", path_glob="*.bin", action="a",
                 max_size_bytes=1024)
    assert eng.match(_evt("/x.bin", size=2048)) == []
    e = FsEvent(kind=FsEventKind.CREATED, path="/x.bin",
                size_bytes=512, ts=time.time() + 10)
    assert len(eng.match(e)) == 1


def test_match_with_require_path_contains():
    eng = _fresh()
    eng.add_rule(name="x", path_glob="*.pdf", action="a",
                 require_path_contains="Invoices")
    assert eng.match(_evt("/Downloads/x.pdf")) == []
    e = FsEvent(kind=FsEventKind.CREATED,
                path="/Downloads/Invoices/x.pdf",
                ts=time.time() + 10)
    assert len(eng.match(e)) == 1


# ---- rate limiting -----------------------------------------------------

def test_rate_limit_suppresses_repeat_fire():
    eng = _fresh()
    eng.add_rule(name="x", path_glob="*.txt", action="a",
                 rate_limit_sec=10.0)
    e1 = FsEvent(kind=FsEventKind.CREATED, path="/x.txt", ts=100.0)
    e2 = FsEvent(kind=FsEventKind.CREATED, path="/x.txt", ts=101.0)
    assert len(eng.match(e1)) == 1
    # Within rate-limit window → no match.
    assert eng.match(e2) == []


def test_rate_limit_clears_after_window():
    eng = _fresh()
    eng.add_rule(name="x", path_glob="*.txt", action="a",
                 rate_limit_sec=1.0)
    e1 = FsEvent(kind=FsEventKind.CREATED, path="/x.txt", ts=100.0)
    e2 = FsEvent(kind=FsEventKind.CREATED, path="/x.txt", ts=102.0)
    assert len(eng.match(e1)) == 1
    assert len(eng.match(e2)) == 1


# ---- placeholder substitution -----------------------------------------

def test_placeholders_substituted():
    args = {"title": "{name}", "url": "file://{path}",
            "tag": "{ext}", "literal": 5}
    e = FsEvent(kind=FsEventKind.CREATED,
                path="C:/Downloads/report.pdf")
    out = _resolve_placeholders(args, e)
    assert out["title"] == "report"
    assert out["url"] == "file://C:/Downloads/report.pdf"
    assert out["tag"] == "pdf"
    assert out["literal"] == 5


def test_unknown_placeholder_kept_literal():
    out = _resolve_placeholders({"x": "{nope}"},
                                _evt("/x.txt"))
    assert out["x"] == "{nope}"


# ---- dispatch end-to-end ----------------------------------------------

def test_dispatch_calls_dispatcher_per_matched_rule():
    eng = _fresh()
    r = eng.add_rule(name="invoices",
                     path_glob="**/*.pdf",
                     action="notion_create_page",
                     action_args={"title": "{filename}"})
    calls = []
    eng.dispatch(_evt("/Downloads/inv.pdf"),
                 lambda action, args: calls.append((action, args)) or {"ok": True})
    assert len(calls) == 1
    assert calls[0][0] == "notion_create_page"
    assert calls[0][1]["title"] == "inv.pdf"


def test_dispatch_records_errors():
    eng = _fresh()
    eng.add_rule(name="x", path_glob="*.txt", action="a")

    def boom(action, args):
        raise RuntimeError("boom")
    results = eng.dispatch(_evt("/x.txt"), boom)
    assert results[0]["dispatched"] is False
    assert "boom" in results[0]["error"]


def test_disabled_rule_does_not_fire():
    eng = _fresh()
    r = eng.add_rule(name="x", path_glob="*.txt", action="a",
                     enabled=False)
    assert eng.match(_evt("/x.txt")) == []
