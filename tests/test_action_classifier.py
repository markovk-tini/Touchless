"""Unit tests for :mod:`hgr.live_api.action_classifier`.

The classifier is pure / deterministic, so each test asserts the full
``{kind, target, confidence}`` shape it cares about and avoids exercising
real I/O.

Author: Konstantin Markov
"""
from __future__ import annotations

import pytest

from hgr.live_api.action_classifier import (
    ACTION_KINDS,
    classify_tool_action,
    classify_user_text,
)


# ---------------------------------------------------------------------------
# tool-action tests
# ---------------------------------------------------------------------------

def test_write_file_ok_classified_as_file_touch() -> None:
    out = classify_tool_action(
        "write_file",
        {"path": "c:/HGR App v1.0.0/notes.md", "base_dir": "c:/HGR App v1.0.0"},
        {"status": "ok", "bytes": 42},
    )
    assert out["kind"] == "file_touch"
    assert out["target"] == "c:/HGR App v1.0.0/notes.md"
    assert out["payload"]["tool"] == "write_file"
    assert out["payload"]["base_dir"] == "c:/HGR App v1.0.0"
    assert out["payload"]["size_bytes"] == 42
    assert out["confidence"] >= 0.9


def test_failed_status_returns_unknown() -> None:
    out = classify_tool_action(
        "write_file",
        {"path": "c:/foo.txt"},
        {"status": "error", "error": "permission denied"},
    )
    assert out["kind"] == "unknown"
    assert out["confidence"] == 0.0


def test_needs_confirmation_returns_unknown() -> None:
    # Regression guard — overwrite gated by user confirmation must NOT be
    # classified as a file_touch.
    out = classify_tool_action(
        "write_file",
        {"path": "c:/foo.txt"},
        {"status": "needs_confirmation"},
    )
    assert out["kind"] == "unknown"


def test_open_in_editor_with_folder_is_new_project() -> None:
    out = classify_tool_action(
        "open_in_editor",
        {
            "folder_path": "c:/some-new-thing",
            "editor": "code",
            "file_to_open": "README.md",
        },
        {"status": "ok"},
    )
    assert out["kind"] == "new_project"
    assert out["target"] == "c:/some-new-thing"
    assert out["payload"]["editor"] == "code"
    assert out["payload"]["file"] == "README.md"
    assert out["confidence"] >= 0.8


def test_open_app_classified_as_tool_use() -> None:
    out = classify_tool_action(
        "open_app",
        {"app": "Notepad"},
        {"status": "ok"},
    )
    assert out["kind"] == "tool_use"
    assert out["target"] == "Notepad"
    assert out["confidence"] >= 0.8


def test_open_url_learning_content_is_pattern_seed() -> None:
    out = classify_tool_action(
        "open_url",
        {"url": "https://www.youtube.com/watch?v=abc"},
        {"status": "ok"},
    )
    assert out["kind"] == "pattern_seed"
    assert "youtube.com" in out["target"].lower()


def test_open_url_generic_is_tool_use() -> None:
    out = classify_tool_action(
        "open_url",
        {"url": "https://www.example.com/store"},
        {"status": "ok"},
    )
    assert out["kind"] == "tool_use"
    assert out["target"] == "https://www.example.com/store"


def test_unknown_tool_falls_back_to_tool_use() -> None:
    out = classify_tool_action(
        "some_brand_new_thing",
        {"foo": "bar"},
        {"status": "ok"},
    )
    assert out["kind"] == "tool_use"
    assert out["target"] == "some_brand_new_thing"


def test_empty_tool_name_returns_unknown() -> None:
    out = classify_tool_action("", {}, {"status": "ok"})
    assert out["kind"] == "unknown"


# ---------------------------------------------------------------------------
# user-text tests
# ---------------------------------------------------------------------------

def test_remember_phrase_is_fact() -> None:
    out = classify_user_text("Remember that my landlord's name is Sam")
    assert out["kind"] == "fact"
    assert "sam" in (out["target"] or "").lower()
    assert out["confidence"] >= 0.8


def test_my_x_is_y_is_fact() -> None:
    out = classify_user_text("My phone number is 555-0100")
    assert out["kind"] == "fact"
    assert out["payload"]["attribute"] == "phone number"
    assert out["payload"]["value"].endswith("555-0100")


def test_i_prefer_is_preference() -> None:
    out = classify_user_text("I prefer dark mode in everything")
    assert out["kind"] == "preference"
    assert out["payload"]["polarity"] == "prefer"
    assert out["confidence"] >= 0.85


def test_i_always_is_preference() -> None:
    out = classify_user_text("I always commit on green tests")
    assert out["kind"] == "preference"
    assert out["payload"]["polarity"] == "always"


def test_i_never_is_preference() -> None:
    out = classify_user_text("I never push directly to main")
    assert out["kind"] == "preference"
    assert out["payload"]["polarity"] == "never"


def test_unknown_project_path_is_new_project() -> None:
    out = classify_user_text(
        "let's look at c:\\some-other-project\\src today",
        known_project_roots=["c:/HGR App v1.0.0", "c:/touchless-website"],
    )
    assert out["kind"] == "new_project"
    assert out["target"].startswith("c:/some-other-project")


def test_known_project_path_is_episode() -> None:
    out = classify_user_text(
        "open c:/HGR App v1.0.0/src/hgr/live_api/tool_executor.py please",
        known_project_roots=["c:/HGR App v1.0.0"],
    )
    assert out["kind"] == "episode"
    assert out["payload"]["known_project"] is True


def test_empty_text_is_unknown() -> None:
    assert classify_user_text("")["kind"] == "unknown"
    assert classify_user_text(None)["kind"] == "unknown"
    assert classify_user_text("   ")["kind"] == "unknown"


def test_random_chatter_is_unknown() -> None:
    out = classify_user_text("what's the weather like tomorrow")
    assert out["kind"] == "unknown"
    assert out["confidence"] == 0.0


def test_preference_takes_priority_over_my_x_is_y() -> None:
    # 'I prefer my coffee is black' could ambiguously match 'my X is Y',
    # but preference should win.
    out = classify_user_text("I prefer my coffee black")
    assert out["kind"] == "preference"


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tool,args,result",
    [
        ("write_file", {"path": "x"}, {"status": "ok"}),
        ("open_app", {"app": "X"}, {"status": "ok"}),
        ("open_url", {"url": "https://example.com"}, {"status": "ok"}),
        ("totally_unknown", {}, {"status": "ok"}),
        ("write_file", {}, {"status": "error"}),
    ],
)
def test_tool_classifications_are_in_whitelist(tool, args, result) -> None:
    out = classify_tool_action(tool, args, result)
    assert out["kind"] in ACTION_KINDS
    assert 0.0 <= out["confidence"] <= 1.0
    assert isinstance(out["payload"], dict)


@pytest.mark.parametrize(
    "text",
    [
        "remember that my wifi password is hunter2",
        "I prefer tabs over spaces",
        "open c:/HGR App v1.0.0",
        "open c:/totally-new",
        "small talk",
        "",
    ],
)
def test_user_text_classifications_are_in_whitelist(text) -> None:
    out = classify_user_text(text, known_project_roots=["c:/HGR App v1.0.0"])
    assert out["kind"] in ACTION_KINDS
    assert 0.0 <= out["confidence"] <= 1.0


def test_classify_tool_is_idempotent() -> None:
    args = {"path": "c:/foo.txt"}
    res = {"status": "ok", "bytes": 100}
    a = classify_tool_action("write_file", args, res)
    b = classify_tool_action("write_file", args, res)
    assert a == b


def test_classify_user_is_idempotent() -> None:
    a = classify_user_text("I prefer dark mode")
    b = classify_user_text("I prefer dark mode")
    assert a == b
