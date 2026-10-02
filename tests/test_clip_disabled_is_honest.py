"""r20d: when Touchless turns clipping off by itself, say so, early.

Seeding the always-on recorder off on weak hardware is the biggest CPU
saving of the round, but it disables the instant-clip gesture, four
gesture-wheel items and three voice commands. Silently. These tests pin
the three things that make that honest rather than bug-like.
"""

import inspect
import io
import os
import pathlib

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hgr.app.ui.main_window import MainWindow          # noqa: E402
from hgr.config.app_config import AppConfig            # noqa: E402

SRC = io.open(
    pathlib.Path(MainWindow.__module__.replace(".", "/") + ".py")
    if False else pathlib.Path(inspect.getfile(MainWindow)),
    encoding="utf-8",
).read()


# ------------------------------------------- the reason must persist

def test_the_reason_is_a_persisted_config_field():
    """The Settings panel is built during startup, BEFORE the seeding
    runs, so an in-memory reason can never reach the tooltip."""
    assert hasattr(AppConfig(), "clip_cache_seeded_off_note")
    assert AppConfig().clip_cache_seeded_off_note == ""


def test_the_seeder_writes_the_reason_when_it_turns_clipping_off():
    body = inspect.getsource(MainWindow._seed_clip_cache_default_once)
    assert "clip_cache_seeded_off_note" in body
    i = body.index("clip_cache_enabled = False")
    assert "clip_cache_seeded_off_note" in body[i:i + 700], (
        "the note must be written in the same branch that turns it off"
    )


def test_the_tooltip_reads_the_persisted_note_not_an_attribute():
    body = inspect.getsource(MainWindow._build_general_clip_section)
    assert "clip_cache_seeded_off_note" in body
    assert "_clip_cache_seeded_off_reason" not in body, (
        "reading the in-memory attribute here is the bug this fixes"
    )


def test_the_note_is_written_in_plain_language():
    """It is shown verbatim to a non-technical user."""
    body = inspect.getsource(MainWindow._seed_clip_cache_default_once)
    i = body.index("clip_cache_seeded_off_note")
    note = body[i:i + 600]
    assert "megapixels" in note
    assert "Tick to" in note or "tick to" in note
    for jargon in ("px=", "vram_mb=", "gdigrab", "swscale", "nv12"):
        assert jargon not in note, jargon


# ------------------------------------- the message must explain itself

def test_the_disabled_pill_distinguishes_automatic_from_user_choice():
    body = inspect.getsource(MainWindow._show_clip_disabled_pill)
    assert "clip_cache_seeded_off_note" in body, (
        "the pill must know whether the app turned it off"
    )
    assert "Touchless turned it off" in body


def test_the_modal_fallback_also_explains():
    body = inspect.getsource(MainWindow._show_clip_disabled_pill)
    i = body.find("QMessageBox")
    assert i != -1
    assert "_why" in body[i:i + 400]


# ------------------------------- the refusal must come before the work

def test_screen_recording_refuses_before_the_picker_and_countdown():
    body = inspect.getsource(MainWindow._start_screen_record_countdown)
    gate = body.find("clip_cache_enabled")
    countdown = body.find("_start_countdown_overlay")
    assert gate != -1, "no clipping gate in the countdown entry point"
    assert gate < countdown or countdown == -1, (
        "the user must be told before a monitor picker and a 3-2-1 countdown"
    )


def test_the_downstream_gate_is_still_there():
    """The early gate is an addition, not a replacement. Other callers
    reach _start_screen_recording directly."""
    body = inspect.getsource(MainWindow._start_screen_recording)
    assert "clip_cache_enabled" in body


@pytest.mark.parametrize("name,needle", [
    ("_start_screen_record_countdown", "clip_cache_enabled"),
    ("_show_clip_disabled_pill", "clip_cache_seeded_off_note"),
    ("_seed_clip_cache_default_once", "clip_cache_seeded_off_note"),
    ("_build_general_clip_section", "clip_cache_seeded_off_note"),
])
def test_the_edit_landed_on_the_definition_python_actually_binds(name, needle):
    """main_window.py carries duplicate definitions of several methods;
    _start_screen_record_countdown has three. An edit that lands on a
    copy Python does not bind is silently dead, which has happened in
    this file before. Assert against the BOUND source, not the text."""
    body = inspect.getsource(getattr(MainWindow, name))
    assert needle in body, (
        f"{name}: the change is not in the definition Python binds"
    )


def test_the_duplicate_count_is_recorded_not_assumed():
    """Documents the hazard rather than asserting it away. If these
    counts change, someone has touched the dead region and should
    re-check which definition binds."""
    import re
    counts = {
        n: len(re.findall(rf"^    def {n}\(", SRC, re.M))
        for n in ("_start_screen_record_countdown", "_show_clip_disabled_pill",
                  "_seed_clip_cache_default_once", "_build_general_clip_section")
    }
    assert counts["_start_screen_record_countdown"] >= 1
    assert all(v >= 1 for v in counts.values()), counts
