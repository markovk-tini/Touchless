"""The "Save debug bundle?" prompt must not fire on a normal close.

It used to run on EVERY real close, which reads as the app nagging you
on the way out, and it also intercepted the build smoke gate's own
shutdown -- the gate saw a clean exit with no marker and failed the
build.

It is gated rather than deleted, because it is the ONLY live path to a
debug bundle: the other two call sites are in main_window.py's dead
region (`MainWindow.closeEvent` binds at ~37565 while the copy at
~39419 never runs, and `_stop_screen_recording` binds at ~36419).
Deleting it would have removed the field-diagnosis tool the support
workflow depends on.

These are WIRING checks, and deliberately so -- `closeEvent` needs a
real QWidget to invoke, so what is verified here is that the opt-in is
actually in the bound method's code and that the shipped debug launcher
turns it on. Driven through AST rather than text matching, so a comment
mentioning the variable cannot satisfy them.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

from hgr.app.ui.main_window import MainWindow


ENV_VAR = "HGR_DEBUG_BUNDLE_ON_CLOSE"
ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "installers" / "windows" / "Touchless_Debug.ps1"


def _live_close_event_tree():
    """AST of the closeEvent Python actually binds, not a dead twin."""
    src = textwrap.dedent(inspect.getsource(MainWindow.closeEvent))
    return ast.parse(src)


def test_close_prompt_is_gated_on_the_env_opt_in():
    """`_skip_bundle` must include a `not <opt-in>` term.

    Without that term the prompt is on by default again, which is the
    whole behaviour being removed.
    """
    tree = _live_close_event_tree()
    gated = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "_skip_bundle"
            for t in node.targets
        ):
            continue
        # Look for a UnaryOp(Not) anywhere in the assigned expression.
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.Not):
                gated = True
    assert gated, (
        "_skip_bundle no longer contains a negated opt-in term -- the "
        "debug-bundle prompt is back on by default for every close"
    )


def test_the_opt_in_reads_the_env_var():
    """The gate must come from the environment, not a config field a
    normal user could trip, and not a hardcoded True."""
    tree = _live_close_event_tree()
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == ENV_VAR:
            found = True
    assert found, (
        "the live closeEvent does not reference %s -- the opt-in is not "
        "wired to the environment" % ENV_VAR
    )


def test_prompt_construction_is_still_reachable():
    """Gated, not deleted. If this disappears there is no live way left
    to produce a debug bundle at all."""
    src = inspect.getsource(MainWindow.closeEvent)
    assert "DebugBundleSavePromptDialog" in src, (
        "the only live debug-bundle call site is gone -- field diagnosis "
        "has no entry point"
    )


def test_debug_launcher_turns_the_prompt_back_on():
    """The support path is "run Touchless_Debug.bat, close the app, send
    me the bundle". That only works if the launcher sets the var."""
    assert LAUNCHER.exists(), LAUNCHER
    text = LAUNCHER.read_text(encoding="utf-8", errors="replace")
    # Strip comment lines so the header prose cannot satisfy this.
    code = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )
    assert ENV_VAR in code, (
        "%s is not SET in Touchless_Debug.ps1 (only mentioned in a "
        "comment?) -- the support path is broken" % ENV_VAR
    )
    assert "'1'" in code or '"1"' in code


def test_launcher_is_shipped_in_the_bundle():
    """A launcher that does not reach users is no support path."""
    spec = (ROOT / "builder" / "windows" / "hgr_app.spec").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "Touchless_Debug" in spec, (
        "hgr_app.spec does not ship Touchless_Debug.* -- the opt-in would "
        "be unreachable for an installed user"
    )


# Author: Konstantin Markov
