"""Source guards for docs/PERFORMANCE_CHECKPOINT.md §2.4 / §3.3 / §3.8 / §3.9.

These read noop_engine.py as TEXT so they run without a camera, Qt or
MediaPipe. They exist because the same regression has now been shipped
three times under three different names (r53 `_user_chose_kick`, the
sticky opt-in, and the r17 "cross-session restore" block): a camera
driver write gated on the sticky `camera_force_short_shutter_user_chose`
flag or on a bare exposure readback threshold.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "hgr" / "app" / "integration" / "noop_engine.py"


def _source() -> str:
    return SRC.read_text(encoding="utf-8")


def test_kick_gate_is_literal():
    s = _source()
    assert "_should_kick = _armed or _known_generic_needs_kick" in s
    # The sticky flag may still be READ for logging, but it must never
    # be part of the kick expression.
    for line in s.splitlines():
        if line.strip().startswith("_should_kick ="):
            assert "_user_chose" not in line, line
    assert "_now_off_from_prior_session" not in s


def test_no_user_chose_gated_driver_write():
    """A cap.set(...) within 12 lines after a user_chose conditional is
    the §3.3/§3.8 pattern. The r53 branch that only ASSIGNS apply_hint
    from the flag is fine; a write is not."""
    lines = _source().splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "camera_force_short_shutter_user_chose" not in line and "_user_chose" not in stripped:
            continue
        if not stripped.startswith(("if ", "elif ", "and ", "or ", "_now_off")):
            continue
        window = "\n".join(lines[i : i + 12])
        assert "cap.set(" not in window, f"user_chose-gated driver write near line {i + 1}:\n{window}"


def test_dshow_auto_exposure_helper_exists_and_no_bare_075_then_30_pairs():
    """On DirectShow CAP_PROP_AUTO_EXPOSURE=3.0 means MANUAL (cap_dshow.cpp:
    cvRound(v)==1 -> auto). Every auto-on write must go through the
    backend-aware helper; the only 3.0 write allowed is inside it."""
    s = _source()
    assert "def _dshow_auto_exposure_on(" in s
    # Only one 3.0 write in the whole file (the helper's non-DSHOW branch).
    assert len(re.findall(r"CAP_PROP_AUTO_EXPOSURE,\s*3\.0", s)) == 1
    # No raw "0.75" auto-on writes outside the helper.
    helper_start = s.index("def _dshow_auto_exposure_on(")
    helper_end = s.index("class GestureWorker(QObject):", helper_start)
    outside = s[:helper_start] + s[helper_end:]
    assert "CAP_PROP_AUTO_EXPOSURE, 0.75" not in outside


def test_r54_low_luma_net_is_gone():
    s = _source()
    assert "_r54_low_luma_streak" not in s
    assert "_r54_short_shutter_safety_disabled" not in s
    assert "r54_auto_safety_net" not in s


def test_r55_marker_never_reads_user_chose_or_bare_threshold():
    """The r55 pre-open restore must key on the persisted marker only."""
    s = _source()
    start = s.index("def _r55_restore_before_open(")
    end = s.index("def _preflight_short_shutter_for_ffmpeg(", start)
    body = s[start:end]
    assert "camera_force_short_shutter_user_chose" not in body
    assert "_current_exp_looks_short" not in body
    assert "camera_short_shutter_latched_for" in body
    # It must never write to the live engine cap: only a local throwaway `pre`.
    assert "self._cap" not in body
    assert "cap.set(" not in body.replace("pre.set(", "")


def test_dshow_helper_semantics_with_fake_cap():
    """Exercise the helper's decision logic without importing the engine
    module (which pulls Qt/cv2/MediaPipe): compile just that function."""
    s = _source()
    start = s.index("def _dshow_auto_exposure_on(")
    end = s.index("\n\nclass GestureWorker(QObject):", start)
    func_src = s[start:end]

    class _Cv2:
        CAP_PROP_AUTO_EXPOSURE = 21

    class _Cap:
        def __init__(self, backend):
            self._backend = backend
            self.writes = []

        def getBackendName(self):
            return self._backend

        def set(self, prop, val):
            self.writes.append((prop, val))
            return True

    import io
    import sys as _sys

    ns = {"cv2": _Cv2, "sys": _sys}
    exec(compile(func_src, "<helper>", "exec"), ns)
    fn = ns["_dshow_auto_exposure_on"]

    dshow = _Cap("DSHOW")
    assert fn(dshow) is True
    assert [v for _, v in dshow.writes] == [0.75]

    msmf = _Cap("MSMF")
    assert fn(msmf) is True
    assert [v for _, v in msmf.writes] == [0.75, 3.0]

    unknown = _Cap("")
    fn(unknown)
    expected = [0.75] if _sys.platform.startswith("win") else [0.75, 3.0]
    assert [v for _, v in unknown.writes] == expected
    _ = io  # keep import for symmetry with engine module style


# ---------------------------------------------------------------------------
# r18 review guards (main_window.py is read as text; the bound definitions
# were verified with inspect.getsourcelines when these were written).
# ---------------------------------------------------------------------------
MAIN_WINDOW = Path(__file__).resolve().parents[1] / "src" / "hgr" / "app" / "ui" / "main_window.py"


def test_stop_engine_restore_is_ledger_gated():
    """The STOP-time camera-control restore may only write controls the
    engine itself wrote this session (its ledger). An unfiltered
    restore(_before, _changes) would revert Synapse / OBS changes."""
    s = MAIN_WINDOW.read_text(encoding="utf-8")
    assert '_ours = [c for c in _changes if c.get("property") in _written]' in s
    assert "_dc.restore(_before, _changes)" not in s
    assert "_dc.restore(_before, _ours)" in s


def test_engine_writers_record_to_ledger():
    s = _source()
    assert "def _note_driver_write(" in s
    # preflight, preflight-release, live ON, rollback, kick, OFF restore
    assert s.count('self._note_driver_write("Exposure")') >= 6
    assert "self._driver_writes_this_session = set()" in s


def test_r55_pre_open_restore_not_called_in_open_camera():
    s = _source()
    start = s.index("    def _open_camera(self):")
    end = s.index("\n    def ", start + 10)
    body = s[start:end]
    assert "self._r55_restore_before_open(" not in body


def _fn_source(name: str) -> str:
    """Source of one GestureWorker method, comments stripped.

    Text guards over the whole 16k-line module match anything, including
    a comment that merely mentions the pattern. Scoping to the method
    and dropping comments is what makes them mean what they say.
    """
    import inspect
    import re

    import hgr.app.integration.noop_engine as _NE

    src = inspect.getsource(getattr(_NE.GestureWorker, name))
    return "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())


def test_throwaway_preflight_writes_are_lock_bounded():
    s = _source()
    assert "def _with_graph_lock(" in s
    # r23: the exposure VALUE is now chosen per camera from the
    # advertised frame rate (exposure_policy.exposure_for_camera), so
    # this pins what the test is actually about -- the write going
    # through the graph lock -- rather than the old -6.0 literal.
    assert "_with_graph_lock(pre.set, cv2.CAP_PROP_EXPOSURE, _exp_target)" in s
    # r25: assert the CALL, not its exact spelling. This used to pin the
    # one-line form `..._short_shutter_exposure_for(device_name)`, which
    # broke the moment the call grew a `driver_fps=` argument -- a false
    # failure for a change that did exactly what the guard wants.
    _pre = _fn_source("_preflight_short_shutter_for_ffmpeg")
    assert "self._short_shutter_exposure_for(" in _pre
    assert "device_name" in _pre
    assert "_with_graph_lock(_dshow_auto_exposure_on, pre, log_tag=\"[preflight-release]\")" in s


def test_both_exposure_write_sites_use_the_policy():
    """r23: there are exactly two short-shutter exposure writes and both
    must go through `_short_shutter_exposure_for`, not a literal.

    The second site (inside `_apply_default_capture_tuning`) is the one
    the field rig actually executes: when the ffmpeg open fails and we
    fall back to an ordinary OpenCV capture, r22 deliberately keeps the
    hint alive there, because otherwise Lite and GPU sit on auto
    exposure at ~10 fps.
    """
    s = _source()
    # Both sites call the policy. Matched per-function rather than by
    # exact call text so that adding an argument -- r25 added
    # `driver_fps=`, the camera's own CAP_PROP_FPS readback, because the
    # learned-capability source is only ever populated on the ffmpeg
    # branch and Default therefore always fell back to -6.0 -- does not
    # read as a regression.
    _pre = _fn_source("_preflight_short_shutter_for_ffmpeg")
    _live = _fn_source("_apply_default_capture_tuning")
    assert "self._short_shutter_exposure_for(" in _pre, "preflight site"
    assert "self._short_shutter_exposure_for(" in _live, "live cap site"
    assert "device_name" in _pre
    assert "_display_name" in _live
    # still exactly two write sites, and neither writes a literal
    assert s.count("self._short_shutter_exposure_for(") == 2
    assert "cv2.CAP_PROP_EXPOSURE, -6.0)" not in s


def test_camera_identity_is_never_read_from_a_field_that_does_not_exist():
    """PERFORMANCE_CHECKPOINT §2.11 records this bug twice already, and
    r23 shipped it a third time: `CameraInfo` is
    (index, backend, backend_name, display_name) -- there is no `name`
    and no `device_name`. Reading one yields "" and every downstream
    per-camera lookup silently degrades to its default, which is
    invisible precisely because the default is the old behaviour.
    """
    from hgr.app.camera.camera_utils import CameraInfo
    import dataclasses

    fields = {f.name for f in dataclasses.fields(CameraInfo)}
    assert "display_name" in fields
    assert "name" not in fields, "CameraInfo grew a `name` field - revisit this guard"

    # Strip comments: a comment that NAMES the bug is useful
    # documentation, and the fix carries one. Only real code counts.
    code = "\n".join(
        re.sub(r"#.*$", "", line) for line in _source().splitlines()
    )
    for bad in ("self._camera_info, 'name'", 'self._camera_info, "name"',
                "self._camera_info, 'device_name'", 'self._camera_info, "device_name"',
                "self._camera_info.name", "self._camera_info.device_name"):
        assert bad not in code, f"reads a CameraInfo field that does not exist: {bad}"


def test_the_exposure_policy_decision_is_always_logged():
    """A breadcrumb that only fires on the interesting value makes
    "never learned" and "caller passed an empty name" indistinguishable
    in a field bundle. Both must be named explicitly."""
    s = _source()
    i = s.index("def _short_shutter_exposure_for(")
    body = s[i:s.index("\n    def ", i + 10)]
    assert "[exposure-policy]" in body
    assert "caller bug" in body
    assert "no capabilities" in body
    # it must also say HOW the cache entry was matched, so a field bundle
    # distinguishes "exact name" from "fell back to the only camera"
    assert "_via" in body
    # the log must NOT be conditional on the value differing from default
    assert "if value != DEFAULT_EXPOSURE:" not in body


def test_the_display_stack_self_disables_when_no_lift_is_needed():
    """r23 / §3.5: the CLAHE + saturation stage used to key on the latch
    alone, so it ran at full strength on an already-bright frame -- the
    "washed-out / dim live view" §3.5 records as a reverted wrong path.
    It must now defer to the gamma stage's own decision."""
    s = _source()
    i = s.index("def _apply_display_local_contrast_if_needed(")
    body = s[i:s.index("\n    def ", i + 10)]
    assert 'getattr(self, "_ss_disp_target_gamma", None) is None' in body
    # and that check must come before any cv2 work
    assert body.index("_ss_disp_target_gamma") < body.index("cv2.cvtColor")
