"""r22: the exposure hint is dead on an ffmpeg capture, not in a mode.

Field log, 2026-09-25. Lite and GPU ran SLOWER than Default on the same
machine, with worse hand tracking. Default held 30 fps; Lite sat near 10
and latched the degraded low-fps mode.

The cause was one gate. Entering Lite/GPU attempts an ffmpeg capture; on
this webcam it failed and fell back to an ordinary OpenCV capture. But
the tuning that applies the short-shutter exposure skipped anyway,
because it asked "does this mode want ffmpeg?" rather than "am I on an
ffmpeg capture?". So the camera was left on auto exposure -- a long
shutter in a dim room, which is both the low frame rate and the blur:

    r49-short-shutter] skip: reason=wants_ffmpeg (exposure hint is dead
                       in this mode); classifier=True unstuck=False

The hint is genuinely dead on a real ffmpeg capture, where ffmpeg owns
the DirectShow graph and cv2 .set() reaches nothing. It is very much
alive on an OpenCV capture, whatever mode asked for it.
"""

import inspect
import re

from hgr.app.integration import noop_engine as NE

BODY = inspect.getsource(NE.GestureWorker._apply_default_capture_tuning)


def _gate_line():
    for line in BODY.splitlines():
        if "_wants_ffmpeg_cap()" in line and "_low_fps_active" in line and line.strip().startswith("if"):
            return line
    return ""


def test_the_skip_requires_an_actual_ffmpeg_capture():
    gate = _gate_line()
    assert gate, "the mode-skip gate is gone entirely"
    assert "_on_ffmpeg_cap" in gate, (
        "the gate still skips on mode alone, so an OpenCV fallback keeps "
        "auto exposure and the long shutter that caused ~10 fps"
    )


def test_the_capture_type_is_detected_before_the_gate():
    gate_at = BODY.index(_gate_line())
    detect_at = BODY.index("FfmpegMjpegCapture")
    assert detect_at < gate_at, "capture type must be known before gating"


def test_detection_survives_every_open_result_shape():
    """open_camera_by_index returns a tuple; other paths hand back a bare
    capture or None. None of them may raise here."""
    src = BODY[:BODY.index(_gate_line())]
    assert "isinstance(open_result, tuple)" in src
    assert 'hasattr(open_result, "set")' in src
    assert "except Exception" in src


def test_a_real_ffmpeg_capture_still_skips():
    """Unchanged behaviour where the hint really cannot land."""
    gate = _gate_line()
    assert "self._wants_ffmpeg_cap() or self._low_fps_active" in gate, (
        "the original mode conditions must still be required"
    )
    assert gate.strip().endswith("and _on_ffmpeg_cap:")


def test_the_breadcrumb_says_which_condition_held():
    """A field log has to distinguish 'skipped on an ffmpeg cap' from the
    old 'skipped because the mode wanted one'."""
    assert "_on_ffmpeg_cap" in BODY
    reasons = re.findall(r'"(wants_ffmpeg[^"]*|low_fps_active[^"]*)"', BODY)
    assert reasons, "no skip reason is logged"
    assert all(r.endswith("_on_ffmpeg_cap") for r in reasons), (
        f"skip reasons must name the capture, got {reasons}"
    )


def test_the_unstick_helper_still_runs_in_the_skip_branch():
    """It is the only exposure repair left when we DO skip."""
    i = BODY.index(_gate_line())
    assert "_unstick_inherited_short_shutter" in BODY[i:], (
        "the inherited-dark-camera repair was lost from the skip branch"
    )
