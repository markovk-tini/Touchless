"""r20: the camera's real name must survive onto CameraInfo.

open_camera_by_index used to hardcode "Camera N (DirectShow)" as the
display name. Nothing matches that in either classifier keyword list, so
classify_camera_shutter_hint returned None for every camera opened that
way, including the cheap UVC webcams the classifier exists for. The
knock-on effect was that the kick which returns a stuck exposure to Auto
could never fire, and a webcam left latched at a very short manual
exposure by an earlier session stayed dark forever.

These tests pin the whole chain: Qt name -> CameraInfo.display_name ->
classifier verdict.
"""

import pytest

from hgr.app.camera import camera_utils as CU


class FakeCap:
    def isOpened(self):
        return True

    def read(self):
        return True, None

    def release(self):
        pass

    def set(self, *a):
        return True

    def get(self, *a):
        return 0.0


@pytest.fixture
def opened(monkeypatch):
    """open_camera_by_index with the cv2 and threading layers stubbed."""
    def _open(qt_names):
        monkeypatch.setattr(CU, "_qt_video_device_names", lambda: list(qt_names))
        monkeypatch.setattr(
            CU, "try_open_camera",
            lambda index, backend, read_attempts=0: FakeCap(),
        )
        monkeypatch.setattr(CU, "ThreadedCvCapture", lambda cap, **kw: cap)
        result = CU.open_camera_by_index(0, max_index=1)
        return result[0] if isinstance(result, tuple) else result
    return _open


# ------------------------------------------------- the field webcam

def test_a_generic_uvc_name_survives_and_classifies_as_generic(opened):
    info = opened(["FULL HD 1080P Webcam"])
    assert info.display_name == "FULL HD 1080P Webcam (Camera 0)"
    assert CU.classify_camera_shutter_hint(info.display_name) is True


def test_the_old_hardcoded_name_would_have_classified_as_unknown():
    """Documents the bug this fixes, so nobody reintroduces it."""
    assert CU.classify_camera_shutter_hint("Camera 0 (DirectShow)") is None


# ------------------------------------------------ the reference rig

def test_a_premium_camera_name_classifies_as_premium(opened):
    info = opened(["Razer Kiyo Pro"])
    assert info.display_name == "Razer Kiyo Pro (Camera 0)"
    assert CU.classify_camera_shutter_hint(info.display_name) is False


def test_the_reference_rigs_actual_reported_name_stays_unknown(opened):
    """The dev machine's Kiyo Pro enumerates as "USB Video Device", which
    is in neither keyword list. It must stay unknown, because unknown is
    what keeps the kick from firing on that machine."""
    info = opened(["USB Video Device"])
    assert info.display_name == "USB Video Device (Camera 0)"
    assert CU.classify_camera_shutter_hint(info.display_name) is None


# --------------------------------------------------------- fallback

def test_it_falls_back_to_the_old_label_when_qt_says_nothing(opened):
    """Qt can return an empty list on some driver combinations. The old
    string is still better than no CameraInfo at all."""
    info = opened([])
    assert info.display_name.startswith("Camera 0 (")
    assert CU.classify_camera_shutter_hint(info.display_name) is None


def test_an_index_past_the_qt_list_falls_back(opened):
    info = opened([""])
    assert info.display_name.startswith("Camera 0 (")


def test_the_suffix_is_the_shape_the_device_resolver_strips(opened, monkeypatch):
    """resolve_dshow_device_for_index strips a trailing "(Camera N)"
    before matching against ffmpeg's device names, so the name must
    carry exactly that shape or the exact-match branch breaks."""
    import hgr.app.camera.ffmpeg_capture as FC

    info = opened(["FULL HD 1080P Webcam"])
    # monkeypatch, NOT a bare assignment: a bare assignment here leaked
    # into every later test in the session and silently disabled the
    # device-cache tests, which count real enumerations.
    monkeypatch.setattr(
        FC, "list_dshow_video_devices", lambda *a, **kw: ["FULL HD 1080P Webcam"]
    )
    assert FC.resolve_dshow_device_for_index(
        0, qt_name_hint=info.display_name
    ) == "FULL HD 1080P Webcam"
