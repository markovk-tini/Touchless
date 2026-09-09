from __future__ import annotations

from types import SimpleNamespace

from hgr.app.integration.noop_engine import GestureWorker
from hgr.config.gesture_bindings import pose_id_for_static_label


def test_thumb_up_is_hidden_from_banner() -> None:
    label, active = GestureWorker._hide_unnamed_recognizer_label("thumb_up", True)
    assert label == ""
    assert active is False
    label, active = GestureWorker._hide_unnamed_recognizer_label("thumb_down", True)
    assert label == ""
    assert active is False


def test_thumb_up_is_not_a_bound_pose() -> None:
    assert pose_id_for_static_label("Right", "thumb_up") is None
    assert pose_id_for_static_label("Right", "thumb_down") is None


def test_empty_frame_skip_never_skips_while_hand_is_tracked() -> None:
    skip, ticks = GestureWorker._empty_frame_skip_inference(True, 99)
    assert skip is False
    assert ticks == 0


def test_empty_frame_skip_every_other_only() -> None:
    ticks = 0
    skips = []
    for _ in range(8):
        skip, ticks = GestureWorker._empty_frame_skip_inference(False, ticks)
        skips.append(skip)
    assert skips == [True, False, True, False, True, False, True, False]


def test_builtin_swipe_in_flight_from_label() -> None:
    pred = SimpleNamespace(dynamic_label="swipe_right", dynamic_candidates=())
    assert GestureWorker._builtin_open_hand_swipe_in_flight(pred) is True
    idle = SimpleNamespace(dynamic_label="neutral", dynamic_candidates=())
    assert GestureWorker._builtin_open_hand_swipe_in_flight(idle) is False


def test_builtin_swipe_in_flight_from_score() -> None:
    pred = SimpleNamespace(
        dynamic_label="neutral",
        dynamic_candidates=(SimpleNamespace(label="swipe_left", score=0.50),),
    )
    assert GestureWorker._builtin_open_hand_swipe_in_flight(pred) is True
    weak = SimpleNamespace(
        dynamic_label="neutral",
        dynamic_candidates=(SimpleNamespace(label="swipe_left", score=0.32),),
    )
    assert GestureWorker._builtin_open_hand_swipe_in_flight(weak) is False


def test_repeat_circle_suppressed_on_two_and_three() -> None:
    class _W:
        _suppress_builtin_horizontal_swipe = False

        def _pose_sequence_owns_motion(self, now=None):
            return False

    w = _W()
    two = SimpleNamespace(dynamic_label="repeat_circle", stable_label="two")
    assert GestureWorker._effective_dynamic_label(w, two) == "neutral"
    three = SimpleNamespace(dynamic_label="repeat_circle", stable_label="three")
    assert GestureWorker._effective_dynamic_label(w, three) == "neutral"
    one = SimpleNamespace(dynamic_label="repeat_circle", stable_label="one")
    assert GestureWorker._effective_dynamic_label(w, one) == "repeat_circle"


def _fake_worker_backend(backend: str, complexity: int):
    class _W:
        engine = SimpleNamespace(
            detector=SimpleNamespace(
                runtime=SimpleNamespace(backend=backend),
                model_complexity=complexity,
            )
        )

    return _W()


def test_engine_landmarks_not_reused_on_onnx_or_lite() -> None:
    assert (
        GestureWorker._custom_runner_can_use_engine_landmarks(
            _fake_worker_backend("onnx-directml", 1)
        )
        is False
    )
    assert (
        GestureWorker._custom_runner_can_use_engine_landmarks(
            _fake_worker_backend("mediapipe-cpu", 0)
        )
        is False
    )
    assert (
        GestureWorker._custom_runner_can_use_engine_landmarks(
            _fake_worker_backend("mediapipe-cpu", 1)
        )
        is True
    )
    assert (
        GestureWorker._custom_runner_can_use_engine_landmarks(
            _fake_worker_backend("mediapipe-tasks-gpu", 1)
        )
        is True
    )


def test_engine_landmarks_reused_on_darwin_onnx_and_lite(monkeypatch) -> None:
    monkeypatch.setattr("hgr.app.integration.noop_engine.sys.platform", "darwin")
    assert (
        GestureWorker._custom_runner_can_use_engine_landmarks(
            _fake_worker_backend("onnx-coreml", 1)
        )
        is True
    )
    assert (
        GestureWorker._custom_runner_can_use_engine_landmarks(
            _fake_worker_backend("mediapipe-cpu", 0)
        )
        is True
    )


def test_custom_mp_skip_reuses_cache_without_treating_as_lost() -> None:
    class _Runner:
        def extract_hands(self, frame):
            return [("lm", "Right")]

    class _W:
        _custom_runner_slow_path_skip_ratio = 2
        _custom_runner_slow_path_counter = 1
        _last_custom_mp_hands = None
        _custom_gesture_runner = _Runner()

        def _custom_runner_can_use_engine_landmarks(self):
            return False

    w = _W()
    result = SimpleNamespace()
    first_hands, first_sampled = GestureWorker._custom_hands_this_frame(w, result, object())
    assert first_sampled is True
    assert first_hands == [("lm", "Right")]
    skip_hands, skip_sampled = GestureWorker._custom_hands_this_frame(w, result, object())
    assert skip_sampled is False
    assert skip_hands == [("lm", "Right")]


def test_custom_hands_prefers_engine_thread_payload() -> None:
    class _Runner:
        def extract_hands(self, frame):
            raise AssertionError("GUI must not extract when payload is attached")

    class _W:
        _custom_gesture_runner = _Runner()

        def _custom_runner_can_use_engine_landmarks(self):
            return False

    w = _W()
    result = SimpleNamespace(custom_hands_payload=([("cached", "Left")], True))
    hands, sampled = GestureWorker._custom_hands_this_frame(w, result, object())
    assert sampled is True
    assert hands == [("cached", "Left")]


def test_pick_custom_hand_empty() -> None:
    assert GestureWorker._pick_custom_hand([]) == (None, None)
    assert GestureWorker._pick_custom_hand([("lm", "Right")]) == ("lm", "Right")


def test_loop_template_detected_from_low_intent_magnitude() -> None:
    from types import SimpleNamespace

    from hgr.custom_gestures.dynamic_classifier import DynamicGestureClassifier

    fake = SimpleNamespace(
        _templates=(SimpleNamespace(intent_magnitude=0.20),),
        _template_reversals=(0,),
    )
    assert DynamicGestureClassifier.has_loop_or_complex_templates(fake) is True
    linear = SimpleNamespace(
        _templates=(SimpleNamespace(intent_magnitude=2.5),),
        _template_reversals=(0,),
    )
    assert DynamicGestureClassifier.has_loop_or_complex_templates(linear) is False
