from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class HandRuntime:
    hands_module: object
    drawing_utils: object | None
    hand_connections: object | None
    backend: str = "mediapipe-cpu"   # "mediapipe-cpu" | "mediapipe-tasks-gpu" | "onnx-directml"


# Module-level log-gate flags. Each mode toggle rebuilds HandDetector, which
# calls load_hand_runtime, which used to emit its "gpu selected" / "gpu
# failed" logs unconditionally — the same messages fired on every Lite Mode
# toggle, every GPU Mode toggle, every auto-engage disengage, spamming the
# log with duplicates. These flags gate each log to first-fire per process
# so mode toggles don't repeat the same message. The reset function is only
# for tests — production code should NEVER reset (the point is one-shot).
_LOGGED_TASKS_GPU_SELECTED = False
_LOGGED_ONNX_DIRECTML_SELECTED = False
_LOGGED_TASKS_CONSTRUCTION_FAIL = False
_LOGGED_ONNX_CONSTRUCTION_FAIL = False
_LOGGED_NO_GPU_PATH = False
# The new C2 log: fires ONCE per process when prefer_gpu=True but the
# runtime resolver landed on CPU MediaPipe. Answers "is Lite/GPU Mode
# actually engaging DirectML on my machine?" without spamming.
_LOGGED_PREFER_GPU_FELL_BACK_TO_CPU = False


def _reset_runtime_log_gates_for_test() -> None:
    """Test hook. Do NOT call from production code — one-shot logging
    is intentional to prevent mode-toggle spam."""
    global _LOGGED_TASKS_GPU_SELECTED, _LOGGED_ONNX_DIRECTML_SELECTED
    global _LOGGED_TASKS_CONSTRUCTION_FAIL, _LOGGED_ONNX_CONSTRUCTION_FAIL
    global _LOGGED_NO_GPU_PATH, _LOGGED_PREFER_GPU_FELL_BACK_TO_CPU
    _LOGGED_TASKS_GPU_SELECTED = False
    _LOGGED_ONNX_DIRECTML_SELECTED = False
    _LOGGED_TASKS_CONSTRUCTION_FAIL = False
    _LOGGED_ONNX_CONSTRUCTION_FAIL = False
    _LOGGED_NO_GPU_PATH = False
    _LOGGED_PREFER_GPU_FELL_BACK_TO_CPU = False


def _log_once(gate_name: str, message: str) -> None:
    """One-shot log helper. Sets the module-level gate flag to True and
    writes to stderr on first call; subsequent calls with the same gate
    are silent. Never raises; log-writing errors are swallowed so a
    failing stderr can't take down the gesture pipeline."""
    global _LOGGED_TASKS_GPU_SELECTED, _LOGGED_ONNX_DIRECTML_SELECTED
    global _LOGGED_TASKS_CONSTRUCTION_FAIL, _LOGGED_ONNX_CONSTRUCTION_FAIL
    global _LOGGED_NO_GPU_PATH, _LOGGED_PREFER_GPU_FELL_BACK_TO_CPU
    if gate_name == "tasks_gpu_selected":
        if _LOGGED_TASKS_GPU_SELECTED:
            return
        _LOGGED_TASKS_GPU_SELECTED = True
    elif gate_name == "onnx_directml_selected":
        if _LOGGED_ONNX_DIRECTML_SELECTED:
            return
        _LOGGED_ONNX_DIRECTML_SELECTED = True
    elif gate_name == "tasks_construction_fail":
        if _LOGGED_TASKS_CONSTRUCTION_FAIL:
            return
        _LOGGED_TASKS_CONSTRUCTION_FAIL = True
    elif gate_name == "onnx_construction_fail":
        if _LOGGED_ONNX_CONSTRUCTION_FAIL:
            return
        _LOGGED_ONNX_CONSTRUCTION_FAIL = True
    elif gate_name == "no_gpu_path":
        if _LOGGED_NO_GPU_PATH:
            return
        _LOGGED_NO_GPU_PATH = True
    elif gate_name == "prefer_gpu_fell_back":
        if _LOGGED_PREFER_GPU_FELL_BACK_TO_CPU:
            return
        _LOGGED_PREFER_GPU_FELL_BACK_TO_CPU = True
    else:
        # Unknown gate — log anyway (better to spam than silently swallow
        # a message when the gate name is a bug).
        pass
    try:
        sys.stderr.write(message)
        if not message.endswith("\n"):
            sys.stderr.write("\n")
        sys.stderr.flush()
    except Exception:
        pass


def _load_mediapipe_cpu_runtime() -> HandRuntime:
    """Load the legacy mediapipe.solutions.hands path. This is the
    path Touchless has always used and the safe fallback for every
    GPU attempt that doesn't reach the GPU."""
    import mediapipe as mp

    last_error: Exception | None = None
    try:
        hands_module = mp.solutions.hands
        drawing_utils = getattr(mp.solutions, "drawing_utils", None)
        hand_connections = getattr(hands_module, "HAND_CONNECTIONS", None)
        return HandRuntime(hands_module, drawing_utils, hand_connections, backend="mediapipe-cpu")
    except Exception as exc:
        last_error = exc

    try:
        from mediapipe.python.solutions import drawing_utils  # type: ignore
        from mediapipe.python.solutions import hands as hands_module  # type: ignore

        hand_connections = getattr(hands_module, "HAND_CONNECTIONS", None)
        return HandRuntime(hands_module, drawing_utils, hand_connections, backend="mediapipe-cpu")
    except Exception as exc:
        last_error = exc

    version = getattr(mp, "__version__", "unknown")
    detail = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown import error"
    raise ImportError(
        "Unable to load MediaPipe Hands for gesture tracking. "
        f"Installed mediapipe version: {version}. Last import error: {detail}"
    )


def _try_load_gpu_runtime() -> HandRuntime | None:
    """Attempt the GPU-accelerated path. Returns None if GPU isn't
    reachable on this machine — the caller falls back to CPU
    MediaPipe transparently."""
    from .gpu_probe import probe_gpu_paths

    probe = probe_gpu_paths()
    if not probe.has_any_gpu_path:
        _log_once(
            "no_gpu_path",
            "[hand_runtime] gpu_mode requested but no GPU inference path is reachable. "
            "Falling back to MediaPipe CPU.\n"
            f"{probe.diagnostic()}"
        )
        return None

    # Path 1 — MediaPipe Tasks API HandLandmarker with GPU delegate.
    # Same models as solutions.hands so accuracy is identical;
    # speedup comes from MediaPipe's Vulkan / OpenGL ES delegate
    # when reachable. Construction failure (delegate can't reach a
    # GPU context, .task asset missing, etc.) → return None and
    # let the caller fall back to CPU MediaPipe.
    if probe.mediapipe_tasks_importable and probe.tasks_gpu_delegate_present:
        try:
            from .tasks_runtime import build_tasks_gpu_runtime

            hands_module = build_tasks_gpu_runtime()
            if hands_module is not None:
                _log_once(
                    "tasks_gpu_selected",
                    "[hand_runtime] gpu_mode active: MediaPipe Tasks GPU delegate "
                    "selected. Inference accuracy matches CPU MediaPipe; speedup "
                    "depends on whether the delegate can reach a GPU context on "
                    "this machine (it transparently runs on CPU otherwise).",
                )
                hand_connections = getattr(hands_module, "HAND_CONNECTIONS", None)
                return HandRuntime(
                    hands_module=hands_module,
                    drawing_utils=None,
                    hand_connections=hand_connections,
                    backend="mediapipe-tasks-gpu",
                )
        except Exception as exc:
            _log_once(
                "tasks_construction_fail",
                f"[hand_runtime] Tasks-API GPU path construction failed: "
                f"{type(exc).__name__}: {exc!s}. Falling back to MediaPipe CPU.",
            )

    # Path 2 — onnxruntime + DirectML on a custom palm-detect +
    # landmark pipeline. Real Windows GPU path. We try this AFTER
    # Tasks-API because Tasks-API would be lower-effort if it
    # worked, but in practice on Windows it raises
    # NotImplementedError so we end up here for any actual GPU.
    if probe.onnxruntime_importable and probe.onnxruntime_directml_provider:
        try:
            from .onnx_runtime import build_onnx_directml_runtime

            hands_module = build_onnx_directml_runtime()
            if hands_module is not None:
                _log_once(
                    "onnx_directml_selected",
                    "[hand_runtime] gpu_mode active: ONNX Runtime + DirectML "
                    "selected. Palm detection + hand landmark inference run on "
                    "the GPU. Accuracy matches MediaPipe CPU (same model weights "
                    "from OpenCV Zoo).",
                )
                hand_connections = getattr(hands_module, "HAND_CONNECTIONS", None)
                return HandRuntime(
                    hands_module=hands_module,
                    drawing_utils=None,
                    hand_connections=hand_connections,
                    backend="onnx-directml",
                )
        except Exception as exc:
            _log_once(
                "onnx_construction_fail",
                f"[hand_runtime] ONNX/DirectML path construction failed: "
                f"{type(exc).__name__}: {exc!s}. Falling back to MediaPipe CPU.",
            )

    return None


def load_hand_runtime(*, prefer_gpu: bool = False) -> HandRuntime:
    """Pick a hand-tracking runtime. When prefer_gpu is True we try
    the GPU path first; on any failure we transparently fall back
    to the CPU MediaPipe runtime so gesture detection keeps working.

    Callers that want CPU unconditionally (low_fps mode etc.) pass
    prefer_gpu=False or omit the kwarg. The GestureWorker reads
    config.gpu_mode and threads it through here.
    """
    if prefer_gpu:
        gpu_runtime = _try_load_gpu_runtime()
        if gpu_runtime is not None:
            return gpu_runtime
        # The user (or Lite Mode's prefer_gpu=True default) asked for
        # GPU inference and we couldn't deliver it. Individual reasons
        # are logged above by _try_load_gpu_runtime (no probe path,
        # Tasks-API construction failure, ONNX construction failure).
        # This SUMMARY log fires once per process so the user can see
        # in the app.log — even when they didn't scroll up through the
        # provider-list details — that Lite/GPU Mode is running on CPU
        # MediaPipe. Answers the diagnostic question "did DirectML
        # actually engage for me?" without the user needing to inspect
        # the per-mode timing logs and infer from engine-work durations.
        _log_once(
            "prefer_gpu_fell_back",
            "[hand_runtime] prefer_gpu=True but the runtime resolver landed on "
            "CPU MediaPipe. Lite/GPU Mode's engine-work speedup will NOT engage "
            "on this machine — hand-tracking inference runs on CPU MediaPipe "
            "XNNPACK exactly as in Normal Mode. See the earlier [hand_runtime] "
            "lines in this log for the specific reason (no GPU probe path / "
            "Tasks-API GPU delegate failed / ONNX Runtime DirectML provider "
            "unavailable). This message logs once per process; toggling modes "
            "will not repeat it.",
        )
    return _load_mediapipe_cpu_runtime()

# Author: Konstantin Markov
