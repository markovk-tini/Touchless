"""Detect what GPU-accelerated inference paths are reachable on this
machine, without raising or printing on import.

The gesture pipeline can in principle accelerate on the GPU through
two distinct stacks, with very different reliability characteristics
on Windows:

  1. MediaPipe Tasks API HandLandmarker with `BaseOptions.delegate =
     Delegate.GPU`. Same models as our existing `solutions.hands`
     code path, so accuracy is identical when it works. On Windows
     this delegate goes through OpenGL ES / Vulkan and frequently
     falls back to CPU silently — but when it does engage, the
     speedup is 2-3x with zero accuracy risk.

  2. ONNX Runtime with the DirectML execution provider on a
     custom palm-detect + landmark pipeline. Reliable but requires
     hand-rolled preprocessing, anchor decoding, NMS and ROI
     rotation matching MediaPipe's internals. Multi-day effort and
     an extra ~80 MB of provider DLLs in the installer.

This module just answers "is each path importable / instantiable
right now". Actual inference lives elsewhere — the `runtime.py`
loader uses these answers to pick a path or fall back to CPU
MediaPipe. The Settings toggle uses them to decide whether the
GPU Mode checkbox should be enabled or shown disabled with a
tooltip explaining the user has no GPU path available.

The probe NEVER imports `onnxruntime-directml` or constructs a
HandLandmarker eagerly at app startup — too slow + side-effecty.
It only checks "could I import these modules cheaply" + reads
provider lists. Real construction happens lazily when the user
flips GPU Mode on.
"""
from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class GpuProbeResult:
    """Snapshot of which GPU-accelerated inference paths are
    plausibly reachable. `path_summary()` formats it for the
    Settings tooltip when no path is available."""

    mediapipe_tasks_importable: bool
    tasks_gpu_delegate_present: bool   # the enum exists, not necessarily that it works
    onnxruntime_importable: bool
    onnxruntime_directml_provider: bool  # true iff "DmlExecutionProvider" is in get_available_providers()
    onnxruntime_coreml_provider: bool = False  # true iff CoreMLExecutionProvider is listed (macOS)
    onnxruntime_providers: tuple[str, ...] = ()  # full provider list ort returned (or empty if import failed)
    directml_dll_loadable: bool | None = None    # ctypes.WinDLL('DirectML.dll') succeeded; None on non-Windows
    directml_dll_error: str = ""                  # OS error string when WinDLL failed (eg "[WinError 5] Access is denied")
    error_notes: tuple[str, ...] = ()

    @property
    def _tasks_gpu_path_ok(self) -> bool:
        return self.mediapipe_tasks_importable and self.tasks_gpu_delegate_present

    @property
    def _ort_dml_path_ok(self) -> bool:
        # The ONNX/DML branch additionally requires the DirectML.dll
        # WinDLL probe to have succeeded (or to be skipped on
        # non-Windows where directml_dll_loadable is None) — otherwise
        # the provider being listed in get_available_providers() is
        # misleading: session creation will throw at LoadLibrary
        # time, and we'd mis-report "GPU available" while the real
        # engine swap silently falls back to CPU.
        return (
            self.onnxruntime_importable
            and self.onnxruntime_directml_provider
            and (self.directml_dll_loadable is None or self.directml_dll_loadable is True)
        )

    @property
    def _ort_coreml_path_ok(self) -> bool:
        return self.onnxruntime_importable and self.onnxruntime_coreml_provider

    @property
    def has_any_gpu_path(self) -> bool:
        return (
            self._tasks_gpu_path_ok
            or self._ort_dml_path_ok
            or self._ort_coreml_path_ok
        )

    def path_summary(self) -> str:
        """Human-readable one-liner for the Settings tooltip."""
        parts: list[str] = []
        if self._tasks_gpu_path_ok:
            parts.append("MediaPipe Tasks GPU delegate")
        if self._ort_dml_path_ok:
            parts.append("ONNX Runtime + DirectML")
        if self._ort_coreml_path_ok:
            parts.append("ONNX Runtime + CoreML")
        if parts:
            return "GPU paths available: " + ", ".join(parts)
        return self._failure_summary()

    def _failure_summary(self) -> str:
        """Describe WHY no GPU path is reachable in one actionable line.
        Branches on what the probe actually found so the user gets a
        specific hint instead of the generic 'no GPU detected'."""
        if sys.platform == "darwin":
            if not self.onnxruntime_importable:
                return (
                    "No GPU inference path detected — ONNX Runtime failed to load."
                )
            if not self.onnxruntime_coreml_provider:
                providers = ", ".join(self.onnxruntime_providers) or "<none>"
                return (
                    "No GPU inference path detected — CoreML is missing "
                    f"(providers: {providers})."
                )
            return "No GPU inference path detected on this Mac."
        # Case A: ONNX wheel didn't even import. Almost always the
        # bundle is corrupt or AV ate the pyd.
        if not self.onnxruntime_importable:
            return (
                "No GPU inference path detected — ONNX Runtime failed to load. "
                "Most likely your antivirus quarantined a file under "
                "Touchless\\_internal\\onnxruntime\\. Add the Touchless install "
                "folder to your antivirus exclusions and reopen Touchless."
            )
        # Case B: ONNX imported but DirectML.dll fails to load. Defender
        # is the usual culprit; old GPU drivers second.
        if self.directml_dll_loadable is False:
            err = self.directml_dll_error or "unknown OS error"
            return (
                "No GPU inference path detected — DirectML.dll could not be "
                f"loaded ({err}). Add the Touchless install folder to your "
                "antivirus exclusions, then reopen Touchless. If that doesn't "
                "help, update your graphics driver."
            )
        # Case C: DirectML.dll loads but DmlExecutionProvider isn't in
        # the providers list. Means onnxruntime built without DML, or
        # the wrong wheel slipped in. Shouldn't happen in shipped builds
        # but useful diagnostic if a dev install is mis-wired.
        if self.onnxruntime_importable and not self.onnxruntime_directml_provider:
            providers = ", ".join(self.onnxruntime_providers) or "<none>"
            return (
                "No GPU inference path detected — ONNX Runtime loaded but its "
                f"DirectML provider is missing (providers: {providers}). This "
                "build was packaged without onnxruntime-directml; reinstall "
                "Touchless from the official installer."
            )
        # Catch-all (no detail captured). Leaves it generic but honest.
        return "No GPU inference path detected on this machine."

    def diagnostic(self) -> str:
        """Verbose multi-line dump for stderr / a debug panel."""
        providers_txt = ", ".join(self.onnxruntime_providers) if self.onnxruntime_providers else "<none>"
        dml_loadable_txt = (
            "n/a (non-Windows)" if self.directml_dll_loadable is None
            else ("yes" if self.directml_dll_loadable else f"no ({self.directml_dll_error or 'unknown error'})")
        )
        lines = [
            f"  mediapipe.tasks importable      : {self.mediapipe_tasks_importable}",
            f"  tasks GPU delegate enum present : {self.tasks_gpu_delegate_present}",
            f"  onnxruntime importable          : {self.onnxruntime_importable}",
            f"  onnxruntime DML provider listed : {self.onnxruntime_directml_provider}",
            f"  onnxruntime CoreML provider     : {self.onnxruntime_coreml_provider}",
            f"  onnxruntime providers           : {providers_txt}",
            f"  DirectML.dll loadable           : {dml_loadable_txt}",
        ]
        if self.error_notes:
            lines.append("  errors during probe:")
            for note in self.error_notes:
                lines.append(f"    - {note}")
        return "\n".join(lines)


@lru_cache(maxsize=1)
def probe_gpu_paths() -> GpuProbeResult:
    """Cheap detection of GPU-capable inference backends. Cached so
    the Settings UI + the runtime loader can call it cheaply on
    every render. Repeated calls don't re-import or re-list
    providers."""
    errors: list[str] = []

    # 1. MediaPipe Tasks API path. Two-stage check:
    #    (a) the `vision` module + `Delegate.GPU` enum import
    #        cleanly. The enum has shipped in mediapipe for a
    #        while, so this passes on Windows even though the
    #        underlying delegate isn't supported.
    #    (b) try to actually serialize a BaseOptions(delegate=GPU)
    #        to its protobuf. The mediapipe Windows wheel raises
    #        `NotImplementedError("GPU Delegate is not yet
    #        supported for Windows")` here — which is what tells
    #        us the runtime can't actually use the delegate.
    # Without (b) the probe falsely reports "GPU available" on
    # Windows and the Settings toggle confuses the user.
    mp_tasks_importable = False
    tasks_gpu_present = False
    try:
        from mediapipe.tasks.python import vision as mp_vision  # noqa: F401
        from mediapipe.tasks.python.core.base_options import BaseOptions

        mp_tasks_importable = True
        delegate_attr = getattr(BaseOptions, "Delegate", None)
        delegate_gpu = getattr(delegate_attr, "GPU", None) if delegate_attr is not None else None
        if delegate_gpu is not None:
            try:
                # Construction-time validation of GPU delegate
                # support. We pass a deliberately bogus model
                # path because all we want is the to_pb2() call
                # to fire — that's the line that raises on
                # Windows. If it raises NotImplementedError, the
                # delegate isn't usable; if it raises any other
                # error (file not found etc.) the delegate IS
                # supported and we'd just hit the path-not-found
                # later, which is the OK case.
                opts = BaseOptions(model_asset_path="__probe__", delegate=delegate_gpu)
                opts.to_pb2()
                tasks_gpu_present = True
            except NotImplementedError as exc:
                errors.append(f"mediapipe.tasks GPU: {exc!s}"[:160])
            except Exception:
                # Any other exception means the delegate
                # serialised; the failure was downstream
                # (model_asset_path doesn't exist etc.) which is
                # fine for the probe — we only care whether the
                # delegate itself is supported.
                tasks_gpu_present = True
    except Exception as exc:
        errors.append(f"mediapipe.tasks: {type(exc).__name__}: {exc!s}"[:160])

    # macOS: the GPU delegate enum serialises, but ConvertToGpu then
    # CHECK-fails on the camera ImageFrame format. Never report it as
    # a usable path.
    if sys.platform == "darwin":
        tasks_gpu_present = False
        errors.append(
            "mediapipe.tasks GPU: disabled on macOS (CVPixelBuffer abort)"
        )

    # 2. ONNX Runtime + DirectML provider path: importing
    # onnxruntime is cheap; listing available providers tells us
    # whether onnxruntime-directml is installed. The Cloudflare
    # Pages build doesn't include DirectML by default, so this
    # tends to be False on most user machines.
    ort_importable = False
    ort_dml = False
    ort_coreml = False
    providers: list[str] = []
    try:
        import onnxruntime as ort

        ort_importable = True
        try:
            providers = list(ort.get_available_providers())
        except Exception as exc:
            errors.append(f"onnxruntime.get_available_providers: {type(exc).__name__}: {exc!s}"[:160])
            providers = []
        ort_dml = "DmlExecutionProvider" in providers
        ort_coreml = "CoreMLExecutionProvider" in providers
    except Exception as exc:
        errors.append(f"onnxruntime import: {type(exc).__name__}: {exc!s}"[:160])

    # 3. Direct DirectML.dll loadability probe (Windows only).
    # The DML provider being listed in get_available_providers() does
    # NOT prove the actual DirectML.dll will load at session-creation
    # time — Microsoft Defender, restrictive antivirus, or a corrupted
    # bundle can block the LoadLibrary call later, which surfaces as a
    # session-creation exception users have no way to diagnose. By
    # explicitly trying ctypes.WinDLL("DirectML.dll") here we capture
    # the OS error code (5 = access denied / quarantined; 126 =
    # not found; 193 = bad image / arch mismatch) and surface it in
    # the Settings tooltip so the user knows whether to whitelist
    # the install folder vs reinstall vs update their GPU driver.
    # We also try the bundled copy under
    # _internal\onnxruntime\capi\DirectML.dll so a broken global
    # search path still gets the right answer for Touchless's bundle.
    directml_loadable: bool | None = None
    directml_err = ""
    if sys.platform.startswith("win"):
        directml_loadable = False
        candidates: list[str] = ["DirectML.dll"]
        try:
            import onnxruntime as _ort_mod  # noqa: F401 — already imported above on success path

            ort_pkg_dir = Path(_ort_mod.__file__).parent
            bundled = ort_pkg_dir / "capi" / "DirectML.dll"
            if bundled.exists():
                candidates.append(str(bundled))
        except Exception:
            pass
        for candidate in candidates:
            try:
                ctypes.WinDLL(candidate)
                directml_loadable = True
                break
            except OSError as exc:
                directml_err = f"{candidate}: {exc}"[:200]
            except Exception as exc:
                directml_err = f"{candidate}: {type(exc).__name__}: {exc}"[:200]

    return GpuProbeResult(
        mediapipe_tasks_importable=mp_tasks_importable,
        tasks_gpu_delegate_present=tasks_gpu_present,
        onnxruntime_importable=ort_importable,
        onnxruntime_directml_provider=ort_dml,
        onnxruntime_coreml_provider=ort_coreml,
        onnxruntime_providers=tuple(providers),
        directml_dll_loadable=directml_loadable,
        directml_dll_error=directml_err,
        error_notes=tuple(errors),
    )

# Author: Konstantin Markov
