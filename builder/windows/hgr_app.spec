# Touchless PyInstaller spec for Windows
# Place this file at builder/windows/hgr_app.spec and run from the repo root.

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_dynamic_libs, collect_data_files

ROOT = Path.cwd()
SRC = ROOT / "src"
ASSETS = ROOT / "assets"
GESTURE_GUIDE = ROOT / "GestureGuide"
WHISPER_BUNDLES = [ROOT / "whisper.cpp", ROOT / "whisper_bundle"]
LLAMA_ROOT = ROOT / "llama.cpp"
ICON = ASSETS / "icons" / "touchless_icon.ico"

datas = []
binaries = []
hiddenimports = []

for package_name in ("PySide6", "shiboken6", "mediapipe"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# WebRTC phone camera (Settings -> Camera -> "Connect Phone") decodes the
# stream in a hidden Chromium page via QtWebEngine. That ships as part of
# PySide6 (collected above), but QtWebEngine needs its runtime side-cars
# bundled too: QtWebEngineProcess.exe, the ICU data, locales, and the
# resource .pak files. PyInstaller's PySide6 hook normally adds these;
# collect_all('PySide6') above already pulls them. If a future PySide6
# refactor splits QtWebEngine into PySide6-Addons as a separate import
# name, collect it explicitly here too (skip silently if absent).
for package_name in ("PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineCore"):
    try:
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hiddenimports
    except Exception:
        pass


# onnxruntime-directml: ships native DLLs (DirectML.dll, the DML
# execution provider, the providers_shared shim, plus a few
# Microsoft.AI.MachineLearning runtime files). PyInstaller's
# automatic detection misses some of these because they're loaded
# via LoadLibrary from C++. collect_all picks them up reliably and
# also includes the pure-Python `onnxruntime.capi`,
# `onnxruntime.providers` etc. submodules the runtime touches when
# initialising a DML session. Skip silently if the package isn't
# installed (some dev machines build CPU-only); the runtime
# already falls back to MediaPipe CPU when DML isn't reachable, so
# a missing wheel here just means GPU Mode is a no-op on that build.
try:
    ort_datas, ort_binaries, ort_hidden = collect_all("onnxruntime")
    datas += ort_datas
    binaries += ort_binaries
    hiddenimports += ort_hidden
except Exception:
    pass

# The app uses several dynamic imports and optional Windows-only controllers.
hiddenimports += collect_submodules("hgr")
hiddenimports += [
    "cv2",
    "numpy",
    "PIL",
    "psutil",
    "keyboard",
    "sounddevice",
    "comtypes",
    "pycaw",
    # WASAPI loopback bridge for clip-cache system-audio capture.
    # See src/hgr/app/ui/wasapi_loopback.py — no released ffmpeg
    # has a `wasapi` indev, so we capture render-endpoint output
    # in Python and pipe raw PCM into ffmpeg's stdin.
    "pyaudiowpatch",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    # Google Picker dialog (app/ui/google_picker_dialog.py) hosts the
    # Picker widget inside a QWebEngineView and bridges the PICKED
    # file_id back to Python via QWebChannel. QtWebEngineWidgets is
    # already collected by the collect_all("PySide6.QtWebEngineWidgets")
    # block above, but the WebChannel module is separate — listing it
    # here ensures the frozen bundle can register the pickerBridge
    # object even if collect_all misses it on a stripped PySide6 build.
    "PySide6.QtWebChannel",
    # ---- Iris ambient-tools deps (zero-setup for end users) ----------
    # Each is optional at runtime (graceful degrade), but we want them
    # PRESENT in the build so a shipped user gets toast / per-app volume
    # / SMTC media / UIA selection / clipboard WITHOUT any pip install.
    # Listed here even though Iris itself is excluded today; ready for
    # when the exclude flips. collect_all() below pulls submodules too.
    "uiautomation",
    "win11toast",
    "winsdk",
    "winsdk.windows.ui.notifications",
    "winsdk.windows.media.control",
    "winsdk.windows.data.xml.dom",
    "winrt",
    "win32clipboard",
    "win32process",
    "win32api",
    "win32con",
    # Outlook desktop COM bridge — free email reading without any
    # OAuth / API verification fees. See
    # src/hgr/live_api/connectors/outlook_com_connector.py.
    "win32com",
    "win32com.client",
    "pythoncom",
    "pywintypes",
    "mcp",  # MCP bridge core SDK
    "mcp.client",
    "mcp.client.stdio",
    "httplib2",
    "google_auth_httplib2",
    # Phase-1 trust substrate: Recycle Bin routing for delete_file so
    # destructive ops are RECOVERABLE. Required by safety_gate.
    "send2trash",
]
# Pull every submodule of the listed optional deps so dynamic / lazy
# imports inside them work in a frozen build (winsdk in particular has
# a huge surface area of generated submodules).
for _opt in ("winsdk", "winrt", "uiautomation", "win11toast", "mcp", "pycaw", "comtypes", "pyaudiowpatch"):
    try:
        hiddenimports += collect_submodules(_opt)
    except Exception:
        # Missing on this dev machine = the build channel will pick it
        # up later. Don't fail the spec.
        pass
hiddenimports = list(dict.fromkeys(hiddenimports))

# PyAudioWPatch bundles its own PortAudio DLL (libportaudio64bit.dll
# and a _portaudio*.pyd extension module) under the wheel. Hidden-
# import alone is not enough — PyInstaller's automatic dynamic-lib
# detection misses the package-relative DLLs, so explicitly collect
# both binaries and data files. Without this the frozen app's
# `import pyaudiowpatch` works on the dev box (because pip site-
# packages is on sys.path) but fails on a clean install with a
# cryptic "_portaudio not found" message — and our WASAPI loopback
# bridge silently degrades to "system audio skipped" with no UI
# feedback. Wrapped in try/except matching the existing
# collect_submodules pattern above: if pyaudiowpatch isn't installed
# on this build machine, the build still succeeds and the runtime
# falls back to no-system-audio.
try:
    binaries += collect_dynamic_libs("pyaudiowpatch")
except Exception:
    pass
try:
    datas += collect_data_files("pyaudiowpatch")
except Exception:
    pass

for source_path, target_name in (
    (ASSETS, "assets"),
    (GESTURE_GUIDE, "GestureGuide"),
    # NOTE: the Iris Cortex web assets (vendored Three.js + HTML/JS) used
    # to be bundled here, but Iris is excluded from shipping builds (see
    # `excludes` below). Including the static demo files alongside the
    # excluded Python code is dead weight at best and "looks like an
    # unshipped feature leaked into the build" at worst. Re-add the
    # tuple below if/when Iris ships:
    #     (SRC / "hgr" / "live_api" / "cortex" / "web", "hgr/live_api/cortex/web"),
):
    if source_path.exists():
        datas.append((str(source_path), target_name))

# Build-channel marker. Written to the bundle root so the runtime
# helper hgr.utils.runtime_paths.build_channel() can read it. The
# value comes from the TOUCHLESS_BUILD_CHANNEL env var that
# build_windows.bat sets ('store' when STORE=1, 'website' otherwise).
# 'store'   -> in-app GitHub auto-updater stays OFF (the Microsoft
#              Store delivers updates).
# 'website' -> GitHub auto-updater is the update path (default).
import os  # noqa: E402 — late import is fine in a PyInstaller spec
_channel = os.environ.get("TOUCHLESS_BUILD_CHANNEL", "website").strip().lower()
if _channel != "store":
    _channel = "website"
_channel_marker = ROOT / "build" / "build_channel.txt"
_channel_marker.parent.mkdir(parents=True, exist_ok=True)
_channel_marker.write_text(_channel, encoding="utf-8")
datas.append((str(_channel_marker), "."))

# Bundle ffmpeg.exe alongside Touchless.exe so the camera fallback
# path can use it. Why we need ffmpeg in the bundle: cv2.VideoCapture
# under DirectShow constructs a filter graph in-process — and a
# buggy third-party filter (Canon EOS Webcam Utility's filter has a
# documented segfault) takes the whole Touchless process down. The
# ffmpeg subprocess approach (`ffmpeg -f dshow -i video=<name>`)
# instantiates the same DirectShow graph in a CHILD process, so a
# crash there only kills ffmpeg.exe, not Touchless. Used as a safe
# fallback when the in-process MSMF and DSHOW paths can't open the
# camera (e.g., older EOS Webcam Utility versions that only register
# a DirectShow filter, no Media Foundation Frame Source).
#
# Source: locates ffmpeg via shutil.which("ffmpeg"), which honours
# PATH on the build machine. The dev currently has the gyan.dev
# static build at C:\Users\<...>\ffmpeg-7.0.2-full_build\bin\. Static
# build is self-contained — no DLLs to bundle alongside it.
import shutil  # noqa: E402 — late import is fine in PyInstaller spec
_ffmpeg_src = shutil.which("ffmpeg")
if _ffmpeg_src and Path(_ffmpeg_src).exists():
    binaries.append((_ffmpeg_src, "."))
else:
    raise RuntimeError(
        "ffmpeg.exe not found on PATH. Install ffmpeg (e.g. the "
        "gyan.dev static build) and add its bin/ directory to PATH "
        "before building Touchless. The bundled ffmpeg is required "
        "for the EOS Webcam Utility fallback capture path."
    )


def _collect_whisper_runtime(roots):
    """Bundle only the whisper runtime files the app actually uses.

    Looks under each provided root (whisper.cpp and whisper_bundle), accepts
    binaries at either `build/bin/Release/` (MSBuild layout) or `build/bin/`
    (flat layout), and always maps them into the canonical
    `whisper.cpp/<build>/bin/Release/` output path so the runtime finder works
    the same way in the packaged app regardless of dev-side layout.

    Dropping the CMake build scaffolding keeps the installer under the
    Windows MAX_PATH limit that Inno Setup enforces on every compressed path.
    """
    keep_ext = {".exe", ".dll", ".bin", ".pdb"}
    # Whisper model filter: ship medium.en (best quality) + small.en
    # (fallback). The runtime resolver in live_api/local_backend.py
    # prefers medium when present, falls back to small if medium
    # ever goes missing — so users get the best dictation accuracy
    # by default with a safety net.
    # - medium.en (~1.4 GB) → primary dictation model (better accuracy
    #   on tricky words, unusual names, noisy environments).
    # - small.en (~465 MB) → kept as fallback (handles voice commands
    #   and clean dictation almost as well; never want to be without
    #   any model if medium fails to load for any reason).
    # - base/tiny variants stay excluded (small handles every realistic
    #   short-transcript better than them).
    # - for-tests-*.bin are repository test fixtures, not runtime files.
    #
    # STORE builds ship ONLY small.en. The Microsoft Store caps EXE/MSI
    # package size, and the ~1.5 GB medium.en model pushes the monolithic
    # installer past it. The slim Store build (~1.1 GB) is fully
    # functional on small.en; users who want higher dictation accuracy
    # pull medium.en at runtime via the in-app "Voice Recognition
    # Upgrade" download (optional DLC — does NOT block app use, so it's
    # Store-policy compliant, unlike an install-time downloader).
    # Channel is read from TOUCHLESS_BUILD_CHANNEL (set by
    # build_windows.bat: 'store' when STORE=1, else 'website').
    # v1.1.7.6 (dad rig 2026-08-21): removed medium.en from the website
    # allowlist. Shipping the 1.5 GB medium.en model in the installer
    # bloated the payload from ~1.75 GB (1.1.7) to ~3.16 GB (1.1.7.5),
    # and Windows Defender flagged the unsigned 1.5 GB binary blob during
    # extraction — dad's install failed 3× with "failed to extract"
    # errors. Behavior now matches the Store build: ship only small.en
    # (~490 MB, sufficient for the default dictation flow), and let
    # users who want higher accuracy pull medium.en at runtime via the
    # in-app "Voice Recognition Upgrade" download. Same code path the
    # Store build already used; no functional regression.
    _channel_for_models = os.environ.get("TOUCHLESS_BUILD_CHANNEL", "website").strip().lower()
    MODEL_ALLOWLIST = {"ggml-small.en.bin"}
    collected = []
    seen_models: set[str] = set()
    seen_binaries: set[tuple[str, str]] = set()

    for root in roots:
        if not root.exists():
            continue
        models_dir = root / "models"
        if models_dir.exists():
            for model_file in models_dir.glob("*.bin"):
                if model_file.name in seen_models:
                    continue
                if model_file.name not in MODEL_ALLOWLIST:
                    # Skip oversized / test-only whisper models.
                    continue
                seen_models.add(model_file.name)
                collected.append((str(model_file), "whisper.cpp/models"))
        for build_dir_name in ("build", "build_cuda", "build_vulkan", "build_stream"):
            build_bin = root / build_dir_name / "bin"
            if not build_bin.exists():
                continue
            source_dirs = [build_bin / "Release", build_bin]
            target = f"whisper.cpp/{build_dir_name}/bin/Release"
            for source_dir in source_dirs:
                if not source_dir.exists():
                    continue
                for entry in source_dir.iterdir():
                    if not entry.is_file():
                        continue
                    if entry.suffix.lower() not in keep_ext:
                        continue
                    key = (build_dir_name, entry.name)
                    if key in seen_binaries:
                        continue
                    seen_binaries.add(key)
                    collected.append((str(entry), target))
    return collected


datas += _collect_whisper_runtime(WHISPER_BUNDLES)


def _collect_llama_runtime(llama_root):
    """Bundle the llama.cpp runtime binaries used by the local Live API agent.

    The Live API local backend (`src/hgr/live_api/local_backend.py`) and the
    grammar corrector (`src/hgr/voice/llama_server.py`) both spawn
    `llama-server.exe` from `llama.cpp/build_<backend>/bin/`. We mirror the
    whisper bundling strategy: only ship the runtime files (.exe / .dll /
    .bin), not the CMake scaffolding, and preserve the canonical layout
    so the existing discovery code doesn't need a packaged-vs-source switch.

    GGUF model files are intentionally NOT bundled — they're 3-5 GB each
    which would balloon the installer past the Inno Setup MAX_PATH limits
    and double the download. Models live in
    `~/Documents/TouchlessVoiceModels/` and the user downloads them once.

    Future Phase 2 (vision) will use `llama-mtmd-cli.exe` /
    `llama-qwen2vl-cli.exe` from the same bin/ dir, which is why we copy
    every .exe rather than just llama-server.exe.
    """
    if not llama_root.exists():
        return []
    keep_ext = {".exe", ".dll"}
    collected = []
    seen: set[tuple[str, str]] = set()
    for build_dir_name in ("build_cuda", "build_vulkan", "build_cpu", "build"):
        bin_dir = llama_root / build_dir_name / "bin"
        if not bin_dir.exists():
            continue
        # MSBuild emits to bin/Release; CMake to bin directly. Accept both
        # and remap to the canonical bin/ layout the discovery code uses.
        for source_dir in (bin_dir / "Release", bin_dir):
            if not source_dir.exists():
                continue
            target = f"llama.cpp/{build_dir_name}/bin"
            for entry in source_dir.iterdir():
                if not entry.is_file():
                    continue
                if entry.suffix.lower() not in keep_ext:
                    continue
                key = (build_dir_name, entry.name)
                if key in seen:
                    continue
                seen.add(key)
                collected.append((str(entry), target))
    return collected


datas += _collect_llama_runtime(LLAMA_ROOT)

a = Analysis(
    [str(ROOT / "run_app.py")],
    pathex=[str(ROOT), str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Touchless Assistant ("Iris" / Live API agent) is NOT shipped yet — keep it
    # OUT of the bundle entirely (the source tree keeps it for dev; the UI entry
    # point is also dev-gated). Excluding the package means its code AND its
    # exclusive deps (rapidocr, websocket-client, etc.) are pruned, since
    # nothing else in the app reaches them. The only importer is a lazy,
    # env-gated import in main_window, so excluding these can't break startup.
    excludes=[
        "PyQt5", "PyQt6", "PySide2",
        "hgr.live_api",
        "hgr.app.ui.live_assistant_window",
    ],
    noarchive=False,
)

# --- Never ship Windows' own runtime libraries (1.1.9 stop-ship) ------------
# PyInstaller resolves each binary's DLL imports by searching the build
# machine's PATH. Whatever it finds gets copied into `_internal/`, which is on
# the frozen app's DLL search path AHEAD of System32 — so a stray build-machine
# DLL silently shadows the system one for every user.
#
# That is exactly how 1.1.9 shipped broken. PySide6 6.10+ made Qt6Core.dll a
# hard (non-delay-load) importer of `icuuc.dll`, and the PySide6 wheel ships no
# ICU at all — upstream expects Windows' own ICU (System32, Windows 10 1703+).
# The 1.1.9 build ran from a shell where Anaconda's `Library\bin` was reachable,
# so PyInstaller bundled conda's ICU 73. That build exports version-SUFFIXED
# symbols (`ucnv_open_73`), while Qt imports the plain names (`ucnv_open`), so
# every launch died with:
#     ImportError: DLL load failed while importing QtGui:
#     The specified procedure could not be found.
# 1.1.8.1 was built without conda on PATH, bundled no ICU, and worked — the Qt
# binaries are byte-identical between the two releases, so the ONLY difference
# was the build environment. Bundling ICU is also wrong even when the symbols
# match: a newer Windows `icuuc.dll` is a stub that forwards to `icu.dll`, which
# doesn't exist on older Windows, so copying it breaks those machines too.
#
# `ucrtbase.dll` + the `api-ms-win-*.dll` stubs are the same class of bug. They
# shipped in earlier releases without an obvious failure, but they load a SECOND
# C runtime alongside System32's (both were confirmed mapped into the running
# process), giving the process two CRT heaps and two locale states — a known
# source of `0xc0000409` __fastfail aborts. Windows 10+ always provides these,
# so drop them and use exactly one system CRT.
#
# Filtering here rather than sanitising PATH in build_windows.bat keeps the
# guarantee attached to the build definition, so a build started from any shell
# (conda-activated or not) produces the same bundle.
def _is_system_runtime_dll(dest_path: str) -> bool:
    name = Path(dest_path).name.lower()
    if name.startswith("api-ms-win-") and name.endswith(".dll"):
        return True
    if name == "ucrtbase.dll":
        return True
    # icuuc.dll / icuin.dll / icudt73.dll / icu.dll and their versioned names.
    if name.endswith(".dll") and name.startswith(("icuuc", "icuin", "icudt", "icu.")):
        return True
    return False


_stripped_system_dlls = sorted(
    Path(entry[0]).name for entry in a.binaries if _is_system_runtime_dll(entry[0])
)
a.binaries = [entry for entry in a.binaries if not _is_system_runtime_dll(entry[0])]
print(
    f"[spec] stripped {len(_stripped_system_dlls)} build-machine system DLL(s) "
    f"so the app uses Windows' own copies: "
    + ", ".join(_stripped_system_dlls[:6])
    + (" ..." if len(_stripped_system_dlls) > 6 else "")
)
for _required_system_dll in ("icuuc.dll", "ucrtbase.dll"):
    if any(
        Path(entry[0]).name.lower() == _required_system_dll for entry in a.binaries
    ):
        raise RuntimeError(
            f"{_required_system_dll} is still in the bundle — the system-DLL "
            "filter above did not catch it. Shipping it will break startup."
        )

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Touchless",
    console=False,
    icon=str(ICON) if ICON.exists() else None,
    disable_windowed_traceback=False,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="Touchless",
    upx=False,
)

# Author: Konstantin Markov
