# Touchless — Windows → macOS Portability Audit

> **Status:** Authoritative source of truth for the macOS port. Consolidated from
> 15 subsystem/cross-cutting audits. Apple Silicon (arm64) is the target.
> Last consolidated: 2026-06-21.

## Executive summary

Touchless is **substantially portable to macOS** — far more than a naive read of
the Windows-heavy git status would suggest. The product is PySide6 + a pure-compute
OpenCV/MediaPipe/numpy pipeline, and the bulk of the codebase (gesture
classification, tracking, feature extraction, smoothing, overlays' Qt layer,
custom-gesture training, voice text normalization, command parsing, telemetry,
config, all cloud connectors, the Iris orchestration core) runs on Apple Silicon
**with zero or trivial changes**. The codebase already follows a disciplined
`platform.system()` / `sys.platform` branch pattern, several modules already ship
working `Darwin` branches (notably `native_overlay._apply_macos_overlay`,
`camera_utils` AVFoundation, `system_actions`, `subprocess_utils.launch_external`,
`kicad_cli_connector`), and almost nothing is *truly* impossible.

The work concentrates in a thin but critical **platform-control layer** plus a
**brand-new packaging/distribution pipeline**. The 3–4 biggest blockers are:

1. **OS input synthesis & cursor control** — every `SetCursorPos` / `mouse_event` /
   `SendInput` / `keybd_event` path (the core gesture→cursor driver, custom-gesture
   keystrokes, Iris/voice text injection, media keys) must move to Quartz
   `CGEvent*` + `CGWarpMouseCursorPosition`, gated behind the **Accessibility** and
   **Input Monitoring** TCC permissions that macOS will *not* grant programmatically.
2. **Window introspection & control of *other* apps** — `EnumWindows` / `GetWindowTextW`
   / `SetWindowPos` / UIAutomation → CGWindowList + Accessibility `AXUIElement`. Window
   *titles* of foreign apps are redacted without **Screen Recording**, and control needs
   **Accessibility**. `uia_controller.py` is the single biggest source rewrite.
3. **Native GPU/ML binaries** — bundled CUDA/Vulkan/DirectML whisper.cpp + llama.cpp +
   `onnxruntime-directml` do not exist on Apple Silicon and must be rebuilt with **Metal**
   (`-DGGML_METAL=ON`) / swapped for `onnxruntime` CoreML EP.
4. **Distribution** — Inno Setup `.exe`, Azure Trusted Signing, and the `.bat`+robocopy+registry
   auto-updater have **no macOS survivors**; a brand-new `.app`/`.pkg`/`.dmg` + codesign +
   notarize + Sparkle pipeline is required (without modifying any Windows build file).

**Genuinely not-possible on macOS:** per-application audio volume/metering & the
ducker (`pycaw IAudioSessionManager2` has no public CoreAudio equivalent), Classic
Outlook COM/MAPI automation, Microsoft Phone Link SMS/iMessage bridge, and the
Microsoft Store update channel. Each has a documented fallback or replacement
backend.

**Rough effort:** Core port (camera + tracking + cursor + overlay + basic
controllers running on a Mac) is a few focused weeks given hardware. Full feature
parity including Iris desktop-control (UIA rewrite), the Metal native rebuilds, and a
notarized auto-updating distribution pipeline is a substantially larger effort.
Many phases **cannot be verified without Mac hardware** (see the porting plan).

---

## Port progress log (code implemented on `feat/macos-port`)

Landed and confirmed working on a Mac (user-verified): camera + MediaPipe
tracking at parity fps (root cause was the instant-clip rolling buffer grabbing
the screen on the GUI thread — moved off-thread via Quartz `CGDisplayCreateImage`),
gesture recognition + gating, hand cursor + clicks, drawing, gesture wheels
(no focus-steal), voice command + dictation (loud & soft), Spotify control
(AppleScript playback + Web-API search/library + connect pill), foreign-window
control (close/min/max via AX), media keys, screenshot + countdown overlay,
instant-clip export, and screen recording (video).

Landed in code, inert until the Mac packaging pipeline exists (no Mac build yet):

- **Single-instance lock** — `fcntl.flock` on `~/Library/Application Support/
  Touchless/touchless.lock` (`single_instance._acquire_mac`).
- **Auto-start on login** — per-user LaunchAgent plist + `launchctl`
  (`autostart._set_enabled_mac`).
- **Metal-aware native finders** — whisper/llama resolvers now search
  `build_metal/` + extensionless Mach-O names and prefer the Metal backend on
  macOS (`whisper_stream`, `whisper_refiner`, `llama_server`, `local_backend`).
- **Auto-update apply path** — `release_checker` matches macOS assets
  (`Touchless.pkg`, `Touchless_Mac_Update_<ver>.zip`) with distinct SHA markers;
  `updater` writes a detached `_apply_update.sh` that `ditto`-swaps the `.app`
  (rollback on failure) + clears quarantine + relaunches; `.pkg` path hands off
  to the GUI Installer via `open`. SHA-256 verification shared; Authenticode
  no-ops off Windows.

Still open: recording **mic audio** (needs ffmpeg present — avfoundation path
staged but gated), first-run **permission onboarding wizard**, **system audio**
capture (BlackHole/ScreenCaptureKit), and the full **.pkg packaging + Metal
binary builds + codesign/notarize** pipeline (needs Mac + Apple Developer creds).

---

## How to read this

Every feature below carries one of four verdicts:

| Verdict | Emoji | Meaning |
|---|---|---|
| `works_asis` | ✅ | Runs on macOS Apple Silicon with **no code change** (at most a dependency-resolution or packaging detail). Pure-compute, stdlib, cross-platform wheels, or code that already has a clean `Darwin` branch / safe non-Windows no-op. |
| `needs_adaptation` | 🟡 | A **localized, well-understood swap** of a Windows API for a known macOS equivalent (pyobjc/Quartz/CoreAudio/NSWorkspace/`open`). The surrounding logic is reusable; difficulty ranges trivial→high but the path is clear. |
| `hard_possible` | 🟠 | Achievable but a **substantial rewrite to a different object model**, with reduced fidelity vs Windows, and/or gated behind hard TCC permissions. Plan as a from-scratch reimplementation, not a translation. |
| `not_possible` | 🔴 | **No macOS equivalent** of the mechanism or the underlying capability. Either excluded from the macOS build or replaced by a different feature with a different backend (which we count as a new feature, not a port). |

TCC = Apple's **Transparency, Consent & Control** permission system. Where a feature
needs a permission, it is named (Camera, Microphone, Screen Recording, Accessibility,
Input Monitoring, Automation, Full Disk Access, Speech Recognition).

---

## Compatibility matrix

Every feature from every report, sorted by subsystem.

| Subsystem | Feature | Windows impl | Verdict | macOS approach | Difficulty | TCC permission |
|---|---|---|---|---|---|---|
| app-core | App bootstrap / Qt startup / debug-log tee / icon | `SetCurrentProcessExplicitAppUserModelID` (try/except) | 🟡 | Drop AppUserModelID; .app `Info.plist` CFBundleIdentifier; `.icns` icon | trivial | — |
| app-core | Single-instance lock + action forwarding | `CreateMutexW` + `RegisterWindowMessageW`/`FindWindowW`/`PostMessageW` | 🟡 | `fcntl.flock` lockfile / AF_UNIX socket; URL-scheme handoff or distributed notification | medium | — |
| app-core | Taskbar Jump List (Pause/Settings/Quit) | comtypes `ICustomDestinationList`/`IShellLinkW` | 🟡 | `NSApplicationDelegate applicationDockMenu:` (direct callback, no IPC) | medium | — |
| app-core | Camera enumeration (device list + EOS safeguards) | `CAP_DSHOW`/`CAP_MSMF`, QMediaDevices | ✅ | Already branches to `CAP_AVFOUNDATION` on Darwin; first open prompts Camera TCC | trivial | Camera |
| app-core | Camera capture pipeline (threaded cv2 reader) | cv2.VideoCapture + numpy + threading | ✅ | OS-agnostic; AVFoundation-backed cv2 transparently | trivial | Camera |
| app-core | FFmpeg MJPEG high-FPS capture (Lite Mode) | `ffmpeg -f dshow` + pipe | 🟡 | `ffmpeg -f avfoundation`; optional perf path (plain capture already works) | medium | Camera |
| app-core | OS cursor control engine (move/click/drag/scroll) | `SetCursorPos`, `mouse_event`, `GetSystemMetrics` | 🟡 | Quartz `CGWarpMouseCursorPosition` + `CGEventCreateMouseEvent`/`CGEventPost`; mind Y-origin/Retina | high | Accessibility |
| app-core | Foreground/fullscreen window introspection | `GetForegroundWindow`/`EnumWindows`/`GetWindowTextW`/`MonitorFromWindow` | 🟠 | `NSWorkspace.frontmostApplication` (free) but foreign titles need Screen Recording; AX for per-window | high | Screen Recording, Accessibility, Automation |
| app-core | Process priority boost for fullscreen game | `GetCurrentProcess`/`SetPriorityClass` | 🟡 | `os.nice`/QoS; mostly droppable (DWM-throttle is Windows-specific) | low | — |
| app-core | Click-through always-on-top overlays | `SetWindowLongPtr` WS_EX_*; `SetWindowPos` HWND_TOPMOST; DWM | ✅ | `native_overlay._apply_macos_overlay` already implemented (NSWindow level + collectionBehavior) | low | — |
| app-core | System actions: open Chrome/Settings/Finder | `ShellExecuteW` | ✅ | Already has Darwin `open`/`open -a` branches | trivial | — |
| app-core | GitHub auto-updater: download + apply | `ShellExecuteW` + `.bat` (Expand-Archive/robocopy/reg) + Inno /SILENT | 🟡 | Sparkle, or `ditto`-replace `.app` + relaunch via `open`; download path portable | high | — |
| app-core | Update artifact verify (SHA-256 + Authenticode) | hashlib + `WinVerifyTrust`/crypt32 | 🟡 | SHA-256 unchanged; `codesign --verify`/`spctl --assess` keyed to Team ID | medium | — |
| app-core | Microsoft Store update checker | winget msstore / `ms-windows-store://` | 🔴 | No MS Store on macOS; exclude from build (GitHub checker handles direct builds) | trivial | — |
| app-core | Subprocess launch helpers (no-console-flash) | `ShellExecuteW`; `CREATE_NO_WINDOW`/`SW_HIDE` | ✅ | Already cross-platform (`open`; returns `{}` off-Windows) | trivial | — |
| app-ui | Click-through always-on-top overlay (wheels/drawing) | WS_EX_LAYERED/NOACTIVATE/TOOLWINDOW + HWND_TOPMOST | ✅ | macOS branch ships (NonactivatingPanel + setLevel + collectionBehavior); Qt `WA_TransparentForMouseEvents`→`setIgnoresMouseEvents_` | low | — |
| app-ui | Frameless title bar + native edge-drag resize | `WM_NCHITTEST` via `nativeEvent` | 🟡 | Replace resize with `QWindow.startSystemResize(Qt.Edge)` (cross-platform); frameless+painted buttons port | medium | — |
| app-ui | DWM caption/text color tint on dialogs | `DwmSetWindowAttribute` CAPTION/TEXT color | 🟡 | NSWindow `titlebarAppearsTransparent` + `backgroundColor`, or frameless+Qt paint | low | — |
| app-ui | Win11 DWM corner/border/non-client tweaks | `DwmSetWindowAttribute` corner/border/NC | ✅ | Borderless NSWindow has no DWM artifacts; gated to Windows, no-ops | trivial | — |
| app-ui | Topmost watchdog (mini viewer + Iris window) | `SetWindowPos` HWND_TOPMOST + 2s re-pin | 🟡 | `NSWindow.setLevel_` + collectionBehavior; route through shared helper | low | — |
| app-ui | System tray icon (state border + menu + pause) | `QSystemTrayIcon`/`QMenu` (Shell_NotifyIcon) | ✅ | Maps to `NSStatusItem`; menu-bar icons ideally template images (optional) | trivial | — |
| app-ui | Animated startup splash (masked text wave) | Qt frameless translucent + DWM tweak | ✅ | Pure Qt renders via NSWindow; DWM no-ops | trivial | — |
| app-ui | Drawing-file lookup via Windows Search Index | `ADODB.Connection` Search.CollatorDSO / SystemIndex | 🟡 | Spotlight `mdfind` / `NSMetadataQuery`; os.walk fallback already cross-platform | low | — |
| app-ui | Taskbar jump-list action routing | `nativeEvent` + `RegisterWindowMessage` IDs | 🟠 | Dock menu `applicationDockMenu:`; no inter-process window-message analog | medium | — |
| app-ui | Restart-as-administrator (elevated relaunch) | `ShellExecuteW 'runas'` / `IsUserAnAdmin` | 🟡 | `osascript … with administrator privileges` / Authorization Services; or hide on macOS | low | — |
| app-ui | Multi-monitor virtual-screen spanning + DPI | `QGuiApplication.screens()` (Qt) | ✅ | Identical on macOS (Retina via devicePixelRatio); combine with CanJoinAllSpaces for overlays | trivial | — |
| app-ui | Pure-Qt widgets (settings/tutorial/debugger/wizards/pickers) | PySide6 only | ✅ | Qt6 ships arm64 wheels; maps to AppKit | trivial | — |
| core-pipeline | MediaPipe Hands runtime + tracking | mediapipe.solutions.hands + cv2 + draw_landmarks | ✅ | Same arm64 wheels; CPU/XNNPACK inference identical | trivial | — |
| core-pipeline | Landmark + gesture stability smoothing | numpy EMA / confidence gating | ✅ | Pure numpy; identical | trivial | — |
| core-pipeline | Static classifiers + features + geometry | numpy/math | ✅ | Pure compute | trivial | — |
| core-pipeline | Dynamic motion classifiers + scaffold | numpy + `time.monotonic` + deque | ✅ | Cross-platform | trivial | — |
| core-pipeline | GestureBackend pipeline orchestration | numpy/dataclass glue | ✅ | Cross-platform glue | trivial | — |
| gesture | GPU path probe + provider selection | `WinDLL('DirectML.dll')`; DmlExecutionProvider; MP `to_pb2()` trick | 🟡 | Gate on `CoreMLExecutionProvider`; drop DirectML.dll probe; relax MP delegate probe | medium | — |
| gesture | GPU hand tracking via ONNX + DirectML | `InferenceSession(providers=['DmlExecutionProvider',…])` | 🟡 | Swap providers to `['CoreMLExecutionProvider','CPUExecutionProvider']`; pre/post-proc verbatim | medium | — |
| gesture | GPU hand tracking via MediaPipe Tasks GPU delegate | `BaseOptions(delegate=GPU)` (dead on Win) | 🟡 | Metal-backed on Apple Silicon — *better* path than Windows | low | — |
| gesture | Hand-runtime selection + CPU fallback | dispatcher over backends | 🟡 | Add `onnx-coreml` to enum; CPU fallback identical | low | — |
| gesture | Hand detector pipeline (de-dup/handedness/smoothing) | numpy + cv2 | ✅ | Unchanged; verify upstream cv2.flip selfie-mirror | trivial | — |
| gesture | Face-exclusion filter (BlazeFace worker thread) | mediapipe + threading | ✅ | Portable with arm64 mediapipe | trivial | — |
| gesture | Overlay rendering + pure-compute recognition/analysis | cv2 drawing + numpy | ✅ | Fully portable; bulk of subsystem | trivial | — |
| gesture | Voice status/selection overlay window | `native_overlay.apply_overlay` (user32/dwmapi) | 🟡 | pyobjc NSWindow level/ignoresMouseEvents/collectionBehavior; Qt flags give ~80% | medium | — |
| gesture | Standalone gesture test window (dev harness) | Qt + imported MouseController/ChromeController | 🟡 | Gate Windows-only controller imports; may exclude from initial build | medium | Camera, Accessibility, Input Monitoring |
| custom-gestures | Action: keystroke/hotkey/text | private `WinDLL('user32').SendInput` (VK map, KEYEVENTF_UNICODE) | 🟡 | `CGEventCreateKeyboardEvent`+`CGEventPost`; `CGEventKeyboardSetUnicodeString`; kVK_* table; map win→Cmd | medium | Accessibility, Input Monitoring |
| custom-gestures | Action: open_file | `os.startfile` | 🟡 | Darwin `subprocess.Popen(['open', …])` already written | trivial | — |
| custom-gestures | Action: open_url | stdlib `webbrowser.open` | ✅ | Cross-platform | trivial | — |
| custom-gestures | Action: run_command | `subprocess shell=True` (cmd.exe) | 🟡 | Works via /bin/sh; saved command strings are shell-specific (UX/docs) | low | Automation |
| custom-gestures | Registry storage + JSON + thumbnails | pathlib/json under `~/.hgr_app` | ✅ | `Path.home()` resolves; optional move to Application Support | trivial | — |
| custom-gestures | Sharing bundle export/import (.tlg zip) | zipfile/io/json | ✅ | Round-trips cross-platform | trivial | — |
| custom-gestures | Static recorder + features + classification | numpy | ✅ | arm64 numpy | trivial | — |
| custom-gestures | Dynamic recording/training/runtime | numpy DTW | ✅ | Only OS touch is `fire_once`→action.py | trivial | — |
| custom-gestures | Drawing-res cache + builtin profiles + description | json/pathlib/numpy | ✅ | Stdlib+numpy | trivial | — |
| custom-gestures | Live runner: private MediaPipe + state machine | cv2/numpy/mediapipe/threading | ✅ | arm64 wheels; downstream `fire_once` only | trivial | — |
| debug-controllers-a | Mouse warp/move/click/drag/scroll | `SetCursorPos`/`mouse_event`/`GetSystemMetrics` | 🟡 | Quartz `CGWarpMouseCursorPosition`/`CGEvent*`; NSScreen bounds; Y-flip | low | Accessibility, Input Monitoring |
| debug-controllers-a | System volume/mute + media key | `pycaw IAudioEndpointVolume`; `keybd_event` VK_VOLUME/MEDIA | 🟡 | CoreAudio `kAudioDevicePropertyVolumeScalar`/Mute; media via `NX_KEYTYPE_PLAY` CGEventPost | medium | Accessibility, Input Monitoring |
| debug-controllers-a | Per-app audio: volume/peak/ducking | `pycaw IAudioSessionManager2`/`ISimpleAudioVolume`/`IAudioMeterInformation` | 🔴 | No public per-process output volume/metering on macOS; ducker can't port with parity | very_high | — |
| debug-controllers-a | Window introspection & control (title/min/max/close) | user32 `EnumWindows`/`ShowWindow`/`PostMessageW` WM_CLOSE | 🟠 | `NSWorkspace` identity; AXUIElement title/min/close; maximize has no equivalent | high | Accessibility, Screen Recording |
| debug-controllers-a | App launch/catalog/search/Settings/mailto/Outlook | `ShellExecuteW`/`winreg` App Paths/Search.CollatorDSO/`ms-settings` | 🟡 | NSWorkspace/LaunchServices/`open`; Info.plist+mdfind catalog; `x-apple.systempreferences`; Outlook select N/A | high | Automation |
| debug-controllers-a | Overlays + pure-compute paint logic | Qt flags + `native_overlay`; cv2/QPainter | 🟡 | Paint runs as-is; swap `apply_overlay` for NSWindow level/ignoresMouseEvents | low | — |
| debug-controllers-b | Chrome launch/focus window | psutil + `EnumWindows`/`SetForegroundWindow`; Popen | 🟡 | `open -a Google Chrome`/NSWorkspace; runningApplications; CGWindowList/AppleScript | medium | Screen Recording, Automation |
| debug-controllers-b | Chrome keyboard shortcuts | `keybd_event` Ctrl/Shift/Alt | 🟡 | Quartz `CGEvent*`; map Ctrl→Cmd, Alt+Left→Cmd+`[` | medium | Accessibility |
| debug-controllers-b | Chrome open URL/search | Popen / `ShellExecuteW` | 🟡 | `open -a Google Chrome url`; URL logic unchanged | low | — |
| debug-controllers-b | YouTube media-key controls | `EnumWindows`+`GetWindowTextW`+`keybd_event` page hotkeys | 🟡 | CGWindowList title + NSRunningApplication activate + CGEvent; Cmd+9; hotkeys identical | medium | Accessibility, Screen Recording |
| debug-controllers-b | YouTube skip-ad template match + click | `GetWindowRect`+`ImageGrab`+`matchTemplate`+`mouse_event` | 🟡 | ScreenCaptureKit capture; cv2 portable; CGEvent click; Retina point/pixel scaling | high | Screen Recording, Accessibility |
| debug-controllers-b | YouTube UIA accessibility actions | PowerShell `System.Windows.Automation` Invoke/Toggle | 🟠 | Rewrite on AXUIElement; Chrome web AX behind force-renderer-accessibility flag, unreliable | very_high | Accessibility, Screen Recording |
| debug-controllers-b | Spotify launch/focus desktop client | `os.startfile`/`winreg` App Paths/`EnumWindows` | 🟡 | `open -a Spotify`/NSWorkspace; runningApplications `com.spotify.client`; drop winreg | medium | Screen Recording, Automation |
| debug-controllers-b | Spotify Web API + OAuth PKCE + playlists | urllib HTTPS + http.server callback | ✅ | urllib/http.server/webbrowser/secrets all cross-platform | trivial | — |
| debug-controllers-b | Discord RPC over IPC pipe + OAuth | named pipe `discord-ipc-N` (os.name nt) | 🟡 | `socket.socket(AF_UNIX)` at `$TMPDIR/discord-ipc-N`; protocol/OAuth identical | medium | Automation |
| debug-controllers-b | MS Office automation (Word/Excel/PPT) | comtypes IDispatch COM | 🟡 | No COM; AppleScript/Apple Events via ScriptingBridge/`osascript`; full rewrite | high | Automation |
| debug-controllers-b | Text input: clipboard paste + Unicode keystroke | `SendInput` KEYEVENTF_UNICODE; clipboard CF_UNICODETEXT; `AttachThreadInput` | 🟡 | `CGEventKeyboardSetUnicodeString`+CGEventPost; NSPasteboard+Cmd+V; no AttachThreadInput analog | medium | Accessibility, Input Monitoring |
| debug-controllers-b | Win+H dictation toggle + Notepad fallback | `SendInput` Win+H; launch notepad.exe | 🟠 | No Win+H analog; rely on own whisper pipeline; open TextEdit as target | high | Accessibility |
| debug-controllers-b | Voice listener: whisper.cpp/faster-whisper | platform gate; sounddevice WASAPI; whisper-cli.exe; CUDA probe | 🟡 | Drop gate; sounddevice CoreAudio; rebuild whisper.cpp arm64 Metal; CUDA→CPU | medium | Microphone |
| debug-controllers-b | Voice listener: SAPI system-speech fallback | PowerShell System.Speech | 🟡 | `SFSpeechRecognizer` (Speech framework via pyobjc) or drop tier | medium | Microphone, Speech Recognition |
| debug-controllers-b | Phone camera HTTPS server + SSE (QR) | aiohttp + ssl | ✅ | Fully portable; LAN may trigger one-time firewall accept | trivial | — |
| debug-controllers-b | Phone camera TLS cert + LAN IP | cryptography x509 + socket | ✅ | Pure crypto+socket | trivial | — |
| debug-controllers-b | Phone camera WebRTC via hidden QtWebEngine | hidden QWebEngineView + aiohttp + numpy | ✅ | QtWebEngine ships arm64 (VideoToolbox); spot-check off-screen flags | low | — |
| voice | Streaming dictation (faster-whisper/CTranslate2) | `WhisperModel` device cuda/cpu | 🟡 | arm64 wheels; CUDA→CPU clean fallback; optional whisper.cpp Metal/mlx | low | Microphone |
| voice | Mic capture (sounddevice InputStream, callback) | PortAudio WASAPI callback | ✅ | CoreAudio host API; already uses safe callback mode | trivial | Microphone |
| voice | SAPI dictation fallback | PowerShell + System.Speech | 🟡 | Self-disables off-Win; optional SFSpeechRecognizer streamer | medium | Microphone, Speech Recognition |
| voice | Backend auto-detect (CUDA→Vulkan→CPU) | `nvidia-smi`/`vulkaninfo`; .exe probe | 🟡 | Add `metal` tier; strip `.exe`; drop GPU probes | low | — |
| voice | whisper-cli refiner subprocess | whisper-cli.exe + CREATE_NO_WINDOW | 🟡 | arm64 `whisper-cli`; creationflags no-op | low | Microphone |
| voice | llama-server grammar-correction host | llama-server.exe + sockets + urllib | 🟡 | Metal llama-server; `-ngl`; drop `.exe` | low | — |
| voice | Grammar/dictation correction scheduler | threading + HTTP | ✅ | Stdlib threading | trivial | — |
| voice | Dictation text normalization | regex | ✅ | Pure compute | trivial | — |
| voice | Insert dictated/sent text into focused app | `SendInput` + clipboard (via text_input_controller) | 🟠 | CGEvent + NSPasteboard + Cmd+V; the one hard TCC wall in voice flow | high | Accessibility, Input Monitoring |
| voice | Voice command parsing + intent ranking | difflib + regex | ✅ | Pure compute; add macOS vocab aliases | trivial | — |
| voice | Voice profile store + training export | json/pathlib | ✅ | `Path.home()` resolves | trivial | — |
| voice | Save-prompt parsing | regex (drive-letter/UNC) | 🟡 | Add POSIX `/…`/`~` path branch | low | — |
| voice | Live dictation backend selection | picks Whisper else SAPI | ✅ | OS-agnostic; Whisper stays available | trivial | — |
| live-api-core | Mic capture for Iris session | sounddevice RawInputStream (PortAudio) | ✅ | CoreAudio AUHAL unchanged | trivial | Microphone |
| live-api-core | TTS / earcon playback | sounddevice RawOutputStream | ✅ | CoreAudio default output | trivial | — |
| live-api-core | Secret vault (encrypted token store) | DPAPI `CryptProtectData` + SQLite | 🟡 | macOS Keychain (`SecItem*`/`keyring`); else plaintext leak | low | — |
| live-api-core | Screen capture for vision context | `ImageGrab.grab(all_screens)` + `EnumDisplayMonitors` | 🟡 | ScreenCaptureKit/`CGDisplayCreateImage`; NSScreen layout; CGWindowList titles | medium | Screen Recording |
| live-api-core | On-screen OCR find/click | ImageGrab + rapidocr + SendInput/SetCursorPos | 🟠 | rapidocr portable; SCK capture; CGEvent input; **Y-flip + Retina** coordinate rework | high | Screen Recording, Accessibility |
| live-api-core | UIA controller (read/click app AX trees) | comtypes `IUIAutomation` Invoke/Value/Toggle/Text | 🟠 | Full AXUIElement rewrite (~1000 lines); biggest single rewrite | very_high | Accessibility |
| live-api-core | Tool executor (clicks/typing/hotkeys/scroll) | MouseController/TextInputController (SendInput) | 🟡 | Quartz CGEvent rewrite; map ctrl→cmd; NSPasteboard | high | Accessibility, Input Monitoring |
| live-api-core | Window management (move/close/max/place) | `ShowWindow`/`SetWindowPos`/`EnumDisplayMonitors` | 🟠 | AX `kAXPosition`/`kAXSize`/`kAXMinimized`; no true maximize; AX-cooperative apps only | high | Accessibility |
| live-api-core | Launch apps / open URLs / open files | `ShellExecuteW` + App Paths registry | 🟡 | NSWorkspace/`open`/`open -a`; rebuild candidate map for .app paths | medium | — |
| live-api-core | System-state signals for interruption gate | winreg ConsentStore/`SHQueryUserNotificationState`/`GetLastInputInfo`/power | 🟠 | psutil signals stay; idle via `CGEventSourceSecondsSinceLastEventType`; battery via IOKit; mic/cam-in-use & DND degrade to UNKNOWN | high | Accessibility |
| live-api-core | Send-to-trash safe delete | send2trash; `SHFileOperationW` fallback | ✅ | send2trash supports macOS Trash; fallback is dead code | trivial | — |
| live-api-core | Consent jurisdiction detection | `GetUserGeoID`/`GetGeoInfoW` | 🟡 | `NSLocale.currentLocale.countryCode`; same classifier | low | — |
| live-api-core | Foreground repo focus watcher | `GetForegroundWindow`/`GetWindowTextW` + IDE regex | 🟡 | NSWorkspace + AX focused-window title; regex reusable | medium | Accessibility |
| live-api-core | Hot prewarm of cold imports | pre-import win32clipboard | 🟡 | Guard/remove win32clipboard; add NSPasteboard import | trivial | — |
| live-api-core | Local backend (bundled whisper.cpp + llama-server) | build_cuda/vulkan/cpu .exe; nvidia-smi | 🟡 | Single Metal build dir; strip `.exe`; drop nvidia-smi (easier than Windows) | medium | Microphone |
| live-api-core | Cloud/compute orchestration core | websockets/urllib/asyncio/sqlite | ✅ | Platform-agnostic; calls into ported control modules | trivial | — |
| live-api-connectors | Connector base class + registry | pure Python ABC | ✅ | Runs identically | trivial | — |
| live-api-connectors | Google connectors (gmail/cal/docs/sheets/slides/drive) | google-api-python-client + run_local_server | ✅ | `webbrowser`+loopback work; verify token path under home | trivial | — |
| live-api-connectors | Microsoft 365 / Graph connectors | MSAL acquire_token_interactive (loopback) | ✅ | MSAL loopback+browser portable; no WAM broker requested | trivial | — |
| live-api-connectors | Spotify connector | shared SpotifyController (Web API) | ✅ | Pure HTTP+OAuth | trivial | — |
| live-api-connectors | Outlook (Graph/draft) connector — compose/send | `os.startfile` + `keybd_event` Ctrl+Enter | 🟡 | `webbrowser`/NSWorkspace; CGEvent send shortcut (Cmd+Return/Cmd+Shift+D per target) | medium | Accessibility, Input Monitoring, Automation |
| live-api-connectors | Outlook desktop COM connector (zero-auth) | win32com `Outlook.Application` MAPI | 🔴 | No COM/MAPI; route to Graph, or partial AppleScript (no Restrict/Sort parity) | very_high | Automation |
| live-api-connectors | Volume connector (system + per-app) | pycaw IAudioEndpointVolume + IAudioSessionManager2 | 🟠 | System volume via CoreAudio/AppleScript; per-app **not_possible** → narrow tool list | high | Automation |
| live-api-connectors | Media connector (global keys + now-playing) | `keybd_event` VK_MEDIA; winsdk SMTC | 🟠 | CGEventPost NX_KEYTYPE_PLAY; now-playing via private MediaRemote.framework | high | Accessibility, Input Monitoring |
| live-api-connectors | Phone Link connector (SMS/iMessage) | AppX UIA + `ms-phone:` + EnumWindows | 🔴 | No Phone Link; rewrite to Messages.app AppleScript + `chat.db` (Full Disk Access) | very_high | Automation, Accessibility, Full Disk Access |
| live-api-connectors | MCP bridge connector | mcp SDK stdio_client + npx default | 🟡 | Subprocess portable; augment PATH for Homebrew (GUI-launched .app has minimal PATH) | low | — |
| live-api-connectors | KiCAD CLI connector | subprocess + per-OS discovery | ✅ | Already mac-aware (`_MAC_LOCATIONS`, darwin branch) | trivial | — |
| live-api-connectors | Shared subprocess/launch helpers | `ShellExecuteW`; CREATE_NO_WINDOW | 🟡 | Add darwin `open` branch; hidden kwargs already no-op | low | — |
| utils-telemetry-config | Hidden subprocess spawning | CREATE_NO_WINDOW + STARTUPINFO | ✅ | Early-returns `{}` off-Windows (no console concept) | trivial | — |
| utils-telemetry-config | Launch external app / URI | `ShellExecuteW` | ✅ | Darwin `open`/`open -a` branch exists (mind arg semantics) | trivial | — |
| utils-telemetry-config | Auto-start on login | winreg HKCU Run key | 🟡 | `~/Library/LaunchAgents/*.plist` + launchctl, or SMAppService | low | — |
| utils-telemetry-config | Runtime/resource path resolution | `sys._MEIPASS`/`sys.frozen` + build_channel.txt | ✅ | PyInstaller sets same on macOS; pathlib neutral | trivial | — |
| utils-telemetry-config | Telemetry event client | urllib + threading + queue | ✅ | Cross-platform; `platform` stamps `darwin` | trivial | — |
| utils-telemetry-config | Stable anonymous install-id | winreg MachineGuid → SHA-256 | 🟡 | IOKit `IOPlatformUUID` → SHA-256; else churns uuid4 (breaks metrics) | low | — |
| utils-telemetry-config | Config load/save + migrations | pathlib/json atomic os.replace | ✅ | Identical on APFS; optional Application Support relocation | trivial | — |
| utils-telemetry-config | Default save locations | `~/Pictures` / `~/Videos` | 🟡 | `~/Pictures` ok; **`~/Videos`→`~/Movies`** (silent $HOME fallback otherwise) | low | — |
| utils-telemetry-config | Single-instance + focus + jumplist forwarding | named mutex + FindWindow/PostMessageW | 🟡 | flock/AF_UNIX + NSApp.activate; jumplist→Dock menu/URL scheme | medium | — |
| utils-telemetry-config | Static registries (bindings/telemetry consts) | pure data | ✅ | Platform-agnostic | trivial | — |
| runtime-paths | PyInstaller/source resource root | `sys._MEIPASS`/parents[3] | ✅ | Identical; .app _MEIPASS handled | trivial | — |
| runtime-paths | App config + settings dir (.touchless) | `Path.home()/.touchless` + legacy migrate | ✅ | Resolves; optional Application Support | trivial | — |
| runtime-paths | Iris/live_api private-data dir (~20 modules) | `LOCALAPPDATA or ~/.local/share` | 🟡 | Central `app_data_dir()`→`~/Library/Application Support/Touchless` | low | — |
| runtime-paths | Model + binary search roots | `Documents/...` + `whisper-stream.exe` | 🟡 | GGUF dir resolves; need Mach-O binaries + strip `.exe` + `bin` not `bin/Release` | medium | — |
| runtime-paths | OAuth token + connector config storage | `Documents/Touchless/...` | ✅ | Resolves; relocate to Library/Keychain for privacy (Documents is iCloud-synced) | low | — |
| runtime-paths | Temp/cache scratch dirs | `tempfile.gettempdir()` + literal TEMP/TMP | ✅ | `gettempdir()`=$TMPDIR; add TMPDIR fallback for outlook literal | trivial | — |
| runtime-paths | Install dir convention + writability probe | `{localappdata}\Programs\Touchless`; exe parent | 🟡 | `/Applications/Touchless.app`; climb to .app root from `Contents/MacOS` | medium | — |
| runtime-paths | Auto-start on login (Run key) | winreg Run key | 🟡 | LaunchAgent plist via plistlib + launchctl | medium | — |
| runtime-paths | In-app auto-updater | `.bat`/robocopy/reg/Authenticode/ShellExecuteW | 🟠 | Sparkle, or `ditto`+codesign/spctl verify + `open` relaunch | high | — |
| runtime-paths | PyInstaller native runtime bundling | whisper/llama Windows build trees + onnxruntime DirectML | 🟡 | Mach-O Metal binaries; onnxruntime CoreML; build_channel neutral | medium | — |
| runtime-paths | Hardcoded `C:\` app/exe locations | literal Program Files/Public Desktop | 🟡 | `/Applications`+`~/Applications`+mdfind; degrade to no-op today | low | — |
| native-deps | pywin32 (win32gui/api/process/com) | Win32 window/process/COM | 🔴 | pyobjc (Cocoa/Quartz/ApplicationServices) + AppleScript per call-site | high | Accessibility, Automation |
| native-deps | comtypes (COM client) | COM/OLE bridge | 🔴 | No COM runtime; UIA→AX, audio→CoreAudio per consumer | high | Accessibility |
| native-deps | pycaw (per-app + system volume) | WASAPI IAudioSessionManager2 | 🟡 | System via CoreAudio; **per-app has no macOS API** | high | Microphone |
| native-deps | onnxruntime-directml | DmlExecutionProvider | 🟡 | plain `onnxruntime` + CoreMLExecutionProvider (same import name — install only one) | low | — |
| native-deps | mediapipe==0.10.21 | x86_64-centric pin | 🟡 | Bump to a version with osx_arm64 wheel, else Rosetta | medium | Camera |
| native-deps | numpy/opencv/Pillow/matplotlib/scipy | cross-platform wheels | ✅ | arm64 wheels | trivial | Camera |
| native-deps | sounddevice (PortAudio) | WASAPI | ✅ | arm64 wheel, CoreAudio backend | trivial | Microphone |
| native-deps | faster-whisper (CTranslate2) | native lib | ✅ | arm64 wheels (CPU/Accelerate) | trivial | — |
| native-deps | rapidocr-onnxruntime + pyclipper/shapely/pyyaml | --no-deps to protect DirectML | ✅ | arm64 wheels; on mac install WITH deps (plain onnxruntime is desired) | trivial | Screen Recording |
| native-deps | pypdf / python-docx | pure Python | ✅ | Identical | trivial | — |
| native-deps | aiohttp | arm64 deps | ✅ | arm64 wheels; firewall prompt on bind | trivial | — |
| native-deps | google-* / httplib2 | pure Python | ✅ | Cross-platform | trivial | — |
| native-deps | mcp | pure Python | ✅ | Installs; adjust node/uvx PATH detection | trivial | — |
| native-deps | msal | pure Python | ✅ | PKCE browser flow works | trivial | — |
| native-deps | websocket-client | pure Python | ✅ | Unchanged | trivial | — |
| native-deps | PySide6 + QtWebEngine | bundled Chromium | 🟡 | universal2/arm64 wheels; Info.plist usage strings or crash on access | medium | Camera, Microphone |
| native-deps | whisper.cpp native build | CUDA/Vulkan + MSVC + vcpkg SDL2 | 🟡 | `-DGGML_METAL=ON` + Xcode clang + Homebrew SDL2; Mach-O | medium | Microphone |
| native-deps | llama.cpp native build | CUDA + MSVC + Ninja | 🟡 | `-DGGML_METAL=ON` + Xcode clang; Mach-O (better parity than CPU) | medium | — |
| packaging | PyInstaller app bundle (.app) | COLLECT one-folder + .ico | 🟡 | NEW `BUNDLE(...)` spec; Info.plist; .icns; drop win32 hiddenimports; arm64 | medium | Camera, Microphone, Screen Recording |
| packaging | Native GPU/ML runtime bundling | CUDA/Vulkan/DirectML trees + nvcc/vcpkg | 🟠 | Rebuild whisper/llama Metal; onnxruntime CoreML; codesign each Mach-O | high | Microphone |
| packaging | Installer (Inno .exe → .pkg) | Inno Setup ISCC + stub downloader | 🟡 | NEW pkgbuild/productbuild/productsign; optional .dmg | medium | — |
| packaging | Code signing & notarization | Azure Trusted Signing + Authenticode | 🟡 | Developer ID + codesign (hardened runtime, inside-out) + notarytool + staple | medium | Camera, Microphone |
| packaging | Auto-update mechanism | GitHub poll + bat/robocopy + WinVerifyTrust | 🟡 | Sparkle (appcast/EdDSA), or Python poll + _apply_update.sh + spctl | medium | — |
| packaging | Microsoft Store submission path | Partner Center EXE-direct | 🔴 | MAS requires App Sandbox (forbids CGEventTap/Accessibility); ship Developer-ID direct | medium | Camera, Microphone, Accessibility, Input Monitoring |
| packaging | macOS test/run script | run_test.py + build_windows.bat | ✅ | run_test.py neutral; NEW run_mac.sh + build_mac.sh | low | — |
| packaging | CI pipeline | manual local Windows build | 🟡 | NEW GitHub Actions macos-14 runner; import certs; notarize; upload R2 | medium | — |
| permissions | Accessibility cursor/keyboard injection | SetCursorPos/SendInput/keybd_event | 🟡 | CGEventPost + CGWarp; grant resets on signature change | high | Accessibility |
| permissions | Read window titles / enumerate windows | GetWindowTextW/EnumWindows | 🟠 | AXUIElement kAXTitle / CGWindowList (name redacted on macOS 14 without SR) | high | Screen Recording, Accessibility |
| permissions | Screen capture for OCR/vision/overlay backdrop | ImageGrab all screens | 🟡 | ScreenCaptureKit/CGDisplayCreateImage; black frames without SR | medium | Screen Recording |
| permissions | Control other apps' UI trees (UIA auto-approve) | comtypes UIAutomationCore | 🟠 | AXUIElement; Electron AX sparse — keep OCR-click fallback | high | Accessibility |
| permissions | Camera for gesture pipeline | cv2 CAP_DSHOW/MSMF | 🟡 | CAP_AVFOUNDATION + NSCameraUsageDescription + one-time prompt | medium | Camera |
| permissions | Microphone for voice/Iris | sounddevice WASAPI | 🟡 | CoreAudio + NSMicrophoneUsageDescription | medium | Microphone |
| permissions | Automation (Chrome/Spotify/Discord/Office/volume) | COM/keybd_event/pycaw/Web API | 🟠 | Spotify+Discord port; Office COM has no port→AppleScript; volume CoreAudio | high | Automation, Accessibility |
| permissions | Input Monitoring (global hotkey/media/autostart) | RegisterHotKey/keybd_event/Run key | 🟡 | Carbon `RegisterEventHotKey` (avoids Input Monitoring) over CGEventTap; LaunchAgent | medium | Input Monitoring |
| permissions | Always-on-top click-through overlay | WS_EX_LAYERED/TRANSPARENT/HWND_TOPMOST | 🟡 | AppKit setLevel/NonactivatingPanel + **add setIgnoresMouseEvents**; can't float above menu bar/secure fullscreen | medium | — |
| permissions | First-run permission onboarding UX | none needed on Windows | 🟠 | NEW wizard; prompt at first use; deep-link on denial; recheck each launch | high | Accessibility, Camera, Microphone, Screen Recording, Automation, Input Monitoring |

---

## macOS permissions model (TCC)

macOS gates capabilities through TCC. Unlike Windows (which mostly synthesizes input
silently), **every** sensitive capability prompts the user, several **cannot** be
granted programmatically, and grants are **bound to the app's code signature** — so a
signature change (or an unstable/ad-hoc signature) **resets all grants**. A stable
**Developer ID** signature is therefore essential, and onboarding must **re-verify
permissions on every launch** and degrade gracefully when denied.

| Permission | Info.plist / API key | What needs it | What breaks without it | How to request |
|---|---|---|---|---|
| **Camera** | `NSCameraUsageDescription` | Gesture pipeline (cv2 AVFoundation), phone-camera WebRTC, QtWebEngine | First `cv2.VideoCapture` open fails / app may crash without the plist string | Auto-prompted on first AVFoundation access; the plist string is mandatory |
| **Microphone** | `NSMicrophoneUsageDescription` | Voice dictation, voice commands, Iris audio, whisper refiner, phone mic | sounddevice stream fails | Auto-prompted on first CoreAudio input access |
| **Screen Recording** | `NSScreenCaptureUsageDescription` (+ system grant) | Iris screen-vision, OCR find/click, YouTube skip-ad capture, **foreign-window titles** (`CGWindowName` redacted otherwise), overlay backdrop | Black capture frames; window titles blank | `CGRequestScreenCaptureAccess()`; user must add app in System Settings → Privacy & Security → Screen Recording; **no auto-grant** |
| **Accessibility** | (no plist key) | All synthetic mouse/keyboard (`CGEventPost`), AXUIElement reads/actions, window move/min/close, foreground-window title reads, idle detection | `CGEventPost` silently dropped (no return value — appears to "succeed"); AX queries fail | `AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: true})` opens the pane; **user must toggle manually**; cannot be auto-granted |
| **Input Monitoring** | (no plist key) | HID-level key/mouse injection on recent macOS; CGEventTap-based global hotkeys | Synthetic key events dropped | Prompted on first relevant API use; user adds app manually. **Prefer Carbon `RegisterEventHotKey` for hotkeys to avoid this entirely.** |
| **Automation** | `NSAppleEventsUsageDescription` | AppleScript control of Office/Chrome/Spotify/Discord/Messages/Outlook/volume | Apple Events rejected; scripted app control no-ops | Per-target consent prompt on first Apple Event to each app; granular (one grant per controlled app) |
| **Full Disk Access** | (no plist key) | Reading `~/Library/Messages/chat.db` for iMessage history | Cannot read message threads | User adds app manually in Privacy pane; **no API prompt** |
| **Speech Recognition** | `NSSpeechRecognitionUsageDescription` | Optional `SFSpeechRecognizer` SAPI-replacement fallback | Apple STT fallback unavailable (whisper still works) | Auto-prompted on first SFSpeechRecognizer use |

**Onboarding contract:** ship a permission wizard that triggers prompts at first use,
deep-links to the correct Privacy pane on denial, and orders requests sensibly
(Accessibility, Camera, Mic, Screen Recording, then Automation / Input Monitoring as
features are used). Because grants reset on signature change, **re-check on each
launch** and surface a friendly "grant X to enable Y" state rather than silently
no-opping. The codebase's existing best-effort/return-clean-error guards make this
tractable.

---

## Native dependency replacements

| Windows dependency | macOS replacement | Notes |
|---|---|---|
| **pywin32** (win32gui/api/process/com) | pyobjc-framework-**Cocoa** / **Quartz** / **ApplicationServices** + AppleScript | Per call-site rewrite; no single drop-in. Marker: `pywin32; sys_platform=="win32"` |
| **comtypes** (COM/OLE) | pyobjc AXUIElement (for UIA) + CoreAudio (for pycaw) | No COM runtime on macOS |
| **pycaw** (system + per-app audio) | pyobjc-framework-**CoreAudio** (system volume) / **AVFoundation** | **Per-app volume has NO macOS API** — system-only |
| **onnxruntime-directml** (DmlExecutionProvider) | **`onnxruntime`** with **`CoreMLExecutionProvider`** | Same import name — install only one. arm64 wheel ships CoreML EP |
| **mediapipe==0.10.21** | mediapipe (version with `macosx_*_arm64` wheel; likely a bump) | CPU/XNNPACK or Metal Tasks delegate; else Rosetta x86_64 |
| **whisper.cpp** CUDA/Vulkan exe | whisper.cpp **`-DGGML_METAL=ON`** Mach-O (+ optional `-DWHISPER_COREML=ON`) | Xcode clang, Homebrew SDL2, ship `ggml-metal.metal` |
| **llama.cpp** CUDA exe | llama.cpp **`-DGGML_METAL=ON`** Mach-O (`llama-server`/`llama-cli`) | `-ngl` GPU offload; GGUF still downloaded at first-run |
| **faster-whisper / CTranslate2** | same (arm64 wheels, CPU/Accelerate) | CUDA→CPU automatic; or prefer whisper.cpp Metal |
| **ffmpeg.exe** (`-f dshow`) | arm64 `ffmpeg` (`-f avfoundation`), codesigned | Optional perf path; plain AVFoundation capture already works |
| **pyaudiowpatch** (WASAPI loopback) | ScreenCaptureKit audio / AVAudioEngine | Loopback capture needs Screen Recording (macOS 13+) |
| **winsdk/winrt** (SMTC now-playing) | private **MediaRemote.framework** (via pyobjc/dlopen) | No public equivalent; maintenance risk |
| **win11toast** (notifications) | `NSUserNotification`/UserNotifications via pyobjc | Out of audited scope but follows the same pattern |
| **DPAPI** (`CryptProtectData`) | macOS **Keychain** (`SecItem*` / `keyring`) | Else OAuth tokens stored plaintext |
| **winreg** (config/MachineGuid/Run key) | plist/Application Support; **IOKit `IOPlatformUUID`**; **LaunchAgent** | IOPlatformUUID survives reinstalls (better than MachineGuid) |
| **PyInstaller win COLLECT + Inno** | PyInstaller **BUNDLE** (.app) + pkgbuild/productbuild + create-dmg | Brand-new spec; do not modify Windows spec |
| **Azure Trusted Signing** | **Developer ID** + `codesign` + `notarytool` + `stapler` | Hardened runtime + entitlements.plist |

**New pyobjc frameworks to add (macOS marker):** `pyobjc-framework-Cocoa`,
`-Quartz`, `-ApplicationServices`, `-CoreAudio`, `-AVFoundation`,
`-ScreenCaptureKit` (macOS 13+), plus `onnxruntime` (CoreML). Split
`requirements.txt` with environment markers (`; sys_platform=="win32"` /
`; sys_platform=="darwin"`) rather than forking files.

---

## The overlay problem

> **Mark this as a known platform difference for users.**

Touchless leans heavily on an **always-on-top, click-through** overlay (gesture
wheels, drawing canvas, countdown/recording/processing/hello overlays, voice-status
overlay). On Windows this is `WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE |
WS_EX_TOOLWINDOW` + `SetWindowPos(HWND_TOPMOST)` reinforced by DWM tweaks.

**What's achievable on macOS:** `native_overlay.py` already ships a working
`_apply_macos_overlay` via pyobjc — it walks `widget.winId()`→NSView→NSWindow and sets
`NSWindowStyleMaskNonactivatingPanel`, `setLevel_(NSStatusWindowLevel)` (with a
fallback level chain), `setCollectionBehavior_(CanJoinAllSpaces | Stationary |
IgnoresCycle | FullScreenAuxiliary)`, `setHidesOnDeactivate_(False)`,
`setFloatingPanel_(True)`. Click-through maps Qt `WA_TransparentForMouseEvents` /
`WindowTransparentForInput` to NSWindow `setIgnoresMouseEvents_(True)` under the Cocoa
plugin. Qt window flags alone deliver roughly 80% of the behavior; the AppKit layer
provides the topmost + no-focus-steal reinforcement.

**Where it falls short (the difference):**
- **Cannot float above the menu bar or above "secure" full-screen apps** the way
  `HWND_TOPMOST` blankets the Windows desktop. `NSStatusWindowLevel` can conflict
  with the menu bar and native full-screen Spaces; the level must be tuned per use
  case (gesture-wheel vs drawing-overlay), which the fallback level chain anticipates.
- **One audited gap:** the permissions report flags that a code path still needs
  `setIgnoresMouseEvents_` explicitly wired in (and `clearColor` for true
  transparency). Verify true pass-through end-to-end (Qt flag → NSWindow) on real
  hardware.
- `NSWindowCollectionBehaviorCanJoinAllApplications` is newer/guarded — confirm
  availability or rely on the `FullScreenAuxiliary` fallback already coded.
- Any overlay that **captures the screen as a backdrop** additionally needs **Screen
  Recording**.

**Recommended fallback:** keep the existing AppKit branch, finish wiring
`setIgnoresMouseEvents_`, tune the window level per overlay type, and accept that over
native full-screen apps and the menu bar the overlay may be occluded — document this
as an expected macOS difference. No TCC permission is required merely to *draw* a
click-through overlay (only screen *capture* needs one).

---

## Not-possible / degraded features

Items to explicitly **mark as a difference** to users. (🔴 = no equivalent; 🟠 =
possible but degraded/rewritten with reduced fidelity.)

- 🔴 **Per-application audio volume / per-app peak metering / the AppSessionDucker**
  (lower a game while Spotify/Chrome play). macOS CoreAudio is **device-level only**;
  there is no public per-process output volume API. *Fallback:* system-wide volume
  control only; per-app tools removed from the catalog (`available()` narrows). True
  parity would require bundling a virtual audio device (e.g. BlackHole) — out of scope.
- 🔴 **Classic Outlook COM/MAPI connector** (zero-auth inbox read / compose /
  calendar with `Restrict`/`Sort`). No COM on macOS; even "new Outlook" on Windows
  lacks COM. *Fallback:* route to the **Microsoft Graph** connector (cloud, OAuth) —
  the recommended path; optionally a partial Outlook-for-Mac AppleScript connector
  later (no Restrict/Sort/multi-Store parity).
- 🔴 **Microsoft Phone Link SMS/iMessage bridge** (AppX driven by UIA + `ms-phone:`).
  Does not exist on macOS. *Fallback:* a brand-new **Messages.app** connector —
  AppleScript `tell application "Messages" to send …` (Automation) for sending, and
  `~/Library/Messages/chat.db` reads (Full Disk Access) for history. A different
  backend, simpler on Apple Silicon, but a rewrite.
- 🔴 **Microsoft Store update channel** (`winget msstore`, `ms-windows-store://`).
  No equivalent. *Fallback:* GitHub ReleaseChecker / Developer-ID direct download
  (the Mac App Store is off the table — see below).
- 🔴 **pywin32 / comtypes runtime.** No macOS port of the libraries; everything that
  imports them is re-platformed onto pyobjc + AppleScript per call-site.
- 🟠 **Iris UIA controller** (read/click any app's accessibility tree, auto-approve
  watcher). Full **AXUIElement** rewrite (~1000 lines, different object model); AX
  attribute availability varies per app and **Electron/Chromium web trees are
  sparse**. *Fallback:* keep the OCR-find-and-click path for apps that don't cooperate.
- 🟠 **YouTube UIAutomation actions** (PowerShell `System.Windows.Automation`).
  AXUIElement rewrite; Chrome web AX behind a force-renderer-accessibility flag and
  unreliable. *Fallback:* `javascript:` via the address bar (Cmd+L), which ports.
- 🟠 **Foreground / foreign-window introspection & control** (titles, geometry,
  move/min/maximize/close of *other* apps). Foreign window **titles are redacted**
  without Screen Recording; control needs Accessibility; **window "maximize" has no
  direct macOS concept** (closest is the AX zoom button or sizing to `visibleFrame`).
  Affects fullscreen-game priority boost and Chrome/YouTube-by-title matching.
  *Fallback:* `NSWorkspace.frontmostApplication` for identity (free); prefer
  AppleScript queries for Chrome tab URLs; degrade gracefully.
- 🟠 **System-state interruption signals** (mic-in-use, camera-in-use, Focus/DND).
  No clean macOS API. *Fallback:* degrade these to `UNKNOWN` (the interruption gate
  already fail-closes); idle/battery/fullscreen have clean equivalents.
- 🟠 **Win+H system dictation toggle.** No stable macOS analog (Dictation hotkey is
  user-configurable). *Fallback:* the app's own whisper pipeline; open TextEdit as
  the target.
- 🟠 **Now-playing metadata (SMTC).** Only via the **private MediaRemote.framework** —
  works but undocumented and a maintenance risk. *Fallback:* per-app AppleScript
  (Spotify/Music) for narrower coverage.
- 🟠 **Mac App Store distribution.** MAS mandates **App Sandbox**, which forbids the
  global CGEventTap / Accessibility input control this app depends on. *Fallback:*
  Developer-ID-notarized direct download (`.pkg`/`.dmg`) as the primary channel; a
  sandbox-limited "lite" MAS build is the only theoretical MAS path.

---

## Packaging & distribution (NEW, separate from Windows)

> **This is a brand-new pipeline. None of the existing Windows files are modified:**
> `builder/windows/hgr_app.spec`, `build_windows.bat`, `installers/windows/hgr_app.iss`,
> `signing/sign-file.bat`, and the `.bat` build scripts stay untouched. Everything
> below lives in **new sibling trees**: `builder/macos/`, `installers/macos/`,
> `signing/macos/`, plus `run_mac.sh` / `build_mac.sh` and a `requirements_mac.txt`.

**1. App bundle (`.app`)** — NEW `builder/macos/hgr_app_mac.spec` using PyInstaller
`BUNDLE(...)` (not COLLECT) with `target_arch='arm64'`, an `.icns` icon (convert via
`iconutil`/`sips`), and an embedded **Info.plist**: `CFBundleIdentifier` (e.g.
`app.touchless`), `LSMinimumSystemVersion` (12.0), and the usage-description strings
(`NSCameraUsageDescription`, `NSMicrophoneUsageDescription`,
`NSScreenCaptureUsageDescription`, `NSAppleEventsUsageDescription`,
optionally `LSUIElement=1` for a menu-bar/agent app). Keep
`collect_all('PySide6','shiboken6','mediapipe')`; **drop** every
`pywin32`/`comtypes`/`pythoncom`/`pycaw`/`winsdk`/`winrt`/`uiautomation`/`win11toast`/`pyaudiowpatch`
hiddenimport and add the pyobjc frameworks. The PySide6 hook handles
`QtWebEngineProcess.app` helpers.

**2. Native GPU/ML rebuilds (the long pole)** — NEW
`builder/macos/_build_whisper_metal.sh` and `_build_llama_metal.sh` (cmake +
`-DGGML_METAL=ON` + Xcode clang + Homebrew SDL2, no nvcc/vcpkg) producing arm64
Mach-O binaries (no `.exe`), bundled under a layout the runtime finder accepts (`bin`,
not `bin/Release`). Swap `onnxruntime-directml` → `onnxruntime` (CoreML). **Schedule
this first** — bundle/sign/notarize all depend on its outputs existing and being
individually codesigned.

**3. Installer** — NEW `installers/macos/`: `pkgbuild` (component pkg → `/Applications`)
+ `productbuild --distribution distribution.xml` (welcome/license/conclusion, arch
requirement) + `productsign` with the **Developer ID Installer** cert. Also a
drag-to-Applications `.dmg` via `create-dmg`/`hdiutil` for the website channel. No
stub-downloader needed (a `.dmg`/`.pkg` can simply BE the full app, or a postinstall
script fetches the GPU/ML payload from R2). No Start-menu / registry Uninstall key.

**4. Codesign + notarize** — NEW `signing/macos/entitlements.plist` with
`com.apple.security.cs.allow-jit`, `allow-unsigned-executable-memory`,
`disable-library-validation` (required for PyInstaller/Python + ctypes/dylib loads),
`com.apple.security.device.camera`, `device.audio-input`. `codesign --force --options
runtime --timestamp --entitlements …` **inside-out** (sign every nested Mach-O —
ffmpeg, whisper, llama, dylibs, QtWebEngineProcess helper — before the outer bundle;
`--deep` is unreliable). Then `xcrun notarytool submit … --wait` and `xcrun stapler
staple`. Gatekeeper enforces trust at launch; the in-app updater verifies downloads
with `codesign --verify`/`spctl --assess` instead of WinVerifyTrust.

**5. Auto-update** — Prefer **Sparkle** (signed EdDSA appcast, atomic `.app`
replacement, Gatekeeper-friendly). If staying in Python, reuse the portable
`release_checker.py` GitHub poller and write a NEW `_apply_update.sh` (wait for PID,
`ditto`/`hdiutil` extract, replace `/Applications/Touchless.app`, `codesign --verify`
+ `spctl --assess` trust gate, `open -n` relaunch). Drop the MS Store path.

**6. NEW mac test/run scripts** — `run_test.py` is platform-neutral and runs under a
mac venv (`.venv/bin/python`). Add `run_mac.sh` (activate venv → run app/test) and
`builder/macos/build_mac.sh` (pip install `requirements_mac.txt` → Metal sub-builds →
pyinstaller mac spec → codesign → pkgbuild/productbuild/productsign → notarize →
staple) — the structural analog of `build_windows.bat`, entirely separate.

**7. CI** — NEW GitHub Actions workflow on a **`macos-14` (Apple Silicon)** runner:
import Developer ID certs from base64 secrets into a temp keychain, build (cache the
compiled native binaries to keep runtimes sane), codesign + entitlements,
pkgbuild/productbuild/productsign, notarize (App Store Connect API key), staple,
generate the Sparkle appcast, upload `.pkg`/`.dmg` to R2 (rclone) / GitHub Release.

---

## Recommended porting plan

> Phases marked **⚠ needs Mac hardware** cannot be meaningfully verified on a
> Windows dev box — TCC prompts, AppKit window levels, CGEvent injection, AVFoundation
> capture, Metal builds, codesign/notarize all require macOS to validate.

### Phase 0 — Platform-abstraction layer
**What:** Introduce a `platform_backend` seam so the portable orchestration core never
imports `ctypes.windll` directly. Build interfaces with `win/` + `mac/` backends for:
`mouse`, `keyboard`, `window`, `accessibility`, `capture`, `secrets`, `signals`,
`paths`, `overlay`. Also centralize paths into one `platform_dirs.py`
(`app_config_dir`/`app_data_dir`/`app_logs_dir`/`models_dir`/`user_documents_dir`),
collapsing the ~20 `LOCALAPPDATA or ~/.local/share` duplications and scattered
`Documents/Touchless` literals. Split `requirements.txt` with env markers.
**Key files:** `debug/mouse_controller.py`, `debug/text_input_controller.py`,
`debug/foreground_window.py`, `live_api/tool_executor.py`, `live_api/uia_controller.py`,
`live_api/screen_ocr.py`, `live_api/system_signals.py`, `live_api/secret_vault.py`,
`app/ui/native_overlay.py`, `utils/runtime_paths.py`, `utils/autostart.py`,
`config/app_config.py`, `requirements.txt`.
**Risk:** Low technically, but a wide refactor; risk is regressing Windows behavior —
keep Windows branches byte-identical. Can be done on Windows without Mac hardware.

### Phase 1 — Core (camera + tracking + cursor + overlay) ⚠ needs Mac hardware
**What:** Get the live pipeline running on a Mac: AVFoundation camera (already
branched), the entirely-portable core-pipeline/gesture compute, the cursor-control
engine reimplemented on Quartz (`CGWarpMouseCursorPosition` + `CGEvent*`), and finish
the `native_overlay` macOS branch (wire `setIgnoresMouseEvents_`, tune levels). This
is the make-or-break vertical slice and surfaces the first TCC walls (Camera,
Accessibility).
**Key files:** `app/camera/*`, `core/**`, `gesture/**`, `debug/mouse_controller.py`,
`app/integration/noop_engine.py`, `app/ui/native_overlay.py`, `app/overlays/overlay.py`.
**Risk:** High-value, high-uncertainty: CGEventPost cursor-injection **latency** for
the real-time gesture loop is unproven; Y-origin/Retina coordinate handling; overlay
level conflicts with full-screen/menu bar. Accessibility cannot be auto-granted.

### Phase 2 — Controllers ⚠ needs Mac hardware
**What:** Reimplement the input/window/app-control controllers behind the Phase-0
seam: keyboard/text synthesis (`CGEventKeyboardSetUnicodeString` + NSPasteboard),
window introspection (NSWorkspace + AXUIElement, accept degraded titles), app launch
(NSWorkspace/`open`), Chrome/YouTube/Spotify/Discord routers (map Ctrl→Cmd; Discord
named-pipe→AF_UNIX), system volume via CoreAudio. **Mark per-app volume, Office COM,
Phone Link, foreign-window maximize as differences** per the not-possible section.
**Key files:** `debug/text_input_controller.py`, `debug/foreground_window.py`,
`debug/desktop_controller.py`, `debug/chrome_controller.py`, `debug/youtube_controller.py`,
`debug/spotify_controller.py`, `debug/discord_controller.py`, `debug/office_controller.py`,
`debug/volume_controller.py`, `debug/media_controller.py`, `live_api/connectors/*`.
**Risk:** Medium-high. Retina scaling breaking skip-ad coordinates; AppleScript per-app
Automation consent friction; Office is a genuine rewrite.

### Phase 3 — Voice / whisper-Metal ⚠ needs Mac hardware (build); logic verifiable on Win
**What:** Add the `metal` backend tier to the three `_resolve_*_executable`
functions, strip `.exe`, drop GPU probes; rebuild whisper.cpp + the refiner with
Metal; confirm faster-whisper/CTranslate2 CPU fallback works day-one. Add POSIX path
branch to `save_prompt._explicit_path`; optional `SFSpeechRecognizer` SAPI replacement.
**Key files:** `voice/whisper_stream.py`, `voice/whisper_refiner.py`,
`voice/llama_server.py`, `voice/sapi_stream.py`, `voice/live_dictation.py`,
`voice/save_prompt.py`, `builder/macos/_build_whisper_metal.sh`.
**Risk:** Low for Python (CPU fallback covers it); the Metal **build** needs Mac. The
real friction (text injection) is the Phase-2 keyboard backend.

### Phase 4 — Iris / live_api ⚠ needs Mac hardware
**What:** The heaviest rewrites: `uia_controller.py` → AXUIElement (budget as
from-scratch), `tool_executor.py` window/click/type, `screen_ocr.py` capture+coordinate
layer (ScreenCaptureKit + Y-flip + Retina), `secret_vault` → Keychain, `system_signals`
API swaps (idle/battery clean; mic/cam/DND → UNKNOWN), `consent_jurisdiction` →
NSLocale, `repo_focus_watcher` → AX, `local_backend` → single Metal build dir. The
orchestration core (`command_router`, `compose`, `config`, `live_api_manager`,
`cortex/`) is works_asis. Narrow connector `available()`/`tools()` per platform so the
model never sees unimplemented tools.
**Key files:** `live_api/uia_controller.py`, `live_api/tool_executor.py`,
`live_api/screen_ocr.py`, `live_api/screen_context.py`, `live_api/secret_vault.py`,
`live_api/system_signals.py`, `live_api/local_backend.py`, `live_api/connectors/*`.
**Risk:** Very high effort on UIA→AX; Screen Recording + Accessibility gating;
MediaRemote private-framework fragility.

### Phase 5 — Packaging ⚠ needs Mac hardware
**What:** Everything in "Packaging & distribution" — new `.app` spec, Metal native
rebuilds bundled and codesigned, `.pkg`/`.dmg`, Developer ID codesign + notarize +
staple, Sparkle/auto-update, `run_mac.sh`/`build_mac.sh`, CI on macos-14. The
permission onboarding wizard lands here (first-run UX, recheck-each-launch).
**Key files (all new):** `builder/macos/hgr_app_mac.spec`, `builder/macos/build_mac.sh`,
`installers/macos/*`, `signing/macos/entitlements.plist`, `run_mac.sh`,
`requirements_mac.txt`, `.github/workflows/build-macos.yml`.
**Risk:** Codesign inside-out / nested helper signing is fiddly (notarization rejects
on the first miss); entitlements interplay; MAS is off the table for the full app.

---

## Open questions for the user

1. **Apple Developer account** — Do you have (or will you enroll in) the Apple
   Developer Program ($99/yr)? It is **mandatory**: Developer ID certs + notarization
   are required, and TCC grants bind to a stable Developer ID signature (ad-hoc
   signing resets permissions on every build/update).
2. **A Mac to build & test on** — Apple Silicon is required to build the Metal
   native binaries, produce/notarize the `.app`, and verify the TCC-gated features
   (Phases 1–5 cannot be validated without it). Do you have one, or should we plan a
   cloud-Mac / CI (`macos-14` runner) approach?
3. **Scope priorities** — Which vertical matters first: the **gesture+cursor core**
   (Phase 1, the demoable slice) or **Iris desktop control** (Phase 4, the biggest
   lift)? This sets the order after Phase 0.
4. **Acceptable degraded features** — Are you OK shipping these as documented
   differences on macOS: **no per-app audio volume / no ducker**, **Outlook via Graph
   only (no Classic-Outlook zero-auth)**, **iMessage via a new Messages.app connector
   (or deferred)**, **no foreign-window maximize**, **overlay may be occluded by
   native full-screen apps / menu bar**, **now-playing via a private framework**?
5. **Distribution channel** — Confirm **Developer-ID direct download** (website
   `.dmg`/`.pkg` + Sparkle/GitHub auto-update) as the macOS channel, given the **Mac
   App Store is incompatible** with the app's Accessibility/CGEventTap requirements
   (App Sandbox). A sandbox-limited "lite" MAS build is the only MAS option — pursue it?
6. **Config/data location & privacy** — OK to relocate macOS config + OAuth tokens
   from `~/.touchless` / `~/Documents/Touchless` into `~/Library/Application Support`
   (and tokens into Keychain) for convention + to keep secrets out of iCloud-synced
   Documents? (Functionally works either way; this is a privacy/idiom call.)
