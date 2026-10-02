# Touchless Performance Checkpoint — v1.1.9.2 (r2)

**Date:** 2026-09-14 (v1.1.9.2 r2 update)
**Author:** Konstantin Markov
**Reference platform:** i5-13500 + RTX 4070, Razer Kiyo Pro
**Status:** Known-good baseline. Four freeze/fps-drop fixes on top of
the first 1.1.9.2 iteration. **Freezes confirmed resolved by user
2026-09-14** — do not regress the four fixes listed below.

> ### ⭐ SAVE POINT — BUILD_ROUND 62, confirmed on BOTH rigs 2026-09-28
>
> First build where the four modes measurably differ **and** improve
> performance on the low-end field rig, not just the reference rig. The
> two-rig requirement is satisfied for the first time. Numbers in
> section 1. Commit `fc0d6c7`, installer md5 `9c24f007e1cd3c02c44a9225e9c122a5`,
> payload md5 `525307344c44c254c4bdec2147532c50` / SHA256 `08b26ae2…`.
> **If performance regresses, restore to this build round before
> diagnosing anything else.**

**v1.1.9.2 fixes on top of v1.1.9.2 baseline:**
1. `is_foreground_fullscreen()` — reject normal maximized Win32 windows
   (WS_CAPTION / WS_THICKFRAME) and Touchless's own PID so
   `_gpu_suppressed_for_fullscreen` no longer latches on maximized Chrome.
   Fixes "GPU mode dropped to 22 fps and stayed".
2. `_CRITICAL_FPS_EXIT_THRESHOLD` 28 → 20 fps + 90 s hard-escape at ≥24 fps
   + engage-time debounce (4 fps margin below the enter threshold). Auto
   low-fps critical tier can no longer strand a user in the 22-27 fps
   dead band.
3. `_maybe_show_spotify_first_active_prompt` — dedup the `save_config`
   disk write so we only touch settings.json when the latch actually
   flips (was 60-160 writes/sec while Spotify's window was open).
   Fixes "freezes when Spotify is open, no hand in frame".
4. `debug_frame_ready` — reinstate 30 Hz throttle with force-emit on
   hand-presence / action-history edges. `raw_frame_ready` stays
   uncapped so live-view fps is unchanged. Restores the pre-v1.1.9.2
   protection that stopped every per-frame handler from being paid at
   Lite/GPU rates.

This file is a **restore point** for the tuning constants and structural choices
that reached the current performance level. When you touch a hot path, cross-
check against this file before committing. When performance regresses, the
first move is to diff the constants below against the current tree.

Every "wrong path" listed in section 3 was actually tried in this codebase and
made things worse. Don't retry them without new evidence.

---

## 1. Measured performance

Reference rig (RTX 4070 / i5-13500 / 32 GB RAM / Razer Kiyo Pro):

| Mode    | Median per-frame | fps ceiling @ 60fps cam | Notes                     |
|---------|-----------------:|------------------------:|---------------------------|
| Default | ~12.0 ms         | 60                      | MediaPipe c=1 CPU         |
| Lite    | ~11.9 ms         | 60                      | MediaPipe c=1 CPU @ 640-w |
| GPU     | ~1.5 ms          | 60                      | ONNX + DirectML           |

### 1.1 BUILD_ROUND 62 — field-confirmed, both rigs (2026-09-28)

These are USER-OBSERVED numbers from the shipped BUILD_ROUND 62 binary,
not bench estimates. **This is the save point.**

**Reference rig** — RTX 4070 / i5-13500 / Razer Kiyo Pro:

| Mode    | fps          | Tracking quality                                   |
|---------|-------------:|----------------------------------------------------|
| GPU     | high 50s–60  | very smooth                                        |
| Lite    | ~33          | consistent, slightly less smooth than GPU          |
| Default | maxing 30    | good — **can draw a full circle without losing the hand** |

Lighting good in all three modes, no issues.

**Field rig** — GTX 960 / generic UVC webcam (the adaptive target):

| Mode              | fps                     | Tracking quality              |
|-------------------|------------------------:|-------------------------------|
| Default           | ~14                     | good                          |
| Boost performance | mid 20s (hand in frame) | good                          |
| Lite              | (above default)         | improves tracking + smoothness slightly |

Lighting good. **"Modes change and improve performance as they should"** —
the first time that has been true on this rig, and the whole point of the
r24/r25/r26 rounds. No dark-live-view complaints in any mode.

The circle test on the reference rig's DEFAULT mode matters as a
regression canary: before r25, drawing a circle lost the hand. Keep it.

**Open ask (not a regression):** field-rig default wants ~20 fps instead
of ~14, i.e. roughly 10 fps more headroom across the board. Anything
attempted for that must leave BOTH tables above intact — the reference
rig must not drop from 60 / 33 / 30, and the field rig must not lose the
mode separation it just gained.

Freeze detector threshold: **200 ms paint gap**. Steady-state paint gap
should stay under this in every mode after the fixes in this checkpoint.

---

## 2. Working values — DO NOT REGRESS

Each value below has a linked rationale in section 3. Restore these exact
values (not "close enough") if a debugging session drifts.

### 2.1 ONNX reacquire constants

`src/hgr/gesture/tracking/onnx_runtime.py` (around line 564):

```python
_REACQUIRE_SCORE_FLOOR = 1.0
_REACQUIRE_WINDOW_FRAMES = 0
_STALE_ROI_RETRY_FRAMES = 2          # ← 2, NOT 0. See §3.4
_MAX_RELAXED_CANDIDATES_PER_FRAME = 0
```

**Failure mode if wrong:** all-zero neutralization → GPU-mode swipes require
sweeping the hand across the full camera view. All-nonzero (the 01d5f81
defaults) → GPU fps drops to ~20.

### 2.2 Lite mode engine build

`src/hgr/app/integration/noop_engine.py`:
- Line 730: `_LITE_PROCESS_WIDTH = 640`
- Line 5219: `HandDetector(max_process_width=self._LITE_PROCESS_WIDTH, prefer_gpu=False)`

Lite is **MediaPipe complexity=1 CPU at 640-wide**, NOT ONNX+DirectML. It is
a cheaper Default, not a lite GPU mode.

**Failure mode if wrong:** using `prefer_gpu=True` or ONNX runtime → Lite
becomes GPU-dependent and stops being the "universal upgrade from Default"
that non-discrete-GPU users need.

### 2.3 GPU mode camera cap default

`src/hgr/app/integration/noop_engine.py`:
- `_apply_perf_camera_path` (~line 5001): default cap is **640x480**
- `_upgrade_to_ffmpeg_capture_if_lite` (~line 5985): default cap is **640x480**

**Failure mode if wrong:** 1280x720 default → GPU fps drops from 55-60 to
20 fps because the higher-res frames dominate the tick budget even when the
inference itself is 1.5 ms.

### 2.4 r53 short-shutter kick gate

`src/hgr/app/integration/noop_engine.py` (~line 6982):

```python
_should_kick = _armed or _known_generic_needs_kick
# NOT: _armed or _known_generic_needs_kick or _user_chose_kick
```

**Failure mode if wrong:** the removed `_user_chose_kick` OR-branch was
sticky-forever — any user who once touched the Short Shutter toggle would
have the kick applied to every subsequent camera open, including the Razer
Kiyo Pro, damaging its exposure state on each launch.

Verdict check is `_classifier_verdict is True` (auto-fire only for a
positively-classified generic UVC), NOT `_classifier_verdict is not False`
(which treated None/unknown as allowed).

### 2.5 Custom-gesture cost gate

`src/hgr/custom_gestures/runner.py` (line 153):

```python
@property
def has_static_gestures(self) -> bool:
    # Returns True ONLY when at least one gesture has kind == "static".
    # Dynamic + sequence gestures do NOT flip this to True.
```

`src/hgr/app/integration/noop_engine.py` (~line 9804):

```python
if runner is not None and getattr(runner, "has_static_gestures", False):
    # ... static gesture cost gate
```

**Failure mode if wrong:** dynamic/sequence gestures fire the ~40ms per-frame
post-processing path in Lite mode → drops the whole engine to ~25 fps even
when no static gesture exists.

### 2.6 Profile cache

`src/hgr/app/integration/noop_engine.py`:
- `_profile_allows_pose(pose_id)` — cached per `active_profile_id`
- `_profile_is_restricting()` — cached per `active_profile_id`

**Failure mode if wrong:** each pose check re-walks the profile allow-list
per frame → measurable cost when many gestures are bound.

### 2.7 Freeze detector

`src/hgr/app/integration/noop_engine.py`:
- Line 4527: `def _maybe_log_freeze(self, *, phase: str, now: float, threshold_ms: float)`
- Line 4602: called from `_note_display_fps` with `phase="paint", threshold_ms=200.0`

Startup grace: does not fire until 30 samples have accumulated (skips the
first ~450ms so app-startup latency doesn't emit false positives).

**Restore instruction:** the emit format is
`[freeze-detector] type=paint_gap gap=Nms mode=X (threshold=200ms)` — keep
this stable so log analysis tooling doesn't break.

### 2.8 Mouse controller (v1.1.9.2)

`src/hgr/debug/mouse_controller.py`:
- `_cached_bounds: tuple | None` on the instance; `virtual_bounds()` reads
  once and caches, saving ~16 GetSystemMetrics per tick.
- `_last_cursor_target: tuple | None` on the instance; `move_normalized`
  skips `SetCursorPos` when the target pixel is unchanged from last call.
- `MouseController._instances: weakref.WeakSet` — every live controller
  auto-registers on `__init__`.
- `MouseController.invalidate_all_bounds_caches()` — classmethod called
  from main_window's `nativeEvent` on WM_DISPLAYCHANGE / WM_DPICHANGED.

`src/hgr/app/ui/main_window.py` (~line 25024):

```python
if msg_id in (0x007E, 0x02E0):   # WM_DISPLAYCHANGE / WM_DPICHANGED
    from ..integration.noop_engine import MouseController as _MC
    _MC.invalidate_all_bounds_caches()
```

**Failure mode if wrong:** without the cache, mouse-control freezes for
200-500ms under GDI display-lock contention or monitor topology changes.

**Behavior change to be aware of:** if the user touches a physical mouse
while their hand is idle at a target, the physical mouse now wins until
the hand moves again (previously the hand re-pinned the cursor every
tick). This is intentional but was flagged.

### 2.9 Spotify startup refresh (v1.1.9.2)

`src/hgr/debug/spotify_controller.py` (around line 215-231):

```python
if self._refresh_token and self._client_id:
    def _startup_refresh() -> None:
        try:
            _ok = self._refresh_access_token()
        except Exception:
            _ok = False
        # ... stderr log
    threading.Thread(
        target=_startup_refresh,
        name="spotify-startup-refresh",
        daemon=True,
    ).start()
```

**Failure mode if wrong:** doing `_refresh_access_token()` synchronously in
`__init__` blocks the constructing thread (usually the GUI thread) for up
to 5 s + DNS/connect time. Shows up as a paint gap at engine-build time.

Accepted trade-off: the first gesture command right after launch may eat
one 401 → refresh → retry round-trip if the refresh hasn't landed yet.

---

### 2.10 r18 guards added 2026-09-24 (provisional; user reports performance back)

- **NVENC startup probe frame = 320x180** (`main_window.py`, search
  `color=black:s=320x180`). Was 64x64 since 1.1.6, below NVENC's minimum on
  Ampere/Ada: the probe failed every session on the RTX 4070, demoted the
  clip cache to QuickSync (no segments) then CPU x264 via the 13 s watchdog
  and a 6 s GUI freeze on recovery, and left a CPU encoder running all
  session. With 320x180 the cache runs on NVENC and reports healthy at 13 s
  with no watchdog failure.
- **`DSHOW_GRAPH_LOCK`** (`camera/threaded_cv_capture.py`): RLock around every
  cv2 capture construction / release / property write; never around read().
  Reproduced crash without it: ntdll 0xC0000374 heap corruption when the
  engine released its capture while the startup camera warmup was still
  constructing one. `ThreadedCvCapture.release()` defers the inner release
  to the reader thread when its 1 s join times out.
- **Camera-control snapshot / restore** (`camera/dshow_controls.py`, wired
  in `start_engine` / `stop_engine`): reads IAMCameraControl +
  IAMVideoProcAmp value AND Auto/Manual flag via COM at START and STOP and
  logs `[camera-controls] session changes: ...`. The restore is **ledger
  gated** (r18 review): it writes back a control only if the engine's own
  `_driver_writes_this_session` ledger says this session wrote it. A flag
  flip made by Synapse / OBS / Windows Camera mid-session is logged as
  "external change left as-is" and never reverted. Every run on the Kiyo
  Pro reads "none (engine wrote: nothing)".
- **Bounded lock waits** (r18 review): every acquire of `DSHOW_GRAPH_LOCK`
  has a timeout (construction = the open timeout, release 1 s then a
  helper thread, set() 2 s), an abandoned constructor can never park the
  lock for the driver's 60-120 s, and `wait_for_pending_releases()` keeps a
  new capture from sharing a device slot with a deferred release.
- **r55 pre-open restore disabled** for r18 (review: it ran before the
  START snapshot and matched on index only). The ON paths still write the
  marker; in-session OFF restores and the Reset button cover the latch.
- **HDR-throttle pill** (`_hdr_throttle_probe`): negotiated >= 50 fps,
  delivered median <= 32 fps over ~12 s, Exposure in Auto -> one pill
  naming HDR in Synapse. Read-only.
- **Rule:** `main_window.py` has duplicated method definitions (see
  OPEN_ISSUES 1.4). Verify with `inspect.getsourcelines` which one Python
  binds before editing; the smoke gate + `[camera-controls]` log lines are
  the proof a change actually runs.

### 2.11 r20 adaptive/Norton work added 2026-09-24

Driven by a field debug bundle from an older rig (Win10 19045, i7-6700,
GTX 960, generic UVC webcam, Norton, 4240x1440 desktop) reporting a dark
preview, ~10 fps with 25 s freezes, and antivirus popups that made Lite
mode unusable. Every item is gated so the reference rig (i5-13500 /
RTX 4070 / Kiyo Pro) takes an identical path.

- **Per-camera ffmpeg-failure memory** (`src/hgr/app/camera/ffmpeg_memo.py`,
  config `camera_ffmpeg_hard_failures`). TWO hard failures at the SAME
  device+resolution are required before the ffmpeg capture attempt is
  skipped. A silent hang never counts, because that is the signature of
  another process holding the DirectShow handle, and a one-strike memo
  would permanently demote a momentarily-busy Kiyo Pro to OpenCV/YUY2.
  Kill switch `HGR_FFMPEG_MEMO=0`. The skip is placed BEFORE the
  short-shutter preflight and before releasing the working OpenCV
  capture; `tests/test_ffmpeg_memo_wiring.py` pins that ordering.
- **ffmpeg open budget** (`open_ffmpeg_cap_with_fps_fallback`,
  `total_budget_seconds=6.0`). The FIRST candidate is always attempted
  regardless of the budget, so a working open can never be starved.
- **DirectShow device list cached** for 10 s, and the verbose device dump
  on the open path is behind `HGR_FFMPEG_DEVICE_DUMP=1`. Together these
  remove one to two ffmpeg spawns per camera-path change.
- **Encoder capability probe persisted** (`ffmpeg_caps_cache_key` /
  `ffmpeg_caps_cache`). The probe spawns eight ffmpeg processes on the
  GUI thread at every launch. The key fingerprints the ffmpeg binary
  (name/size/mtime) plus every display adapter's DriverDesc@DriverVersion
  read from the registry in ~0.1 ms with no subprocess. A runtime encoder
  demotion calls `invalidate_ffmpeg_capabilities_cache`, so a single bad
  session can never strand a capable GPU on libx264.
- **Clip-cache scale filter converts to nv12 before resizing.** gdigrab
  emits BGRA and swscale was resizing 4 bytes/px; `format=nv12,scale=...`
  is ~4x cheaper for the same bicubic kernel (measured 14.1 -> 5.8 CPU-s
  per 6 s wall). Dead code on any desktop at or under 4096 px, which
  includes the reference rig at 3584x1440.
- **Real camera name kept on CameraInfo** (`open_camera_by_index`). It
  hardcoded `"Camera N (DirectShow)"`, which matches nothing in either
  classifier keyword list, so `classify_camera_shutter_hint` returned
  None for the exact generic-UVC cameras it was written for and the r53
  kick could never un-latch an inherited Exposure=-6/Manual. The kick
  gate expression in 2.4 is UNCHANGED; only its input is now correct.
- **Classifier-driven short shutter stays opt-in.** Because the name fix
  makes a positive classification reachable for the first time, the
  auto-apply branch is behind `HGR_CLASSIFIER_AUTOSHUTTER=1`. Short
  shutter trades brightness for frame rate, and a dark preview on first
  launch is the worst first impression this app can make. The Settings
  toggle and `HGR_FORCE_SHORT_SHUTTER` are unaffected.
- **Restores are directional** (`dshow_controls.split_restorable`). A
  restore may hand a control back to Auto, or change a value while the
  flag stays put, but may NEVER turn Auto back into Manual. Restoring a
  camera to the Manual latch it was found in is how a dark preview
  becomes permanent across launches.
- **Driver-write ledger read before `stop()`.** `self._worker.stop()`
  emits `running_state_changed(False)` on the same thread, which runs
  `_cleanup_thread_if_stopped` and sets `self._worker = None`, so the
  ledger read three lines later always saw an empty set and the restore
  never fired.
- **Diagnostics camera list fixed.** `test_my_pc.camera_info()` read
  `c.name` / `c.device_name`; `CameraInfo` only has `display_name`, so
  "Test my PC" spent 2-5 s opening cameras and then reported none.

Second wave, same round:

- **Encoder cache carries `nvenc_modern_presets`.** The first cut of the
  cache stored five keys and dropped this one, which is read at encode
  time to choose `-preset p4` over legacy `medium`. A cache hit would
  have silently downgraded every clip on a capable GPU. The loader now
  rejects any payload missing a probe field, the key carries BUILD_ROUND
  (a previous round fixed a false-negative NVENC probe by changing only
  the probe frame size, with no ffmpeg or version change), and a probe
  where any sub-probe threw is marked degraded and never cached.
- **Clip-cache default is hardware-seeded once** (`clip_cache_default_
  seeded`). `discrete_gpu_vram_mb()` reads the 64-bit
  `HardwareInformation.qwMemorySize` from the display-class registry in
  ~0.1 ms with no subprocess, so it is free of the WMI 4095 MB ceiling.
  The recorder is seeded off only when the desktop is above 4.6 Mpix AND
  VRAM is at or below 3072 MB; an unknown reading always fails open. The
  Enable clipping tooltip then explains why, and the latch means the app
  decides at most once and never overrides the user.
- **Rank-4 'pre-open COM un-latch' was deliberately NOT taken.** The
  camera-name fix makes `classify_camera_shutter_hint` return True for a
  generic UVC, so `_known_generic_needs_kick` and therefore the existing
  2.4 kick now fire on an inherited Exposure=-6/Manual and return it to
  Auto at engine start. That is an automatic exit path for the dark
  preview through a gate that is already reviewed. Adding a fourth
  variant of 'write auto-exposure at start' is exactly the recurrence
  3.3 / 3.8 / 3.9 records three times.

Third wave (r20d), from tracing the landed work against the field rig:

- **An empty ffmpeg enumeration is a failed probe, not an answer.**
  `_run_external_probe` returns "" for ANY failure including a timeout,
  and ffmpeg never legitimately lists zero encoders. Without a guard, a
  probe delayed past its timeout by antivirus yielded an empty encoder
  list, which left clip encoding on CPU x264 for the whole session AND
  was written to the cache, where the loader rejected it every launch
  after. Net effect: permanent silent re-probing with hardware encoding
  off. All three enumeration probes now set `probe_degraded` when empty,
  and the store refuses any result with no encoders.
- **`discrete_gpu_vram_mb` must not take a plain maximum.** An
  integrated GPU can report a large shared-memory figure, so on a machine
  with a weak add-in card plus an iGPU the maximum reports the iGPU and
  hides the hardware the clip-cache seeding exists to detect. It now
  prefers a non-integrated adapter (`INTEGRATED_ADAPTER_HINTS`) and falls
  back to the maximum only when there is no add-in card.

Two behaviours worth writing down because they are easy to mis-explain:

- **The gamma lift is gated on short shutter, and that explains the field
  report exactly.** `_maybe_apply_gamma_lift` and the CLAHE path both
  return early unless `_short_shutter_active_for_display` is True, and
  that flag is only set by the two short-shutter writers. So on a camera
  latched dark by a previous session: Boost ON writes short shutter, sets
  the flag, and the lift brightens the frame artificially, which is the
  "more visible but yellow tint" the user described. Boost OFF left the
  camera at -6 with nothing setting the flag, hence "super dark". r20
  fixes the Boost-OFF case properly by returning the driver to Auto.
- **Brightness and frame rate pull against each other on a cheap UVC.**
  Short shutter was measured at 2-2.5x the frame rate on a cheap Realtek
  UVC (r49, 2026-07-27). Returning such a camera to Auto exposure buys a
  genuinely bright picture and may cost frame rate, because the driver
  lengthens the shutter in a dim room. This is a real trade-off, not a
  regression, and the Boost toggle is the user-facing knob for it. Do not
  "fix" one side of it silently.

Reference-rig verification, three source launches with the engine
autostarted (28 s each, same window):

| launch | encoder probe | frames | display fps | camera at STOP |
|---|---|---|---|---|
| 1 (cold cache) | FINAL, 8 spawns | 835 | 50.0 | session changes: none |
| 2 (warm cache) | CACHED, 0 spawns | 979 | 56.0 | session changes: none |
| 3 (warm + seeder) | CACHED, 0 spawns | 1059 | 60.0 | session changes: none |

Launch 3 also logged `[clip-cache-seed] desktop_px=5160960 vram_mb=12282
-> clip_cache_enabled=True`, and `[ffmpeg_capture] engaged at 640x480 @
60 fps MJPG` on the first candidate in all three.

Measured on the reference rig while doing this work, for future sizing:
`probe_system()` costs 4.1 s warm / 9.6 s cold, almost all of it two
PowerShell `Get-CimInstance` calls plus the camera open loop. WMI
`AdapterRAM` caps at 4095 MB, so a 12 GB card and a 4 GB card are
indistinguishable through it; only values below 4095 are real. Camera
friendly names do not identify models either -- a Kiyo Pro enumerates as
"USB Video Device". Any future hardware tiering must key on measured
behaviour, not on names or on that probe.
### 2.12 r21: ask the camera, and two build-machine gotchas

**The camera will tell you what it supports.** One
`ffmpeg -list_options true -f dshow -i video=<name>` call enumerates
every pin the driver advertises, with format, resolution and maximum
frame rate, without opening a capture. Cached per device name in
`camera_capabilities`, that replaces the 60-then-30-then-retry cascade
with a lookup. Three outcomes: open what it advertises, open at the rate
it actually claims rather than a hopeful 60, or skip the fast path with
no attempt at all. Unknown is never treated as unsupported.

On the reference rig the enumeration also explains the whole feature:
the uncompressed pin tops out at 30 fps while MJPG reaches 60.

**The probe is itself a DirectShow open.** Handing the camera straight
to the capture afterwards races the driver teardown and hangs silently
with empty stderr, which is the same failure the 600 ms settle elsewhere
in the engine exists to prevent. There is now a settle after the probe,
paid once per camera. Unit tests could not have caught this; only
running the app did.

**Bump BUILD_ROUND whenever probe semantics change.** Both the encoder
cache and the camera-capability cache are stamped with
`__version__|BUILD_ROUND`. If the probing logic changes but the stamp
does not, every existing user keeps an answer produced by the old logic.
It was not bumped for r21 because no shipped build had either cache
populated, so every machine learns fresh regardless.

**The smoke gate fails if a fullscreen game has focus.** Touchless
correctly detects a fullscreen foreground app and swaps off GPU
inference; two engine swaps inside the 45 s window leave the marker
reporting `engine_started=false` with frames still counted, and the gate
refuses to sign. This is the product working, not a defect. A build at
09:34 passed with the same game running but NOT focused. Before a
release build, check the foreground, or re-run with `--resume` which
skips the rebuild and goes straight back to the gate.

### 2.13 r22: two bugs the r21 field log exposed

The r21 capability probe worked -- it enumerated the camera and asked
for the advertised 30 fps, and the "could not set video options"
rejection is gone. The fast path still failed, for reasons that had
nothing to do with the probe.

**Every camera open before ffmpeg needs a settle, not just the probe.**
The short-shutter PRE-FLIGHT is a second open/close and handed the
device over 0.40 s later, which is the same teardown race. Raised to
0.60 s. Unverified on the slow rig.

**A diagnosis the system can disprove must be recorded after the
evidence.** `record_failure` never counted a "busy" verdict, on the
reasoning that another process might hold the camera. But if the OpenCV
fallback opens that same camera a second later, nothing held it. The
memo is now written AFTER the fallback with `device_confirmed_free`, so
a disproven verdict counts and an unchallenged one still does not.
Without this the memo never learned and every mode switch re-ran the
whole cascade: seconds of frozen UI plus fresh ffmpeg processes for the
antivirus to prompt about.

**Gate on the thing, not on the intent.** The exposure hint was skipped
whenever the MODE wanted ffmpeg, rather than when the capture actually
WAS ffmpeg. So a failed ffmpeg open left the camera on auto exposure --
a long shutter in a dim room, about 10 fps with visible blur. That is
why Lite and GPU measured SLOWER than Default on the field machine with
worse tracking, which should have been the tell: modes whose purpose is
speed cannot lose to the baseline unless something is being skipped.
The hint is dead only on a real ffmpeg capture, where ffmpeg owns the
DirectShow graph and cv2 .set() reaches nothing.

**Reading the field log:** `[r49-short-shutter] skip: reason=` now ends
in `_on_ffmpeg_cap`, so a future log distinguishes a legitimate skip
from the r21-era wrong one.

### 2.14 r23: the exposure constant was tuned on the wrong camera

The r22 field log showed the short shutter working exactly as designed
and the picture still being unusable: `[r49-short-shutter] applied
EXPOSURE=-6.0, readback exp=-6.0` alongside `[gamma-lift] median=10.0
target=100` held for thirty seconds. The user's words were "dim and
practically black and white".

**`CAP_PROP_EXPOSURE` is log2(seconds), so -6.0 is a 15.6 ms shutter --
a 64 fps ceiling.** The 2026-07-27 A/B that chose it ran on a 60 fps
camera, where that is right. The field webcam advertises
`mjpeg 640x480@30`: one frame every 33.3 ms. We were spending half the
light budget buying frames that sensor cannot produce.

`src/hgr/app/camera/exposure_policy.py` now derives the value from the
frame period instead: the longest whole stop that still fits, clamped to
[-6, -4]. A 30 fps camera gets **-5.0, twice the light at the same
30 fps**; a 60 fps camera still gets -6.0, so the reference rig is a
no-op by construction. The rate comes from the capability data r21
already caches, so this costs no probe, no camera open, no extra
latency, and no new antivirus prompt. Unknown camera falls back to -6.0.

| exposure | shutter | fps ceiling |
|---|---|---|
| -4.0 | 62.50 ms | 16 |
| -5.0 | 31.25 ms | 32 |
| -6.0 | 15.62 ms | 64 |

**The display lift was desaturating by construction.**
`_compensate_short_shutter_for_display` lifted Y and merged the ORIGINAL
Cr/Cb back. YCrCb chroma excursion scales with signal level, so lifting
luma ~8x while leaving chroma alone drops perceived saturation by the
same factor: **measured 20% of true**, which is literally "practically
black and white". The flat `_SAT_GAIN = 1.15` that was supposed to
offset it was ~6.6x too small and moved saturation from 20% to 21%.
It is now derived from the luma gain actually applied, as **one
256x256 uint8 table** (row = the pixel's PRE-lift luma, column = its
chroma), 64 KiB, rebuilt only when gamma moves by more than 0.02.
Chroma is resampled at HALF resolution -- the reason 4:2:0 subsampling
is invisible -- which cuts the gather from 4.7 ms to 1.5 ms. Measured
20% -> 93% of true saturation at median 10, 99% at the new exposure, and
the whole display stack got **faster** (7.6 -> 5.1 ms per frame) because
it replaces two float32 clip/cast passes.

**The gain must be per pixel.** A single gain derived from the frame
median restores the shadows but over-saturates everything brighter than
the median -- measured 148% of true on a lit face, which reads garish.
That scalar-gain design was tried and rejected; do not reintroduce it.

**Keyed on gamma, and that is sufficient** -- the per-pixel gain is
`lut[y] / max(y, 1)` where `lut = (y/255) ** gamma * 255`, a function of
gamma and the pixel's own luma only. The median enters solely through
gamma, so there is nothing left to go stale. The table is still cleared
when the latch drops, so a camera or mode change cannot carry a previous
camera's gain into the next one's first frames.

**The CLAHE stage must follow the gamma stage's verdict, not the latch.**
It used to gate on `_short_shutter_active_for_display` alone while the
gamma stage ALSO early-returns once the median reaches target, so an
already-bright frame still got CLAHE 3.0 plus a saturation push
(measured: saturation +26%, value 179 -> 172, ~3.5 ms/frame). That is
exactly the "washed-out / dim live view" of section 3.5. It now returns
early when `_ss_disp_target_gamma` is None, which makes the whole
display stack self-disabling: 0.010 ms on a well-exposed frame. This
also makes the latch-lifecycle bugs below visually inert.

**Known, deliberately NOT fixed in r23 -- the latch lifecycle.**
`_short_shutter_active_for_display` is raised in two places but lowered
only inside `_release_ffmpeg_preflight`, which early-returns unless
`_ffmpeg_preflight_device` is set -- and the r49 raiser never sets it.
Worse, `_apply_perf_camera_path` assigns `_ffmpeg_preflight_device =
None` on the line before it calls `_release_ffmpeg_preflight(...)`, so
the "leaving ffmpeg mode releases the dark exposure" path has been a
guaranteed no-op. Fixing the ordering is correct, but it turns an
already-reviewed driver write (EXPOSURE=-4 then auto) from never-firing
into firing on the field rig, and section 3.9 records a native access
violation from DShow property writes there. The display gate above
removes the user-visible symptom, so this lands on its own round where a
field regression would be attributable. See OPEN_ISSUES 1.10.

**A low smoke-gate fps is usually the ROOM, not the code.** r23's gate
read 294 frames / 15 fps in GPU mode where the same gate read 1797 / 60
fps at midday. Nothing had regressed: `tools/probe_delivered_fps.py`
with Touchless CLOSED reported `640x480 negotiated=60 DELIVERED=15.0
EXPOSURE=-4.0 frame_mean=64`. The Kiyo Pro's own driver, in a dim
late-afternoon room on auto exposure, had stretched its shutter to
~1/15 s; the app tracked it at ~100% (15.45 vs 15.0), and
`[camera-controls] session changes: none (engine wrote: nothing)`
confirms it never touched the driver. Which is this whole section's
thesis reproducing on the reference rig — brightness and frame rate pull
against each other on any auto-exposure camera, and the dev rig has the
short-shutter hint deliberately off so nothing stops it.

Run that probe BEFORE investigating any dev-rig fps drop; it takes 20
seconds. Gate numbers are only comparable at similar room light. Cross
-check `[ffmpeg_capture] ... luma check passed: median=` between runs
(104.6 midday vs 71.7 late afternoon).

**BUILD_ROUND stays at 57 for r23.** The capability-probe semantics are
unchanged, and `_caps_stamp()` is `version|BUILD_ROUND`, so bumping it
would invalidate every persisted `camera_capabilities` entry -- and the
new exposure policy reads that cache at the initial camera open, before
anything re-learns. Bumping it would have silently reverted fault (A)
to -6.0 on the first launch after the update.

**The antivirus prompts are a spawn-count problem, and the generator was
automatic.** `_apply_perf_camera_path` reads "not currently on an ffmpeg
capture" as "never tried", so every Lite/GPU toggle re-ran the whole
cascade -- and `_engage_auto_low_fps` re-enters it with NO user action on
any rig under `_CRITICAL_FPS_THRESHOLD` (12 fps), which oscillates. The
field rig sat under 12 fps for the whole session. There is now a 90 s
per-device+size re-attempt cooldown, in memory only, cleared on camera
recovery. Deliberately NOT a session-wide condemnation: the two-strike
persisted memo exists so a momentarily-busy premium camera is never
permanently demoted (§2.11), and a one-strike rule would undo that.

**Deliberately NOT done:** the too-dark rollback at
`_auto_applied = _decision_reason == "classifier_generic"` is still dead
for a `config_user_chose` machine. Widening it would let the app undo an
explicit user setting and cost that user the 3x fps the short shutter
actually buys them (the field log measured 30 fps at -6 against ~10 fps
on auto). §2.4/§3.3/§3.8 narrowed that gate over three rounds. Left
alone on purpose; revisit only with a field log that shows -5 is still
too dark.

**Also deliberately NOT done:** gain/brightness compensation at the
driver. It is the textbook answer to a dark sensor, but §3.9 records that
the first new DShow property writes on this exact rig produced a native
access violation. Not worth spending a field cycle on.

### 2.15 r24: four modes, one engine

The field report was "none of the modes do anything and the default
performance is much lower than it should be -- max 10.3 fps in default,
lite, gpu, and with boost performance on". The modes were not similar.
They were the same engine.

**The tier was computed from the wrong input.** At 10.3 fps the rig
trips the critical auto-low-fps gate (`fps < _CRITICAL_FPS_THRESHOLD`,
12.0, held for `_CRITICAL_FPS_ENTER_SECONDS`). `_engage_auto_low_fps`
then re-derived which tier had fired as
`fps < _CRITICAL_FPS_THRESHOLD - _CRITICAL_FPS_ENGAGE_MARGIN` = 8.0.
10.3 is not below 8.0, so a machine that had just tripped the CRITICAL
gate was filed as **"fullscreen"** -- a tier needing 18 fps to exit,
unreachable on a camera delivering 10, and one the hard-escape block
skips outright (`if tier == "critical"`). Permanent latch.

`if self._low_fps_active:` is the FIRST branch of
`_build_engine_for_fps_mode`, so while latched Default, Lite and GPU all
build the same 384-wide complexity-0 detector, and line 6282 also passes
`gpu_mode=... and not self._low_fps_active` = False.

The margin was a v1.1.9.2 guard against transient sub-12 dips during
engine swaps latching the critical tier. Transients are already handled
upstream by `_CRITICAL_FPS_ENTER_SECONDS` -- six continuous seconds under
threshold. **A second fps test at the moment of engaging was never the
right instrument.** The tier is now which gate fired, stated by the
caller: `_engage_auto_low_fps(reason="critical"|"fullscreen")`.

**GPU was unreachable whenever Lite was on.** `elif lite_active:` sat
ahead of the GPU path and hard-codes `prefer_gpu=False`, while the
comment directly below it states the intended ordering is
Default < Lite < GPU. Now `elif lite_active and not prefer_gpu:`.

**Lite has no compute story, by arithmetic.** `_LITE_PROCESS_WIDTH = 640`
and `_NORMAL_PROCESS_WIDTH = 960`, but `detector.process` only downscales
when `width > max_process_width`. At a 640-wide camera neither fires, so
Default and Lite run the identical graph on the identical frame. Lite's
only capture delta is `fps_target = 60.0 if lite_active else 30.0`, and
this camera advertises 30 everywhere. Lite's real speedup was always the
ffmpeg MJPG pin, which has never once opened on that camera. Left as is:
the rig is shutter-bound, not inference-bound, so lowering the width buys
nothing and costs landmark accuracy. See section 2.2 -- that constant
stays.

**`try_open_msmf_mjpg` (camera_utils.py:292) has ZERO callers.** It is
the only path that supplies FOURCC/W/H/FPS to the 3-arg
`cv2.VideoCapture` constructor before negotiation, which is the only way
to actually pin MJPG on Windows. The plain `cap.set(CAP_PROP_FOURCC)` the
code does use is accepted by DShow and silently ignored -- camera_utils
says so itself. Not wired up this round (at 640x480 both yuyv422 and
mjpeg advertise 30, so it buys little on that rig) but it is the obvious
lever for any camera whose uncompressed pin is slower.

**A disproven "busy" verdict is now conclusive on the FIRST occurrence.**
r22 made it count one strike; at two strikes the user still sat through a
second 20-26 s frozen mode switch and a second antivirus prompt to learn
what the first already proved. The Kiyo Pro protection is untouched: when
Synapse really holds the device the OpenCV fallback fails too, so
`device_confirmed_free` is False and the two-strike path still applies.

**Strikes were filed under a key nobody reads.** Both sites consulted the
pre-plan size and recorded the size the r21 capability plan had
rewritten. On any camera whose plan downshifts, the whole r20/r21/r23
memo was dead code and the doomed cascade repeated forever.

**BUILD_ROUND is now identity only, and safe to bump every build.** It
used to be mixed into `_caps_stamp()`, so bumping it discarded the
learned camera capabilities -- which is why r21, r22 and r23 all shipped
as "build round 57" and a field log could not be attributed to a build.
Cache invalidation moved to `CAMERA_CAPS_PROBE_VERSION`. The startup
banner now carries `exe=<name>:<size>:<mtime>`.

**The chain that should now hold on that rig:** memo skips ffmpeg on the
first failure (no repeat stalls) -> the mode setter still calls
`_apply_default_capture_tuning` -> r22's `_on_ffmpeg_cap` gate lets the
shutter land on the OpenCV cap -> r23 writes -5.0 instead of -6.0 (twice
the light at the same 30 fps) -> fps recovers -> r24's correct tier lets
the latch release at 20 fps -> the modes differentiate again.

**What to read in the next bundle, in order:**
1. the startup banner's `exe=` -- is this even the build we shipped?
2. `[perf-auto] auto-engaging low_fps: reason=` -- did it latch, which
   tier, and what exit gate is it waiting for
3. `[exposure-policy] ... -> exposure -5.0`
4. `[gamma-lift] median=` -- should read ~20, not ~10
5. `[cap-read] ok_pct` with `HGR_TICK_TIMING=1` -- the only line that
   separates a camera not delivering frames from a pipeline that cannot
   keep up

### 2.16 r24: a bright short shutter, paid for at the driver

The fourth part of the field ask: "boost performance should do the short
shutter speed **without dimming live view**". §2.14 already halved the
darkness by picking the longest shutter that fits one frame period, and
r23 stopped the display lift desaturating. What remains is that a short
shutter genuinely collects less light, and §3.5 says a display-side
stack is the wrong cure — "fighting the camera driver rather than
getting the raw signal right".

`src/hgr/app/camera/light_policy.py` (pure, 38 tests) proposes `Gain`,
then `Gamma`, then `Brightness`, from the driver's own `GetRange` values
that `dshow_controls.snapshot_all` already reads. It never assumes:

* **Gain is often absent.** The field camera exposes Brightness,
  Contrast, Hue, Saturation, Sharpness, Gamma, WhiteBalance and
  BacklightCompensation — no Gain. The physically correct knob simply
  is not there, so the plan falls through.
* **Ranges are driver-specific.** No 0–255 convention. Targets are
  capped at `default + 0.6 * (max - default)`, because a UVC driver's
  raw maximum is usually a washed-out grey card. Field plan: Gamma
  165 → 252 (max 500), then Brightness 0 → 19.
* **Direction is a convention, not a guarantee.** So nothing is claimed:
  measure, write, measure again, and write back anything that darkened
  the frame or moved it under `MIN_USEFUL_GAIN` (6 luma). Verified
  against a simulated backwards-Gamma driver.

Fires only under 70 median luma, at most two attempts, once per device
per session. Kill switch `HGR_SHORT_SHUTTER_LIGHT_LIFT=0`. No-op on the
reference rig — the Kiyo Pro classifies `None`, so the hint never
applies.

#### Where it runs, and why that is the whole design

**Only on a throwaway `cv2.VideoCapture` in the pre-open window.** This
is §3.9's rule verbatim: "runs on a throwaway `cv2.VideoCapture` in the
pre-open window … and never touches the live engine cap or a
`ThreadedCvCapture` whose reader is running."

The first implementation ran on the live cap, at the existing
`CAP_PROP_EXPOSURE` site inside `_apply_default_capture_tuning`, and
argued it was safe because it inherited that write's call site and gate.
**That reasoning was wrong, and it is worth recording why**, because it
sounded convincing: §3.9's rule is not about inheriting a gate, it is
about how many DirectShow Sets are racing the reader thread. What
changed in r17 — the round that killed START on the dad rig with a
native access violation, read at 0x20 — was precisely that number, and
the lift turns one Set into as many as five.
`ThreadedCvCapture.set()` takes `DSHOW_GRAPH_LOCK` but the reader
deliberately does not (module comment: "read() never does, so the hot
path is untouched"), so the lock does not serialise them at all.

The pre-open window is not merely safer, it is where the feature
actually works:

* no reader thread exists, so a blocking `read()` returns genuinely new
  frames — measurement needs no tricks and steals nothing;
* UVC drivers latch these properties at the device, so the values carry
  into the real capture. This is the same mechanism
  `_preflight_short_shutter_for_ffmpeg` has relied on for EXPOSURE since
  r54;
* the live view is already down, so the ≤ 2.5 s budget is not a freeze
  of a running preview.

Four windows, all routed so none can be forgotten:

| Path | Window |
|---|---|
| GPU / Lite (ffmpeg) | inside `_preflight_short_shutter_for_ffmpeg`, reusing the throwaway cap it already opened — zero extra opens |
| every re-open in `_apply_perf_camera_path` | `_open_index_taking_the_light_window` |
| ordinary start (Default + Boost) | `_maybe_lift_light_before_open`, from `_open_camera` |
| live Boost toggle | **none** — see below |

The first attempt covered one of the four re-open sites in
`_apply_perf_camera_path`. All four now go through one helper, and
`tests/test_light_lift_wiring.py` asserts the literal
`open_camera_by_index(` no longer appears in that function.

Device identity pre-open comes from `list_cameras_qt_only()` — Qt's
registry, which does not instantiate a DirectShow filter graph and so
cannot hit the Canon EOS crash path the cv2 probe can.

#### What it deliberately does not do

* **A live Boost toggle gets no driver lift.** The Settings checkbox
  retunes the live cap in place (`main_window` →
  `_apply_default_capture_tuning((None, worker_cap))`), which is not a
  pre-open window. That session keeps the display-side lift, which is
  exactly what it is for; the driver lift lands at the next camera open.
* **No single-camera fallback in `_procamp_for_device`.** Every value in
  that block becomes a `cap.set` target, so another device's ranges must
  never be substituted — and the name would also land in the
  driver-write ledger for a camera the app never touched.
  `_caps_for_exposure_lookup` can afford that fallback because it only
  picks a frame rate.
* **Only KEPT lifts are ledgered.** `_note_driver_write` is called after
  the verdict, never before. STOP restores everything the ledger calls
  ours, so recording a reverted write would silently roll back a Gamma
  the user changed in Synapse mid-session.
* **A rejected write (`cap.set` → False) is not measured or reverted.**
  A False return is the one reliable "it did not land" signal; a True
  return is not (see the r49 note).

#### Audited before shipping

A 24-agent adversarial pass raised 15 findings; 4 survived refutation.
Two were the budget being checked only at the loop head and the routine
freezing the GUI thread on mode toggles — both answered by the redesign
(the deadline is now threaded through every measurement, and the live
call site is gone). One was refuted on a point worth keeping:
`snapshot_all(only_name=...)` does **not** avoid binding other devices
— `_each_filter()` calls `BindToObject` on every video device and
`only_name` filters *after* the yield, so the Canon exposure is
identical either way. The remaining upheld finding is the tutorial gap
in OPEN_ISSUES §1.11.

**What to read in the next bundle:** `[light-lift]`. It logs on every
path including the skips, and "could not measure the frame" and "is not
dark enough" are deliberately different lines — a single silent return
is how a dead feature hides.

### 2.17 r25: the modes finally differ, and r24's crash

Field report on the 09-25 bundle: "Default mode is still really bad,
barely hitting 10fps ... when i turn on boost performance the fps jumps to
30fps but the hand detection isnt much better. Lite mode and gpu mode
still do nothing ... **Before** lite mode was bringing fps up higher and
it made hand tracking very smooth."

**Read the banner first.** That bundle was `build round 57` and its
`system.txt` said `Collected: 2026-09-24 19:06`. r22, r23 and r24 had
none of them ever run on that machine. Three rounds of field diagnosis
went into builds that were never installed. This is why `BUILD_ROUND` is
now identity-only and why the banner carries `exe=<name>:<size>:<mtime>`.

#### The crash r24 shipped

r24 routed all four camera re-opens in `_apply_perf_camera_path` through
`_open_index_taking_the_light_window(index, device_name)`. Three sites
are inside `if want_ffmpeg:`, where `device_name` is bound at 6433. The
fourth is in the `else:` body — so Python made it a function local and
the ffmpeg -> OpenCV restore raised `UnboundLocalError`, *after*
`self._cap = None` and `release_capture_serialised(old_cap)`. Turning
Lite or GPU back OFF left a dead camera. Reference-rig only: the field
rig's ffmpeg never engages, so it never takes that branch.

It shipped because the guard asserted on the function's SOURCE TEXT
(`"open_camera_by_index(" not in body`). **A text assertion cannot see a
name-binding error on a branch it never executes.**
`tests/test_perf_camera_path_branches.py` now calls the function down
both branches, and carries a generic AST check for "read in the else
branch, bound only in the if branch". Mutation-verified: restoring the
bug fails 5 tests.

`device_name` is now bound from `info.display_name` before the split.
It is deliberately NOT the DirectShow name: `resolve_dshow_device_for_index`
spawns `ffmpeg -list_devices` (6 s timeout) on a cache miss, on the GUI
thread — §2.11 records r20 cutting ffmpeg spawns per launch from 10 to 1,
and hoisting that call would hand one back, paid entirely by the
reference rig. `_procamp_for_device` strips the `" (Camera N)"` suffix,
so the Qt name resolves to the same device for free.

The same shape, found in the same round: `_memo_w`/`_memo_h` were bound
only inside `if _memo_dev:` but read at the strike-recording call —
inside a bare `except Exception: pass`, so instead of crashing it
silently swallowed the ffmpeg strike and the doomed 15 s cascade was
never remembered. Both now bind before the split; they depend only on
config flags, never on the device name.

#### Why Default was 10 fps

10 fps is a ~100 ms shutter. The same camera reports `driver_fps=30.0`
the instant Boost forces a short exposure, and advertises 30 fps at
640x480 on BOTH mjpeg and yuyv422 (`fourcc=844715353` = `YUY2`, so the
MJPG hint is being ignored and does not matter). **Light, not format,
was the limit.**

r23's per-camera exposure policy was supposed to fix this and could not:
`_camera_caps_learn` has exactly two callers and both are on the ffmpeg
branch, so **Default never learns the camera** and
`_short_shutter_exposure_for` always fell back to -6.0 — the constant it
exists to replace. The driver was volunteering the answer in the same
log line as the write. `_short_shutter_exposure_for` now takes
`driver_fps` and uses it when no capability is learned:
30 fps -> **-5.0** (31 ms, twice the light, still inside a 33 ms frame),
60 fps -> -6.0 unchanged, 0/None/absurd -> -6.0 historical fallback.
No probe, no camera open, no ffmpeg spawn, no antivirus prompt.

#### Why Boost gave 30 fps and no better tracking

At -6.0 the frame measured **median luma 10 of 255**. MediaPipe was
being handed a near-black image; frame rate cannot help that. r24's
driver light lift was built for exactly this and was dead on the field
rig in two independent ways, both introduced by r24:

* `_maybe_lift_light_before_open` returned early when
  `preferred_camera_index` was None — which is the default for every
  user who never opened the camera dropdown, and is what the field
  config shows. It now resolves to the first available camera, the same
  way `open_preferred_or_first_available` does.
* `_lift_light_pre_open` spent its once-per-device token BEFORE opening
  the camera, so a single transient disabled the lift for the session.
  `_recover_light_after_short_shutter` now returns whether it actually
  measured, and the token is spent only on a reading.

Traced against the field rig's real config: luma **13.7 -> 57.7**.

#### Why Lite did nothing — and why it used to work

`HandDetector.process` downscales only when
`width > max_process_width` (detector.py:120). `_LITE_PROCESS_WIDTH` was
**640** and the camera delivers exactly **640**, so `640 > 640` is False
and Lite ran the identical graph on the identical frame as Default. Its
only surviving delta was `stable_frames=1`, which changes gesture
latency, not fps.

Lite's real lever used to be `model_complexity=0`; **02e343a deleted
it**, because c=0's noisier landmarks read as "laggy hand" on the 60 fps
reference rig. The replacement width only bites on cameras delivering
more than 640 — i.e. the reference rig's ffmpeg-MJPG path, never the
cheap-UVC OpenCV path Lite exists to rescue. A textbook
[[two-rig]] regression: the fix for one rig removed the only thing
helping the other.

`_LITE_PROCESS_WIDTH = 480` restores a real saving on any camera at or
above 640 wide — ~56% of the pixels — with complexity still 1, so
02e343a's finding is respected. §2.2's pinned contract is "Lite is
MediaPipe complexity=1 CPU, NOT ONNX+DirectML"; `prefer_gpu` stays False
and that is unchanged.

#### Why GPU mode was unanswerable

`providers=["DmlExecutionProvider", "CPUExecutionProvider"]` is a
preference ORDER. When the DML EP cannot claim the graph, ONNX Runtime
silently runs on the CPU — and `get_providers()` had **zero callers in
the repo**. The app did not know whether it was on the GPU, so neither
could a bundle. It now logs the bound providers unconditionally, and
`[hand_runtime]`, `[onnx_runtime]` and `[detector]` are no longer
filtered out of the debug bundle's perf-triage file. `[detector]` is
new and reports delivered width vs the lever — the one number that says
whether Lite is doing anything.

#### Known, not fixed this round

* The ~15 s ffmpeg cascade runs synchronously on the Qt GUI thread
  (`GestureWorker` is a plain QObject; no `moveToThread` in `src/hgr`
  outside the updater). The 20 s `paint_gap` in the field log is that.
* `_release_ffmpeg_preflight` hands exposure back to AUTO on the
  FAILURE path too, so a Lite/GPU attempt that falls back to OpenCV with
  Boost on lands on auto exposure — strictly worse than Default+Boost.
* `_directml_auto_demoted`, the sticky GPU safety net, is read at 6663
  and assigned nowhere in the repo.
* GPU mode still hard-couples the inference tier to the ffmpeg camera
  path, so on a rig where ffmpeg always fails "GPU mode" is mostly a
  doomed camera cascade.

**What to read in the next bundle, in order:** the banner's `exe=`;
`[exposure-policy]` (expect `-5.0`, and "no learned capabilities;
driver-reported 30 fps"); `[light-lift]` (expect a median well above
10); `[detector] delivered width=... -> downscaling to 480` in Lite;
`[onnx_runtime] session providers:` in GPU.

## 3. Wrong paths — DO NOT RETRY

Every item here was tried in this codebase and made performance worse or
broke user-visible behavior. Recorded so future debugging doesn't repeat.

### 3.1 MSMF-first camera path

**Tried:** Setting `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=1` and
`OPENCV_VIDEOIO_PRIORITY_MSMF=10000`, plus an MSMF-first path in noop_engine.
**Result:** 4-5 s delays on every camera open when MSMF failed to negotiate.
No improvement to Default or GPU mode fps.
**Status:** Fully reverted. Do not reintroduce without new evidence that
MSMF has an advantage over DShow + ffmpeg-MJPG on the target hardware.

### 3.2 ONNX+DirectML Lite mode

**Tried:** Making Lite mode use ONNX Runtime with DirectML at a lower
resolution than GPU mode.
**Result:** Lite became GPU-dependent, defeating its purpose as the
"universal upgrade from Default" for users with no discrete GPU.
**Status:** Rewritten as MediaPipe c=1 CPU @ 640-wide. See §2.2.

### 3.3 r53 kick auto-fire for Kiyo Pro

**Tried:** First gate was `_classifier_verdict is not False` — but Kiyo Pro
presents as generic "Camera 0 (DirectShow)" so `verdict=None`, and None is
"not False" → kick fired.
Second attempt kept `_user_chose_kick` in the OR chain → sticky-forever
once the user touched the toggle.
**Result:** Kiyo Pro exposure state got damaged on every camera open.
**Status:** Gate is now `_armed or _known_generic_needs_kick` only. See §2.4.

### 3.4 Aggressive reacquire neutralization

**Tried:**
```python
_STALE_ROI_RETRY_FRAMES = 0     # was
_REACQUIRE_WINDOW_FRAMES = 0
_MAX_RELAXED_CANDIDATES_PER_FRAME = 0
_REACQUIRE_SCORE_FLOOR = 1.0
```
**Result:** GPU-mode swipes required sweeping the hand across the full
camera view because the tracker dropped the hand mid-swipe on the first
motion-blur frame, then palm-detect needed the hand centered again.
**Status:** Restored `_STALE_ROI_RETRY_FRAMES = 2` (30-60 ms bridge for
motion blur). Other three stay at 0 to avoid the 20-frame relaxed-threshold
fps regression that motivated the original neutralization. See §2.1.

### 3.5 Gamma + CLAHE live-view post-processing

**Tried:** Enabling gamma correction and CLAHE on the display path.
**Result:** Washed-out / dim live view. Root cause: fighting the camera
driver rather than getting the raw signal right.
**Status:** Reverted. Correct fix is Synapse HDR-off + manual exposure at
the driver level, no display-side stack.

### 3.6 "Runs on any 2015 CPU" tier copy

**Tried:** First-draft Test-My-PC dialog copy said Default runs on any
64-bit Windows PC with a 2015 CPU + 4 GB RAM.
**Result:** Ignored concurrent-app impact. A 2015 dual-core running
Touchless at 30-40% CPU tanks the user's browser and video calls.
**Status:** Rewritten to lead with "leaves room for browsers and video
calls" and to name concrete comfort thresholds (8th-gen quad-core+).
See `feedback_perf_tier_framing.md` in memory.

### 3.7 Assuming Spotify caused every freeze

**Tried:** First-pass diagnosis blamed spotify_controller for the 2016 ms
freeze during mouse control.
**Result:** Wrong. The workflow audit ranked cap.read() UVC driver stall
as the #1 suspect (canonical 2 s dropped-frame shape, no Spotify log line
in the freeze window). Spotify's per-frame polls are already cache-gated /
off-thread.
**Status:** Spotify startup refresh moved off-thread (§2.9) as a durable
fix for the earlier logs' pattern. Camera-off-thread deferred as a
follow-up structural change; not fixed in this checkpoint.

### 3.8 Sticky `_user_chose_kick` opt-in

See §3.3. Recorded separately here because it took two rounds to unlearn:
"user toggled" is a one-time event, not a permanent config latch.

### 3.9 r17 "cross-session AUTO_EXPOSURE restore" (third recurrence of §3.3/§3.8)

**Tried (2026-09-23, commit 67e9e21, removed in eae5713):** A block in the
OFF branch of `_apply_default_capture_tuning` that fired when
`camera_force_short_shutter_user_chose` was True, `_armed` was False and
the raw `CAP_PROP_EXPOSURE` readback looked "short" (< -2.5) or "manual",
then wrote `CAP_PROP_AUTO_EXPOSURE` 0.75 → 3.0 to the LIVE engine cap.
Intended to un-latch a cheap UVC left at EXPOSURE=-6 by a prior session.
**Result:** Two failures at once.
1. On the reference rig the Kiyo Pro reads `EXPOSURE=-4.0` in normal auto
   mode, so the readback threshold was a false positive and the write
   knocked the camera back into auto/HDR (60 → 25 fps). User-visible
   regression the same day: "ok but now performance is bad".
2. On the dad rig (generic "FULL HD 1080P Webcam", lite_mode → ffmpeg cap
   fails → fresh OpenCV cap wrapped in `ThreadedCvCapture`) these were the
   first-ever property writes r17 issued that r16 did not, and r17 died
   on START with a native access violation (read at 0x20) while r16 ran
   fine on the same rig. Leading hypothesis: DSHOW property Set racing the
   reader thread's `cap.read()` on the same filter graph.
**Why the pre-commit audit missed it:** it checked every §2 constant
literally (kick-gate grep passed, `_should_kick` untouched) but did not ask
whether the new block was §3.3/§3.8 *semantically* — a driver write gated
on the sticky user_chose flag under a new name.
**Status:** Removed. Rules going forward:
- Any hunk that calls `cap.set(CAP_PROP_*)` must be reviewed against every
  §3 item, asking: "what does the Kiyo Pro read back in normal auto mode?"
  (−4.0) and "is this gated on anything sticky or on a bare readback
  threshold?" A yes to either is a §3.3/§3.8 reintroduction.
- Cross-session driver restore, if ever needed, keys on an explicit
  per-camera marker the ON path wrote to config, runs on a throwaway
  `cv2.VideoCapture` in the pre-open window (where
  `_preflight_short_shutter_for_ffmpeg` already runs), and never touches
  the live engine cap or a `ThreadedCvCapture` whose reader is running.
- Frozen builds are smoke-launched locally (START via `HGR_AUTOSTART_ENGINE=1`)
  before any upload. r13–r17 each shipped a bug a 60-second launch would
  have caught.

**r18 audit addendum (2026-09-24):**
- The (E) -> dad-rig START crash link is only *reachable* in Default
  mode: `_apply_default_capture_tuning` returns before any cap.get/set
  when Lite / GPU / Low-FPS is on (`_wants_ffmpeg_cap()` gate). With
  dad's persisted `lite_mode=True` the block never ran; the crash is
  more likely a pre-existing DirectShow graph-rebuild race (FOURCC/FPS
  writes on a live `ThreadedCvCapture`, release after a timed-out
  join). Classify the next field report from the Event Log faulting
  module: `cv2.pyd` = videoInput side, `ntdll` = freed critsec /
  release race, `quartz`/`qcap`/`ksproxy` = graph teardown, a Norton
  module = SafeCam injection.
- **DirectShow AUTO_EXPOSURE semantics (from cap_dshow.cpp 4.x):**
  `cap.set(CAP_PROP_AUTO_EXPOSURE, v)` -> Auto only when
  `cvRound(v) == 1` (0.75 qualifies; it writes the driver DEFAULT
  exposure with the Auto flag). Any other value (0.25, **3.0**, the
  -1.0 read sentinel) -> Manual at the current exposure. Every
  "0.75 then 3.0" pair the engine used for years therefore ended in
  MANUAL. Fixed in r18 via `_dshow_auto_exposure_on()`; the guard test
  `tests/test_short_shutter_gate_guard.py` forbids raw pairs.
  `cap.get(CAP_PROP_AUTO_EXPOSURE)` is unimplemented on DirectShow
  (always -1.0), so a stashed auto flag can never be written back.
- r17's r54 low-luma net was removed outright (it disengaged the
  gamma lift in the normal working state of short shutter and was
  keyed on a never-assigned name).
- Cross-session latch handling is now r55: the two ON writers persist
  `camera_short_shutter_latched_for = "<index>|<name>"`; a later open
  with short shutter OFF restores auto on a throwaway cap before the
  engine's capture exists, bounded to 3 attempts, cleared only on a
  verified readback. Empty marker = no-op (Kiyo Pro never sets it).

---

## 4. Test My PC benchmark (CLI debug tool)

Not in the shipping UI. This is a dev / debug tool only.

CLI: `python tools/test_my_pc.py [--warmup N] [--measure N]`

Library module (imported by the CLI): `src/hgr/diagnostics/test_my_pc.py`
- `probe_system()` returns CPU/RAM/GPU/DirectML/camera info
- `MODES` = list of 3 mode presets (Default, Lite, GPU)
- `benchmark_mode(label, ...)` — pure function, callback-based progress
- `benchmark_all(...)` — iterates MODES
- `recommend(results, gpu_info)` → `(mode, reason, lines)`

Recommendation heuristic (1 ms hysteresis prevents flapping):
- GPU wins if `gpu.median_ms + 1.0 < lite.median_ms` AND DirectML available.
- Else Lite wins if `lite.median_ms + 1.0 < default.median_ms`.
- Else Default.

The library was designed with callback-based progress + cancel hooks so
a future in-app dialog could reuse it without a rewrite — if we ever
decide to promote it from debug-only.

---

## 5. Restore procedure

If Touchless performance regresses, follow this order:

1. **Diff constants against §2.** Every value here has been proven; grep for
   the constant name and compare literal values.
2. **Re-run Test My PC in the dialog.** Compare median-ms per mode against
   §1's reference numbers. A drift >5 ms/frame in any mode means the engine
   is off-config or the hardware is throttling.
3. **Grep for the freeze-detector emits.** If freezes are back, correlate
   the mode + gap-ms against the wrong paths in §3.
4. **Check the `[tick-timing]` line's `real_rate` vs `counter_fps`.** If
   they diverge (paint rate crashes while tick loop stays alive), the
   freeze is in the paint pipeline, not the tick. Look at Win32 syscalls
   on the GUI thread first (mouse controller cache, foreground probe).
5. **Check `[cap-read] ok_pct`.** A drop from 80%+ to <20% during a freeze
   indicates ffmpeg-side stall (camera driver hiccup) → the fix is a
   camera worker thread, not a GUI-thread change.

---

## 6. Deferred follow-ups (out of scope for this checkpoint)

These are known real issues but were deemed too structural for this cycle:

1. **`cap.read()` UVC driver stall** — canonical 2 s dropped-frame shape
   observed in field logs during mouse-control sessions. Fix requires a
   camera worker thread with a bounded frame queue. See workflow audit
   scratchpad for design notes.
2. **Clip-cache ffmpeg-stdin backpressure** — `_capture_clip_cache_frame`
   in main_window.py writes to ffmpeg-nvenc stdin on the GUI thread. Any
   NVENC hiccup can block on `write()` for seconds. Fix: bounded writer
   thread with drop-oldest queue.
3. **Foreground process-name probe on UWP apps** — 500-2000 ms on UWP /
   MSIX foreground windows. Fix: move to a shared system-probe worker.
4. **Engine-swap on GUI thread** — `_swap_engine_safely` /
   `_apply_perf_camera_path` document 1-3 s stalls. Fix: engine-swap
   worker with pause/swap/resume protocol.

Track these in `OPEN_ISSUES.md` under a "Performance follow-ups" heading
if this checkpoint gets stale.

---

*End of checkpoint.*
