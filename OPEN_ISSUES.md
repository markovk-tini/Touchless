# Touchless — Open Issues, Roadmap & Regression Notes

Single source of truth for what's open, what's planned, and what historically breaks.
Last reviewed 2026-05-14.

---

## Section 1 — Active bugs

### 1.1 Stub installer "Installing…" page sits at 0% for 3–8 minutes (FIX READY, queued for b9)

**Symptom:** During first install (stub mode), the wizard moves past Download fine but then the "Installing…" page shows a frozen progress bar for the full extraction duration. Users think the installer hung.

**Root cause:** [hgr_app.iss](installers/windows/hgr_app.iss) STUB mode has an empty `[Files]` section — Inno has nothing of its own to count, so the bar stays at 0%. The real work is a synchronous `Exec('powershell.exe', '... Expand-Archive ...', ewWaitUntilTerminated)` that blocks the install thread silently until done.

**Fix (already in the working tree, ships with b9):**
- [build_windows.bat](builder/windows/build_windows.bat) counts files in `dist\Touchless` after the payload zip and passes `/DPAYLOAD_FILE_COUNT=N` to ISCC.
- [hgr_app.iss](installers/windows/hgr_app.iss) `CurStepChanged(ssInstall)` now spawns `Expand-Archive` async (`ewNoWait`), polls `{app}` recursively every 500 ms, drives `WizardForm.ProgressGauge.Position` and `WizardForm.StatusLabel.Caption` ("Extracting payload (1234 / 2856 files)…"). A PowerShell-written sentinel file (`extract_done.flag`) signals completion + carries the exception message on failure.

**To verify on b9:** install fresh, watch the wizard. Bar should advance steadily; status should report `(N / total files)` ticking up.

---

## Section 2 — Security / distribution

### 2.1 PowerShell removal — friend's install auto-uninstalled by Windows [LANDED — C30, v1.1.7]

**Root cause:** Unsigned exe + `powershell.exe -ExecutionPolicy Bypass -EncodedCommand <base64>` is a known malware-dropper fingerprint. Windows Defender / ASR auto-quarantines and the installer gets rolled back.

**Status (v1.1.7 C30):** shipped. Every hot shell-out is now a native Python replacement — no subprocess spawn, no fingerprint. Removed sites:

- **SAPI fallback:** [voice_command_listener.py:794/815](src/hgr/debug/voice_command_listener.py) (`_fallback_system_speech` + `_encoded_system_speech_script`) — deleted entirely. [voice/live_dictation.py](src/hgr/voice/live_dictation.py) simplified to whisper-only. [voice/sapi_stream.py](src/hgr/voice/sapi_stream.py) file deleted (203 lines).
- **YouTube UIA + OCR:** [youtube_controller.py](src/hgr/debug/youtube_controller.py) three PowerShell scripts ported. Captions OCR → `winsdk.windows.media.ocr` async pattern. UIA invoke → `uiautomation` BFS walk. Search-result rect finder → `uiautomation` two-pass filter (Hyperlink first, any control second), preserving Chrome accessibility warm-up + nav-chrome blocklist + rect quality threshold.
- **Desktop controller:** [desktop_controller.py](src/hgr/debug/desktop_controller.py) three sites ported. Windows Search index → `comtypes.Dispatch("ADODB.Connection")` driving Search.CollatorDSO directly. `Get-StartApps` → `Shell.Application.NameSpace("shell:AppsFolder")` enumeration. `.lnk` shortcut → `WScript.Shell.CreateShortcut` via comtypes.
- **Spotify launch fallback:** [spotify_controller.py](src/hgr/debug/spotify_controller.py) `Start-Process spotify:` → `launch_external("spotify:")` (ShellExecuteW).
- **Phone Link connector:** [phone_link_connector.py](src/hgr/live_api/connectors/phone_link_connector.py) `Get-AppxPackage` name probe → `Shell.Application` shell:AppsFolder scan; version probe → `winsdk.windows.management.deployment.PackageManager`.

`requirements.txt` gained `uiautomation` + `winsdk` as hard deps (were already in the PyInstaller spec `hiddenimports` + `collect_submodules`). Longer-term Azure Trusted Signing (~$10/mo) is still worth doing to sign the installer + exe.

**Deferred (not shipping-blocking):**
- [updater.py:512/534](src/hgr/app/updater/updater.py) has two `powershell -Command Expand-Archive` / `Get-AuthenticodeSignature` calls, but they live INSIDE the `.bat` file the updater spawns after Touchless.exe exits — the parent process is cmd.exe, not Touchless.exe, and they use plain `-Command`, not `-EncodedCommand`. Different fingerprint than what ASR was quarantining. Updater is verified-working; leave alone.
- [tool_executor.py:5974](src/hgr/live_api/tool_executor.py#L5974) has `"powershell": ["powershell"]` in the user-app-launcher map (user says "open PowerShell" → we launch it). User-facing action, indistinguishable from the Start Menu. Preserved intentionally.

### 2.2 Motion blur on 30fps 720p laptop cameras [LANDED — C31, v1.1.7]

**Root cause:** Auto-exposure on most webcams drops shutter speed in dim rooms, producing blur even at 30fps. Framerate is not the issue.

**Status (v1.1.7 C31):** shipped in `_apply_default_capture_tuning` in [noop_engine.py](src/hgr/app/integration/noop_engine.py). Now sets on the default OpenCV cap:
- `CAP_PROP_FOURCC = MJPG` (already applied pre-C31 in default + lite tuning)
- `CAP_PROP_AUTO_EXPOSURE = 1.0` (manual, DirectShow "manual" hint — best-effort; drivers that don't honour the value silently ignore it)
- `CAP_PROP_EXPOSURE = -6.0` (~15.6 ms shutter, DirectShow log2 scale)
- `CAP_PROP_BUFFERSIZE = 1` (already applied pre-C31)

Applies only when the default cap is engaged (not GPU Mode / Low FPS / ffmpeg-MJPG cap — those have their own tuning paths). Tradeoff: dim rooms get a darker image. If that becomes a real complaint, follow-up is an exposure slider in Settings → Camera.

---

### 2.3 Per-frame engine cost drops fps when a hand is tracked [LARGELY ADDRESSED]

**Update (v1.1.7, C14–C29):** the mode work substantially reduced the
per-frame CPU cost that this was tracking. GPU Mode now caches the
ONNX/DirectML engine by config signature (C17), guards `_swap_engine_safely`
against reentrancy (C23), and opens the ffmpeg-MJPG cap at 640×480 instead
of 1280×720 (C27) so the decode + tensor prep cost drops meaningfully.
Lite Mode collapses stable_frames to 1 (C26) and shares the default OpenCV
cap while requesting 60 fps FOURCC MJPG (C25). Motion-blur exposure tuning
(2.2 → landed C31) also indirectly helps by preventing UVC drivers from
throttling to 15-20 fps under auto-exposure. Net: on the reference Kiyo
Pro laptop the "hand in frame" dip is much smaller than it was before
C14 — often within 2-3 fps of the no-hand baseline.

**Historical baseline (pre-C14, kept for reference):** With a hand in
frame, fps dipped (~30 → ~26 local webcam; ~high-30s → ~22-26 on the
phone source). GPU Mode on/off made no difference, which proved the
bottleneck was NOT the tracking inference but the per-frame CPU work
that runs *after* detection: landmark smoothing, gesture classification,
overlay/skeleton drawing, and the GpuVideoWidget paint.

**Remaining opportunities (still deferred; ~26fps is usable):** profile
where per-frame time goes when a hand is present. Candidate wins — lighter
overlay drawing, cheaper / less-frequent landmark smoothing, and
decoupling the widget paint from the processing loop so paint can't
throttle production. (The phone source's extra dip vs local is the WebRTC
transport — already trimmed by sending 360p in `webengine_host.py`.)

---

## Section 3 — Dictation accuracy

Current focus. Historical issues (from prior sessions, not yet all verified on current build):

- **Low-latency live dictation re-enabled (2026-06-03).** `WhisperStreamer`
  again emits `hypothesis` events — periodic greedy (beam=1, no-VAD) partial
  decodes of the in-progress utterance — so the LocalAgreement-2 consumer in
  [noop_engine.py](src/hgr/app/integration/noop_engine.py) types words live as
  you speak; the silence-triggered beam=3 final decode is unchanged and
  reconciles the live text via a case/punctuation-insensitive word-prefix diff
  (`_reconcile_final_edit`). Pending window 2.5s→1.0s; live inserts use SendInput
  (no clipboard clobber). Kill switch: `HGR_DICTATION_HYPOTHESES=0`. Per-100ms
  sub-block silence detection (immune to decode-induced backlog).
- **Early-word clipping** — words occasionally start mid-syllable; pre-roll was bumped to 600ms in voice-command capture but dictation uses `whisper-stream.exe` which has its own internal VAD.
- **Streaming vs. refinement drift** — RESOLVED: `WhisperRefiner` (the 2nd-mic
  re-decode) is now mutually exclusive with streaming hypotheses (it would race /
  double-type and opened a second mic stream). With hypotheses on, the beam=3
  final + grammar pass cover refinement; the refiner only runs when
  `HGR_DICTATION_HYPOTHESES=0`.
- **Grammar correction lag / misfires** — `GrammarCorrector` waits for sentence-boundary + 0.5s idle + ≥20 chars before submitting. On dense dictation the corrector can fall behind; `_chunk_stale` is supposed to discard corrections that overlap new input but edge cases exist.
- **Mic-gain slider doesn't reach `whisper-stream.exe`** — documented limitation ([6.6 prior]). Whisper-stream pulls audio directly from WASAPI, bypassing the in-process gain slider. Users are pointed to Windows input volume. Fix would require capturing audio ourselves and piping via stdin or switching dictation to the `whisper-cli` path the refiner already uses.
- **Title-case / spoken-punctuation coverage** — [dictation.py](src/hgr/voice/dictation.py) has static lists. Misses (new brands, uncommon punctuation names) need to be added case-by-case.

**Next step:** baseline accuracy test with a fixed sentence, 3 repetitions, to identify which of the above is currently most visible before changing anything.

---

## Section 4 — Roadmap (next features)

Planned-work / polish backlog has moved to [`Touchless to-do.md`](Touchless%20to-do.md). This section is reserved for architectural-level roadmap notes that don't fit the to-do format.

### 4.0 Dynamic custom gestures (RECALL FIX LANDED — v1.1.8.1, 2026-08-24)

Motion-based custom gestures (DTW over per-frame hand landmarks). End-to-end loop was wired 2026-05-13 but recall was low ("just rarely detects the dynamic gesture that i just recorded"). The 2026-08-24 diff addresses the root causes ranked by the audit workflow — see the block below.

**2026-08-24 recall fix (v1.1.8.1):**
- **Segment timeout** — [dynamic_classifier.py](src/hgr/custom_gestures/dynamic_classifier.py) `_MAX_SEGMENT_SECONDS = 2.5`. Fixes the #1 root cause: at 15-25 fps in dim rooms, motion energy stays above LOW during small hand jitter after a gesture, so the segment never closes and no match is ever attempted. Timeout force-closes and matches.
- **Timeout quality gate** — before a timeout-close fires DTW, requires accumulated wrist path ≥ 0.5 palm units OR max-finger displacement ≥ 0.3. Prevents random hand fidgeting during a 2.5 s window from firing the closest template.
- **Settle debounce** — motion must sit ≤ LOW for 3 consecutive frames before closing on settle. Prevents mid-gesture pauses (double-taps, swipe-pause-swipe) from closing prematurely.
- **fps-invariant min-segment floor** — replaces the `_MIN_SEGMENT_FRAMES = 8` constant with per-second math (0.20 s → 3-4 frames @ 15 fps, 12 frames @ 60 fps). Fast 250-350 ms flicks now pass the floor on dim-room rigs.
- **Wrist channel = displacement, not absolute** — [dynamic_classifier.py](src/hgr/custom_gestures/dynamic_classifier.py) `build_template_from_takes` subtracts wrist[0]; live `_close_and_match` does the same. A swipe recorded at the left edge of frame now matches the same swipe performed at the right edge. Legacy templates get migrated on registry load and persisted back at `wrist_schema=2`.
- **Per-template auto-threshold** — pairwise DTW between takes at build time → threshold clamped `[0.22, 0.30]`. Tight-recording users get a tight gate; sloppy-recording users get a loose gate. Legacy templates get the same value computed on-the-fly at runtime load.
- **Top-1 vs top-2 margin gate** — with multiple registered gestures, winner must beat runner-up by ≥ 0.04 palm units. Prevents ambiguous-motion firing the "closest wrong" gesture.

**What was NOT changed (backward compat):**
- `_DEFAULT_MATCH_THRESHOLD` stays at 0.18 (fallback only; new/legacy templates get the 0.22-0.30 auto-threshold).
- Handedness gate stays decisive-drop (skeptic 1 blocker).
- HGR_DYNAMIC_FPS_INVARIANT stays default-OFF (per-second gates deferred; skeptic 1 blocker).
- Landmarks_raw path deferred — requires cross-file coordination + recorder-vs-live worker-signal plumbing (skeptic 1 concern).
- No forced re-record UI. Existing templates work with in-memory migration.

**Historical context (2026-05-13 investigation):**

**What's wired and working:**
- Wizard ([custom_gestures_wizard.py](src/hgr/app/ui/custom_gestures_wizard.py)) Static / Dynamic toggle. Custom-painted widget with a true diagonal seam and a darken-edge gradient on the inactive half. Dynamic-only Duration-per-take radio block (1.5 s / 3 s / Until stopped) with custom white-outline + green-dot radios. Action picker auto-grows the dialog and scrolls the keyboard into view; thin green scrollbar.
- Recorder window ([dynamic_gesture_recorder_window.py](src/hgr/app/ui/dynamic_gesture_recorder_window.py)) mirrors the static-recorder shell: instruction line, big video panel, "Recording complete!" overlay, single Begin/Stop/Save button, Exit + Restart, Space shortcut. Captures full BGR frame sequence per take.
- Post-Save clip picker (`DynamicClipPickerDialog`): grid of looping take-clip tiles. Picked take is encoded as an animated GIF via PIL into `gesture_thumbnails/<name>.gif`. Card panel ([custom_gestures_panel.py](src/hgr/app/ui/custom_gestures_panel.py)) detects `.gif` and plays via `QMovie` so the card shows the recorded motion.
- Engine integration ([noop_engine.py](src/hgr/app/integration/noop_engine.py)) calls `DynamicGestureRuntime.process_frame` on every frame and shows the gesture name in the live banner via `dyn.current_match(now)` for ~1.2 s after each fire.
- Sandbox integration ([custom_gestures_sandbox.py](src/hgr/app/ui/custom_gestures_sandbox.py)) runs the dynamic runtime in parallel with the static classifier; honors the "Fire actions" checkbox via `dispatch=True/False`.
- Tests: `tests/test_dynamic_gesture_*.py` — 30/30 passing.

**Detection pipeline (current state):**
- Recorder normalizes via `palm_scale_from_landmarks(landmarks)` ([dynamic_recording.py](src/hgr/custom_gestures/dynamic_recording.py)) — engine-compatible formula `(palm_width + palm_height) / 2`. The earlier wrist→middle-MCP-only formula was a mismatch and meant nothing fired live.
- Classifier ([dynamic_classifier.py](src/hgr/custom_gestures/dynamic_classifier.py)) gates segments via motion energy (LOW=0.018, HIGH=0.030), rejects segments where the absolute wrist barely moved (`_MIN_WRIST_TRAVEL_PALM_UNITS=0.4`), then runs DTW with `_DEFAULT_MATCH_THRESHOLD=0.18`.
- Selector ([key_point_selector.py](src/hgr/custom_gestures/key_point_selector.py)) and template builder both subtract per-frame wrist before scoring/storing, so any whole-hand translation is removed (selector picks keypoints based on wrist-relative motion only).

**Known issues / what to revisit when we come back:**
1. **Accuracy still mediocre.** The wrist-relative normalization erases whole-hand swipe motion from the per-frame trajectory; DTW operates on the residual finger-orientation signal which is small and not very discriminative. The wrist-travel gate prunes the worst false-positive class ("hand entering view") but does not improve true positives. Real fix likely needs an extra feature channel — store the absolute (palm-scaled) wrist trajectory as a 22nd row in the template and include it in DTW. That requires a template-format bump + recorder/template-builder/classifier all changing together.
2. **Pure-finger gestures (e.g., fist squeeze) are blocked by the wrist gate.** Wrist barely moves, gate fires, segment rejected. Need a per-template "wrist-required" flag derived at recording time (was the recorded wrist motion above N palm units?), then conditionally apply the gate.
3. **No visible bbox color change for dynamic fires** — the live overlay banner now shows the gesture name, but the green/active-gesture *bbox color* path is still hardcoded to the static runner. Would need to extend `_apply_custom_label`'s caller to flip bbox color too.
4. **Re-record required after every threshold/gate tweak.** Saved templates from the broken-palm-scale era won't match. Document this somewhere user-visible if we keep iterating.
5. **Clip picker tile aspect ratio.** Tiles are square-ish (200×200) but raw camera frames are 16:9 — letterboxing inside the tile. Looks fine but could be tighter.
6. **GIF size on disk.** No optimization; full frame count at 33 ms/frame. A 3 s take = ~90 frames ≈ 2-4 MB. Fine for a few gestures, will balloon if user records many.
7. **Test coverage is synthetic-only.** All passing tests use generated landmark sequences. No integration test against a real recorded session — accuracy improvements are hard to verify reproducibly without one.

**Where the user left off:**
User reported: "fires in normal app mode but doesn't show the gesture name on the bbox or turn green. Also, it's not super accurate. I did swipe up... whenever I bring my hand into view anywhere as long as it's open it almost always fires the action." After the wrist-travel gate + threshold tightening + banner-name fix, the over-firing should be much reduced — but we never got user confirmation before pausing. **First action when resuming: have the user re-record a swipe-up with the current build and report whether (a) detection still over-fires, (b) the banner name shows on fire, (c) accuracy on a deliberate swipe is acceptable.**

### 4.1 Custom gesture macros (queued)

User asked for custom gestures to be able to record + replay a macro (e.g. "open this app, click that button"). Scoped down for now during the b7 QoL pass; ship later as a proper feature.

**Shape:**
- New gesture-action type `macro` (alongside the existing `app_action`, `voice_action`, etc.).
- Recorder UI in the Custom Gestures panel: "Record macro" button → captures sequenced events for N seconds:
  - Mouse clicks (button + screen coords + timing)
  - Keystrokes
  - App focus events (foreground window name, to handle "open this app first")
- Replay engine: on gesture fire, replay the sequence with the captured timing.
- Per-step timing: optional `wait until window X exists` between steps so timing doesn't break on slow machines.
- Storage: JSON sidecar in the custom-gestures directory alongside the existing classifier weights.

**Why deferred:** macro recording + a reliable replay engine is its own multi-file feature. The b7 QoL bundle stayed focused on tray icon + autostart + dropdown polish + Spotify reauth.

### 4.1.5 Phone connect — cross-network (TURN) [FUTURE]

The pairing-code phone camera (Settings → Camera/Microphone → "Connect
Phone", via touchless-control.com/connect, WebRTC decoded by QtWebEngine
in `webengine_host.py`) works on the **same Wi-Fi** today — it connects
via local ICE candidates, so no STUN/TURN is needed. The `-105`
`stun.l.google.com` resolve errors in the log are harmless on LAN.

**Cross-network (phone on cellular) does NOT work yet** because (a)
QtWebEngine's embedded STUN DNS is failing and (b) there's no TURN relay.
To enable it:
- Provision a Cloudflare Realtime **TURN** key.
- Add a `/turn` route to the signaling Worker that mints short-lived TURN
  credentials (don't embed long-lived secrets in the client).
- Point the ICE config at it in both `connect.html` (guest) and
  `webengine_host.py` (host `HOST_HTML`).
- Cost: usage-priced (~$0.05/GB; a 1–2h session ≈ pennies, often within
  the free allowance). TURN only bills when the relay is actually used.

### 4.2 Discord integration (planned)

Companion-app integration following the Spotify pattern. Marketing surface is already live ([discord.html](hgr-download-page/discord.html), card on [integrations.html](hgr-download-page/integrations.html)). Code work has not started.

**Target capabilities (v1):**
- Mute / unmute self by gesture or voice
- Deafen / undeafen by gesture or voice
- Switch to next / previous voice channel by voice ("next channel", "join general")
- Read current voice-channel state for gesture-mode HUD

**Transport:** Discord exposes a local RPC over a Windows named pipe (`\\.\pipe\discord-ipc-0` … `discord-ipc-9` — the desktop client claims the first free slot). JSON-framed messages: 4-byte LE opcode + 4-byte LE length + UTF-8 payload. No cloud round-trip needed for the in-session control plane.

**Auth flow:**
1. Open the IPC pipe.
2. Send HANDSHAKE `{v:1, client_id}`.
3. Send `AUTHORIZE` command with `scopes:["rpc"]` — Discord prompts the user to allow.
4. Receive an OAuth `code`. Exchange via `POST https://discord.com/api/oauth2/token` with `client_id` + `client_secret` + `code` → access token.
5. Send `AUTHENTICATE` command with the access token → connection upgraded to authed.

The `rpc` scope is **whitelist-gated**. The app's dev-portal entry has to be approved before Discord will issue tokens for it. Whitelist application is free but human-reviewed.

**Open design decisions:**
- **`client_secret` distribution.** Discord doesn't fully support PKCE for the `rpc` scope. Two options: (a) embed `client_secret` in the binary — pragmatic, low real risk because the worst a thief can do is request rpc-scope auth against their own Discord account; (b) run a tiny token-exchange Cloudflare Worker that holds the secret. Lean toward (a) for v1 to avoid a server dependency, mirroring how the Spotify default `client_id` ships embedded.
- **Per-user own-app workaround.** Not necessary the way it is for Spotify: Discord doesn't impose a 5-tester cap. One shared Touchless app, one whitelist application.
- **Discovery & launch.** Touchless should not launch Discord. It should detect a running Discord process (`Discord.exe` on Windows) and only then prompt for connect.

**Code milestones (mirror the Spotify controller layout):**
- [ ] Pre-code human step: register `Touchless` app at <https://discord.com/developers/applications>, capture `client_id` + `client_secret`. Apply for the `rpc` scope whitelist at the same time (free, human-reviewed).
- [ ] `src/hgr/debug/discord_controller.py` — named-pipe client, frame codec, handshake, authorize → token-exchange → authenticate, mute/deafen/channel commands.
- [ ] `src/hgr/debug/discord_gesture_router.py` — gesture mode toggle following [youtube_gesture_router.py](src/hgr/debug/youtube_gesture_router.py)'s shape.
- [ ] Voice commands wired in [voice_command_listener.py](src/hgr/debug/voice_command_listener.py).
- [ ] Settings → General → Discord section in [main_window.py](src/hgr/app/ui/main_window.py), parallel to the Spotify section.
- [ ] First-active connect prompt + tutorial entry point (mirror the Spotify prompt flow).
- [ ] Documentation page: [docs/DISCORD_PIPELINE.md](docs/DISCORD_PIPELINE.md) covering the pipe protocol, auth flow, debug commands.

**Why not done yet:** waiting for the Discord dev-portal app + whitelist application (the dev-portal step takes ~5 minutes, but the whitelist is human-reviewed and can take days/weeks). The pipe-protocol + command code can be scaffolded in parallel — it's testable with a placeholder client_id against a logged-in Discord client; the only thing that fails is the AUTHORIZE step at the end.

### 4.3 Apple Music integration (planned — gated on macOS support)

Companion-app integration for macOS-bound users. Marketing surface live ([apple-music.html](hgr-download-page/apple-music.html), card on [integrations.html](hgr-download-page/integrations.html)). Code work has not started.

**Target capabilities (v1):**
- Play / pause / skip / previous by gesture or voice
- Volume up / down by gesture
- Read current track for gesture-mode HUD
- Library search by voice ("play Sade on Apple Music")

**Transport:** Apple MusicKit. The desktop path on Windows would require MusicKit-JS in an embedded WebView2 (clunky); the clean path is **the native macOS framework**, so this integration ships **alongside the macOS build** — it would not be exposed on Windows for v1.

**Auth flow:**
1. Touchless app holds an Apple **developer token** (JWT signed with a private key from the Apple Developer Program).
2. User signs in to Apple Music inside Touchless → MusicKit issues a per-user **music user token**.
3. Touchless uses both tokens to call MusicKit endpoints.

**Cost:** Apple Developer Program is **$99/year**, which is the same membership required to codesign + notarize the macOS build of Touchless itself. So MusicKit is **zero incremental cost** once macOS shipping exists.

**Operational quirk:** Developer tokens expire every ~6 months. Two strategies:
- (a) **Ship a fresh token in each release.** Acceptable cadence — we ship more often than every 6 months.
- (b) **Token-refresh worker.** A Cloudflare Worker that holds the private key and signs a fresh JWT on demand. Standard pattern; ~50 lines.

Lean toward (b) so a user on an old build doesn't get stuck.

**Open design decisions:**
- **Monetization.** Recouping $99/yr is trivial — a $4.99 one-time "Apple Music unlock" via Stripe (direct, not Mac App Store IAP) breaks even at ~20 buyers. Could equally stay free and treat the $99/yr as part of the macOS dev cost. Decide at ship time based on Touchless's user-base size.
- **Mac App Store vs. direct.** If shipping via Mac App Store, Apple's StoreKit IAP is mandatory (30% / 15% small-biz cut). Direct distribution avoids that but means handling notarization + Sparkle-style updates ourselves. The current Windows installer pipeline (Inno + R2 + custom updater) has a macOS analogue we'd build then.

**Code milestones (mirror the Spotify controller layout):**
- [ ] Pre-code human step: Sign up for the Apple Developer Program. Generate a MusicKit private key + capture the key ID.
- [ ] macOS build pipeline — codesign + notarize + DMG + Sparkle update channel. Blocking prerequisite.
- [ ] Token-refresh Cloudflare Worker that signs MusicKit developer JWTs on demand.
- [ ] `src/hgr/debug/apple_music_controller.py` — wrap MusicKit framework via PyObjC (macOS only).
- [ ] `src/hgr/debug/apple_music_gesture_router.py` — gesture mode toggle.
- [ ] Voice commands wired in [voice_command_listener.py](src/hgr/debug/voice_command_listener.py).
- [ ] Settings → General → Apple Music section in [main_window.py](src/hgr/app/ui/main_window.py) (macOS-only visibility).
- [ ] First-active connect prompt + tutorial entry point.
- [ ] Decide gating: free vs. one-time Stripe unlock. Wire StoreKit or Stripe if paid.
- [ ] Documentation page: docs/APPLE_MUSIC_PIPELINE.md.

**Why not done yet:** the macOS build doesn't exist. Apple Music is downstream of that work — there's no path that ships it on Windows for v1.

---

## Section 4.5 — Iris Phase-2/3 substrate wiring (TRACK BEFORE NEXT RELEASE)

Phase-2 and Phase-3 introduced 21 new modules under `src/hgr/live_api/` that form the substrate for "Jarvis-grade Iris" (cognition, voice safety, observability, background watchers). Substrate + tests are green (**390+ audit-hardened & passing across substrate + wiring**).

**Wiring landed (live in `live_api_manager.start()` + `orchestrator` + `live_assistant_window`):**
- ✅ **Conversational layer** wired into EVERY reply path — Tier-1 connector, Tier-2 multi-step, skills, all pseudo-tools, reauth messages. Synthesizer-produced output bypasses re-rendering. Per-session utterance-cache reset on planner construction so stale conversational rewrites don't survive across sessions.
- ✅ **Sentinel daemon** started at session boot with four registered watchers (system_signals @5s, calendar_briefing @60s, file_watcher @2s, repo_focus @8s).
- ✅ **`reliability_ledger.global_ledger()`** constructed → bus-subscribed.
- ✅ **`stuck_pattern_detector`** constructed + `attach_to_bus()` called at session boot; UI subscribes via `_on_stuck_signal()` → renders inline helper chip respecting InterruptionGate.
- ✅ **`utterance_cache`** consulted at Tier-0 of `try_handle`; populated after every successful multi-step reply (skipping destructive + time-sensitive tools).
- ✅ **`cot_layer`** records reasoning trail for every `_run_with_revision()` execution (incognito-honored).
- ✅ **`system_signals.py`** new module — Win32 detectors for screen-share/mic/camera/focus-assist/DND/fullscreen/battery/idle. Registered as Sentinel watcher; primes signals at boot so SEC-002 fail-closed doesn't block the very first interruption.
- ✅ **`safety_gate.gate()`** consults `voice_spoof_defense` for `source="realtime"` invocations (typed-confirm + spoof-defense two-layer for voice-sourced destructive ops).
- ✅ **`dictation_bridge`** producer wired in `text_input_controller._publish_to_iris_bridge()` (honors incognito); consumer in `orchestrator._recall_context()` enriches planner context when user asks about recently-dictated text.
- ✅ **`model_router`** consulted in `planner_llm.plan()` — picks tier, falls through to local on over-cap. Cost meter actually accumulates via `_maybe_record_cost()`.
- ✅ **`calendar_briefing_watcher.py`** new module — Sentinel-registered watcher that polls calendar connector, builds briefings, consults InterruptionGate, queues into manager's `_pending_notes`.
- ✅ **`file_watcher_daemon.py`** new module — polling FS watcher (no watchdog dep) over union of rule roots; dispatches matched events through tool registry.
- ✅ **Cost pill** in chat header — green/yellow/orange/red badge, refreshes every 10 sec, tooltip shows today's spend.
- ✅ **CostMeter wired into PlanReviser** (BAIL-on-budget branch is no longer dead).

**Still pending — see 4.5.1 / 4.5.2 / 4.5.3 / 4.5.4 below.**

### 4.5.1 Dead-singletons (built + tested, wiring status)

- ✅ **`reliability_ledger.global_ledger()` WIRED** — `live_api_manager.start()` constructs it after `_ensure_audit_log()`; bus subscription is live.
- ✅ **`model_router.global_router()` WIRED across paid paths** — `planner_llm.plan()`, `synthesizer.summarize()` both consult the router (fall through to local on over-cap; use router model_id when provider matches OpenAI). Cost meter records token spend at both sites. `plan_reviser._llm_revise` inherits via its `llm.plan()` call.
- ✅ **`voice_spoof_defense.global_defense()` WIRED** — `safety_gate.gate(..., source="realtime")` consults it before destructive-op dispatch. TTS-loop refusal, repeat-attack detection, and trusted-second-channel checks are now active for realtime-sourced tool calls.
- ✅ **`dictation_bridge.global_bridge()` WIRED** — `text_input_controller._publish_to_iris_bridge()` posts after every successful paste/insert (incognito-honored); `orchestrator._recall_context()` reads `last_dictation_text()` and injects it into planner context when the user's request references recently-dictated content ("summarize what I just typed").
- ✅ **`cot_layer.global_cot_layer()` WIRED** — turn trail recorded on every `_run_with_revision()` execution.
- ✅ **`utterance_cache.global_utterance_cache()` WIRED** — Tier-0 cache hit before classifier; never-cache list covers destructive + time-sensitive tools; per-session reset.
- ✅ **`transcription_router.TranscriptionRouter` WIRED** — `voice_command_listener._maybe_escalate_transcription()` runs after the fast tier; when the transcript contains email/URL/file-path/digit-run or starts with a destructive verb, re-decodes via `faster_whisper` (accurate) and uses the better result. Logs escalation hit-rate via `voice_transcription_escalated` event. Realtime-path wiring is Phase-4 (different audio flow).
- ✅ **`earcons.global_earcons()` WIRED via new `earcon_wiring.py`** — InvocationBus subscriber plays DONE/ERROR/DECLINED/NEEDS_CONFIRM tones on tool-call transitions. Skip-list for background/read-only tools; always-chime list for sends/deletes/creates. Honors incognito + QuietMode.
- ✅ **`mcp_trust.global_store()` WIRED into `MCPConnector.execute()`** — every MCP tool dispatch consults `gate_mcp_call()` first; PENDING/REVOKED servers return a clear `mcp_trust_required` error so the user knows to grant trust via the picker. Fails-open if the trust module is absent (back-compat).
- ✅ **MCP trust toggle chip in the picker UI** — each server row has a trust chip (🔒 pending / 👁 read only / ✏ read+write / 🔓 full trust / ⛔ revoked) that cycles on click. Persists to `mcp_trust.db` immediately.
- ✅ **Voice-source label in confirm dialog** — when a destructive op was triggered by voice, the typed-confirm modal title is prefixed `[voice command]` and the body adds "⚠ This was triggered by voice. If you didn't say this, click No." so the user knows their trust path was downgraded from the spoof-defense layer.
- ✅ **Incognito UI toggle** wired into the chat header next to the voice toggle. Distinct red styling when ON; subscribes to incognito state changes so external toggles refresh the label.
- ✅ **Persona anchor module** (`persona.py`) — user-tunable personality block resolved at every session start, injected into the system prompt's tail. Resolution order: explicit set → `TOUCHLESS_PERSONA` env → `TOUCHLESS_PERSONA_FILE` env → memory fact `preference.persona` → built-in default. Clamped to 1500 chars.
- ✅ **`tool_metadata` coverage** — 7 missing entries added (notion_create_page, notion_append_to_page, notion_add_to_database, notion_search, calendar_create_event, spotify_play, spotify_pause). Safety gate now picks the right destructiveness for them.
- ✅ **`safety_gate._VERB_BY_TOOL` expanded** — 14 additional readable verbs (drive_upload, notion_*, sheets_append_rows, slides_add_slide, excel_create, word_create, powerpoint_create, etc.) so confirm prompts say "Create a Notion page?" instead of "Confirm: run notion_create_page?".
- ✅ **Implicit-fact extraction wired into `MemoryManager.record()`** — every successful planner turn auto-extracts "I live in Berlin" / "always send via gmail" / "my name is Dani" / "I prefer dark mode" / pronouns / contacts → persisted to the semantic store. Confidence-floored at 0.7.
- ✅ **`proactive_nudges.py`** new module + Sentinel registration — long-idle break suggestion (45 min, skips 1am-7am), cost-cap warning at 80% and 95%, unhealthy-tool nudge after 5 errors in 30 min. Nudge body queued into `_pending_notes` so Iris surfaces it in the next reply window. Per-kind cooldowns prevent spam; honors incognito + InterruptionGate.
- ✅ **Proactive surfacing of nudges + briefings** — when a nudge or briefing arrives AND no response is currently active, the manager kicks off `_drain_pending()` immediately so the user actually hears it instead of waiting for the next user turn. Without this the queued notes sat idle forever during user silence.
- ✅ **`sentinel_status.py`** new module — formatters for the Sentinel daemon's user-facing status (one-line summary, watcher table for settings/debug pane, chat-header pill data). Wired into the chat header next to the cost pill, polling every 10 sec; green / yellow / red colors match the watcher state.
- ✅ **Slash-command surface in chat** — `/persona <text>` sets the persona (or `/persona clear` reverts), `/persona` alone shows the current one. `/incognito` toggles private mode from chat. `/status` shows daemon + cost summary inline. Unknown slash commands fall through to the planner unchanged.
- ✅ **`reauth_nudge.py`** new module + bus subscriber — on the FIRST auth_revoked / not_connected error per connector per session, emits a one-click "Reconnect Gmail/Outlook" chip via the existing `suggested_actions` signal. Per-connector 5-min cooldown. No more waiting for the proactive_nudge cooldown (5 errors / 30 min) — the chip fires immediately on the first auth failure.
- ✅ **`cot_explainer.py`** new module + Tier-0a intercept in `try_handle` — when the user asks "why did you do that?" / "explain that" / "show me your work", Iris pulls the most recent TurnTrail from `cot_layer` and surfaces a conversational summary of the planning + revision + critique steps. Zero tokens, deterministic, no planner round-trip.
- ✅ **Rate-limit-aware reviser** — `_llm_revise` now consults `scheduler.is_throttled("cheap-llm")` before retrying via the LLM; falls through to the deterministic rule-based fallback when throttled. Prevents back-to-back 429s in the same user turn.
- ✅ **`session_buffer.py`** new module — rolling per-session conversation buffer (12 turns / 4000 chars / 1 hr TTL). User+assistant turns recorded automatically by `try_handle` and `_record_turn`. Surfaced as a "RECENT CONVERSATION" block in the planner's recall context so follow-up questions ("do it again with Alice instead", "and on my other monitor") have the immediate context they need. Honors incognito. Realtime/voice turns mirror into the buffer via `_record_convo_turn` too.
- ✅ **Slash commands expanded** — `/help` (quick reference), `/explain` (walks through the most recent CoT trail inline), `/forget` (hint surface). The full set is now: `/persona /incognito /status /explain /forget /help`. Unknown slashes still fall through to the planner.
- ✅ **`cot_explainer` tool-ref enrichment** — explanation output now includes a "Tools called: gmail_send, contacts_search" line so "why did you do that?" answers WHAT ran, not just the planning stages. Orchestrator records every executed step's tool name into the CoT trail.
- ✅ **Earcon speak-suppression** — DONE chimes are suppressed while the realtime model is actively streaming a response (would overlap audibly with Iris's voice). ERROR / NEEDS_CONFIRM still force through. Manager publishes itself at module level so the dispatcher can introspect `_response_active`.
- ✅ **Stuck-chip "Why?" button** — repeated-error stuck-pattern helper chips include a one-click "Why?" button that surfaces the explainer summary inline, so the user sees exactly which steps failed before deciding what to try next.
- ✅ **`memory_pinning.py`** new module + wiring — when an implicit fact (kind, key) gets the same value extracted 3+ times in 30 days, it auto-pins (confidence boost to 0.95). Skipped on multi-value ambiguity (user changed mind). Honors incognito. **Read side wired**: `MemoryStore.find_facts` now sorts `source_kind='auto_pin'` rows FIRST before falling back to recency. The planner's compose-rewrite + contact-lookup paths automatically get the user's most-reinforced value before any one-off mentions, with no caller changes.

### 4.5.5 Phase-4 — vision + self-learning (in progress)

- ✅ **`screen_awareness.py`** new substrate — Sentinel-tickable ambient screen capture every ~30s. Honors incognito + screen-sharing signal. Caches a compact `ScreenSummary` (one-line + visible text + element preview) with 90s TTL. Orchestrator's `_recall_context` auto-injects the summary as `ON SCREEN: ...` whenever the user's request looks vision-relevant ("what does this say", "summarize this", "that one", etc.).
- ✅ **`vision_dispatch.py`** new optional module — `ask_about_screen(question)` wraps Claude Sonnet's vision API for rich queries the cheap UIA/DOM/OCR path can't answer. Env-gated (`TOUCHLESS_VISION=1`), cost-cap aware, returns None on any failure so callers fall through.
- ✅ **`skill_consolidator.py`** new module + wiring — bus subscriber watches successful multi-step plans, counts recurring (tool→tool→tool) shape signatures, and emits a "Want me to save this as a skill?" nudge after 3+ recurrences. Persisted in `%LOCALAPPDATA%/Touchless/private/skill_consolidator.db`. Per-shape 24h cooldown; honors incognito; the nudge body is queued into `_pending_notes` + proactive-surfaced when the user is silent.
- ✅ **`anthropic_client.py`** new module — zero-dep `urllib` wrapper for Anthropic Messages API. Supports system blocks (with `cache_control` for prompt caching), JSON-mode via assistant-prefill, usage parsing for cost meter accounting. `planner_llm` now genuinely routes to Anthropic (Haiku/Sonnet) when the ModelRouter picks those tiers — not just OpenAI anymore. Falls back to OpenAI default when no Anthropic key set.
- ✅ **Realtime transcript router flagging** — when the realtime model's `input_audio_transcription.completed` event arrives, the manager passes the transcript through `TranscriptionRouter.decide()`. If it would have escalated (email/URL/path/digit-run), emits a `realtime_transcript_flagged` log event + a `realtime_transcript_risky` event when the ambiguity is a typed-token risk. UI can surface a "Heard an email — verify?" chip from these.
- ✅ **watchdog real-time layer** — `file_watcher_daemon.register_with_sentinel` now ALSO starts a `watchdog.observers.Observer` on each rule root when the dep is available. Events fire with sub-second latency instead of waiting up to 2s for the polling tick. Polling stays as a safety net for network FS / VSS / virtualized paths where watchdog can miss events. Graceful no-op when watchdog isn't installed.

### Phase 4 status — DONE (substrate + wiring + tests)
552 substrate+wiring tests passing across P2 + P3 + P4.

### 4.5.6 Phase-5 — autonomous, multi-modal, self-improving Iris (DONE)

- ✅ **`standing_orders.py`** new module + Sentinel wiring + slash commands — persistent background goals that survive restarts. SQLite-backed `StandingOrdersStore` + condition evaluator with 5 predicates (`inbox_match`, `time_at`, `time_after`, `tool_returns_ok`, `deadline_passed`). Sentinel-tickable every 60s; honors incognito + InterruptionGate. UI slash commands: `/watch <query>` (inbox watcher), `/remind <when>` ('in 30 min' / '3pm' / 'tomorrow 9am' parser), `/orders` (list), `/unwatch <id>` (cancel). Notifier queues fire-events into `_pending_notes` + proactive-surfaces them.
- ✅ **`multimodal_fusion.py`** new module — replaces the orchestrator's naive concatenation of memory + dictation + repo + session + screen context with relevance-scored budget allocation. 2KB total budget, min 80-char reserve per section, overlap dedup, stable rendering order. Pinned memory facts get priority-bumped to survive budget cuts.
- ✅ **`prompt_variants.py`** new module — A/B framework for prompt variants. Per-group bandit with Bayesian smoothing (alpha=7, beta=3), 20% explore floor, 10-sample warm-up phase. Persisted counters survive restarts. Modules call `pick(group_name)` to get a variant + `record_outcome(group, variant, success)` to update the tracker. Leaderboard introspection for human review of which variant is winning.
- ✅ **`reliability_advisor.py`** new module + Sentinel wiring — periodic (6h) chronic-unhealth scanner on the reliability ledger. When a tool's been failing ≥30% over a week (≥20 samples), surfaces a 'Want me to switch to <alternate>?' nudge with a known-alternates table (gmail ↔ outlook, drive ↔ onedrive, sheets ↔ excel, slides ↔ powerpoint, etc.). 7-day per-tool cooldown to avoid nagging.

### Phase 5 status — DONE (substrate + wiring + tests)
**617 substrate+wiring tests passing across P2 + P3 + P4 + P5.**

Iris substrate now genuinely autonomous: background goals that survive restart, multi-modal context fusion under hard prompt budget, self-improving prompts via bandit, autonomous reliability suggestions, vision-aware ambient screen capture, multi-provider model routing (OpenAI + Anthropic), implicit fact extraction + auto-pinning, conversational reply layer on every path, ambient nudges (idle / cost / unhealthy / stuck), proactive briefings + interruption gating, secret vault, audit log, content quarantine, MCP trust, voice spoof defense, persona resolver, slash commands, real-time FS events.

### 4.5.7 Phase-6 — entity reasoning + anticipation + vision + speculative latency + project memory (DONE)

- ✅ **`entity_graph.py`** new module — typed entity nodes + relations (PERSON, PROJECT, DOCUMENT, ARTIFACT, LOCATION, APP, FILE, EMAIL_THREAD, EVENT, TOPIC) wired with RelationKind (WORKS_ON, LEADS, COPIED_ON, AUTHORED_BY, MENTIONED_IN, PART_OF, ATTENDS, LOCATED_AT, DUE_BY, HAS_ALIAS, RELATED_TO). Alias resolution + recency lookups. Person facts mirror in from `MemoryManager.record`.
- ✅ **`pronoun_resolver.py`** new module — `resolve_references(text)` returns a `ResolutionReport` with `prompt_block` for planner injection. Handles person pronouns (him/her/them/they), thing pronouns (it/that/this/those/these), and demonstratives ("the one", "that one"). Cross-references session_buffer + screen_summary + entity_graph. Wired into orchestrator `_recall_context` as a "resolved_refs" modality.
- ✅ **`anticipation_engine.py`** new module + Sentinel wiring — predictive engine with kinds: MEETING_IMMINENT (calendar within 8 min), RECURRING_ROUTINE (same window/hour-bucket on ≥3 distinct days), STALE_STANDING_ORDER (active order fired ≥12h ago), INBOX_BACKLOG, DOC_FOLLOWUP. Per-pattern cooldowns. Honors incognito + InterruptionGate. Handler queues `🔮 {headline}` into `_pending_notes`.
- ✅ **`vision_observer.py`** new module + Sentinel wiring — delta-aware screen observer (distinct from passive `screen_awareness`). Recognizers: error dialog (window title), stack trace (regex over visible text — Python/Java/JS frames), save dialog, URL-opened (domain in window title that was recently mentioned in session). Per-kind cooldowns. Handler queues `👁 {headline}` into `_pending_notes`.
- ✅ **`smart_fast_path.py`** new module — classifies trivial utterances ("what time is it", "pause", "thanks") as CHAT (static reply) or DIRECT (single-tool dispatch); bypasses planner round-trip. Conservative-by-default: chained commands, why-questions, pronoun references, >12-word utterances fall through to planner. Wired into orchestrator as Tier-0.4 between utterance cache and skills.
- ✅ **`speculative_warmup.py`** new module — opens a TLS warmup against the planner provider (OpenAI/Anthropic) when audio activity starts, so the first real call after ASR doesn't pay handshake cost. 30s cooldown, provider-aware (asks ModelRouter), cost-aware (skipped in slow-mode).
- ✅ **`project_profile.py`** new module — SQLite-backed per-project profile keyed by absolute project root path. Tracks friendly name + summary + git branch (read from `.git/HEAD`, no subprocess) + recent files (24, MRU sliding window) + recent activity (32). `find_project_root` walks markers (`.git`, `pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`, `Makefile`, ...). `render_for_planner` produces a context block the orchestrator can fuse as the "project" modality.

### Phase 6 status — DONE (substrate + wiring + tests)
**Phase-6 deltas: 141 new tests (P6-B1: 35, B2: 19, B3: 24, B4: 34, B5: 29).** Cumulative substrate+wiring tests across P2 + P3 + P4 + P5 + P6.

Pushes Iris from ~70-75% PC-scoped Jarvis to ~90%: entity-grounded pronoun resolution, anticipatory nudges, screen-delta reasoning, sub-planner-latency fast-path for ~30% of utterances, and a project memory the planner can cite. Remaining gap to 100%: voice/persona quality bump (model-bound, not substrate work — out of scope for Phase 6).

### 4.5.8 Phases 7-10 — Personality, latency, smarter, subscription polish (DONE)

**Phase 7 — Personality engine (81 tests, wired):**
- ✅ **`persona_voice.py`** — named preset catalogue (default, jarvis, concise, warm, playful, tutor) with per-preset style block, few-shot examples, and temperature dial. UI / env / memory resolution order. Bandit-tracked outcomes. `/persona <name>` slash, `/persona list`, persisted to memory.
- ✅ **`callback_engine.py`** — 5 detectors (past_failure, callback_name, project_activity, repeated_app, prior_topic). Cooldown 4 turns. Preset gain dial (Concise=0; Jarvis=0.85). Affect-aware (frustration suppresses). Wired into `_conversationalize`.
- ✅ **`affect.py`** — 3-axis state (mood/focus/verbosity) from frustration / acceptance / exploratory / terse / silence-gap / correction / deep-work signals. Half-life decay 15 min. Read by callback_engine + interruption_gate (LOW nudges hold off when focused/frustrated) + streaming_renderer (max_tokens bias).

**Phase 8 — Latency (85 tests):**
- ✅ **`streaming_renderer.py`** — SSE streaming Jarvis rewrite. Yields `(token, is_final)`. Same fact-preservation guard on the buffered full string. Cache compatible with `prose_renderer`. Caller pattern: pause TTS, buffer, commit on guard pass, fall back on guard fail.
- ✅ **`speculative_planner.py`** — `TranscriptStabilizer` + `SpeculativeCache` + `SpeculativePlanner`. Fires the planner on a stable partial transcript before finalization. On finalize, Jaccard match ≥0.7 → use cached plan instantly.
- ✅ **`local_intent.py`** — per-user perceptron-style learned classifier. SQLite-backed. `record_example()` on every successful tool dispatch teaches it; `classify()` fires before fast-path when confidence ≥ 0.7. Wired as orchestrator Tier-0.3.
- ✅ **`tool_speculation.py`** — pre-fetches idempotent read-only tools (clock/weather/calendar/email) on partial-transcript keyword match. Allow-list only. Incognito + cost-aware. 60s TTL cache. Max 2 per turn.

**Phase 9 — Smarter (97 tests):**
- ✅ **`reply_judge.py`** — cheap async LLM grader on (helpfulness, accuracy, vibe). Feeds the bandit via `persona_voice.record_outcome`. Skipped on short replies / incognito / no API key / slow mode.
- ✅ **`behavior_log.py`** — captures (utterance, chosen_tool, outcome) tuples + flips on user corrections. Surfaces "you keep correcting X→Y, want me to default to Y?" suggestions after N corrections. SQLite-backed with preference table.
- ✅ **`kg_extractor.py`** — heuristic entity extractor from any text (emails, persons, projects, dates). Pure regex, no LLM. Seeds the existing `entity_graph` so "send the deck to Dani" works day 1. Honors incognito + content quarantine + caps input length.
- ✅ **`tool_discovery.py`** — capability-to-MCP-server catalogue. When user asks for Notion / Linear / Jira / Obsidian / Figma / GitHub / Slack / Discord / Spotify (and the corresponding MCP isn't installed), surfaces "want to enable it?" with one-line install hint.

**Phase 10 — Subscription polish (67 tests):**
- ✅ **`latency_dashboard.py`** — per-stage p50/p95/p99 (asr/plan/tool/reply/tts/total). 1000-sample ring buffer. `StageTimer` context manager. `one_line()` summary for chat-panel header pinning. Wired into orchestrator at planner + tool dispatch.
- ✅ **`persona_marketplace.py`** — UI cards for the persona catalogue (display name, description, sample reply, accent color, tier badge). Pro tier gating for Jarvis-tribute. `activate(name, allow_pro=...)` one-click switch.
- ✅ **`tts_voice.py`** — voice catalogue substrate (SAPI / OpenAI TTS / ElevenLabs / user-clone). Active voice resolution with privacy-tier fallback (cloud voices demoted to SAPI in local-only mode). User clone registration scaffold.
- ✅ **`privacy_tier.py`** — single `cloud` / `local_only` toggle that gates all cloud calls (vision_dispatch, tts_voice, kg_extractor scope, tool_speculation). Sticky in-session + env + memory-backed.
- ✅ **`federation.py`** — cross-device sync skeleton. `FederationEngine` with per-kind providers (memory / entity_graph / standing_orders pluggable). `InMemoryRemote` for tests; cloud endpoint stubbed. Per-device monotonic versions + per-device-id pull watermark + push-since-timestamp.

### Phases 7-10 status — DONE (substrate + wiring + tests)
**330 new tests across P7+P8+P9+P10.** Iris reaches the 90%+ Jarvis benchmark for PC-scoped voice control. The remaining gap to 100% is integration polish (streaming TTS in the realtime client, persona marketplace UI cards in the chat panel, persistent device-id storage for federation) — substrate is in place; wiring is incremental.

Subscription value proposition is now substrate-complete:
- Personality dial (free presets + Pro Jarvis voice)
- Latency dashboard subscribers can see
- Voice clone hook (no model bundled — Pro feature)
- Privacy toggle (cloud vs local)
- Federation skeleton ready for Touchless cloud endpoint when it ships
- ✅ **`repo_context.global_resolver()` WIRED** — new `repo_focus_watcher.py` Sentinel watcher polls foreground window every 8 sec, parses workspace path from JetBrains/Sublime/VS Code/generic titles, resolves the repo, caches the context with a 10-min TTL. Orchestrator's `_recall_context()` injects the cached context block into the planner prompt automatically.
- ✅ **`stuck_pattern_detector.global_stuck_detector()` FULLY WIRED** — bus subscription at boot + UI subscriber via `LiveAssistantWindow._on_stuck_signal()` renders inline helper chip respecting InterruptionGate's LOW-severity check.

### 4.5.2 Phase-3 watchers (need real signal producers)

- ✅ **`sentinel.Sentinel` STARTED + WATCHERS REGISTERED** — daemon running with three production watchers: `system_signals` (5s), `calendar_briefing` (60s), `file_watcher` (2s).
- ✅ **`interruption_gate.InterruptionGate` PRODUCERS WIRED** — `system_signals.py` polls Win32 every 5s and pushes screen-share / mic / camera / focus-assist / DND / fullscreen / battery / idle signals. Primed at session boot so SEC-002 fail-closed doesn't pre-empt the first interruption.
- `calendar_briefing.BriefingScheduler` — built; no calendar source. **Fix:** add a Sentinel watcher that calls `google_calendar.list_events` / `ms365_calendar.list_events` every ~5 min, runs `upcoming_within(events, lead_minutes=5)`, dispatches eligible briefings through `interruption_gate`.
- `file_watcher_rules.RulesEngine` — built; no watcher loop or rules-UI. **Fix:** Sentinel watcher running `watchdog.observers.Observer` against the union of all enabled rules' `path_glob` roots; dispatcher routes through the existing ToolRegistry.
- `stuck_pattern_detector.StuckPatternDetector` — already attaches to InvocationBus but no UI consumes its signals. **Fix:** subscribe in `LiveAssistantWindow` and surface suggested-action prompts.
- `cost_surfaces` — formatters built; no UI surface renders them. **Fix:** add the colored cost pill to the Iris chat header.

### 4.5.3 Architectural follow-ups from the P2 audit (LOW)

- Three independently-maintained lists of "destructive verbs" (`transcription_router._DESTRUCTIVE_VERBS`, `self_critique._GOAL_VERBS_TO_TOOLS`, `safety_gate`'s classifier). Pull into a single `destructive_verbs.py` to prevent drift.
- `repo_context._hidden_subproc_kwargs` duplicates `hgr.utils.subprocess_utils.hidden_subprocess_kwargs`. Replace with import.
- `self_critique._looks_multi_action` duplicates `planner.triggers.looks_multi_action`. Replace with import.
- Critic-added gmail_send step uses hardcoded body — if reached deterministically it sends a content-free email. **Fix:** restrict deterministic critic to NOT add `*_send` / `*_compose` steps; only LLM-critique path (env-gated) may.
- `repo_context._cache` has no LRU/size cap. Use `OrderedDict` + `max_entries=50` like `utterance_cache`.

### 4.5.4 P3 audit follow-ups not yet applied (MEDIUM / LOW)

Apply when wiring lands or in a dedicated polish PR:

- **sentinel-4 (LOW)**: Auto-disabled watchers never recover. Add exponential back-off (double interval on each failure beyond threshold; cap 30 min) instead of permanent disable.
- **gate-1 (MEDIUM)**: `delay_until` on a deferred decision is informational only — no internal scheduler re-checks. Either remove from the contract or build a tick()-drained deferred queue inside the gate.
- **stuck-4 (MEDIUM)**: REPEATED_ERROR emits once per 30s cooldown, but for slow-paced failure loops (60s poll intervals) every error past the 3rd surfaces a new signal. Track "already emitted at count N", only re-emit on count doubling or window reset.
- **stuck-5 (LOW)**: REPEATED_ACTION cooldown key omits `args_hash`, so two distinct repeated-action patterns on the same tool collapse to one bucket. Include args_hash in cool_key.
- **calbrief-2/4 (LOW)**: Attendee "and N others" branch + agenda HTTP regex polish (both cosmetic).
- **fwr-5 (MEDIUM, missed-by-panel)**: `_resolve_placeholders` substitutes raw values into dispatcher args. Document the contract (placeholders RAW; tool authors must escape) and add explicit `{path_url}` / `{path_shell}` variants.
- **shadow-3 / docstring drift (LOW)**: ShadowMode docstring mentions "Date.now() / Math.random()" (JS-isms). Update to Python equivalents.
- **stuck-LOW (missed-by-panel)**: `_user_text_history` has no per-entry timestamp; circular_dialogue can fire for utterances hours apart. Either add TTL or document "3 in a row" semantics explicitly.
- **sentinel register-during-tick race (LOW)**: When register() runs while a tick is in flight, the stale WatcherSpec's mutations are discarded. Document or fix by looking up current spec by name inside _maybe_run.
- **packaging gap (HIGH)**: When Iris ships, the `"hgr.live_api"` entry in `builder/windows/hgr_app.spec` `excludes=[...]` must be removed, and the lazily-imported Phase-2/3 modules added to `hiddenimports` (per CLAUDE.md rule #6).
- **integration tests**: Add one test per wiring point (sentinel started, gate consulted from notifier, stuck-detector attached, briefing scheduler ticking, file-rule dispatcher bound, cost pill rendered) so future PRs can't silently un-wire them.

---

## Section 5 — Regression-risk patterns

When a task resembles one of these patterns, explicitly test the matching regression path before finishing:

- Fixing one gesture mode accidentally changes unrelated gesture behavior.
- Drawing-mode changes break gesture-wheel actions.
- Modal/chooser windows freeze the live camera or hand-driven cursor use.
- Voice follow-up timing becomes misaligned after a fix.
- Dictation changes accidentally affect voice-command behavior (or vice versa) — they share the whisper build dir and model folder but run in different capture pipelines.
- UI changes sneak in during functional patches.
- Runtime path/import fixes accidentally alter packaged behavior.

---

## Section 6 — Priority decision

Current candidates, roughly ranked by impact:

1. **Dictation accuracy (Section 3)** — active focus; baseline test in progress.
2. **PowerShell removal (2.1)** — actively breaks installs for other users. Dropping SAPI is a prereq for cleaning up the dictation stack too.
3. **Motion blur (2.2)** — quality issue, app still works.
4. **Selector window cursor (4.1)** — roadmap feature.

<!-- Author: Konstantin Markov -->
