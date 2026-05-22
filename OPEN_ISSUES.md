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

### 2.1 PowerShell removal — friend's install auto-uninstalled by Windows

**Root cause:** Unsigned exe + `powershell.exe -ExecutionPolicy Bypass -EncodedCommand <base64>` is a known malware-dropper fingerprint. Windows Defender / ASR auto-quarantines and the installer gets rolled back.

**Status:** plan agreed, not yet implemented. Current `powershell.exe` call sites:
- [voice_command_listener.py:804](src/hgr/debug/voice_command_listener.py#L804) — System.Speech fallback via `-EncodedCommand` (strongest ASR trigger)
- [youtube_controller.py:636](src/hgr/debug/youtube_controller.py#L636) — UIAutomation (`System.Windows.Automation`)
- [youtube_controller.py:868](src/hgr/debug/youtube_controller.py#L868) — WinRT OCR (`Windows.Media.Ocr.OcrEngine`)
- [desktop_controller.py:1107](src/hgr/debug/desktop_controller.py#L1107), [:1745](src/hgr/debug/desktop_controller.py#L1745), [:2147](src/hgr/debug/desktop_controller.py#L2147) — misc shell/automation calls (review each to see if it needs removal or can stay as a plain non-encoded invocation)

**Agreed replacements:**
- UIA → `uiautomation` pip package (COM wrapper over UIAutomationClient).
- OCR → `winsdk.windows.media.ocr` (official WinRT projection).
- SAPI fallback → **drop entirely**. Whisper is the primary path; the fallback rarely fires and introduces the worst ASR fingerprint. Dropping it also removes [sapi_stream.py](src/hgr/voice/sapi_stream.py) and simplifies [live_dictation.py](src/hgr/voice/live_dictation.py).
- Update `requirements.txt` and PyInstaller hidden-imports in [hgr_app.spec](builder/windows/hgr_app.spec).
- Longer-term: Azure Trusted Signing (~$10/mo) to sign installer + exe.

### 2.2 Motion blur on 30fps 720p laptop cameras

**Root cause:** Auto-exposure on most webcams drops shutter speed in dim rooms, producing blur even at 30fps. Framerate is not the issue.

**Status:** not implemented. MediaPipe tracks prior-frame landmarks so the app still works with some blur, but accuracy degrades noticeably in dim rooms and during fast motion.

**Plan:**
- In [camera_utils.py](src/hgr/app/camera/camera_utils.py), set on the capture:
  - `CAP_PROP_FOURCC = MJPG` (always, not just low-fps mode)
  - `CAP_PROP_AUTO_EXPOSURE = 1` (manual, DirectShow)
  - `CAP_PROP_EXPOSURE = -6` (short shutter, DirectShow log2 scale)
  - `CAP_PROP_BUFFERSIZE = 1` (always)
- Tradeoff: short exposure darkens dim rooms. If that becomes a problem, expose an exposure slider in Settings.

---

### 2.3 Per-frame engine cost drops fps when a hand is tracked [LOOK LATER]

With a hand in frame, fps dips (~30 → ~26 local webcam; ~high-30s → ~22-26
on the phone source). **GPU Mode on/off makes no difference**, which proves
the bottleneck is NOT the tracking inference (GPU-accelerated) but the
per-frame CPU work that runs *after* detection: landmark smoothing, gesture
classification, overlay/skeleton drawing, and the GpuVideoWidget paint.
Affects every camera source equally.

**To investigate later:** profile where per-frame time goes when a hand is
present. Candidate wins — lighter overlay drawing, cheaper / less-frequent
landmark smoothing, and decoupling the widget paint from the processing
loop so paint can't throttle production. Not urgent; ~26fps is usable.
(The phone source's extra dip vs local is the WebRTC transport — already
trimmed by sending 360p in `webengine_host.py`.)

---

## Section 3 — Dictation accuracy

Current focus. Historical issues (from prior sessions, not yet all verified on current build):

- **Early-word clipping** — words occasionally start mid-syllable; pre-roll was bumped to 600ms in voice-command capture but dictation uses `whisper-stream.exe` which has its own internal VAD.
- **Streaming vs. refinement drift** — `WhisperStreamer` emits fast hypotheses; `WhisperRefiner` later replaces spans with a fuller-context decode. Occasional visible flicker during the swap.
- **Grammar correction lag / misfires** — `GrammarCorrector` waits for sentence-boundary + 0.5s idle + ≥20 chars before submitting. On dense dictation the corrector can fall behind; `_chunk_stale` is supposed to discard corrections that overlap new input but edge cases exist.
- **Mic-gain slider doesn't reach `whisper-stream.exe`** — documented limitation ([6.6 prior]). Whisper-stream pulls audio directly from WASAPI, bypassing the in-process gain slider. Users are pointed to Windows input volume. Fix would require capturing audio ourselves and piping via stdin or switching dictation to the `whisper-cli` path the refiner already uses.
- **Title-case / spoken-punctuation coverage** — [dictation.py](src/hgr/voice/dictation.py) has static lists. Misses (new brands, uncommon punctuation names) need to be added case-by-case.

**Next step:** baseline accuracy test with a fixed sentence, 3 repetitions, to identify which of the above is currently most visible before changing anything.

---

## Section 4 — Roadmap (next features)

Planned-work / polish backlog has moved to [`Touchless to-do.md`](Touchless%20to-do.md). This section is reserved for architectural-level roadmap notes that don't fit the to-do format.

### 4.0 Dynamic custom gestures (in progress — paused 2026-05-13)

Motion-based custom gestures (DTW over per-frame hand landmarks). The first end-to-end loop works in the live engine — recording → save → live-fire — but accuracy is mediocre and several UX/visual polish items are still open. Pausing here to revisit later.

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
