# Touchless 1.1.4 — release notes

## What's new (user-facing)

- **Clip V2 — rebuilt clip recording for sub-300 ms sync and full
  reliability.** A dual-stream architecture (separate system audio + mic
  bridges, OBS-style) replaces the prior single-pipe path. Long clips
  (2 min / 5 min) no longer drift, the mic no longer goes garbled or silent
  mid-clip, and the 5-minute crash is gone. New cap-aware safety nets and
  per-stream rate compensation keep audio + video aligned even when
  Windows hot-swaps the default audio endpoint mid-clip.

- **One-fist clip gesture.** Hold a left-hand fist for 0.5 s to save an
  instant clip at your default duration — no voice command needed. The
  previous thumbs-up trigger has been retired (it conflicted with the
  YouTube like gesture); voice-cancel still takes priority so a brief
  fist won't accidentally fire.

- **2 min and 5 min clip durations.** Say "clip the last two minutes" /
  "clip the last five minutes" and Touchless captures up to a full 5
  minutes of rolling buffer. Existing 60 s / 30 s commands still work
  exactly as before.

- **Settings reorganized — single Clip Presets section.** The split
  "Clip & Record" + "Clip Audio" panes have been merged into one
  Clip Presets section so all clip-related controls live in one place.
  Settings search indexes the new fields, including the clip-gesture
  binding.

- **Voice command capture rebuilt.** Switched to callback-driven audio
  capture (the polling path was corrupting WAVs on Razer Kiyo Pro / UVC
  webcam mics, producing "low robotic" audio and Whisper hallucinating
  YouTube taglines). Voice commands are now accurate and fast on the
  Kiyo Pro, default English-only model upgraded from `distil-large-v3`
  to `medium.en` (24-layer decoder, far better for short commands), and
  end-of-speech detection waits a configurable ~3 s so natural mid-
  sentence pauses no longer cut you off.

- **Tighter volume popup.** The system volume overlay is half the width
  it was, the percent text is smaller and centered, and the noisy
  "Adjusting / Ready / Idle" labels are gone — only "Muted" appears
  when actually muted.

- **Mode-toggle pill no longer cuts text off.** "Mouse Mode: ON / OFF"
  and "Drawing Mode: ON / OFF" pills had a sizing mismatch between
  resize and paint that clipped the "ON" / "OFF" in half. Fixed.

- **Processing-clip pill stacks above the voice command pill.** When
  you say "clip that" the "Processing 60 s clip" pill now stacks above
  the "Executing command" pill with a small gap, and smoothly slides
  down to take its place when the voice pill disappears — instead of
  the two overlapping.

- **Volume endpoint rebind no longer breaks.** Fixed a regression where
  pycaw's audio-device-identity check used `.GetId()` (deprecated) and
  the overlay stopped tracking volume after a default-device switch.

## Store submission — "What's new" (plain text)

Rebuilt clip recording (sub-300 ms sync, no drift on long clips, no more
mid-clip mic dropouts or 5-minute crash). One-fist gesture instantly saves
a clip — no voice needed. New 2 min and 5 min clip durations. Voice
commands rebuilt for accuracy on webcam mics (medium.en model, callback-
driven capture). Tighter volume overlay, fixed mode-toggle pill cutoff,
processing-clip pill stacks above voice pill with smooth slide-down.

---

## Maintainer notes (not user-facing)

- The Touchless Assistant ("Iris") remains intentionally **excluded**
  from this build (PyInstaller `excludes` in `hgr_app.spec`) and dev-
  gated in the UI (`TOUCHLESS_ENABLE_ASSISTANT=1`). Not shipped to
  users yet.

- Clip V2 cache buffer was scaled 65 s → 305 s with a migration shim
  for existing users (config field `clip_cache_duration_seconds`). 2-min
  and 5-min clips were previously truncated to the 65 s baseline.

- voice_command_listener now uses sd.InputStream **callback** mode
  (not `stream.read()` polling). The polling path on Windows WASAPI
  shared-mode corrupts audio on some UVC drivers. See `memory/`
  `project_wasapi_callback_vs_read.md` for the historical context.

### GitHub release body markers (website / GitHub auto-updater)

Append to the GitHub release body so the in-app updater can find the
full installer on R2. Fill the size in bytes (optional) from the
uploaded file. Add the SHA-256 if you want the dual-verify path.

```markdown
<!-- full-installer-url: https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/windows/v1.1.4/Touchless_Installer.exe -->
<!-- full-installer-size:  -->
<!-- full-installer-sha256:  -->
```

Attach `Touchless_App_Update_1.1.4.zip` as the GitHub release asset
(this is the small ~50-150 MB zip the auto-updater prefers for
existing users; only first-time installers download the full installer
from R2).

### Microsoft Store submission

Submit `release/Touchless_Store_Installer.exe` (produced by `STORE=1`
build) to Partner Center. Use these Installer parameters:

- Silent install:    `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /NOCANCEL`
- Silent uninstall:  `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART`

The Store build sets `build_channel=store` so the in-app GitHub
auto-updater stays OFF and the Store owns updates for Store-installed
copies. No cross-channel update overlap.
