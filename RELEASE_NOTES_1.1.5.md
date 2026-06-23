# Touchless 1.1.5 — release notes

## What's new (user-facing)

- **Clip recording sync is airtight again.** Audio and video stay in
  perfect sync across every clip length — 60&nbsp;s, 2&nbsp;min, 5&nbsp;min,
  overlapping clips, clips taken while a previous export is finishing.
  Fixes the v1.1.4-era issue where the audio could drift 2&ndash;3
  seconds behind video on long clips.

- **Clip audio is loud.** Both your microphone and system audio in
  saved clips now sit at full level &mdash; no more "I have to crank
  the playback volume just to hear myself talking." A clean +18&nbsp;dB
  on the mic chain combined with the corrected mix levels means clips
  recorded at normal speaking volume play back clearly.

- **Voice commands now hear you at any speed.** Phrases like
  "open YouTube on Google Chrome", "play Spotify", "clip last
  two minutes" recognise fully even when spoken quickly, with natural
  pauses, or on a quiet mic &mdash; the new VAD logic gives you up to
  a full 1&nbsp;second of grace after every spoken peak before counting
  silence, so trailing words don't get cut off as "Open." or
  "Clip la-".

- **The mic gain slider in Settings actually persists now.** Move it
  once, click Save, and the value sticks across worker restarts. The
  v1.1.4 slider would silently revert to default whenever the engine
  restarted (which happens on most settings changes), so users who
  thought they'd set their gain higher were still capturing at the
  auto-classified default.

- **One-fist clip works again.** The left-hand fist-hold clip gesture
  no longer produces a single-frame video file. (v1.1.4 was stamping
  the gesture timestamp with `time.monotonic()` instead of
  `time.time()`, so the clip-export's trim window calculated a
  ~56-year duration and clamped to zero frames.)

- **Settings page doesn't yank to the top on Save.** Changes still
  apply instantly, the view just stays where you were editing &mdash;
  no more re-scrolling after every change.

- **System-audio bridge no longer cuts out mid-clip during quiet
  music passages.** Combined with the sync fix above, music + voice
  clips now record reliably even through song-to-song crossfades or
  notification audio sessions briefly taking over.

- **Internal cleanup so clips don't crash on export when no audio
  captured.** Closed an `UnboundLocalError` in the clip-export path
  that fired whenever a user toggled both audio sources off and
  tried to clip a video-only recording.

## Store submission &mdash; "What's new" (plain text)

Clip recording sync is airtight again across every duration (60 s,
2 min, 5 min, overlapping clips). Clip audio is loud at full level
on both mic and system audio. Voice commands hear full phrases at
any speed including quiet mics with natural pauses. Mic gain slider
in Settings now persists across worker restarts. Fist-hold clip
gesture fixed (was producing single-frame files). Settings page no
longer yanks to top on Save.

---

## Maintainer notes (not user-facing)

- The mic gain slider in Settings now writes BOTH `mic_input_gain`
  AND `mic_input_gain_auto` to config + spawns a background
  `save_config` thread. v1.1.4 only wrote the auto flag, so the gain
  value was lost on every worker restart. See `_on_mic_test_gain_changed`
  in main_window.py around line 14323.

- VAD silence logic now has a 1.0-second grace counter after every
  trigger-crossing peak (`SILENCE_GRACE_BLOCKS = 25`). Quiet trailing
  speech below silence_threshold no longer accumulates end-of-utterance
  silence while the user is still speaking. See
  voice_command_listener.py around line 866-877 and 1149-1190.

- Clip export amix now uses `weights=1 1:normalize=0` (was `2 3` with
  default normalize=1). With normalize=1, mix output was
  `(sys*2 + mic*3) / 5` &mdash; mic at 60%, sys at 40%. With
  normalize=0, both streams pass through at full level. Combined with
  the new `volume=18dB` mic boost, mic in clips is ~13&times; louder
  than v1.1.4 (+22 dB).

- WASAPI sys-bridge `long_stall_threshold` reverted from 3.0 to 0.5
  seconds. The 3.0 value caused AAC segment files to carry fewer
  samples than their wall-clock span (no silence chunks written
  during quiet patches), which the export's safety-net trim then
  mis-aligned &mdash; the audio "delay" symptom. 0.5 s matches v1.1.4
  and earlier; sync is airtight.

- The Touchless Assistant ("Iris") remains intentionally **excluded**
  from this build (PyInstaller `excludes` in `hgr_app.spec`) and dev-
  gated in the UI (`TOUCHLESS_ENABLE_ASSISTANT=1`).

### GitHub release body markers (website / GitHub auto-updater)

Append to the GitHub release body so the in-app updater can find the
full installer on R2:

```markdown
<!-- full-installer-url: https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/windows/v1.1.5/Touchless_Installer.exe -->
<!-- full-installer-size:  -->
<!-- full-installer-sha256:  -->
<!-- app-update-zip-sha256:  -->
```

Attach `Touchless_App_Update_1.1.5.zip` as a GitHub release asset
(this is the small ~135 MB zip that v1.1.4 users will auto-download
as their in-place update).

### Microsoft Store submission

Submit `release/Touchless_Store_Installer.exe` (produced by `STORE=1`
build) to Partner Center. Installer parameters:

- Silent install:    `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /NOCANCEL`
- Silent uninstall:  `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART`

Order: publish GitHub release FIRST, then submit to Microsoft Store
Partner Center. This ensures v1.1.4 Store users get the fast 135 MB
auto-update path (the `StoreUpdateChecker` fetches the GitHub release
for the matching version to find the small-zip URL).
