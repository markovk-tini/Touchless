# Touchless — Open Issues, Roadmap & Regression Notes

Single source of truth for what's open, what's planned, and what historically breaks.
Last reviewed 2026-04-21.

---

## Section 1 — Active bugs

*(none currently open)*

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
