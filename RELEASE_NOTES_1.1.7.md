Feature + polish release. Recording quality tiers, retooled left-hand gestures, adaptive short-shutter for cheap webcams, refreshed dialog chrome across the whole app, plus dozens of UI fixes.
## Touchless 1.1.7 Release Notes:

## Recording quality

- **New Quality Level tier picker** in Settings → General → Clip Presets — pick Low, Normal, or High. Low compresses hard for smaller files; Normal is the balanced default; High goes near-lossless at 60 fps for editable footage on machines with an NVIDIA / Intel / AMD hardware encoder.
- **Show more expander** under the tier picker explains what each tier changes in plain English (fps, quality, file size, hardware needs) — no more guessing.
- **30-second clip duration preset** added. You can now set Clip Duration to 30 s, 1, 2, or 5 min. Filenames pick up the actual duration (`Touchless_Clip_30s_1.mp4`).

## Gestures

- **Left-hand TWO is now the clip gesture.** Hold the "two" pose (index + middle up) for half a second to save an instant clip of the last N seconds — same as saying "clip that" out loud.
- **Left-hand FIST does nothing anymore.** Was a voice-cancel + backup clip trigger; retired to keep it clean.
- **Dictation mode temporarily removed.** Was previously on left-hand TWO. Coming back in a future release with better accuracy.
- Drawing-mode swipes (left / right to undo / clear) now fire from the index-only "one" pose you already draw with, so you don't have to switch hand shapes mid-stroke.
- Drawing has a new **drift dead-zone**: the pen stays still when your hand does, even if MediaPipe wobbles the landmarks. No more ink trails from a still hand.

## Camera & tracking

- **Adaptive short-shutter for generic UVC webcams** (Realtek, Sonix, Chicony, and other cheap USB webcams). When GPU mode is on, Touchless writes a short-exposure hint to the camera BEFORE ffmpeg claims it, so MediaPipe gets sharp frames with much less motion blur. Premium cameras (Kiyo Pro, Brio, C920) are left alone.
- **Tightened landmark smoother** so the hand skeleton is more stable on a still hand. Roughly ~20 % less per-frame jitter on GPU mode.
- **Camera settings page** — the scrollbar no longer appears in the default state.
- **Force Short Shutter checkbox** is now labeled "Boost performance for older cameras" with a Show-more explanation, and honours the user's explicit off choice even when the auto-classifier thinks the camera would benefit.

## Popup chrome (visual overhaul)

- **All 18 dialogs now use the Touchless indigo chrome** — Spotify setup wizard, Discord setup wizard, phone connect popups, Iris confirm-action, custom-gesture recorder / sandbox / wizard, dynamic-gesture recorder, tutorial, drawing-chooser, MCP servers picker, voice picker, and the update dialog. Previously showed the OS-default black or white title bar on Win10.
- **Sharp square X close button** (32 × 32) on every popup, on both Win10 and Win11.
- **Live View window title bar** now matches the rest of the app (was OS default before).
- Fixed the r51 issue where the Spotify wizard's Next / Finish buttons were invisible — a stray widget-level QSS rule was cascading into them.

## Settings search

- **New search entries** for Quality Level, Clip Duration Presets, Boost Performance, Reapply Camera Tuning, Discord / Spotify setup wizards, and Connect Phone. Type any of them into the search bar.
- **Dropdown width tracks the widest result** — no more oversized popup for short queries. Long labels elide with "…" instead of forcing the popup wider.
- **Vertical extent** now stretches down to just above the Back button when you have many results.
- **Single-character queries** (`q`, `s`, etc.) only match labels — no more surfacing every entry that happened to have a `q` in its keyword blob.
- Clicking a result while you're **already on that section** now scrolls to the top instead of doing nothing.

## Overlay toggles

- **Text pop-ups OFF** now correctly silences the clip flow: the "Clipping last N seconds" pill and the "Clip saved to …" toast both honour the toggle. Previously only the "where should I save this?" voice prompt was gated.

## Other polish

- **Loading bar visibly reaches 100 %** on the clip processing pill before it disappears, so you can see the clip actually finished.
- **Tooltips shortened** across ~14 sites so hover text is a single readable phrase instead of a paragraph.
- **Debug-log-location popup on exit** removed — was cluttering shutdown.
- **Top-right diagnostic HUD** on the home page removed — was leftover dev instrumentation.
- **Connect Phone description** shortened; the details now live under Show more.
- **Iris backend / MCP / Voice picker** hidden from search until Iris ships.

## Auto-update

If you're on v1.0.6 or later, Touchless will offer to update itself on next launch. Click "Update now" and the app swaps to v1.1.7 automatically.

If you're on an older build, download a fresh installer from [touchless-control.com](https://touchless-control.com).

<!-- full-installer-url: https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/windows/v1.1.7/Touchless_Installer.exe -->
