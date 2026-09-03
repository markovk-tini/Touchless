Hotfix + custom-gesture release. The in-app update popup works again (it was invisible in 1.1.7 and crashed in 1.1.8 / 1.1.8.1), and custom motion gestures detect more reliably.

## Touchless 1.1.9 Release Notes:

## Updates (fixes 1.1.7 / 1.1.8 / 1.1.8.1)

- **Update popup is visible again.** The 1.1.7 chrome change hid it behind the main window; 1.1.8 and 1.1.8.1 also crashed the dialog on open. Opening the app now surfaces the prompt on screen with the new version, download size, **What's new**, and **Download Update**.
- **Tray balloon is a real fallback.** Clicking it brings the update dialog forward if the popup is ever behind another window.
- **Store and website installs use the same dialog.** Website builds poll GitHub; Store builds poll the Store and pull matching GitHub notes when they exist. **Download Update** applies the in-app updater (small app-zip when available) or opens the Store page if that is the only option.
- **Moved-install updates.** The updater and the sign-in autostart entry follow the app if you moved the install folder.
- If **Install updates automatically** is on (the default) and the update is a small app-zip, launch can apply it silently with a tray toast instead of the popup. Turn that off in Settings → General to always see What's new first.

## Custom gestures

- **Motion gestures detect more reliably.** Fast circles and long paths were missing; a short hand entering the frame could false-fire a wave. Matching now scales with how fast the recorded gesture actually moved, and a gesture is not confirmed from a window shorter than the recording.
- **Pose sequences.** Hold an ordered list of static poses (for example 3 → 2 → 1) as a custom gesture, with dwell / max-hold / max-gap timing.
- **Recorded-path preview.** After recording a motion gesture you can open a diagram of the takes.

## Tracking

- **ONNX hand reacquire.** GPU-mode tracking recovers the hand more cleanly after it leaves and re-enters the frame.

## GitHub release markers (added at publish time)

```
full-installer-url: https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/windows/v1.1.9/Touchless_Installer.exe
full-installer-size: <bytes after rclone>
```

Do not attach `Touchless_Installer.exe` to the GitHub release. Attach `Touchless_App_Update_1.1.9.zip` if this build produces one.

<!-- Author: Konstantin Markov -->
