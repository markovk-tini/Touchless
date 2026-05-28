# Touchless 1.1.2 — release notes

## What's new (user-facing)

- **Phone camera connects more reliably.** When you use your phone as a
  camera, it now appears as a camera source — and the Disconnect button shows —
  only *after* the phone has actually connected, instead of immediately. No more
  empty/premature phone entries before a connection exists.

- **Smoother, more precise drawing.** Drawing mode is steadier and far less
  twitchy — the cursor no longer drifts when you hold your hand still, and it
  moves at a more controllable speed. In the camera ("switch view") mode your
  strokes now follow your fingertip exactly (1:1), and the hand-tracking overlay
  no longer covers your drawing.

- **One-click in-app updates.** Touchless now reliably tells you in-app when a
  new version is available and lets you install it with one click — including
  the Microsoft Store edition, which now surfaces Store updates inside the app
  instead of only through the Store.

## Store submission — "What's new" (plain text)

Phone camera now connects more reliably (only appears once actually connected).
Smoother, more precise drawing — steadier cursor, 1:1 fingertip tracking in
camera view, no overlay covering your strokes. In-app update notifications so
you can update with one click.

---

## Maintainer notes (not user-facing)

- The Touchless Assistant ("Iris") is intentionally **excluded** from this
  build (PyInstaller `excludes` in `hgr_app.spec`) and dev-gated in the UI
  (`TOUCHLESS_ENABLE_ASSISTANT=1`). Not shipped to users yet.

### GitHub release body markers (website / GitHub auto-updater)

Append to the GitHub release body so the in-app updater can find the full
installer on R2. Fill the size in bytes (optional) from the uploaded file.

```markdown
<!-- full-installer-url: https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/windows/v1.1.2/Touchless_Installer.exe -->
<!-- full-installer-size:  -->
```

Attach `Touchless_App_Update_1.1.2.zip` as the GitHub release asset.
