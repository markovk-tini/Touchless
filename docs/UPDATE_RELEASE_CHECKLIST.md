# Update Release Checklist

**The single walk-through for every Touchless update.** Covers build,
sign, all three distribution channels (Cloudflare R2, Microsoft Store,
GitHub Releases), website, and the critical in-app popup + update-button
verification. Nothing else — this is the whole list.

**Order matters.** The sections are numbered in the order they must
happen. Skipping ahead has burned us before (Store users seeing 404s,
users getting a popup that points at a not-yet-uploaded R2 file).

**Boxes are stop-ship.** If a box goes red, the release halts. No
"we'll fix it in a hotfix" — a hotfix is what we call a bug that
already reached users.

**AI-agent note:** you (Claude) must walk this list top-to-bottom and
REPORT green/red per box before running any `git push`, `rclone`,
Partner Center action, or website deploy. Do not skip. Do not batch.

---

## 1. Pre-build sanity (before touching the build machine)

- [ ] `src/hgr/__init__.py` `__version__` == the intended tag (e.g.
      `"1.1.9"`)
- [ ] `installers/windows/hgr_app.iss` `#define MyAppVersion` == same
- [ ] `git status` — nothing uncommitted in `src/hgr/app/updater/`, the
      installer script, or `src/hgr/__init__.py`. A WIP change to the
      updater is the #1 way a release ships broken.
- [ ] Run tests: `python -m pytest tests/ -q` → 0 failed.
- [ ] Run the popup smoke test explicitly:
      `python -m pytest tests/test_update_dialog_smoke.py -q` → 8 passed.
      This sentinel catches the NameError class of bug that shipped in
      1.1.8 and 1.1.8.1, the 1.1.7 frameless z-order flags, and the
      dead tray-balloon fallback (`TouchlessTrayIcon.showMessage`).
- [ ] Import smoke:
      `python -c "from hgr.app.updater import update_dialog, updater, store_updater, release_checker; print('OK')"`

## 2. Build + sign (produce all three artifacts)

- [ ] Run `builder\windows\build_windows.bat` (no `SKIP_SIGNING` for a
      real release).
- [ ] Confirm three artifacts land in `release\`:
      - `Touchless_Installer.exe` (Cloudflare stub)
      - `Touchless_Payload_vX.Y.Z.zip` (Cloudflare payload)
      - `Touchless_Store_Installer.exe` (monolithic for Store)
- [ ] Verify each is signed:
      `signtool verify /pa /v release\Touchless_Installer.exe` (exit 0)
      Repeat for the store installer.
- [ ] Signing MUST be Azure Artifact Signing via `dotnet sign` with the
      Individual "Konstantin Markov" cert. `signtool + dlib` fails
      silently and has bitten us before — do not use it.

## 3. Upload to Cloudflare R2 (put files in place BEFORE anything
       announces them)

Order inside this section doesn't matter — all three uploads are
independent files.

- [ ] Upload the Cloudflare stub installer:
      ```
      rclone copyto release/Touchless_Installer.exe \
        r2:hgr-downloads/windows/vX.Y.Z/Touchless_Installer.exe \
        --s3-upload-cutoff=100M --s3-chunk-size=100M
      ```
- [ ] Upload the payload zip:
      ```
      rclone copyto release/Touchless_Payload_vX.Y.Z.zip \
        r2:hgr-downloads/windows/vX.Y.Z/Touchless_Payload_vX.Y.Z.zip \
        --s3-upload-cutoff=100M --s3-chunk-size=100M
      ```
- [ ] Upload the Store installer to BOTH paths (Partner Center's saved
      URL sometimes assumes the flat path):
      ```
      rclone copyto release/Touchless_Store_Installer.exe \
        r2:hgr-downloads/windows/vX.Y.Z/store/Touchless_Store_Installer.exe \
        --s3-upload-cutoff=100M --s3-chunk-size=100M
      rclone copyto release/Touchless_Store_Installer.exe \
        r2:hgr-downloads/windows/vX.Y.Z/Touchless_Store_Installer.exe \
        --s3-upload-cutoff=100M --s3-chunk-size=100M
      ```
- [ ] Verify all four public URLs return HTTP 200 with correct
      Content-Length. Silent-fail uploads have shipped before.
      ```
      curl -I https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/windows/vX.Y.Z/Touchless_Installer.exe
      curl -I https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/windows/vX.Y.Z/Touchless_Payload_vX.Y.Z.zip
      curl -I https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/windows/vX.Y.Z/store/Touchless_Store_Installer.exe
      curl -I https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/windows/vX.Y.Z/Touchless_Store_Installer.exe
      ```

## 4. Submit to Microsoft Partner Center FIRST

Store certification takes 24-72 hours. Kick it off early so it's in the
pipeline while everything else finishes. Store submission depends on the
R2 upload in section 3 being complete + reachable — do NOT start this
before section 3's curl checks pass.

- [ ] Partner Center → your app → **Update app** → new submission draft.
- [ ] **Packages** → edit **Package URL** to the new version:
      `https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/windows/vX.Y.Z/store/Touchless_Store_Installer.exe`
      Historically this URL sits stale from the previous submission —
      1.1.8 shipped a 404 because we forgot to update it. Take a
      screenshot after editing so you can prove it was updated.
- [ ] Architecture: `x64` (Touchless is x64-only).
- [ ] App type: `EXE`.
- [ ] Installer parameters unchanged:
      `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /NOCANCEL`
- [ ] Exit codes unchanged (0=success, 2/5=cancel, 3=in-progress,
      8=reboot, misc=1/4/7).
- [ ] Store listing → What's new field describes this version's changes.
- [ ] Availability → Markets = all (default), publish date = "as soon as
      possible after certification".
- [ ] Submit for certification. Note the submission ID.
- [ ] **STOP HERE and wait for one of:**
      - Partner Center emails "certification passed" and the submission
        shows "In the Store".
      - The user tells you the Store is ready.
      Do not proceed to section 5 until this is true — otherwise
      GitHub-Releases users get pushed to a version whose Store variant
      is still pending review.

## 5. Publish GitHub Release (this is what fires the in-app popup)

The moment the GitHub Release goes public (non-draft, non-pre-release),
every currently-running Touchless install polling `release_checker` will
see it on its next check and try to surface the update popup. That
means: if the R2 files aren't uploaded yet (section 3) or the popup is
broken (see section 7), users hit the failure the same minute you push.
Only do this AFTER section 3 is green and section 4 has confirmed Store
acceptance.

- [ ] Draft or edit the release for tag `vX.Y.Z`.
- [ ] Release body includes these markers so `release_checker` can find
      the Cloudflare installer:
      ```
      full-installer-url: https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/windows/vX.Y.Z/Touchless_Installer.exe
      full-installer-size: <bytes>
      ```
- [ ] Attach the app-zip auto-update asset if this release ships one
      (name: `Touchless_App_Update_X.Y.Z.zip`). This is what enables the
      silent app-zip auto-update path.
- [ ] Do **not** attach `Touchless_Installer.exe` itself — too big;
      it lives on Cloudflare.
- [ ] Push the tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
- [ ] Publish the release (untick Draft, untick Pre-release).

## 6. Update the website IMMEDIATELY after GitHub Release

The website's download button is what new users hit. If it points at the
old version they miss the new one. Per the standing rule ("website edits
go to preview branch first; only push to prod on explicit approval"),
this is a two-step operation.

- [ ] Update `c:\touchless-website` — bump the version string and the
      download link on the download page and the release-notes page.
- [ ] Push to `redesign` branch → Cloudflare Pages preview URL updates.
- [ ] Confirm preview looks correct.
- [ ] **Ask the user: "push website to production?"** Do not push to
      `main` without explicit confirmation.
- [ ] On approval, push `main`. Verify `touchless-control.com` shows the
      new version.

## 7. The critical popup + update-button verification (do this on
       a real machine, not the dev checkout)

Once sections 5 and 6 are live, users' installs will start polling for
the update. Do the verification NOW so you catch any breakage in the
same session, not from a user report the next day.

- [ ] On a clean machine (or VM) that has the PREVIOUS shipped version
      installed (not the dev build — the actual N-1 installer downloaded
      from Cloudflare), launch the app.
- [ ] Within 15 seconds, the update popup surfaces **visibly on-screen**.
      Alt+Tab is NOT required to find it. If it's invisible → HALT and
      investigate immediately. This exact bug shipped in 1.1.7 + 1.1.8.
- [ ] Popup shows the correct new version string in the title.
- [ ] Popup shows the correct download size text.
- [ ] Windows tray shows a balloon notification within 15 seconds as a
      fallback.
- [ ] Click "Download Update". Progress bar appears. Download completes.
- [ ] App restarts on the new version. Confirm the About panel version
      string matches the intended new version.
- [ ] Uninstall + reinstall the previous version. This time click
      "Later" instead of Download. Re-launch. Popup re-appears
      (assuming no dismissal was persisted for this version).
- [ ] **If auto-update is enabled + this release is app-zip type:**
      restart the previous version. Silent download + apply completes
      without any dialog. Tray balloon confirms the new version. App
      restart shows new version.

## 8. Post-publish sanity (last-line-of-defense)

- [ ] Download the Cloudflare installer through touchless-control.com
      (not directly via the R2 URL). Install on a machine that doesn't
      already have Touchless. First-launch succeeds — camera opens,
      main window renders, no error dialogs.
- [ ] Open Store (from a different machine or a signed-in test account)
      → your app page → verify the Store version has updated (or is at
      least "installing"). If not yet propagated, this is Microsoft's
      client-side auto-update timing (1-7 days) — nothing to fix, but
      note it so you know when the Store cohort will start seeing it.
- [ ] Update `MEMORY.md` if this release established anything new worth
      remembering across sessions.

## 9. If ANY box turned red

Do not ship. Roll back:

- **If GitHub Release published**: mark it as draft or delete it. This
  is the ONE user-facing announcement — pulling it stops new update
  checks from finding it (existing check-and-download cycles in the
  moment already saw it and may have started).
- **If website updated**: revert the `main` push. Users going to the
  download page fall back to N-1 which is still on Cloudflare.
- **If R2 uploaded**: leave the files — they don't hurt sitting there.
- **If Store submitted**: cancel the submission in Partner Center if
  it's still in Draft or Certification. If it's already In the Store,
  submit a rollback pointing at the previous version's Package URL.

Then fix the underlying bug + add a test that pins the failure so it
can't return + re-walk from section 1.

---

## Referenced from

- `../PUBLISHING_POLICY.md` — the stop-ship rules governing all publish
  paths
- `RELEASE_PROCESS.md` — the step-by-step build procedure
- `../CLAUDE.md` — instructs the AI agent to load PUBLISHING_POLICY (and
  thus this file) before any release-touching task
