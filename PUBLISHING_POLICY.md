# PUBLISHING POLICY — Touchless

**Purpose.** Every shipped bug in the last three releases could have been
caught by a checklist item that didn't exist yet. This file is the
checklist. Every publish (GitHub Release, Cloudflare R2 upload, Microsoft
Store submission) MUST satisfy every rule below. **No exceptions, no
"just this once", no "we'll fix it in a hotfix".** A "hotfix" is what we
call a bug that already reached users.

**Where this lives.** Root of the repo (`PUBLISHING_POLICY.md`) so it's
visible at a glance in the file tree. Referenced from `CLAUDE.md` so the
AI agent loads it every session. Referenced from `docs/RELEASE_PROCESS.md`
so the step-by-step release doc cross-links it.

**Enforcement is dual.** Machine-checkable items are enforced by tests
that CI (or `run_test.py` locally) must run green. Human-eyeball items
are a physical walk-through per release. Both are stop-ship.

---

## STOP-SHIP RULES (all must be green before a publish)

### Automated — must pass in the test suite

- [ ] `python -m pytest tests/ -q` returns 0 failures. Any red test halts
      the release. No `@pytest.mark.skip` added in the release commit to
      make a red test go away.
- [ ] `python -c "from hgr.app.updater.update_dialog import UpdateDialog; ..."`
      instantiates `UpdateDialog` with a mock `ReleaseInfo` and confirms
      `dlg.isVisible()` is True and `hasattr(dlg, '_body')` is True.
      This is `tests/test_update_dialog_smoke.py` and it must pass.
      Reason: 1.1.8 and 1.1.8.1 both shipped a NameError inside
      `UpdateDialog.__init__` that made the dialog impossible to
      construct. A single instantiation test would have caught it.
- [ ] Version comparison test: `_is_newer("1.1.9", "1.1.9rc1")` returns
      True; `_is_newer("1.1.8.1", "1.1.8")` returns True. These pin the
      PEP 440 semantics we depend on.
- [ ] Import smoke: `python -c "from hgr.app.ui import main_window;
      from hgr.app.updater import update_dialog; from hgr.app.updater
      import updater; from hgr.app.updater import store_updater;
      print('OK')"` returns "OK" with no traceback.

### Human eyeball — must pass on a real machine

- [ ] **Update-popup smoke test** (procedure in `docs/RELEASE_PROCESS.md`
      section "Update-popup smoke test"). On a previous shipped version,
      point release_checker at the pending release and confirm the popup
      surfaces on-screen within 15 seconds, sits above the main window,
      renders the version + size text correctly, and "Download Update"
      is clickable.
- [ ] **Silent auto-update path** (if the release is app-zip): a user with
      `auto_update_enabled=true` on the previous version relaunches, the
      download+apply completes silently, and a tray balloon confirms the
      new version. If no tray balloon fires, HALT.
- [ ] **Signed binaries.** Every .exe in the installer is Azure Artifact
      Signed via `dotnet sign` (not signtool+dlib — that path fails
      silently). Verify with `signtool verify /pa /v release\Touchless_Installer.exe`
      exits 0.
- [ ] **Version consistency triple-check.** All three sources agree on
      the version string:
      - `src/hgr/__init__.py` `__version__`
      - `installers/windows/hgr_app.iss` `#define MyAppVersion`
      - The Git tag being pushed
      Enforced by `python tools/validate_release.py vX.Y.Z`.

### Distribution — must be verified before announcing

- [ ] **R2 upload verified reachable.** After `rclone copyto` to
      `r2:hgr-downloads/windows/vX.Y.Z/...`, `curl -I` on the public
      URL returns 200 with the right Content-Length. Silent-fail
      uploads have bitten us before.
- [ ] **GitHub Release drafted with markers.** The release body contains
      the `full-installer-url:` and `full-installer-size:` markers
      pointing at the R2 URL, so the in-app `release_checker` can find
      the installer.
- [ ] **Microsoft Store — Partner Center package URL updated.** If we're
      pushing to Store too, the Partner Center submission's Package URL
      field is edited to point at the NEW version's `store/` subfolder
      (e.g. `windows/v1.1.8.2/store/Touchless_Store_Installer.exe`).
      Historically the URL is often stale from the previous submission.
- [ ] **Dual-hosted Store installer.** The signed Store installer exists
      at BOTH `v<X.Y.Z>/store/Touchless_Store_Installer.exe` AND
      `v<X.Y.Z>/Touchless_Store_Installer.exe`. Partner Center's saved
      URL sometimes assumes the flat path.

---

## POST-PUBLISH RULES

- [ ] **Verify installability from GitHub Releases.** Download the
      Cloudflare installer via the touchless-control.com download page
      (not directly), install on a machine that doesn't already have
      Touchless, and confirm first-launch succeeds.
- [ ] **Verify auto-update from N-1.** Install version N-1, launch it,
      let the in-app updater find N and either (a) show the visible
      dialog or (b) silently apply the app-zip. Both paths must reach a
      usable install of N.
- [ ] **Post to website.** Update `touchless-control.com` version string
      and download link. Push to preview branch first (redesign), then
      main only on explicit approval per the standing policy.

---

## HISTORY OF WHAT WE'RE PREVENTING

Every line in this file exists because a version shipped without it.

- **1.1.7:** r51 chrome refactor introduced frameless-window z-order bug
  → update popup invisible to all Windows users → nobody could see
  update prompts. Not caught because there was no post-build "does the
  popup appear" test.
- **1.1.8:** same bug rode through unchanged. Not caught because the
  release focus was elsewhere (Spotify, installer polish) and no
  release-level regression check existed for the update dialog.
- **1.1.8 and 1.1.8.1:** `UpdateDialog._build_ui` referenced a bare
  `body` name that raised NameError on construction. Not caught
  because no unit test ever instantiated `UpdateDialog`. The 1.1.8.1
  hotfix polished a class that was crashing at __init__.
- **1.1.8:** Microsoft Store Partner Center package URL was stale from
  1.1.7 → Store users got a 404. Not caught because there was no
  "does the Partner Center URL match the actual upload path" step.
- **1.1.8.1:** Store users can't discover the update through the app
  because the in-app popup that would tell them to check the Store is
  the same broken popup. Users had to manually open Store → Library →
  Get updates.

---

## WHEN THIS DOC CHANGES

Every release that ships a user-visible regression MUST add a new
STOP-SHIP rule here BEFORE the hotfix ships. The rule is what proves
we've absorbed the lesson. No rule = we'll do it again.

**AI-agent note:** You (Claude / Codex / whoever) are expected to read
this file at the start of any session that touches the release, updater,
installer, or Store submission code. If asked to "publish 1.1.x", walk
the full step-by-step in `docs/UPDATE_RELEASE_CHECKLIST.md` item by
item and REPORT green/red per line before running any `git push`,
`rclone copyto`, or Partner Center action.

## See also

- **`docs/UPDATE_RELEASE_CHECKLIST.md`** — the walkable list for every
  update: pre-build → sign → Cloudflare uploads → Partner Center
  submission → GitHub Release → website → popup verification →
  post-publish sanity, in the exact order they must happen.
