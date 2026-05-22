# Microsoft Store submission — EXE installer path

This doc covers the workflow to ship `Touchless_Installer.exe` to the
Microsoft Store **without** repackaging as an MSIX. Microsoft opened
the Store to traditional EXE and MSI installers in 2022 — same binary
that the website serves, just resubmitted through Partner Center.

## Why EXE-direct instead of MSIX

- **Same binary as the website.** The Inno Setup `.exe` we already
  produce IS the Store package. No second build, no second SKU, no
  drift between channels.
- **No re-signing or repackaging.** The Azure Trusted Signing
  signature on the EXE is the only signature the Store cares about.
- **No code changes.** MSIX-packaged apps run in a restricted
  container (different file-system view, no broad registry write,
  no helper-process spawn). The Inno installer + the Touchless
  runtime work as-is.
- **MSIX is optional later.** If we ever need the appcontainer
  benefits (PWA-style isolation, true sandboxing for the Store)
  we can wrap the same payload then. Today, EXE-direct is the
  fastest path to Store validation.

## What the Store requires from an EXE submission

Per Microsoft's "Bring your existing apps to the Microsoft Store"
docs, an EXE installer must:

1. **Be code-signed** by a publicly trusted CA. ✓ Already true —
   Azure Trusted Signing certificate, `Konstantin Markov` (Individual).
2. **Support silent install** with a documented command-line flag.
   ✓ Inno Setup supports `/SILENT` and `/VERYSILENT`. The Store's
   installer-orchestration uses whichever flag we declare in the
   Partner Center submission.
3. **Support silent uninstall** with a matching flag. ✓ Inno
   creates `unins000.exe` that accepts `/SILENT` and `/VERYSILENT`.
4. **Write a proper Add/Remove Programs entry** with publisher,
   version, uninstall command. ✓ The Inno installer already does this
   (`HKLM\...\Uninstall\{2C4EE680-53F5-4D83-92A8-ADF4D2D8794E}_is1` —
   AppId matches the `.iss` file).
5. **Run without elevation when possible.** ✓ Per-user install path
   (`%LOCALAPPDATA%\Programs\Touchless`) — no UAC prompt on a clean
   machine.
6. **Detect and handle existing installs.** ✓ Inno's `AppId` is a
   fixed UUID; subsequent installs detect the prior install and
   upgrade-in-place.

## Required fields on the Partner Center submission

When creating a Store submission for `Touchless_Installer.exe`:

| Field | Value |
|---|---|
| **Identity / Package family name** | Reserved at Partner Center (e.g. `KonstantinMarkov.Touchless`) |
| **Publisher display name** | `Konstantin Markov` |
| **Application name** | `Touchless` |
| **Installer URL** | The R2 URL — `https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/windows/v1.1.0/Touchless_Installer.exe`. Store can also host the file. |
| **Installer type** | `Exe` |
| **Install command-line** | `/SILENT /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS` |
| **Uninstall command** | `"{app}\unins000.exe" /SILENT` |
| **Install scope** | Per-user (matches `DefaultDirName={localappdata}\Programs\{#MyAppName}`) |
| **Architecture** | x64 |
| **Locales** | en-US |
| **Age rating** | Use IARC questionnaire — for Touchless, *Productivity* category, no concerning content |
| **Privacy policy URL** | `https://touchless-control.pages.dev/privacy.html` |
| **Support contact** | `konstantinvmarkov@gmail.com` |

## Things that will trigger Store certification reviewers

1. **SmartScreen warning during install.** Currently expected on the
   first few hundred installs while the Azure Trusted Signing cert
   gains reputation. Store reviewers know this and accept it; the
   submission form has a "code-signed, gaining reputation" note we
   can flag.
2. **Capabilities not declared.** Touchless uses Camera + Microphone.
   The Store submission has a "Properties → Capabilities" section
   where these MUST be ticked, otherwise reviewers fail the build
   for undeclared device access.
3. **Auto-update prompts.** Our GitHub-poll updater
   (`release_checker.py`) currently runs unconditionally. For Store
   submissions Microsoft prefers the Store handle the update
   delivery — but for EXE-direct submissions this is OK as long as
   our updater UI clearly says *"download will continue when you
   click Update"* and doesn't auto-restart silently. The current
   `update_dialog.py` UX meets this bar.

## Submission steps (one-time setup)

1. **Reserve the app name** at <https://partner.microsoft.com/dashboard/>
   → Apps and games → New product → MSIX or PWA app → Reserve name
   `Touchless`. (Yes, even for EXE submissions the reservation flow
   uses the same form.)
2. **Create a new Submission**:
   - Pricing and availability → Free, all markets except
     restricted-by-default ones (we have no special encryption).
   - Properties → Category: Productivity, Subcategory: General.
     Declare Camera + Microphone capabilities.
   - Age ratings → IARC: complete questionnaire, expect *PEGI 3
     / ESRB E* rating.
   - Packages → Upload `Touchless_Installer.exe`. Select
     "Allow Microsoft to make changes to my app's behavior on
     end-user PCs" → **No** (we don't want Store to wrap us in MSIX).
   - Store listings → en-US:
     - Description: pull from `hgr-download-page/index.html`'s
       tagline + features.
     - Screenshots: at least 1, recommend 4-8 (1366×768 or higher).
     - Hero image: 1920×1080 of the home screen.
3. **Submit for certification.** Reviewer turnaround is 24-72 hours
   for the first submission, generally faster after.

## Subsequent updates

After 1.1.0 is approved, every future Touchless release goes through
the SAME submission flow with the new installer URL. The Store will:

1. Notice the new submission.
2. Diff metadata changes (skipped if you only update the installer URL).
3. Push the update to anyone who installed via the Store.

Our GitHub-poll auto-updater also still runs for those users — when
both channels offer the same version, the duplicate-update path is
idempotent (extract-over-install + version-equal-check), so no harm.

## Update mechanism reference (already shipped)

For users who installed from the website:

| Asset | Size | How it's applied | UAC? |
|---|---|---|---|
| `Touchless_App_Update_<ver>.zip` | 50-150 MB | `_apply_update.bat` waits for app to exit, unzips over install, relaunches | No |
| `Touchless_Installer.exe` | ~2.4 GB | Re-runs Inno installer silently | Only on legacy Program Files installs |

For users who installed from the Store, the Store's update channel
handles it. Our in-app updater will still attempt to run; the version
checks make it a no-op when the Store has already updated us.

<!-- Author: Konstantin Markov -->
