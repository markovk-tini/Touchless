; Touchless Windows installer
; Place this file at installers/windows/hgr_app.iss
;
; -- Build modes --
;   Default  : STUB installer. Tiny (~5-15 MB) Setup.exe that
;              downloads Touchless_Payload_v<version>.zip from R2
;              at install time and unpacks it into {app}. Mirrors the
;              Adobe / Discord / Spotify pattern; lets the website
;              link feel like a few-second click.
;   /DMONOLITHIC=1 : the original behavior — Setup.exe carries the
;              entire dist/Touchless tree inside it (~2.4 GB). Kept
;              as the "offline edition" for users on planes / air-
;              gapped machines, behind the MONOLITHIC=1 env flag in
;              builder/windows/build_windows.bat.
;
; -- Defines passed in by build_windows.bat --
;   /DPAYLOAD_URL=<full https URL>     (stub mode, required)
;   /DPAYLOAD_FILE=<filename only>     (stub mode, required)
;   /DPAYLOAD_SHA256=<lowercase hex>   (stub mode, required — verifies the
;                                       downloaded zip before extraction)
;   /DMONOLITHIC=1                     (optional — switches to embedded zip)

#define MyAppName "Touchless"
#define MyAppVersion "1.1.9.2"
#define MyAppPublisher "Konstantin Markov"
#define MyAppExeName "Touchless.exe"
#define DistDir "..\..\dist\Touchless"
#define IconFile "..\..\assets\icons\touchless_icon.ico"

#ifndef MONOLITHIC
  #define STUB
#endif

#ifdef STUB
  #ifndef PAYLOAD_URL
    #error "STUB build requires /DPAYLOAD_URL=https://..."
  #endif
  #ifndef PAYLOAD_FILE
    #error "STUB build requires /DPAYLOAD_FILE=filename.zip"
  #endif
  #ifndef PAYLOAD_SHA256
    #error "STUB build requires /DPAYLOAD_SHA256=lowercase-hex"
  #endif
  ; File count of the dist tree (the build script counts and passes
  ; this in). Used to drive the extraction-progress bar — if the
  ; build script forgets, fall back to a sensible default so the bar
  ; still moves vaguely correctly instead of staying at 0%.
  #ifndef PAYLOAD_FILE_COUNT
    #define PAYLOAD_FILE_COUNT "2000"
  #endif
  ; r19: sizes used by the free-space precheck, in MEGABYTES. The build
  ; script measures both and passes them in; the fallbacks are the
  ; measured 1.1.9.2 values so a hand-run ISCC still checks something.
  #ifndef PAYLOAD_ZIP_MB
    #define PAYLOAD_ZIP_MB "1673"
  #endif
  #ifndef PAYLOAD_TREE_MB
    #define PAYLOAD_TREE_MB "3408"
  #endif
#endif

[Setup]
AppId={{2C4EE680-53F5-4D83-92A8-ADF4D2D8794E}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; r23: identity metadata. The stub shipped with a blank version
; resource (FileVersion 0.0.0.0, no copyright, no original name),
; which reads as an anonymous binary to reputation engines and
; leaves Add/Remove Programs with no publisher link. These are the
; fields Explorer's Details tab and Norton's file record read.
AppPublisherURL=https://touchless-control.com
AppSupportURL=https://touchless-control.com
AppUpdatesURL=https://touchless-control.com
AppCopyright=Copyright (C) 2026 {#MyAppPublisher}
VersionInfoVersion={#MyAppVersion}
VersionInfoProductVersion={#MyAppVersion}
VersionInfoProductName={#MyAppName}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} Setup
VersionInfoCopyright=Copyright (C) 2026 {#MyAppPublisher}
VersionInfoOriginalFileName=Touchless_Installer.exe
; Per-user install under %LOCALAPPDATA%\Programs\Touchless. Avoids
; UAC entirely — the app folder is user-writable, so subsequent
; auto-updates can replace files without prompting the user for
; admin approval. Same approach Discord/Slack/VS Code (User Installer)
; use. Trade-off: each Windows user installs separately, which is
; fine for the friends-and-family scale we ship at.
;
; v1.1.7 fix: PrivilegesRequiredOverridesAllowed=dialog was REMOVED.
; It popped a "just for me / all users" choice dialog on startup;
; users clicking "all users" (whether by accident, or because it looked
; like the safer default) then hit Inno's SetupErrorRoleMustBeAdmin
; error — "You must be logged in as an administrator when installing
; this program" — and the install died. Standard-user accounts with
; no admin creds available anywhere on the box were completely locked
; out of a working installer. With the override removed, there's no
; dialog and no choice — Inno silently installs to %LOCALAPPDATA%.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#MyAppName}
; FSL-1.1-Apache-2.0 license shown on the License Agreement page so
; users see the terms before installing. Path is relative to the
; .iss file location (installers/windows/).
LicenseFile=..\..\LICENSE
DefaultGroupName={#MyAppName}
OutputDir=..\..\release
OutputBaseFilename=Touchless_Installer
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#MyAppExeName}
ChangesAssociations=no
DisableProgramGroupPage=yes
; Auto-update support: when the running app's updater launches us
; with /CLOSEAPPLICATIONS, Inno Setup gracefully closes Touchless.exe
; before replacing files (otherwise the in-use .exe blocks the
; upgrade). /RESTARTAPPLICATIONS in the updater command line plus
; the [Run] entry below put Touchless back up automatically.
; CloseApplications=force tells Inno to actually kill the running
; Touchless.exe instead of asking nicely. The default (`yes`) sends
; WM_CLOSE / WM_ENDSESSION which Qt apps may not handle cleanly within
; Inno's timeout, producing the "Setup was unable to automatically
; close all applications" dialog users hit during the 1.1.3 Store
; update. `force` skips the negotiation and just terminates the
; process — safe here because Touchless's auto-save handlers run on
; QApplication.quit() in the update path, and after that the .exe is
; just a frozen-bundle wrapper with no in-flight state worth
; preserving.
CloseApplications=force
CloseApplicationsFilter=*.exe,*.dll
RestartApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked
; NOTE: the prior "defender_exclusion" task (which added the install
; folder to Microsoft Defender exclusions) was REMOVED in b8. Even
; with Flags: unchecked it was producing UAC prompts during install
; because users were ticking the checkbox without realizing the
; "Verb: runas" Add-MpPreference call would trigger elevation. The
; whole install is now 100 % per-user under %LOCALAPPDATA% and
; UAC-free. Users whose GPU mode falls back to CPU because Defender
; quarantined DirectML.dll can add the exclusion manually via
; Windows Security -> Virus & threat protection -> Exclusions.

[InstallDelete]
; v1.1.9 rebuild — remove Windows system DLLs that older builds wrongly
; shipped inside the bundle. 1.1.9 shipped Anaconda's ICU 73 in
; `_internal\` (PyInstaller resolves DLL imports off the build machine's
; PATH), and `_internal\` precedes System32 on the frozen app's DLL search
; path, so that copy shadowed Windows' own ICU. Its symbols are
; version-suffixed (`ucnv_open_73`) while PySide6 6.10+ Qt6Core imports the
; plain names, so the app could not start at all:
;   ImportError: DLL load failed while importing QtGui:
;   The specified procedure could not be found
; Inno does NOT delete orphaned files on an in-place upgrade, and neither
; does the app-zip updater (it only carries Touchless.exe + assets). Without
; this section, "updating" a broken 1.1.9 — including the Microsoft Store's
; own Update button, which runs this installer over the existing folder —
; would leave the shadowing DLL behind and the app would keep crashing.
; `builder\windows\hgr_app.spec` now strips these at build time; this
; section heals machines that already have them.
; NOTE: only `_internal\*.dll` at the top level is matched. QtWebEngine's
; `PySide6\resources\icudtl.dat` is a required data file and is NOT touched.
Type: files; Name: "{app}\_internal\icuuc.dll"
Type: files; Name: "{app}\_internal\icuin.dll"
Type: files; Name: "{app}\_internal\icu.dll"
Type: files; Name: "{app}\_internal\icudt*.dll"
Type: files; Name: "{app}\_internal\ucrtbase.dll"
Type: files; Name: "{app}\_internal\api-ms-win-*.dll"

[Files]
#ifdef MONOLITHIC
; Original embedded-payload behavior. Pulls the entire PyInstaller
; bundle into the installer at compile time. Used only when the
; build is invoked with MONOLITHIC=1 (offline-edition build).
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif
#ifdef STUB
; STUB mode has no embedded files — everything ships in the
; downloaded payload zip. The [Code] section handles the download
; and extraction.
#endif

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon; IconFilename: "{app}\{#MyAppExeName}"

[Run]
; b8: Microsoft Defender exclusion was removed (it was the only
; UAC trigger in the install). See the comment under [Tasks].
;
; Two Launch entries, deliberately split:
;   1. Interactive: postinstall + skipifsilent + Description text.
;      This is the classic Inno "Run Touchless when finished?"
;      checkbox shown only when the user runs the installer
;      themselves. Skipped under silent install.
;   2. Silent: runasoriginaluser + RestartApplications behaviour.
;      The 1.1.3 Store update finished without relaunching the app
;      because the only Launch entry above had skipifsilent and
;      RestartApplications=yes alone doesn't relaunch when no
;      Touchless process was running at the start of install (the
;      app-zip path quits the app gracefully before the helper runs).
;      The silent entry fires unconditionally under /VERYSILENT so
;      Store users see the app come back automatically after an
;      update, matching the manual-install UX.
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
Filename: "{app}\{#MyAppExeName}"; Flags: nowait runasoriginaluser; Check: WizardSilent

; [UninstallRun] removed in b8 along with the install-time Defender
; exclusion. With nothing to undo, the uninstall is also UAC-free.

#ifdef STUB
[Code]
// Stub installer download + extract logic. Uses Inno Setup 6.1+'s
// built-in CreateDownloadPage + DownloadTemporaryFile (no third-
// party plugins). Flow:
//   1. NextButtonClick(wpReady) shows the download page with a
//      progress bar; user can't proceed until the download lands
//      (or they cancel).
//   2. CurStepChanged(ssInstall) unpacks the zip into the install
//      directory via PowerShell's Expand-Archive (built into
//      Windows 10+).
//   3. The post-Install [Run] entries (Defender exclusion + Launch
//      checkbox) and [Icons] / [UninstallRun] all reference the
//      install directory, which is populated by step 2 — no
//      other plumbing changes.
//
// SHA256 verification is built into DownloadPage.Add — if the
// downloaded zip doesn't match PAYLOAD_SHA256 (baked in at
// compile time by build_windows.bat), Inno raises an exception
// before extraction runs.

var
  DownloadPage: TDownloadWizardPage;

// v1.1.9.2 (r19): free-space precheck.
//
// The stub needs room for BOTH the downloaded zip and the extracted
// tree at the same time:
//     {tmp}  <- PAYLOAD_ZIP_MB   (the .zip, deleted only after setup)
//     {app}  <- PAYLOAD_TREE_MB  (the extracted files)
// On a default install both live on C:, so the peak is the sum.
// Measured for 1.1.9.2: zip 1673 MB, extracted tree 3408 MB / 6031 files.
// Before r19 there was NO check at all: a user short on space got
// "Payload extraction failed after 3 attempts: tar.exe exit code 1"
// with no indication that disk space was the cause.
function DriveRoot(const Path: String): String;
begin
  Result := ExtractFileDrive(Path);
  if Result <> '' then
    Result := Result + '\';
end;

function FreeMBOn(const Path: String): Int64;
var
  FreeBytes, TotalBytes: Int64;
begin
  Result := -1;
  if GetSpaceOnDisk64(DriveRoot(Path), FreeBytes, TotalBytes) then
    Result := FreeBytes div 1048576;
end;

// Returns '' when there is enough room, else a user-facing message.
function CheckFreeSpace(): String;
var
  TmpDir, AppDir, TmpRoot, AppRoot: String;
  TmpFree, AppFree, NeedTmp, NeedApp, NeedBoth: Int64;
begin
  Result := '';
  NeedTmp := {#PAYLOAD_ZIP_MB};
  NeedApp := {#PAYLOAD_TREE_MB};
  TmpDir := ExpandConstant('{tmp}');
  AppDir := ExpandConstant('{app}');
  TmpRoot := DriveRoot(TmpDir);
  AppRoot := DriveRoot(AppDir);
  TmpFree := FreeMBOn(TmpDir);
  AppFree := FreeMBOn(AppDir);
  // Unknown free space (network path, odd volume) -> do not block.
  if (TmpFree < 0) or (AppFree < 0) then
    Exit;
  if CompareText(TmpRoot, AppRoot) = 0 then begin
    NeedBoth := NeedTmp + NeedApp;
    if TmpFree < NeedBoth then
      Result :=
        'Not enough free disk space on drive ' + TmpRoot + #13#10#13#10 +
        'Touchless needs about ' + IntToStr(NeedBoth div 1024) + ' GB free to install:' + #13#10 +
        '  - ' + IntToStr(NeedTmp div 1024) + ' GB for the download' + #13#10 +
        '  - ' + IntToStr(NeedApp div 1024) + ' GB for the installed app' + #13#10#13#10 +
        'Free space right now: ' + IntToStr(TmpFree div 1024) + ' GB' + #13#10#13#10 +
        'Free up space (Windows Settings > System > Storage) and run the ' +
        'installer again. The download space is released once setup finishes.';
  end else begin
    if TmpFree < NeedTmp then
      Result :=
        'Not enough free disk space on drive ' + TmpRoot + ' for the download.' + #13#10#13#10 +
        'Needed: about ' + IntToStr(NeedTmp div 1024) + ' GB.  Free right now: ' +
        IntToStr(TmpFree div 1024) + ' GB.' + #13#10#13#10 +
        'Windows downloads the installer payload to this drive even when ' +
        'Touchless is installed elsewhere.'
    else if AppFree < NeedApp then
      Result :=
        'Not enough free disk space on drive ' + AppRoot + ' for the installation.' + #13#10#13#10 +
        'Needed: about ' + IntToStr(NeedApp div 1024) + ' GB.  Free right now: ' +
        IntToStr(AppFree div 1024) + ' GB.';
  end;
end;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage(
    'Downloading Touchless',
    'Please wait while Setup downloads the Touchless payload from the Touchless website.',
    nil);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Attempt: Integer;
  MaxAttempts: Integer;
  LastError: String;
  Succeeded: Boolean;
  SpaceMsg: String;
begin
  if CurPageID = wpReady then begin
    // v1.1.9.2 (r19): refuse BEFORE spending a 1.6 GB download when the
    // machine cannot hold the result. Checked here rather than in
    // PrepareToInstall because the download happens on this click.
    SpaceMsg := CheckFreeSpace();
    if SpaceMsg <> '' then begin
      SuppressibleMsgBox(SpaceMsg, mbCriticalError, MB_OK, IDOK);
      Result := False;
      Exit;
    end;
    // r45: retry the 3-GB payload download up to 4 times before
    // surfacing a hard failure. Inno's DownloadPage.Download uses
    // WinInet's single-shot GET with no resume support, so any
    // transient connection drop mid-transfer (Cloudflare 5xx, ISP
    // hiccup, laptop wake-from-sleep) throws and the user has to
    // restart. Real-world reports: ~30-40% of first-attempt installs
    // over slow/flaky connections were failing. Retrying WITH the
    // Show/Hide cycle keeps the UX identical to a fresh attempt.
    MaxAttempts := 4;
    Succeeded := False;
    LastError := '';
    for Attempt := 1 to MaxAttempts do begin
      DownloadPage.Clear;
      DownloadPage.Add('{#PAYLOAD_URL}', '{#PAYLOAD_FILE}', '{#PAYLOAD_SHA256}');
      if Attempt > 1 then begin
        DownloadPage.SetText(
          'Downloading Touchless (attempt ' + IntToStr(Attempt) +
          ' of ' + IntToStr(MaxAttempts) + ')',
          'The previous attempt was interrupted. Retrying...');
      end;
      DownloadPage.Show;
      try
        try
          DownloadPage.Download;
          Succeeded := True;
        except
          if DownloadPage.AbortedByUser then begin
            Log('Aborted by user.');
            DownloadPage.Hide;
            Result := False;
            Exit;
          end else begin
            LastError := GetExceptionMessage;
            Log('Download attempt ' + IntToStr(Attempt) +
                ' failed: ' + LastError);
          end;
        end;
      finally
        DownloadPage.Hide;
      end;
      if Succeeded then break;
      // Brief backoff before retrying so the network stack gets a
      // moment to settle if the previous failure was transient.
      // 1500 ms is short enough not to feel like a wait, long enough
      // for a Cloudflare edge to reset any half-closed sockets.
      if Attempt < MaxAttempts then
        Sleep(1500);
    end;
    if Succeeded then
      Result := True
    else begin
      SuppressibleMsgBox(
        'Download failed after ' + IntToStr(MaxAttempts) +
        ' attempts. Last error: ' + LastError + #13#10#13#10 +
        'Please check your internet connection and re-run the ' +
        'installer.',
        mbCriticalError,
        MB_OK,
        IDOK);
      Result := False;
    end;
  end else
    Result := True;
end;

// Recursively count files in Dir so the extraction-progress poll
// has a numerator. Cheap on Windows even for 2-3k files (FindFirst/
// FindNext is OS-level enumeration), and the poll only runs every
// 500 ms during extract — well under any real cost concern.
function CountFilesRecursive(const Dir: String): Integer;
var
  FindRec: TFindRec;
begin
  Result := 0;
  if FindFirst(Dir + '\*', FindRec) then begin
    try
      repeat
        if (FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY) = 0 then
          Result := Result + 1
        else if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
          Result := Result + CountFilesRecursive(Dir + '\' + FindRec.Name);
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

// Escape a path for embedding inside a PowerShell single-quoted
// string literal. PS rule: ' is escaped by doubling it ('').
// Without this, any user whose install dir or username contains an
// apostrophe (O'Brien, D'Angelo, "Tom's Apps") silently breaks the
// Expand-Archive command line — PS sees the path as terminated and
// the rest of the command as garbage. This was a 100%-failure-rate
// latent bug for users with apostrophes in their paths.
function PsQuoteEscape(const s: String): String;
var
  i: Integer;
begin
  Result := '';
  for i := 1 to Length(s) do begin
    if s[i] = '''' then
      Result := Result + ''''''
    else
      Result := Result + s[i];
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  FileCount: Integer;
  TotalFiles: Integer;
  ZipPath, ExtractDir, DoneFlag, PsCmd, StatusText, ErrorMsg: String;
  TouchlessExePath, RenameTestPath: String;
  ResultStr: AnsiString;
  Attempt: Integer;
  TouchlessExeSize: LongInt;
  ExeFindRec: TFindRec;
  ExtractOK: Boolean;
  // v1.1.9.2 (r5): tar.exe fast-path (Norton-safer + faster than PS).
  TarExe: String;
  TarArgs: String;
  UseTar: Boolean;
  // v1.1.9.2 (r19): tar stderr capture (attempts 2+).
  TarErrPath: String;
  TarErrText: AnsiString;
  CmdArgs: String;
  SpaceNote: String;
  ExecOK: Boolean;
begin
  if CurStep = ssInstall then begin
    ZipPath := ExpandConstant('{tmp}\{#PAYLOAD_FILE}');
    ExtractDir := ExpandConstant('{app}');
    TarErrPath := ExpandConstant('{tmp}\tar_stderr.txt');
    DoneFlag := ExpandConstant('{tmp}\extract_done.flag');
    TouchlessExePath := ExtractDir + '\' + ExpandConstant('{#MyAppExeName}');
    TotalFiles := {#PAYLOAD_FILE_COUNT};
    if TotalFiles < 1 then TotalFiles := 1;

    if not ForceDirectories(ExtractDir) then
      RaiseException('Could not create install directory: ' + ExtractDir);

    // CRITICAL: kill any running Touchless before extraction. Inno's
    // built-in CloseApplications=force only fires when the [Files]
    // section is replacing files, but our STUB-mode [Files] is empty
    // — the actual file work happens via Expand-Archive below. Without
    // killing the process first, the extraction silently fails to
    // overwrite Touchless.exe (44 MB Python bundle) because Windows
    // refuses to replace a running .exe — every _internal/ file gets
    // updated, but Touchless.exe stays at the old version. Result:
    // user sees "installed successfully" but the app still reports the
    // old version after restart. /F = hard terminate, /T = kill the
    // whole process tree (engine worker, llama-server, whisper-stream).
    // Trailing `& exit 0`: taskkill returns nonzero when the process
    // wasn't running ("not found"), which is FINE; we don't want to
    // abort the install over that.
    Exec(
      ExpandConstant('{cmd}'),
      '/C taskkill /F /IM ' + ExpandConstant('{#MyAppExeName}') + ' /T 2>NUL & exit 0',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

    // v1.1.9.2 (r19): ALSO kill ORPHANED helper processes. THIS WAS THE
    // ROOT CAUSE OF THE "failed to extract" REPORTS.
    //
    // Touchless spawns ffmpeg (clip cache, camera capture, audio
    // bridges), llama-server and whisper-stream as children. `taskkill
    // /T` above only walks the tree of a LIVE Touchless.exe, so when
    // Touchless has already crashed or been force-killed its helpers
    // survive as orphans with NO parent -- Task Manager shows no
    // "Touchless", but an ffmpeg launched from
    // <install>\_internalfmpeg.EXE is still running, writing
    // clip-cache segments in a loop and holding an open handle on its
    // own image file. Windows then refuses to let tar overwrite
    // _internalfmpeg.EXE, tar returns exit code 1, and the install
    // dies at extraction. Field-confirmed 2026-09-24: clip-cache files
    // reappearing after deletion and an undeletable install folder on a
    // machine with 27 GB free, so it was never a disk-space problem.
    //
    // Targeted ON PURPOSE: only helpers whose executable path contains
    // "Touchless" are killed, so an ffmpeg the user installed for their
    // own use is untouched. -Command (not -File) needs no execution
    // policy, so there is no -ExecutionPolicy Bypass for Norton to
    // dislike. Wholly best-effort: any failure here is ignored and the
    // extract retry loop below still reports the real reason.
    Exec(
      'powershell.exe',
      '-NoProfile -NonInteractive -Command "' +
      '$ErrorActionPreference=''SilentlyContinue''; ' +
      'foreach($n in @(''ffmpeg'',''llama-server'',''whisper-stream'',''whisper-server'')){ ' +
      'foreach($p in @(Get-Process -Name $n -ErrorAction SilentlyContinue)){ ' +
      'try{ if($p.Path -and $p.Path -match ''Touchless''){ Stop-Process -Id $p.Id -Force } }catch{} } } ' +
      'exit 0"',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    // Pause so Windows fully releases the file handles + AV finishes
    // its post-mortem scan of the killed process. Bumped 750 -> 1500 ms
    // because the workflow audit flagged that slow systems with real-
    // time AV scanning can still hold the handle past 750 ms.
    Sleep(1500);

    // Reconfigure Inno's progress bar to track real extraction state.
    //
    // r62 HONESTY FIX. Two separate things were wrong with the bar, both
    // reported from the field as "empty, then jumps to full, or stops at
    // 9 / 10":
    //
    //  (a) The live poll below only ever existed in the PowerShell
    //      FALLBACK branch. Attempt 1 -- the path virtually every install
    //      takes -- runs tar with ewWaitUntilTerminated, which blocks this
    //      thread for the whole extraction, so the wizard cannot repaint at
    //      all and the gauge is simply jammed to 100% afterwards. It is not
    //      fixable by polling harder: going async needs either a shell in
    //      the process tree (undoes r19's AppLocker/Norton mitigation, which
    //      exists because installs were being BLOCKED) or a heuristic
    //      "finished" signal. Neither is worth trading for a cosmetic bar.
    //
    //  (b) The numerator was wrong anyway. CountFilesRecursive counts the
    //      files PRESENT in {app}, not the files extracted this run. On a
    //      fresh install those coincide; on an UPGRADE {app} already holds
    //      a full tree, so the count starts at the maximum and the bar is
    //      full before extraction begins. A one- or two-file difference
    //      between the old tree and the new payload is the "9 / 10".
    //
    // So: do not present a proportional bar we cannot honestly fill. Say
    // what is happening and how long it takes, and warn that the window
    // may stop responding -- which is true, and is exactly what made users
    // think it had hung.
    WizardForm.ProgressGauge.Min := 0;
    WizardForm.ProgressGauge.Max := TotalFiles;
    WizardForm.ProgressGauge.Position := 0;
    WizardForm.StatusLabel.Caption :=
      'Extracting ' + IntToStr(TotalFiles) + ' files (' +
      '{#PAYLOAD_TREE_MB} MB). This takes several minutes and the window ' +
      'may stop responding — that is normal, it is not stuck.';
    WizardForm.FilenameLabel.Caption := '';
    WizardForm.Update;

    // v1.1.9.2 (r5): Prefer bundled Windows tar.exe (libarchive) over
    // PowerShell Expand-Archive. Rationale:
    //   * ~2-4x faster on a ~1 GB payload (no PS startup + no progress
    //     reflection cost — libarchive goes straight to disk).
    //   * No powershell.exe process in the install-time tree —
    //     unblocks AppLocker / WDAC corp-locked machines.
    //   * Removes the -ExecutionPolicy Bypass surface Norton
    //     occasionally flags as suspicious.
    //   * Every supported Windows baseline (Win10 1803+ / Win11) ships
    //     tar.exe under C:\Windows\System32\tar.exe.
    // Falls back to the original PowerShell path if tar.exe is missing
    // (very old Win10, custom images) or fails to launch. The 3-attempt
    // AV-race retry wrapper stays around both branches.
    TarExe := ExpandConstant('{sys}\tar.exe');
    UseTar := FileExists(TarExe);
    if UseTar then begin
      // tar -xf <zip> -C <dir> — libarchive handles ZIP natively on
      // Win10 1803+. Quote-wrapped paths tolerate spaces + apostrophes.
      TarArgs := '-xf "' + ZipPath + '" -C "' + ExtractDir + '"';
    end;

    // Build the PowerShell command once (fallback path only, but still
    // populated so the fallback branch can Exec cleanly). PATHS ARE
    // APOSTROPHE-ESCAPED so usernames / install dirs containing '
    // (O'Brien, Tom's Apps) don't break the PS single-quoted string
    // literal.
    PsCmd :=
      '-NoProfile -NonInteractive -ExecutionPolicy Bypass ' +
      '-Command "try { Expand-Archive -LiteralPath ''' + PsQuoteEscape(ZipPath) + ''' ' +
      '-DestinationPath ''' + PsQuoteEscape(ExtractDir) + ''' -Force; ' +
      '''OK'' | Out-File -LiteralPath ''' + PsQuoteEscape(DoneFlag) + ''' -Encoding ascii } ' +
      'catch { $_.Exception.Message | Out-File -LiteralPath ''' + PsQuoteEscape(DoneFlag) + ''' -Encoding ascii }"';

    // EXTRACT RETRY LOOP (3 attempts, 3s pause between). Antivirus
    // scanners + slow disks can race the file replace — extraction
    // may report overall success while having silently failed on a
    // specific locked file. Retry catches transient locks. Mirrors
    // the build-side payload retry loop in build_windows.bat.
    ExtractOK := False;
    ErrorMsg := '';
    for Attempt := 1 to 3 do begin
      // Wipe any stale sentinel from a previous attempt (PS path uses it).
      if FileExists(DoneFlag) then DeleteFile(DoneFlag);

      WizardForm.StatusLabel.Caption := 'Extracting payload (attempt ' +
        IntToStr(Attempt) + ' of 3)...';
      WizardForm.Update;

      if UseTar then begin
        // Synchronous exec — tar returns when extraction completes.
        // ResultCode 0 = success. Progress bar jumps 0 -> 100 in one
        // hop (no live sentinel-poll), but the "Finalizing installation
        // — antivirus scanning..." label below already tells users what's
        // happening, so the UX gap is small vs. the speed win.
        //
        // v1.1.9.2 (r19): attempt 1 runs tar DIRECTLY (no shell in the
        // process tree, which is what AppLocker / Norton see on the
        // happy path). Attempts 2 and 3 run it through cmd.exe with
        // stderr redirected to a file, so a repeat failure reports the
        // DRIVER'S OWN message ("No space left on device", "Cannot
        // open: Permission denied", ...) instead of a bare exit code.
        // Three field failures were diagnosed by guesswork because this
        // output was being thrown away.
        if Attempt = 1 then
          ExecOK := Exec(TarExe, TarArgs, '', SW_HIDE, ewWaitUntilTerminated, ResultCode)
        else begin
          if FileExists(TarErrPath) then DeleteFile(TarErrPath);
          CmdArgs := '/S /C ""' + TarExe + '" ' + TarArgs + ' 2> "' + TarErrPath + '""';
          ExecOK := Exec(ExpandConstant('{cmd}'), CmdArgs, '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
        end;
        if not ExecOK then begin
          // Launch failed — fall back to PowerShell on this attempt.
          UseTar := False;
        end else if ResultCode = 0 then begin
          WizardForm.ProgressGauge.Position := TotalFiles;
          // v1.1.7.10 "Finalizing installation..." label so the user
          // doesn't think the installer hung during Defender scan.
          WizardForm.StatusLabel.Caption :=
            'Finalizing installation — this can take 1-3 minutes while ' +
            'antivirus scans the new files. The installer is not stuck.';
          WizardForm.Update;
          ExtractOK := True;
          Break;
        end else begin
          ErrorMsg := 'tar.exe exit code ' + IntToStr(ResultCode);
          // r19: append tar's own stderr when we captured it.
          if FileExists(TarErrPath) then begin
            if LoadStringFromFile(TarErrPath, TarErrText) then begin
              if Trim(String(TarErrText)) <> '' then
                ErrorMsg := ErrorMsg + ' — ' + Trim(String(TarErrText));
            end;
          end;
        end;
      end;

      if not UseTar then begin
        if not Exec('powershell.exe', PsCmd, '', SW_HIDE, ewNoWait, ResultCode) then begin
          ErrorMsg := 'Could not launch PowerShell. Check AppLocker / WDAC policy.';
          Break;
        end;

        // Poll until the sentinel appears (PowerShell fallback path only).
        while not FileExists(DoneFlag) do begin
          FileCount := CountFilesRecursive(ExtractDir);
          if FileCount > TotalFiles then FileCount := TotalFiles;
          WizardForm.ProgressGauge.Position := FileCount;
          StatusText := 'Extracting payload (' + IntToStr(FileCount) + ' / ' +
                        IntToStr(TotalFiles) + ' files, attempt ' +
                        IntToStr(Attempt) + ')...';
          WizardForm.StatusLabel.Caption := StatusText;
          WizardForm.Update;
          Sleep(500);
        end;
        // r62: set the CAPTION as well as the gauge. Only the gauge was
        // pushed to TotalFiles, so the label kept whatever the final
        // 500 ms poll read -- one tick short, and on an upgrade the count
        // can never reach TotalFiles at all. Never leave a stale numerator
        // on screen.
        WizardForm.ProgressGauge.Position := TotalFiles;
        WizardForm.StatusLabel.Caption := 'Extraction complete (' +
          IntToStr(TotalFiles) + ' files).';
        WizardForm.Update;

        // Sentinel-check on the PowerShell fallback path only. The tar
        // path already Break'd out above on success or set ErrorMsg on
        // failure — it has no DoneFlag file.
        LoadStringFromFile(DoneFlag, ResultStr);
        DeleteFile(DoneFlag);
        if Trim(String(ResultStr)) = 'OK' then begin
          ExtractOK := True;
          // v1.1.7.10 "Finalizing installation..." label so the user
          // doesn't think the installer hung during Defender scan.
          WizardForm.StatusLabel.Caption :=
            'Finalizing installation — this can take 1-3 minutes while ' +
            'antivirus scans the new files. The installer is not stuck.';
          WizardForm.Update;
          Break;
        end;
        ErrorMsg := Trim(String(ResultStr));
      end;

      if Attempt < 3 then begin
        WizardForm.StatusLabel.Caption :=
          'Extract attempt ' + IntToStr(Attempt) +
          ' failed — retrying in 3 seconds...';
        WizardForm.Update;
        Sleep(3000);
      end;
    end;

    if not ExtractOK then begin
      // r19: always state the free space alongside the error - "no space"
      // is by far the most common cause and the least obvious one.
      SpaceNote := '';
      if FreeMBOn(ExtractDir) >= 0 then
        SpaceNote := #13#10#13#10 + 'Free space on ' + DriveRoot(ExtractDir) +
          IntToStr(FreeMBOn(ExtractDir)) + ' MB (about ' +
          IntToStr({#PAYLOAD_TREE_MB}) + ' MB is needed to unpack).';
      RaiseException('Payload extraction failed after 3 attempts: ' + ErrorMsg + SpaceNote
                     + Chr(13) + Chr(10)
                     + 'This is usually caused by antivirus software locking '
                     + 'files during install. Add ' + ExtractDir
                     + ' to Windows Defender exclusions and re-run the installer, '
                     + 'or use the offline edition from the Touchless website.');
    end;

    // POST-EXTRACT VERIFICATION — three layered checks.
    //
    //   1) FileExists: catches "extract produced nothing at target"
    //      (corrupt / empty zip).
    //   2) SIZE check: a v1.1.5 Touchless.exe is 44,467,736 bytes; a
    //      v1.1.6 is 44,883,936. Any v1.1.x is in the 40-50 MB range.
    //      If size is outside [40 MB, 60 MB] something is wrong.
    //   3) Writability probe: rename Touchless.exe to a temp name and
    //      back. If AV / another process is holding the handle, the
    //      rename fails — catches the "fresh file but still locked"
    //      race that would block the user's first launch.
    WizardForm.StatusLabel.Caption := 'Verifying installation files...';
    WizardForm.Update;
    if not FileExists(TouchlessExePath) then
      RaiseException(ExpandConstant('{#MyAppExeName}')
                     + ' was not extracted to ' + ExtractDir + '.'
                     + Chr(13) + Chr(10)
                     + 'The payload zip may be corrupt. Re-download the installer.');

    if FindFirst(TouchlessExePath, ExeFindRec) then begin
      // Touchless.exe is ~44 MB, well under 2 GB. SizeLow alone is
      // safe (would overflow only if the binary ever exceeded 2 GB).
      TouchlessExeSize := ExeFindRec.SizeLow;
      FindClose(ExeFindRec);
      if (TouchlessExeSize < 40000000) or (TouchlessExeSize > 60000000) then
        RaiseException(ExpandConstant('{#MyAppExeName}')
                       + ' has an unexpected size (' + IntToStr(TouchlessExeSize)
                       + ' bytes) after install — extraction likely incomplete.'
                       + Chr(13) + Chr(10)
                       + 'Re-download the installer and try again.');
    end;

    // Writability probe: rename Touchless.exe to a temp name and back.
    // If anyone is holding the file open, RenameFile fails.
    WizardForm.StatusLabel.Caption :=
      'Waiting for antivirus to finish scanning Touchless.exe...';
    WizardForm.Update;
    RenameTestPath := TouchlessExePath + '.locktest';
    if FileExists(RenameTestPath) then DeleteFile(RenameTestPath);
    if RenameFile(TouchlessExePath, RenameTestPath) then
      RenameFile(RenameTestPath, TouchlessExePath)
    else
      RaiseException(ExpandConstant('{#MyAppExeName}')
                     + ' is locked by another process and may not launch '
                     + 'correctly.' + Chr(13) + Chr(10)
                     + 'Wait a moment for any antivirus scan to finish, then '
                     + 'try launching Touchless. If the app fails to start, '
                     + 'reboot and launch again.');
    // v1.1.7.10: final message before returning to Inno's own [Icons]
    // / [Registry] / [Run] stages, which are usually near-instant but
    // benefit from an honest "almost done" label instead of leaving
    // the previous "waiting for antivirus" text hanging.
    WizardForm.StatusLabel.Caption := 'Almost done, setting up shortcuts...';
    WizardForm.Update;
  end;
end;
#endif
