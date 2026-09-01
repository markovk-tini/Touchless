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
#define MyAppVersion "1.1.8.2"
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
#endif

[Setup]
AppId={{2C4EE680-53F5-4D83-92A8-ADF4D2D8794E}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
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
begin
  if CurPageID = wpReady then begin
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
begin
  if CurStep = ssInstall then begin
    ZipPath := ExpandConstant('{tmp}\{#PAYLOAD_FILE}');
    ExtractDir := ExpandConstant('{app}');
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
    // Pause so Windows fully releases the file handles + AV finishes
    // its post-mortem scan of the killed process. Bumped 750 -> 1500 ms
    // because the workflow audit flagged that slow systems with real-
    // time AV scanning can still hold the handle past 750 ms.
    Sleep(1500);

    // Reconfigure Inno's progress bar to track real extraction state.
    WizardForm.ProgressGauge.Min := 0;
    WizardForm.ProgressGauge.Max := TotalFiles;
    WizardForm.ProgressGauge.Position := 0;
    WizardForm.StatusLabel.Caption := 'Extracting payload (0 / ' + IntToStr(TotalFiles) + ' files)...';
    WizardForm.FilenameLabel.Caption := '';

    // Build the PowerShell command once. PATHS ARE APOSTROPHE-ESCAPED
    // so usernames / install dirs containing ' (O'Brien, Tom's Apps)
    // don't break the PS single-quoted string literal.
    PsCmd :=
      '-NoProfile -NonInteractive -ExecutionPolicy Bypass ' +
      '-Command "try { Expand-Archive -LiteralPath ''' + PsQuoteEscape(ZipPath) + ''' ' +
      '-DestinationPath ''' + PsQuoteEscape(ExtractDir) + ''' -Force; ' +
      '''OK'' | Out-File -LiteralPath ''' + PsQuoteEscape(DoneFlag) + ''' -Encoding ascii } ' +
      'catch { $_.Exception.Message | Out-File -LiteralPath ''' + PsQuoteEscape(DoneFlag) + ''' -Encoding ascii }"';

    // EXTRACT RETRY LOOP (3 attempts, 3s pause between). Antivirus
    // scanners + slow disks can race the file replace — Expand-Archive
    // may report overall success while having silently failed on a
    // specific locked file. Retry catches transient locks. Mirrors
    // the build-side payload retry loop in build_windows.bat.
    ExtractOK := False;
    ErrorMsg := '';
    for Attempt := 1 to 3 do begin
      // Wipe any stale sentinel from a previous attempt.
      if FileExists(DoneFlag) then DeleteFile(DoneFlag);

      WizardForm.StatusLabel.Caption := 'Extracting payload (attempt ' +
        IntToStr(Attempt) + ' of 3)...';
      WizardForm.Update;

      if not Exec('powershell.exe', PsCmd, '', SW_HIDE, ewNoWait, ResultCode) then begin
        ErrorMsg := 'Could not launch PowerShell. Check AppLocker / WDAC policy.';
        Break;
      end;

      // Poll until the sentinel appears.
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
      WizardForm.ProgressGauge.Position := TotalFiles;
      // v1.1.7.10: replace the frozen "6047 / 6047 files, attempt 1..."
      // label with an honest "still working" message. The rest of the
      // ssInstall step (verification + rename probe + Inno's own [Icons]
      // / [Registry] stages) can take 2-3 minutes on slower computers,
      // mostly because Windows Defender / Norton is scanning each of the
      // just-extracted files before letting us open them for the probe.
      // Users seeing 6047/6047 with no motion have historically thought
      // the installer hung — this message tells them what's actually
      // happening. Bar stays at 100% because progress is meaningful up
      // to this point; the sub-messages below update as we advance.
      WizardForm.StatusLabel.Caption :=
        'Finalizing installation — this can take 1-3 minutes while ' +
        'antivirus scans the new files. The installer is not stuck.';
      WizardForm.Update;

      LoadStringFromFile(DoneFlag, ResultStr);
      DeleteFile(DoneFlag);
      if Trim(String(ResultStr)) = 'OK' then begin
        ExtractOK := True;
        Break;
      end;
      ErrorMsg := Trim(String(ResultStr));
      if Attempt < 3 then begin
        WizardForm.StatusLabel.Caption :=
          'Extract attempt ' + IntToStr(Attempt) +
          ' failed — retrying in 3 seconds...';
        WizardForm.Update;
        Sleep(3000);
      end;
    end;

    if not ExtractOK then
      RaiseException('Payload extraction failed after 3 attempts: ' + ErrorMsg
                     + Chr(13) + Chr(10)
                     + 'This is usually caused by antivirus software locking '
                     + 'files during install. Add ' + ExtractDir
                     + ' to Windows Defender exclusions and re-run the installer, '
                     + 'or use the offline edition from the Touchless website.');

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
