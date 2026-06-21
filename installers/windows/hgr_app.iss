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
#define MyAppVersion "1.1.4"
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
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
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
begin
  if CurPageID = wpReady then begin
    DownloadPage.Clear;
    DownloadPage.Add('{#PAYLOAD_URL}', '{#PAYLOAD_FILE}', '{#PAYLOAD_SHA256}');
    DownloadPage.Show;
    try
      try
        DownloadPage.Download;
        Result := True;
      except
        if DownloadPage.AbortedByUser then begin
          Log('Aborted by user.');
          Result := False;
        end else begin
          SuppressibleMsgBox(
            'Download failed: ' + GetExceptionMessage,
            mbCriticalError,
            MB_OK,
            IDOK);
          Result := False;
        end;
      end;
    finally
      DownloadPage.Hide;
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

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  FileCount: Integer;
  TotalFiles: Integer;
  ZipPath, ExtractDir, DoneFlag, PsCmd, StatusText: String;
  ResultStr: AnsiString;
begin
  if CurStep = ssInstall then begin
    ZipPath := ExpandConstant('{tmp}\{#PAYLOAD_FILE}');
    ExtractDir := ExpandConstant('{app}');
    DoneFlag := ExpandConstant('{tmp}\extract_done.flag');
    TotalFiles := {#PAYLOAD_FILE_COUNT};
    if TotalFiles < 1 then TotalFiles := 1;

    if not ForceDirectories(ExtractDir) then
      RaiseException('Could not create install directory: ' + ExtractDir);

    // Wipe any stale sentinel from a previous failed install so the
    // poll loop below doesn't immediately think extraction finished.
    if FileExists(DoneFlag) then DeleteFile(DoneFlag);

    // Reconfigure Inno's progress bar to track real extraction state.
    // ProgressGauge.Style stays the default (smooth, determinate);
    // we drive Position from a recursive file count of {app} on a
    // 500 ms poll. Without this the bar sits frozen at 0% because
    // STUB mode has an empty [Files] section — Inno has nothing of
    // its own to count.
    WizardForm.ProgressGauge.Min := 0;
    WizardForm.ProgressGauge.Max := TotalFiles;
    WizardForm.ProgressGauge.Position := 0;
    WizardForm.StatusLabel.Caption := 'Extracting payload (0 / ' + IntToStr(TotalFiles) + ' files)...';
    WizardForm.FilenameLabel.Caption := '';

    // Run Expand-Archive asynchronously, and have PowerShell write a
    // sentinel file when it's done (so the Inno side can poll for
    // completion — Exec(ewNoWait) doesn't return a process handle
    // we could WaitForSingleObject on). On success the sentinel
    // contains the literal "OK"; on failure it contains the
    // exception message so we can surface it to the user.
    PsCmd :=
      '-NoProfile -NonInteractive -ExecutionPolicy Bypass ' +
      '-Command "try { Expand-Archive -LiteralPath ''' + ZipPath + ''' ' +
      '-DestinationPath ''' + ExtractDir + ''' -Force; ' +
      '''OK'' | Out-File -LiteralPath ''' + DoneFlag + ''' -Encoding ascii } ' +
      'catch { $_.Exception.Message | Out-File -LiteralPath ''' + DoneFlag + ''' -Encoding ascii }"';

    if not Exec('powershell.exe', PsCmd, '', SW_HIDE, ewNoWait, ResultCode) then
      RaiseException('Could not launch PowerShell to extract payload.');

    // Poll until the sentinel appears. Cap the displayed count at
    // TotalFiles so a slightly-off PAYLOAD_FILE_COUNT define doesn't
    // overshoot the bar (or stall it at 99% if undershoot).
    while not FileExists(DoneFlag) do begin
      FileCount := CountFilesRecursive(ExtractDir);
      if FileCount > TotalFiles then FileCount := TotalFiles;
      WizardForm.ProgressGauge.Position := FileCount;
      StatusText := 'Extracting payload (' + IntToStr(FileCount) + ' / ' +
                    IntToStr(TotalFiles) + ' files)...';
      WizardForm.StatusLabel.Caption := StatusText;
      WizardForm.Update;
      Sleep(500);
    end;
    // One final update so the bar hits 100% even if the last poll
    // didn't catch the last few files.
    WizardForm.ProgressGauge.Position := TotalFiles;
    WizardForm.Update;

    // Read PowerShell outcome from the sentinel and surface failures.
    LoadStringFromFile(DoneFlag, ResultStr);
    DeleteFile(DoneFlag);
    if Trim(String(ResultStr)) <> 'OK' then
      // CRLF spelled out with Chr() instead of #13#10 because the
      // Inno preprocessor treats any line whose first non-whitespace
      // character is '#' as a directive, and breaking the string
      // before #13#10 made it fail with 'Unknown preprocessor
      // directive' at compile time.
      RaiseException('Payload extraction failed: ' + Trim(String(ResultStr))
                     + Chr(13) + Chr(10)
                     + 'Try running the installer again, or use the '
                     + 'offline edition from the Touchless website if the issue persists.');
  end;
end;
#endif
