@echo off
setlocal enabledelayedexpansion

REM === Touchless Windows release build ====================================
REM
REM Default mode: STUB INSTALLER. Produces a tiny (~5-15 MB) Setup.exe
REM that downloads Touchless_Payload_v<version>.zip from R2 at install
REM time. Both artifacts must be uploaded to R2 -- see the rclone hints
REM at the end of this script.
REM
REM Set MONOLITHIC=1 in env to build the legacy ~2.4 GB embedded
REM installer (offline edition / air-gapped fallback). The payload zip
REM is still produced so the auto-update path stays unchanged.
REM
REM Set SKIP_SIGNING=1 to bypass dotnet sign (dev builds).
REM
REM v1.1.9.2 (r18) build gates + resumability:
REM   [0/6]   pyflakes / compileall preflight   (builder\windows\pyflakes_gate.py)
REM   [0.5/6] ISCC dry-compile of hgr_app.iss   (Pascal errors in ~2 s, not after sign)
REM   [2.5/6] SMOKE GATE: launch the UNSIGNED dist\Touchless\Touchless.exe,
REM           auto-START, verify engine_started + clean exit
REM           (builder\windows\smoke_gate.ps1). SMOKE_SECONDS (default 45),
REM           SMOKE_REQUIRE_CAMERA=1 makes camera_opened=false fatal.
REM   --resume        skip clean + PyInstaller when dist\Touchless\Touchless.exe
REM                   is newer than every file under src\ (+ spec + run_app.py);
REM                   the smoke gate is skipped too if its stamp matches the exe.
REM   --stage sign    start at [3/6]  (freeze already done + smoke-passed)
REM   --stage pack    start at [4/6]  (freeze + sign already done)
REM   --stage upload  only verify release\ artifacts and print the rclone lines
REM ========================================================================

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\..\.."
set "ROOT=%CD%"

set "BUILD_STAGE=all"
set "BUILD_RESUME="
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--stage" (
  set "BUILD_STAGE=%~2"
  shift
  shift
  goto parse_args
)
if /i "%~1"=="--resume" (
  set "BUILD_RESUME=1"
  shift
  goto parse_args
)
echo [ERROR] Unknown argument: %~1  ^(expected --resume ^| --stage sign^|pack^|upload^)
popd
exit /b 1
:args_done
if /i not "%BUILD_STAGE%"=="all" if /i not "%BUILD_STAGE%"=="sign" if /i not "%BUILD_STAGE%"=="pack" if /i not "%BUILD_STAGE%"=="upload" (
  echo [ERROR] Unknown --stage "%BUILD_STAGE%" ^(expected sign ^| pack ^| upload^)
  popd
  exit /b 1
)
if not "%SMOKE_SECONDS%"=="" (set "SMOKE_RUN_SECONDS=%SMOKE_SECONDS%") else (set "SMOKE_RUN_SECONDS=45")
set "SMOKE_DIR=%ROOT%\build\smoke"
set "SMOKE_MARKER=%SMOKE_DIR%\smoke_marker.json"
set "SMOKE_STAMP=%SMOKE_DIR%\smoke_ok.stamp"
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
set "SPEC=%ROOT%\builder\windows\hgr_app.spec"
set "ISS=%ROOT%\installers\windows\hgr_app.iss"
set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

if not exist "%PYTHON%" (
  echo [ERROR] Virtual environment not found at %PYTHON%
  popd
  exit /b 1
)

if not exist "%SPEC%" (
  echo [ERROR] Missing spec file: %SPEC%
  popd
  exit /b 1
)

if not exist "%ISS%" (
  echo [ERROR] Missing Inno Setup script: %ISS%
  popd
  exit /b 1
)

REM Sanity-check for the streaming whisper binary (the actual one
REM hgr_app.spec collects -- the older script looked for whisper-cli.exe
REM which the app doesn't ship). Accepts any of the cuda/vulkan/cpu
REM stream builds; the spec collects whichever ones are present.
set "WHISPER_OK="
if exist "%ROOT%\whisper.cpp\build_cuda\bin\Release\whisper-stream.exe" set "WHISPER_OK=1"
if exist "%ROOT%\whisper.cpp\build_cuda\bin\whisper-stream.exe" set "WHISPER_OK=1"
if exist "%ROOT%\whisper.cpp\build_stream\bin\Release\whisper-stream.exe" set "WHISPER_OK=1"
if exist "%ROOT%\whisper.cpp\build_stream\bin\whisper-stream.exe" set "WHISPER_OK=1"
if exist "%ROOT%\whisper.cpp\build_vulkan\bin\Release\whisper-stream.exe" set "WHISPER_OK=1"
if exist "%ROOT%\whisper.cpp\build_vulkan\bin\whisper-stream.exe" set "WHISPER_OK=1"
if exist "%ROOT%\whisper_bundle\build_cuda\bin\Release\whisper-stream.exe" set "WHISPER_OK=1"
if exist "%ROOT%\whisper_bundle\build_cuda\bin\whisper-stream.exe" set "WHISPER_OK=1"
if exist "%ROOT%\whisper_bundle\build_stream\bin\Release\whisper-stream.exe" set "WHISPER_OK=1"
if exist "%ROOT%\whisper_bundle\build_stream\bin\whisper-stream.exe" set "WHISPER_OK=1"
if exist "%ROOT%\whisper_bundle\build_vulkan\bin\Release\whisper-stream.exe" set "WHISPER_OK=1"
if exist "%ROOT%\whisper_bundle\build_vulkan\bin\whisper-stream.exe" set "WHISPER_OK=1"
if not defined WHISPER_OK (
  echo [ERROR] Could not find whisper-stream.exe under whisper.cpp\ or whisper_bundle\.
  echo         Build whisper.cpp ^(cuda/vulkan/stream variants^) before running this script.
  popd
  exit /b 1
)

set "WHISPER_MODEL_OK="
if exist "%ROOT%\whisper.cpp\models\ggml-medium.en.bin" set "WHISPER_MODEL_OK=1"
if exist "%ROOT%\whisper_bundle\models\ggml-medium.en.bin" set "WHISPER_MODEL_OK=1"
if not defined WHISPER_MODEL_OK (
  echo [ERROR] Missing model: ggml-medium.en.bin
  echo         Checked: whisper.cpp\models and whisper_bundle\models
  popd
  exit /b 1
)

if not exist "%ROOT%\whisper.cpp\models\ggml-silero-v5.1.2.bin" if not exist "%ROOT%\whisper_bundle\models\ggml-silero-v5.1.2.bin" (
  echo [WARN] Optional VAD model not found: ggml-silero-v5.1.2.bin
)

REM Read app version (single source of truth) -- used for payload zip
REM name + URL. Plain findstr + for /f instead of a Python one-liner
REM because cmd's for /f parses parentheses inside the quoted command,
REM which collides with python's m.group(1) and similar.
set "APP_VERSION="
for /f "tokens=2 delims==" %%V in ('findstr /b /c:"__version__" "%ROOT%\src\hgr\__init__.py"') do (
  set "APP_VERSION=%%V"
)
REM Strip surrounding spaces and the outer double quotes around the value.
set "APP_VERSION=%APP_VERSION: =%"
set "APP_VERSION=%APP_VERSION:"=%"
if "%APP_VERSION%"=="" (
  echo [ERROR] Could not read __version__ from src\hgr\__init__.py
  popd
  exit /b 1
)
set "PAYLOAD_FILE=Touchless_Payload_v%APP_VERSION%.zip"
set "PAYLOAD_URL_BASE=https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/windows/v%APP_VERSION%"
set "PAYLOAD_URL=%PAYLOAD_URL_BASE%/%PAYLOAD_FILE%"

REM STORE=1 forces a Microsoft Store-compliant build. Store policy
REM 10.2.9.3 forbids "downloader" installers (the default stub
REM downloads the payload from R2 at install time), so a Store
REM submission MUST be the MONOLITHIC standalone/offline installer.
REM Policy 10.2.9.2 additionally requires silent install — the
REM Partner Center silent args are printed at the end of this build.
REM Setting STORE=1 implies MONOLITHIC=1.
if "%STORE%"=="1" set "MONOLITHIC=1"

if "%MONOLITHIC%"=="1" (
  set "BUILD_MODE=monolithic"
) else (
  set "BUILD_MODE=stub"
)

REM Build-channel marker baked into the bundle (read at runtime by
REM hgr.utils.runtime_paths.build_channel). STORE builds set 'store'
REM so the in-app GitHub auto-updater stays OFF and the Microsoft
REM Store owns updates; every other build is 'website' so the GitHub
REM auto-updater is the update path.
if "%STORE%"=="1" (
  set "TOUCHLESS_BUILD_CHANNEL=store"
) else (
  set "TOUCHLESS_BUILD_CHANNEL=website"
)
echo [info] Build channel:  %TOUCHLESS_BUILD_CHANNEL%

echo [info] Build mode:    %BUILD_MODE%
echo [info] App version:   %APP_VERSION%
echo [info] Payload file:  %PAYLOAD_FILE%
echo [info] Payload URL:   %PAYLOAD_URL%
echo [info] Stage:         %BUILD_STAGE%
if defined BUILD_RESUME echo [info] Resume:        yes ^(skip freeze when dist is fresh^)
echo.

if /i "%BUILD_STAGE%"=="sign" goto stage_sign
if /i "%BUILD_STAGE%"=="pack" goto stage_pack
if /i "%BUILD_STAGE%"=="upload" goto stage_upload

:stage_freeze
REM v1.1.9.2 (r18): static preflight. A SyntaxError or an undefined name
REM (the r16 `NameError: name 'os' is not defined` at first launch; the
REM QRectF NameError on a settings page) is caught here in ~10 s instead
REM of on a family member's PC. Allowlist for known string-annotation
REM false positives: builder\windows\pyflakes_allow.txt.
echo [0/6] pyflakes undefined-name preflight...
"%PYTHON%" "%ROOT%\builder\windows\pyflakes_gate.py"
if errorlevel 1 (
  echo [ERROR] Static preflight failed. Fix the undefined name / syntax error above.
  popd
  exit /b 1
)

REM v1.1.9.2 (r18): ISCC dry-compile with dummy payload defines so a
REM Pascal-script error (r12: GetSetupLogFileName does not exist in Inno)
REM surfaces in ~2 s, not after the 30-min sign step. Always compiles the
REM STUB form (that is where the [Code] section lives, and it embeds no
REM dist files so it works before PyInstaller runs). Output goes to a
REM temp dir and is deleted; release\ is untouched.
if not exist "%ISCC%" (
  REM !ISCC! not %ISCC%: the path contains "(x86)" and a plain expansion
  REM inside this parenthesised block ends the block at parse time.
  echo [ERROR] Inno Setup compiler not found at:
  echo         !ISCC!
  echo         Install Inno Setup 6 or update the ISCC path in build_windows.bat.
  popd
  exit /b 1
)
echo [0.5/6] ISCC dry-compile of hgr_app.iss ^(stub form, dummy payload defines^)...
set "ISS_DRY_DIR=%TEMP%\touchless_iss_dry"
if exist "%ISS_DRY_DIR%" rmdir /s /q "%ISS_DRY_DIR%"
"%ISCC%" /Q ^
  "/O%ISS_DRY_DIR%" ^
  "/DPAYLOAD_URL=https://example.invalid/dry-run/%PAYLOAD_FILE%" ^
  "/DPAYLOAD_FILE=%PAYLOAD_FILE%" ^
  "/DPAYLOAD_SHA256=0000000000000000000000000000000000000000000000000000000000000000" ^
  "/DPAYLOAD_FILE_COUNT=1" ^
  "/DPAYLOAD_ZIP_MB=1" ^
  "/DPAYLOAD_TREE_MB=1" ^
  "%ISS%"
if errorlevel 1 (
  echo [ERROR] hgr_app.iss does not compile. Fix the Inno / Pascal error above before building.
  if exist "%ISS_DRY_DIR%" rmdir /s /q "%ISS_DRY_DIR%"
  popd
  exit /b 1
)
if exist "%ISS_DRY_DIR%" rmdir /s /q "%ISS_DRY_DIR%"

REM --resume: skip clean + PyInstaller when the frozen exe is newer than
REM every bundle input (src\ minus __pycache__, the spec, run_app.py).
REM (Kept out of a parenthesized block: the PowerShell one-liner contains
REM quoted parentheses that cmd's block parser can trip over.)
if not defined BUILD_RESUME goto resume_check_done
powershell -NoProfile -ExecutionPolicy Bypass -Command "$exe = Get-Item -LiteralPath '%ROOT%\dist\Touchless\Touchless.exe' -ErrorAction SilentlyContinue; if (-not $exe) { Write-Host '[resume] dist\Touchless\Touchless.exe missing - full freeze needed'; exit 2 }; $inputs = @(Get-ChildItem -LiteralPath '%ROOT%\src' -Recurse -File | Where-Object { $_.FullName -notmatch '\\__pycache__\\' -and $_.Extension -ne '.pyc' }) + @(Get-Item -LiteralPath '%SPEC%'), (Get-Item -LiteralPath '%ROOT%\run_app.py'); $newest = $inputs | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1; if ($newest.LastWriteTimeUtc -gt $exe.LastWriteTimeUtc) { Write-Host ('[resume] ' + $newest.FullName + ' (' + $newest.LastWriteTime + ') is newer than Touchless.exe (' + $exe.LastWriteTime + ') - full freeze needed'); exit 1 }; Write-Host ('[resume] Touchless.exe (' + $exe.LastWriteTime + ') is newer than every bundle input - skipping clean + PyInstaller'); exit 0"
if %errorlevel% equ 0 goto stage_smoke
echo [info] --resume: dist is stale or missing, running the full freeze.
:resume_check_done

echo [1/6] Cleaning previous build output...
if exist "%ROOT%\build" rmdir /s /q "%ROOT%\build"
if exist "%ROOT%\dist\Touchless" rmdir /s /q "%ROOT%\dist\Touchless"
if exist "%ROOT%\dist\HGR App" rmdir /s /q "%ROOT%\dist\HGR App"
if exist "%ROOT%\release\%PAYLOAD_FILE%" del /q "%ROOT%\release\%PAYLOAD_FILE%"
if exist "%ROOT%\release\Touchless_Installer.exe" del /q "%ROOT%\release\Touchless_Installer.exe"

echo [2/6] Building PyInstaller bundle...
"%PYTHON%" -m PyInstaller "%SPEC%" --noconfirm --clean
if errorlevel 1 (
  echo [ERROR] PyInstaller build failed.
  popd
  exit /b 1
)

if not exist "%ROOT%\dist\Touchless\Touchless.exe" (
  echo [ERROR] Expected bundle missing: dist\Touchless\Touchless.exe
  popd
  exit /b 1
)

:stage_smoke
REM v1.1.9.2 (r18) SMOKE GATE. Runs on the UNSIGNED dist immediately after
REM PyInstaller so a broken build fails in ~60 s, not after the 30-min
REM sign step. Four builds (r14-r17) shipped to a family member without
REM the frozen exe ever being launched; each had a bug a 60-second launch
REM would have caught. Launches Touchless.exe with HGR_AUTOSTART_ENGINE=1
REM + HGR_SMOKE_EXIT_AFTER_S, verifies exit code 0, marker.engine_started,
REM no new faulthandler fatal exception, kills orphans, restores the
REM registry InstallLocation heal_install_location rewrites when the exe
REM runs from dist\. See smoke_gate.ps1 header for the full rule set.
REM NEVER skippable by env var: the only way past this gate is a build
REM that starts.
if not exist "%SMOKE_DIR%" mkdir "%SMOKE_DIR%"
echo [2.5/6] Smoke gate: launching UNSIGNED dist\Touchless\Touchless.exe for %SMOKE_RUN_SECONDS% s...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\builder\windows\smoke_gate.ps1" ^
  -Exe "%ROOT%\dist\Touchless\Touchless.exe" ^
  -RunSeconds %SMOKE_RUN_SECONDS% ^
  -MarkerPath "%SMOKE_MARKER%" ^
  -StampPath "%SMOKE_STAMP%"
if errorlevel 1 (
  echo [ERROR] SMOKE GATE FAILED - the frozen build does not start cleanly. NOT signing, NOT packing.
  echo         Marker: %SMOKE_MARKER%
  echo         Log:    %%LOCALAPPDATA%%\Touchless\logs\touchless_debug.log
  echo         Crash:  %%LOCALAPPDATA%%\Touchless\crash\faulthandler.log
  popd
  exit /b 1
)

:stage_sign
if not exist "%ROOT%\dist\Touchless\Touchless.exe" (
  echo [ERROR] --stage sign: dist\Touchless\Touchless.exe missing. Run the full build first.
  popd
  exit /b 1
)
if not exist "%SMOKE_STAMP%" (
  echo [ERROR] --stage sign: no smoke-gate stamp at %SMOKE_STAMP%. This dist never passed the smoke gate; run without --stage.
  popd
  exit /b 1
)

REM Sign the inner Touchless.exe BEFORE we zip it, so users get a
REM signed exe whether they install via the stub, the monolithic
REM installer, or the auto-update zip path. Skip with SKIP_SIGNING=1.
if not "%SKIP_SIGNING%"=="1" (
  echo [3/6] Signing Touchless.exe...
  call "%ROOT%\signing\sign-file.bat" "%ROOT%\dist\Touchless\Touchless.exe" "Touchless"
  if !errorlevel! neq 0 (
    echo [ERROR] Signing Touchless.exe failed. Set SKIP_SIGNING=1 to bypass for dev builds.
    popd
    exit /b 1
  )
) else (
  echo [3/6] Skipping signing of Touchless.exe ^(SKIP_SIGNING=1^)
)

REM v1.1.9.2 (r13): move Touchless_Debug.bat / .ps1 from _internal\ up to
REM the bundle root BEFORE the sign step so the .ps1 gets a valid
REM Authenticode signature in place (was being hoisted after signing in
REM r11.3, which left the .ps1 unsigned in the shipped payload). Norton
REM SONAR treats an unsigned .ps1 launched via `powershell -ExecutionPolicy
REM Bypass -File` as a canonical dropper fingerprint; a signed script
REM sidesteps the heuristic. Idempotent (missing source is a no-op).
if exist "%ROOT%\dist\Touchless\_internal\Touchless_Debug.bat" (
  move /Y "%ROOT%\dist\Touchless\_internal\Touchless_Debug.bat" "%ROOT%\dist\Touchless\Touchless_Debug.bat" >nul
)
if exist "%ROOT%\dist\Touchless\_internal\Touchless_Debug.ps1" (
  move /Y "%ROOT%\dist\Touchless\_internal\Touchless_Debug.ps1" "%ROOT%\dist\Touchless\Touchless_Debug.ps1" >nul
)

REM v1.1.7.7 (dad rig 2026-08-22): sign every internal .exe under
REM dist\Touchless\_internal too. Norton flagged whisper-server.exe as
REM Win64:Evo-gen [Trj] on 1.1.7.6 installs — a heuristic false positive
REM Norton's engine fires on unsigned OSS C++ binaries (whisper.cpp,
REM llama.cpp, ffmpeg builds). Signing them with our publisher cert
REM bypasses the heuristic.
REM
REM v1.1.9.2 (r13): widened to sign .exe / .dll / .pyd across the whole
REM dist tree (not just _internal\*.exe). The 1.1.7.7 fix only covered
REM the .exe half of the OSS-C++ surface. Dad's 1.1.9.2 install still
REM tripped Norton because the ggml-cuda.dll / whisper.dll / llama.dll
REM DLLs from the same OSS lineage shipped UNSIGNED. Cost: ~20-25 min
REM for the full walk (~500 files); idempotent (already-Valid signatures
REM are skipped so re-runs cost only the Get-AuthenticodeSignature scan).
if not "%SKIP_SIGNING%"=="1" (
  echo [3.5/6] Signing internal binaries ^(.exe / .dll / .pyd, ~20-25 min^)...
  call "%ROOT%\signing\sign-all-internal.bat" "%ROOT%\dist\Touchless"
  if !errorlevel! neq 0 (
    echo [ERROR] Signing internal binaries failed. Set SKIP_SIGNING=1 to bypass for dev builds.
    popd
    exit /b 1
  )
) else (
  echo [3.5/6] Skipping signing of internal binaries ^(SKIP_SIGNING=1^)
)

REM v1.1.9.2 (r13): opportunistically sign Touchless_Debug.ps1 too.
REM Non-fatal — the sign CLI's PowerShell-script support is newer than
REM its PE support and can fail on some environments. If it fails we
REM ship the .ps1 unsigned (same as r11.3); if it succeeds we get an
REM Authenticode block appended and Norton stops flagging the launcher.
if not "%SKIP_SIGNING%"=="1" (
  if exist "%ROOT%\dist\Touchless\Touchless_Debug.ps1" (
    echo [3.6/6] Attempting to sign Touchless_Debug.ps1 ^(non-fatal on failure^)...
    call "%ROOT%\signing\sign-file.bat" "%ROOT%\dist\Touchless\Touchless_Debug.ps1" "Touchless Debug Launcher"
    if !errorlevel! neq 0 (
      echo [WARN] .ps1 signing failed. Non-fatal — build continues.
    )
  )
)

REM v1.1.9.2 (r13): post-sign audit gate. Walks dist\Touchless and
REM fails the build if any .exe/.dll/.pyd is NOT Authenticode-Valid.
REM Turns "everything is signed" from an unverified assumption into
REM an enforced build invariant — the exact hole that let 1.1.7.7 ship
REM a partial-signing pass. Vendor-signed DLLs already return Valid so
REM they pass this gate under their own signer.
if not "%SKIP_SIGNING%"=="1" (
  echo [3.7/6] Auditing signature coverage of dist\Touchless...
  powershell -NoProfile -Command "$u = Get-ChildItem -Recurse -Path '%ROOT%\dist\Touchless' -Include *.exe,*.dll,*.pyd -File | Where-Object { (Get-AuthenticodeSignature $_.FullName).Status -ne 'Valid' }; if ($u) { Write-Host ('UNSIGNED BINARIES: ' + $u.Count) -ForegroundColor Red; $u | ForEach-Object { Write-Host ('  ' + $_.FullName) }; $u.FullName | Set-Content '%ROOT%\release\unsigned-audit.txt'; exit 1 } else { Write-Host '[audit] All .exe / .dll / .pyd under dist\Touchless are Authenticode-Valid.' -ForegroundColor Green; exit 0 }"
  if !errorlevel! neq 0 (
    echo [ERROR] Post-sign audit found unsigned binaries. See release\unsigned-audit.txt.
    popd
    exit /b 1
  )
)

:stage_pack
if not exist "%ROOT%\dist\Touchless\Touchless.exe" (
  echo [ERROR] --stage pack: dist\Touchless\Touchless.exe missing. Run the full build first.
  popd
  exit /b 1
)
if not exist "%SMOKE_STAMP%" (
  echo [ERROR] --stage pack: no smoke-gate stamp at %SMOKE_STAMP%. This dist never passed the smoke gate; run without --stage.
  popd
  exit /b 1
)
REM Pack the dist tree into the payload zip BEFORE running ISCC, so
REM the SHA256 we bake into the stub matches the bytes we'll upload.
if not exist "%ROOT%\release" mkdir "%ROOT%\release"
echo [4/6] Building payload zip ^(release\%PAYLOAD_FILE%^)...
REM Use PowerShell Compress-Archive -- built into Windows, deterministic
REM enough for our purposes, no extra build dependency.
REM
REM Retry loop: PyInstaller writes `_internal/base_library.zip` right
REM before this step runs, and on machines with aggressive antivirus
REM (Defender + 3rd-party AV) the file handle isn't released by the
REM scanner for a second or two. Compress-Archive then errors with
REM "process cannot access the file because it is being used by
REM another process" and the whole build dies. Three attempts with a
REM 5-second sleep between them clears every real-world AV race
REM we've hit; if it's still locked after 15 s, something else is
REM wrong and the build fails for real.
set "PAYLOAD_RETRIES=0"
:payload_zip_retry
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path '%ROOT%\dist\Touchless\*' -DestinationPath '%ROOT%\release\%PAYLOAD_FILE%' -CompressionLevel Optimal -Force"
if errorlevel 1 (
  set /a PAYLOAD_RETRIES+=1
  if !PAYLOAD_RETRIES! lss 3 (
    echo [WARN] Payload zip attempt !PAYLOAD_RETRIES! failed - antivirus lock on dist/_internal/* likely. Retrying in 5 s...
    powershell -NoProfile -Command "Start-Sleep -Seconds 5"
    del "%ROOT%\release\%PAYLOAD_FILE%" 2>nul
    goto payload_zip_retry
  )
  echo [ERROR] Payload zip build failed after 3 attempts.
  popd
  exit /b 1
)
if not exist "%ROOT%\release\%PAYLOAD_FILE%" (
  echo [ERROR] Payload zip not produced.
  popd
  exit /b 1
)
REM Compute SHA256 of the zip and stash for ISCC to bake into the
REM stub. Lowercase hex (Inno's DownloadPage.Add accepts either case
REM but lowercase is the convention).
for /f "delims=" %%H in ('powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath '%ROOT%\release\%PAYLOAD_FILE%').Hash.ToLower()"') do set "PAYLOAD_SHA256=%%H"
if "%PAYLOAD_SHA256%"=="" (
  echo [ERROR] Could not compute SHA256 for payload zip.
  popd
  exit /b 1
)
for %%S in ("%ROOT%\release\%PAYLOAD_FILE%") do set "PAYLOAD_SIZE=%%~zS"
echo [info] Payload size:   %PAYLOAD_SIZE% bytes
echo [info] Payload SHA256: %PAYLOAD_SHA256%

REM Count files in the dist tree so the stub installer can drive a
REM REAL extraction-progress bar instead of the frozen "Installing..."
REM page users currently sit on for several minutes while PowerShell
REM Expand-Archive runs silently. Passed to ISCC as PAYLOAD_FILE_COUNT.
for /f "delims=" %%C in ('powershell -NoProfile -Command "(Get-ChildItem -LiteralPath '%ROOT%\dist\Touchless' -Recurse -File).Count"') do set "PAYLOAD_FILE_COUNT=%%C"
if not defined PAYLOAD_FILE_COUNT set "PAYLOAD_FILE_COUNT=2000"
echo [info] Payload files:  %PAYLOAD_FILE_COUNT%

REM v1.1.9.2 (r19): measure BOTH sizes in MB for the installer's
REM free-space precheck. The stub needs the zip and the extracted tree
REM on disk at the same time (~5.3 GB combined for 1.1.9.2), and before
REM r19 nothing checked that -- a user short on space just got
REM "tar.exe exit code 1" with no explanation. Rounded UP.
for /f "delims=" %%Z in ('powershell -NoProfile -Command "[int][math]::Ceiling(%PAYLOAD_SIZE% / 1MB)"') do set "PAYLOAD_ZIP_MB=%%Z"
if not defined PAYLOAD_ZIP_MB set "PAYLOAD_ZIP_MB=1700"
for /f "delims=" %%T in ('powershell -NoProfile -Command "[int][math]::Ceiling(((Get-ChildItem -LiteralPath '%ROOT%\dist\Touchless' -Recurse -File | Measure-Object Length -Sum).Sum / 1MB))"') do set "PAYLOAD_TREE_MB=%%T"
if not defined PAYLOAD_TREE_MB set "PAYLOAD_TREE_MB=3500"
echo [info] Payload zip:    %PAYLOAD_ZIP_MB% MB   extracted tree: %PAYLOAD_TREE_MB% MB

if not exist "%ISCC%" (
  echo [ERROR] Inno Setup compiler not found at:
  echo         !ISCC!
  echo         Install Inno Setup 6 or update the ISCC path in build_windows.bat.
  popd
  exit /b 1
)

echo [5/6] Building installer ^(%BUILD_MODE%^)...
if "%BUILD_MODE%"=="stub" (
  "%ISCC%" /Q ^
    "/DPAYLOAD_URL=%PAYLOAD_URL%" ^
    "/DPAYLOAD_FILE=%PAYLOAD_FILE%" ^
    "/DPAYLOAD_SHA256=%PAYLOAD_SHA256%" ^
    "/DPAYLOAD_FILE_COUNT=%PAYLOAD_FILE_COUNT%" ^
    "/DPAYLOAD_ZIP_MB=%PAYLOAD_ZIP_MB%" ^
    "/DPAYLOAD_TREE_MB=%PAYLOAD_TREE_MB%" ^
    "%ISS%"
) else (
  "%ISCC%" /Q "/DMONOLITHIC=1" "%ISS%"
)
if errorlevel 1 (
  echo [ERROR] Installer build failed.
  popd
  exit /b 1
)

REM Sign the installer Inno Setup just produced. Same skip flag applies.
if not "%SKIP_SIGNING%"=="1" (
  echo [5.5/6] Signing installer...
  call "%ROOT%\signing\sign-file.bat" "%ROOT%\release\Touchless_Installer.exe" "Touchless Installer"
  if !errorlevel! neq 0 (
    echo [ERROR] Signing installer failed. Set SKIP_SIGNING=1 to bypass for dev builds.
    popd
    exit /b 1
  )
)

echo [6/6] Building app-only update zip ^(small download for incremental updates^)...
"%PYTHON%" "%ROOT%\builder\windows\build_app_update_zip.py"
if errorlevel 1 (
  echo [WARN] App-only zip build failed; full installer is still good.
)

:stage_upload
if /i "%BUILD_STAGE%"=="upload" (
  if not exist "%ROOT%\release\Touchless_Installer.exe" (
    echo [ERROR] --stage upload: release\Touchless_Installer.exe missing.
    popd
    exit /b 1
  )
  if "%BUILD_MODE%"=="stub" if not exist "%ROOT%\release\%PAYLOAD_FILE%" (
    echo [ERROR] --stage upload: release\%PAYLOAD_FILE% missing.
    popd
    exit /b 1
  )
  if not exist "%SMOKE_STAMP%" (
    echo [ERROR] --stage upload: no smoke-gate stamp at %SMOKE_STAMP%. Do not upload a dist that never passed the smoke gate.
    popd
    exit /b 1
  )
)
if not defined PAYLOAD_SIZE if exist "%ROOT%\release\%PAYLOAD_FILE%" for %%S in ("%ROOT%\release\%PAYLOAD_FILE%") do set "PAYLOAD_SIZE=%%~zS"
REM Print the installer size so the operator can sanity-check stub mode
REM (~5-15 MB) vs monolithic mode (~2.4 GB) at a glance.
for %%S in ("%ROOT%\release\Touchless_Installer.exe") do set "INSTALLER_SIZE=%%~zS"

echo.
echo ===============================================================
echo Build complete.  Mode: %BUILD_MODE%   Version: %APP_VERSION%
echo ===============================================================
echo Bundle:           %ROOT%\dist\Touchless
echo Installer:        %ROOT%\release\Touchless_Installer.exe   ^(%INSTALLER_SIZE% bytes^)
echo Payload zip:      %ROOT%\release\%PAYLOAD_FILE%   ^(%PAYLOAD_SIZE% bytes^)
echo App update zip:   %ROOT%\release\Touchless_App_Update_*.zip
echo.
if "%BUILD_MODE%"=="stub" (
  echo Stub mode: BOTH the installer AND the payload zip MUST be uploaded
  echo to R2, otherwise users will get a download error during install.
  echo.
  echo Upload commands ^(rclone^):
  echo   rclone copyto "release\Touchless_Installer.exe" r2:hgr-downloads/windows/v%APP_VERSION%/Touchless_Installer.exe --s3-upload-cutoff=100M --s3-chunk-size=100M
  echo   rclone copyto "release\%PAYLOAD_FILE%" r2:hgr-downloads/windows/v%APP_VERSION%/%PAYLOAD_FILE% --s3-upload-cutoff=100M --s3-chunk-size=100M
) else (
  echo Monolithic mode: upload the installer + the app-update zip to
  echo R2 / GitHub release. The payload zip is for the auto-update path
  echo only; users won't download it directly because everything is in
  echo the installer.
)
echo.

if "%STORE%"=="1" (
  REM Copy the monolithic installer to a Store-distinct name so it
  REM can never be confused with the stub when submitting to Partner
  REM Center. Both are produced as Touchless_Installer.exe by ISCC.
  copy /y "%ROOT%\release\Touchless_Installer.exe" "%ROOT%\release\Touchless_Store_Installer.exe" >nul
  echo ===============================================================
  echo MICROSOFT STORE SUBMISSION
  echo ===============================================================
  echo This is a MONOLITHIC standalone/offline installer — it carries
  echo the full app and downloads nothing at install time, satisfying
  echo Store policy 10.2.9.3 ^(no downloader installers^).
  echo.
  echo Submit this file to Partner Center:
  echo   %ROOT%\release\Touchless_Store_Installer.exe
  echo.
  echo Set these in the Partner Center package "Installer parameters"
  echo so the app installs silently ^(Store policy 10.2.9.2^):
  echo   Silent install:    /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /NOCANCEL
  echo   Silent uninstall:  /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
  echo.
  echo Notes:
  echo   - Do NOT submit the stub installer ^(that was the 10.2.9.3
  echo     rejection^). Always use STORE=1 for Store builds.
  echo   - The install is per-user under %%LOCALAPPDATA%%\Programs and
  echo     needs no elevation, so silent install completes without a
  echo     UAC prompt. The Defender-exclusion and Launch steps are
  echo     skipped under /VERYSILENT ^(skipifsilent^).
  echo.
)

popd
exit /b 0

REM Author: Konstantin Markov
