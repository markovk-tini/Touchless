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
REM ========================================================================

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\..\.."
set "ROOT=%CD%"
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
echo.

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
