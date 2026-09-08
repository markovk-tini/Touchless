@echo off
setlocal enabledelayedexpansion

REM Sign every unsigned .exe under dist\Touchless\_internal\ so AV
REM engines (Norton, Defender, ESET) see a valid publisher signature
REM instead of running heuristic classifiers on unknown binaries.
REM
REM Root cause this step exists: 1.1.7.6 (Aug 2026) was quarantined
REM on dad's rig by Norton as Win64:Evo-gen [Trj] on whisper-server.exe.
REM Norton's Evo-gen is a well-known false-positive detector for OSS
REM C++ binaries (whisper.cpp, llama.cpp, ffmpeg builds). A signed
REM binary from a trusted publisher sidesteps the heuristic entirely.
REM
REM Usage: sign-all-internal.bat <dist_root>
REM   dist_root = the folder holding Touchless.exe (default:
REM               <script_dir>\..\dist\Touchless)

set "DIST=%~1"
if "%DIST%"=="" set "DIST=%~dp0..\dist\Touchless"

if not exist "%DIST%\_internal" (
  echo [sign-all-internal] Not found: %DIST%\_internal
  exit /b 2
)

echo [sign-all-internal] Signing internal .exe files under %DIST%\_internal
set /a SIGNED=0
set /a SKIPPED=0
set /a FAILED=0

for /r "%DIST%\_internal" %%F in (*.exe) do (
  REM Check if already signed. powershell one-shot is slow but only
  REM 50 files worst-case, and this is a build-time step. Skipping
  REM already-signed files means re-running this step is idempotent.
  for /f "usebackq delims=" %%S in (`powershell -NoProfile -Command "(Get-AuthenticodeSignature '%%F').Status"`) do (
    if "%%S"=="Valid" (
      set /a SKIPPED+=1
    ) else (
      call "%~dp0sign-file.bat" "%%F" "Touchless component" >nul 2>&1
      if !errorlevel! equ 0 (
        set /a SIGNED+=1
        echo   [signed] %%F
      ) else (
        set /a FAILED+=1
        echo   [FAIL]   %%F
      )
    )
  )
)

echo [sign-all-internal] Done. Signed: !SIGNED!  Already-signed skipped: !SKIPPED!  Failed: !FAILED!
if !FAILED! gtr 0 exit /b 1
exit /b 0
