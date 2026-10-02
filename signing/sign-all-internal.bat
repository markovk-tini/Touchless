@echo off
setlocal enabledelayedexpansion

REM Sign every unsigned .exe / .dll / .pyd under dist\Touchless\ so AV
REM engines (Norton, Defender, ESET) see a valid publisher signature
REM instead of running heuristic classifiers on unknown binaries.
REM
REM v1.1.9.2 (r13) — extended from .exe-only to walk .exe/.dll/.pyd and
REM to anchor at the entire %DIST% root (not just _internal). Root cause
REM this widening exists: 1.1.7.6 (Aug 2026) was quarantined on dad's rig
REM by Norton as Win64:Evo-gen [Trj] on whisper-server.exe. 1.1.7.7 signed
REM the .exe half. 1.1.9.2 shipped the same OSS-C++ DLLs (ggml-cuda.dll,
REM ggml-vulkan.dll, whisper.dll, llama.dll, mtmd.dll) UNSIGNED, so the
REM heuristic simply moved from the exe to the DLL its exe loads. This
REM build signs those too, which closes the DLL side of that door.
REM
REM Idempotent: already-Valid files are skipped, so re-runs cost only
REM the powershell Get-AuthenticodeSignature scan (~200-500 ms per file).
REM Vendor-signed DLLs (Python, PySide6/Qt, av.libs from FFmpeg builds
REM whose upstream signs them, ffi wheel etc.) already report Valid and
REM are never overwritten.
REM
REM Usage: sign-all-internal.bat [dist_root]
REM   dist_root defaults to <script_dir>\..\dist\Touchless

set "DIST=%~1"
if "%DIST%"=="" set "DIST=%~dp0..\dist\Touchless"

if not exist "%DIST%" (
  echo [sign-all-internal] Not found: %DIST%
  exit /b 2
)

set "LOG=%~dp0..\release\sign-all-log.txt"
if not exist "%~dp0..\release" mkdir "%~dp0..\release" >nul 2>&1
type nul > "%LOG%"

echo [sign-all-internal] Signing .exe / .dll / .pyd under %DIST%
echo [sign-all-internal] Detailed sign CLI output: %LOG%
set /a SIGNED=0
set /a SKIPPED=0
set /a FAILED=0

for %%E in (exe dll pyd) do (
  for /r "%DIST%" %%F in (*.%%E) do (
    for /f "usebackq delims=" %%S in (`powershell -NoProfile -Command "(Get-AuthenticodeSignature '%%F').Status"`) do (
      if "%%S"=="Valid" (
        set /a SKIPPED+=1
      ) else (
        call "%~dp0sign-file.bat" "%%F" "Touchless component" >>"%LOG%" 2>&1
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
)

echo.
echo [sign-all-internal] Done. Signed: !SIGNED!  Already-signed: !SKIPPED!  Failed: !FAILED!
if !FAILED! gtr 0 (
  echo [sign-all-internal] See %LOG% for per-file sign CLI output.
  exit /b 1
)
exit /b 0
