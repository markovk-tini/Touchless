@echo off
setlocal enabledelayedexpansion

REM Sign a single PE file via Azure Artifact Signing using the `sign` CLI.
REM
REM Usage: sign-file.bat <path-to-file> [description]
REM
REM Requirements (one-time setup, see project_signing_setup memory):
REM   - `sign` CLI installed: dotnet tool install --global sign --prerelease
REM   - Active Azure login: az login (token expires hourly)
REM   - Cert profile `touchless-prod` Active in `touchless-signing` account
REM
REM Implementation note: every IF block inside this file uses
REM `!errorlevel! neq 0` (delayed expansion) and avoids unescaped
REM parens inside echo args. cmd parses parens in echo args as block
REM delimiters, which silently breaks IF blocks; that bit us during
REM the first signing wire-up. Keep this file paren-clean.

if "%~1"=="" (
  echo [sign-file] Usage: sign-file.bat ^<path-to-file^> [description]
  exit /b 2
)

set "TARGET=%~1"
set "DESC=%~2"
if "%DESC%"=="" set "DESC=Touchless"

if not exist "%TARGET%" (
  echo [sign-file] File not found: %TARGET%
  exit /b 2
)

REM Verify `sign` CLI is reachable. If not, the dotnet tools dir may not
REM be on PATH for this shell. Explicitly add the standard global tools
REM path before failing so the build doesn't break for a missing PATH
REM entry alone.
where sign >nul 2>&1
if !errorlevel! neq 0 (
  set "PATH=!PATH!;!USERPROFILE!\.dotnet\tools"
  where sign >nul 2>&1
  if !errorlevel! neq 0 (
    echo [sign-file] sign CLI not found. Install with:
    echo            dotnet tool install --global sign --prerelease
    exit /b 3
  )
)

echo [sign-file] Signing: %TARGET%

sign code artifact-signing ^
  --artifact-signing-account "touchless-signing" ^
  --artifact-signing-certificate-profile "touchless-prod" ^
  --artifact-signing-endpoint "https://eus.codesigning.azure.net/" ^
  --azure-credential-type azure-cli ^
  --description "%DESC%" ^
  --description-url "https://touchless-control.com" ^
  --verbosity Information ^
  "%TARGET%"

set "SIGN_RC=!errorlevel!"

if !SIGN_RC! neq 0 (
  echo [sign-file] FAILED: %TARGET%
  echo            Common causes:
  echo              - Azure token expired. Run az login and retry.
  echo              - Network blocked from reaching eus.codesigning.azure.net
  echo              - Cert profile name mismatch. Must be touchless-prod.
  exit /b 1
)

echo [sign-file] Signed: %TARGET%
exit /b 0

REM Author: Konstantin Markov
