<#
.SYNOPSIS
  Build step [2.5/6]: launch the freshly frozen (UNSIGNED) Touchless.exe,
  auto-press START, run for -RunSeconds, and FAIL the build unless the
  app started its engine and exited cleanly.

  Consumes the env hooks in hgr.app.main._install_smoke_hooks:
    HGR_AUTOSTART_ENGINE=1, HGR_SMOKE_EXIT_AFTER_S=N, HGR_SMOKE_MARKER_PATH.

.DESCRIPTION
  Pass/fail rules (any violation -> exit 1 -> build_windows.bat aborts):
    - a Touchless.exe is NOT already running (single-instance mutex would
      make the smoke launch bail with exit 0 and no marker; we refuse to
      kill the operator's live session).
    - process exits by itself within RunSeconds + 45 s (else killed, FAIL)
    - exit code == 0
    - marker JSON exists and parses
    - marker.engine_started == true
    - marker.errors is empty (Traceback / fatal exception tail lines)
    - no NEW "Windows fatal exception" line in
      %LOCALAPPDATA%\Touchless\crash\faulthandler.log since launch,
      ignoring code 0x8001010d (benign handled COM exception)
    - camera policy: camera_opened == false is a WARN (build machines
      without a camera) unless -RequireCamera / SMOKE_REQUIRE_CAMERA=1;
      camera_opened == true with frames_seen == 0 is always a FAIL
      (the r16 "silent hang" shape).

  Side-effect hygiene:
    - Running from dist\ triggers heal_install_location(), which rewrites
      HKCU\...\Uninstall\{2C4EE680-...}_is1 (InstallLocation, Inno Setup:
      App Path, DisplayIcon) to the dist folder, and autostart.heal()
      may repoint HKCU\...\Run\Touchless. Both are snapshotted before
      launch and restored afterwards (finally block).
    - Orphan Touchless.exe / ffmpeg.exe whose image lives under the dist
      folder are killed after the run.

.PARAMETER Exe          Path to dist\Touchless\Touchless.exe (or python.exe
                        for a source-mode dry run, with -Arguments run_app.py).
.PARAMETER RunSeconds   Seconds the app runs before self-exiting (default 45).
.PARAMETER MarkerPath   Where the app writes the JSON marker.
.PARAMETER StampPath    Optional. On success, the exe's LastWriteTimeUtc
                        ticks are written here; if the file already holds
                        the current exe's ticks the run is skipped
                        (build_windows.bat --resume).
.PARAMETER RequireCamera  Make camera_opened == false a FAIL.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [string]$Arguments = "",
    [string]$WorkingDirectory = "",
    [int]$RunSeconds = 45,
    [Parameter(Mandatory = $true)][string]$MarkerPath,
    [string]$StampPath = "",
    [switch]$RequireCamera
)

$ErrorActionPreference = "Stop"
$script:failures = New-Object System.Collections.Generic.List[string]
$script:warnings = New-Object System.Collections.Generic.List[string]

function Fail([string]$msg) { $script:failures.Add($msg); Write-Host "[smoke] FAIL: $msg" -ForegroundColor Red }
function Warn([string]$msg) { $script:warnings.Add($msg); Write-Host "[smoke] WARN: $msg" -ForegroundColor Yellow }
function Info([string]$msg) { Write-Host "[smoke] $msg" }
# reg.exe writes "The operation completed successfully." to STDERR, which
# Windows PowerShell 5.1 turns into a terminating NativeCommandError under
# ErrorActionPreference=Stop (and it would abort the finally block that
# restores the registry). Run it detached and read the exit code instead.
function Invoke-Reg([string[]]$RegArgs) {
    $p = Start-Process -FilePath "$env:SystemRoot\System32\reg.exe" -ArgumentList $RegArgs -Wait -PassThru -WindowStyle Hidden
    return $p.ExitCode
}

if ($env:SMOKE_REQUIRE_CAMERA -eq "1") { $RequireCamera = $true }

# ---------------------------------------------------------------- preflight
if (-not (Test-Path -LiteralPath $Exe)) {
    Fail "exe not found: $Exe"
    exit 1
}
$exeItem = Get-Item -LiteralPath $Exe
$distDir = if ($WorkingDirectory) { $WorkingDirectory } else { $exeItem.DirectoryName }
$exeTicks = [string]$exeItem.LastWriteTimeUtc.Ticks

if ($StampPath -and (Test-Path -LiteralPath $StampPath)) {
    $stamped = (Get-Content -LiteralPath $StampPath -Raw).Trim()
    if ($stamped -eq $exeTicks) {
        Info "stamp matches current exe (LastWriteTimeUtc ticks $exeTicks) - smoke already passed for this dist, skipping."
        exit 0
    }
}

$running = @(Get-Process -Name Touchless -ErrorAction SilentlyContinue)
if ($running.Count -gt 0) {
    Fail ("Touchless.exe is already running (PID " + (($running | ForEach-Object { $_.Id }) -join ",") + "). " +
          "Quit it from the tray first - the single-instance mutex would make the smoke launch bail silently.")
    exit 1
}

$localAppData = $env:LOCALAPPDATA
$faultLog = Join-Path $localAppData "Touchless\crash\faulthandler.log"
$faultOffset = 0
if (Test-Path -LiteralPath $faultLog) { $faultOffset = (Get-Item -LiteralPath $faultLog).Length }

# ---------------------------------------------------------------- registry snapshot
$uninstallKey = 'HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\{2C4EE680-53F5-4D83-92A8-ADF4D2D8794E}_is1'
$runKeyPs     = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
$regBackup    = Join-Path $env:TEMP ("touchless_smoke_uninstall_" + [guid]::NewGuid().ToString("N") + ".reg")
$haveUninstallBackup = $false
$regExportCode = Invoke-Reg @("export", "`"$uninstallKey`"", "`"$regBackup`"", "/y")
if ($regExportCode -eq 0 -and (Test-Path -LiteralPath $regBackup)) {
    $haveUninstallBackup = $true
    Info "registry: uninstall key snapshotted to $regBackup"
} else {
    Info "registry: no per-user uninstall key present (nothing for heal_install_location to rewrite)"
}
$runValueBefore = $null
try { $runValueBefore = (Get-ItemProperty -Path $runKeyPs -Name Touchless -ErrorAction Stop).Touchless } catch { $runValueBefore = $null }

# ---------------------------------------------------------------- launch
$markerDir = Split-Path -Parent $MarkerPath
if ($markerDir -and -not (Test-Path -LiteralPath $markerDir)) { New-Item -ItemType Directory -Force -Path $markerDir | Out-Null }
if (Test-Path -LiteralPath $MarkerPath) { Remove-Item -LiteralPath $MarkerPath -Force }

$env:HGR_AUTOSTART_ENGINE   = "1"
$env:HGR_SMOKE_EXIT_AFTER_S = [string]$RunSeconds
$env:HGR_SMOKE_MARKER_PATH  = $MarkerPath
$env:TOUCHLESS_SIMULATE_UPDATE = "0"   # source-mode dry runs: no synthetic update prompt

$hardTimeout = $RunSeconds + 45
Info "launching $Exe $Arguments  (run $RunSeconds s, hard timeout $hardTimeout s, marker $MarkerPath)"
$exitCode = $null
$proc = $null
try {
    $spArgs = @{ FilePath = $Exe; WorkingDirectory = $distDir; PassThru = $true }
    if ($Arguments) { $spArgs.ArgumentList = $Arguments }
    $proc = Start-Process @spArgs
    $null = $proc.Handle   # cache the handle so ExitCode is readable after exit
    $t0 = Get-Date
    if (-not $proc.WaitForExit($hardTimeout * 1000)) {
        Fail "process did not exit within $hardTimeout s - killing it"
        try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
        Start-Sleep -Seconds 2
    } else {
        $exitCode = $proc.ExitCode
        Info ("process exited with code $exitCode after {0:N1} s" -f ((Get-Date) - $t0).TotalSeconds)
    }
}
finally {
    # ---- orphan cleanup: only images living under the dist folder
    foreach ($name in @("Touchless", "ffmpeg")) {
        foreach ($p in @(Get-Process -Name $name -ErrorAction SilentlyContinue)) {
            $path = $null
            try { $path = $p.Path } catch { $path = $null }
            if ($path -and $path.StartsWith($distDir, [System.StringComparison]::OrdinalIgnoreCase)) {
                Warn "killing orphan $($p.ProcessName) (PID $($p.Id)) from $path"
                try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
            }
        }
    }
    # ---- registry restore (heal_install_location / autostart.heal side effects)
    if ($haveUninstallBackup) {
        $regImportCode = Invoke-Reg @("import", "`"$regBackup`"")
        if ($regImportCode -eq 0) { Info "registry: uninstall key restored (InstallLocation / App Path / DisplayIcon)" }
        else { Warn "registry: reg import of $regBackup failed (exit $regImportCode) - check InstallLocation manually" }
        Remove-Item -LiteralPath $regBackup -Force -ErrorAction SilentlyContinue
    }
    $runValueAfter = $null
    try { $runValueAfter = (Get-ItemProperty -Path $runKeyPs -Name Touchless -ErrorAction Stop).Touchless } catch { $runValueAfter = $null }
    if ($runValueAfter -ne $runValueBefore) {
        if ($null -eq $runValueBefore) {
            Remove-ItemProperty -Path $runKeyPs -Name Touchless -ErrorAction SilentlyContinue
            Info "registry: removed Run\Touchless value the smoke run created"
        } else {
            Set-ItemProperty -Path $runKeyPs -Name Touchless -Value $runValueBefore
            Info "registry: Run\Touchless restored to $runValueBefore"
        }
    }
    Remove-Item Env:HGR_AUTOSTART_ENGINE, Env:HGR_SMOKE_EXIT_AFTER_S, Env:HGR_SMOKE_MARKER_PATH, Env:TOUCHLESS_SIMULATE_UPDATE -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------- verdict
if ($null -ne $exitCode -and $exitCode -ne 0) {
    if ($exitCode -eq 4) { Fail "exit code 4: app wrote the marker but the Qt loop never exited (modal dialog or stop_engine hang) - os._exit fallback fired" }
    else { Fail "exit code $exitCode (expected 0)" }
}

$marker = $null
if (-not (Test-Path -LiteralPath $MarkerPath)) {
    Fail "marker not written: $MarkerPath (app crashed before HGR_SMOKE_EXIT_AFTER_S fired, or this exe predates the smoke hooks)"
} else {
    try { $marker = Get-Content -LiteralPath $MarkerPath -Raw | ConvertFrom-Json }
    catch { Fail "marker is not valid JSON: $($_.Exception.Message)" }
}

if ($null -ne $marker) {
    Info ("marker: engine_started={0} engine_running={1} camera_opened={2} first_frame={3} frames_seen={4} display_fps={5} mode={6} frozen={7}" -f `
        $marker.engine_started, $marker.engine_running, $marker.camera_opened, $marker.first_frame_received, `
        $marker.frames_seen, $marker.display_fps, $marker.mode, $marker.frozen)
    if (-not $marker.engine_started) { Fail "engine_started=false (GestureWorker construction / start_engine failed - see touchless_debug.log)" }
    if (-not $marker.camera_opened) {
        if ($RequireCamera) { Fail "camera_opened=false and camera is required (SMOKE_REQUIRE_CAMERA=1)" }
        else { Warn "camera_opened=false - build machine has no usable camera; camera-open path NOT exercised by this smoke" }
    } elseif ([int]$marker.frames_seen -le 0) {
        Fail "camera opened but frames_seen=0 (capture silent-hang)"
    } elseif (-not $marker.engine_running) {
        Fail "camera opened and frames flowed but engine_running=false at exit (worker died mid-run)"
    }
    if ($marker.errors -and $marker.errors.Count -gt 0) {
        Fail ("marker.errors has {0} Traceback / fatal-exception line(s):" -f $marker.errors.Count)
        $marker.errors | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    }
}

if (Test-Path -LiteralPath $faultLog) {
    $fs = [System.IO.File]::Open($faultLog, 'Open', 'Read', 'ReadWrite')
    try {
        if ($fs.Length -gt $faultOffset) {
            $fs.Seek($faultOffset, 'Begin') | Out-Null
            $sr = New-Object System.IO.StreamReader($fs)
            $newText = $sr.ReadToEnd()
            $bad = @($newText -split "`r?`n" | Where-Object { $_ -match 'Windows fatal exception' -and $_ -notmatch '0x8001010d' })
            if ($bad.Count -gt 0) {
                Fail ("faulthandler.log grew {0} new fatal-exception line(s) during the run:" -f $bad.Count)
                $bad | Select-Object -First 5 | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
            } else {
                Info "faulthandler.log: no new fatal exceptions (benign 0x8001010d ignored)"
            }
        } else { Info "faulthandler.log: unchanged" }
    } finally { $fs.Dispose() }
}

Write-Host ""
if ($script:failures.Count -gt 0) {
    Write-Host ("[smoke] BUILD GATE FAILED - {0} violation(s), {1} warning(s). Marker: {2}" -f $script:failures.Count, $script:warnings.Count, $MarkerPath) -ForegroundColor Red
    exit 1
}
if ($StampPath) {
    $stampDir = Split-Path -Parent $StampPath
    if ($stampDir -and -not (Test-Path -LiteralPath $stampDir)) { New-Item -ItemType Directory -Force -Path $stampDir | Out-Null }
    Set-Content -LiteralPath $StampPath -Value $exeTicks -Encoding ascii
}
Write-Host ("[smoke] PASS ({0} warning(s)). Frozen exe launched, engine started, exited cleanly." -f $script:warnings.Count) -ForegroundColor Green
exit 0

# Author: Konstantin Markov
