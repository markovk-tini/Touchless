<#
.SYNOPSIS
  Kill orphaned Touchless helper processes and remove leftover Touchless
  folders. Run this when "failed to extract" keeps happening, or when a
  Touchless folder refuses to delete.

.DESCRIPTION
  Touchless spawns helper processes: ffmpeg (clip cache, camera capture,
  audio bridges), llama-server (grammar correction) and whisper-stream
  (dictation). If Touchless crashes or is force-killed, these keep
  running as ORPHANS with no parent. Task Manager shows no "Touchless",
  but the helpers are still there, writing clip-cache segments in a loop
  and holding an open handle on their own image file inside the install
  folder.

  That is why:
    * clip_cache files reappear after you delete them
    * the Touchless folder refuses to delete
    * the installer reports "failed to extract" - tar cannot overwrite
      _internal\ffmpeg.EXE while it is running

  SAFETY: only processes whose executable path contains "Touchless" are
  killed. An ffmpeg you installed yourself elsewhere is left alone.

.PARAMETER WhatIf
  Show what would be killed and deleted without changing anything.

.PARAMETER RemoveFolders
  Also delete leftover Touchless install folders and the clip cache.
  Without this, only the orphaned processes are killed.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File Touchless_Cleanup.ps1
  powershell -ExecutionPolicy Bypass -File Touchless_Cleanup.ps1 -RemoveFolders
#>
[CmdletBinding()]
param(
    [switch]$WhatIf,
    [switch]$RemoveFolders
)

$ErrorActionPreference = 'Continue'
$HelperNames = @('ffmpeg', 'llama-server', 'whisper-stream', 'whisper-server', 'whisper-cli', 'Touchless')

function Write-Head([string]$t) { Write-Host ''; Write-Host $t -ForegroundColor Cyan }

Write-Head '=== 1. Looking for Touchless processes and orphaned helpers ==='
$targets = @()
foreach ($n in $HelperNames) {
    foreach ($p in @(Get-Process -Name $n -ErrorAction SilentlyContinue)) {
        $path = $null
        try { $path = $p.Path } catch { $path = $null }
        # Touchless.exe always counts. Helpers count only when their image
        # lives inside a Touchless folder, so unrelated copies are safe.
        $isOurs = $false
        if ($p.ProcessName -eq 'Touchless') { $isOurs = $true }
        elseif ($path -and $path -match 'Touchless') { $isOurs = $true }
        if ($isOurs) {
            $targets += [pscustomobject]@{
                Name = $p.ProcessName; Id = $p.Id
                Path = $(if ($path) { $path } else { '(path unavailable)' })
                Started = $(try { $p.StartTime } catch { $null })
                Proc = $p
            }
        } elseif ($path) {
            Write-Host ("  skipping (not Touchless): {0} (PID {1}) {2}" -f $p.ProcessName, $p.Id, $path) -ForegroundColor DarkGray
        }
    }
}

if ($targets.Count -eq 0) {
    Write-Host '  none found - no orphaned Touchless processes are running.' -ForegroundColor Green
} else {
    $targets | ForEach-Object {
        Write-Host ("  {0,-16} PID {1,-7} started {2}" -f $_.Name, $_.Id, $(if ($_.Started) { $_.Started.ToString('HH:mm:ss') } else { '?' })) -ForegroundColor Yellow
        Write-Host ("      " + $_.Path) -ForegroundColor DarkYellow
    }
    if ($WhatIf) {
        Write-Host '  -WhatIf: nothing killed.' -ForegroundColor Magenta
    } else {
        Write-Head '=== 2. Stopping them ==='
        foreach ($t in $targets) {
            try {
                Stop-Process -Id $t.Id -Force -ErrorAction Stop
                Write-Host ("  stopped {0} (PID {1})" -f $t.Name, $t.Id) -ForegroundColor Green
            } catch {
                Write-Host ("  COULD NOT STOP {0} (PID {1}): {2}" -f $t.Name, $t.Id, $_.Exception.Message) -ForegroundColor Red
            }
        }
        Start-Sleep -Seconds 2
        $left = @()
        foreach ($n in $HelperNames) {
            foreach ($p in @(Get-Process -Name $n -ErrorAction SilentlyContinue)) {
                $pp = $null; try { $pp = $p.Path } catch { }
                if ($p.ProcessName -eq 'Touchless' -or ($pp -and $pp -match 'Touchless')) { $left += $p }
            }
        }
        if ($left.Count -gt 0) {
            Write-Host ("  WARNING: {0} still running. Reboot and run this again before installing." -f $left.Count) -ForegroundColor Red
        } else {
            Write-Host '  all clear.' -ForegroundColor Green
        }
    }
}

Write-Head '=== 3. Leftover Touchless folders ==='
$candidates = @(
    (Join-Path $env:LOCALAPPDATA 'Programs\Touchless'),
    (Join-Path $env:LOCALAPPDATA 'Touchless\clip_cache'),
    (Join-Path $env:TEMP 'hgr_clip_cache')
)
# Only real, existing fixed drives - Join-Path throws on a missing drive.
foreach ($dr in @(Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue | Where-Object { $null -ne $_.Free })) {
    $root = $dr.Name + ':\'
    $candidates += ($root + 'Touchless')
    $candidates += ($root + 'Program Files\Touchless')
    $candidates += ($root + 'Programs\Touchless')
}
$found = @()
foreach ($c in ($candidates | Select-Object -Unique)) {
    if (Test-Path -LiteralPath $c) {
        $sz = 0
        try { $sz = (Get-ChildItem -Recurse -File -LiteralPath $c -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum } catch { }
        $found += [pscustomobject]@{ Path = $c; MB = [int]($sz / 1MB) }
    }
}
if ($found.Count -eq 0) {
    Write-Host '  none found.' -ForegroundColor Green
} else {
    $found | ForEach-Object { Write-Host ("  {0,8} MB  {1}" -f $_.MB, $_.Path) -ForegroundColor Yellow }
    if (-not $RemoveFolders) {
        Write-Host ''
        Write-Host '  Re-run with -RemoveFolders to delete these.' -ForegroundColor Magenta
        Write-Host '  (Uninstall from Settings > Apps first if Touchless is still listed there.)' -ForegroundColor Magenta
    } elseif ($WhatIf) {
        Write-Host '  -WhatIf: nothing deleted.' -ForegroundColor Magenta
    } else {
        Write-Head '=== 4. Deleting them ==='
        foreach ($f in $found) {
            try {
                Remove-Item -LiteralPath $f.Path -Recurse -Force -ErrorAction Stop
                Write-Host ("  deleted " + $f.Path) -ForegroundColor Green
            } catch {
                Write-Host ("  COULD NOT DELETE " + $f.Path) -ForegroundColor Red
                Write-Host ("      " + $_.Exception.Message) -ForegroundColor DarkRed
                Write-Host '      Something still holds a file here. Reboot, then run this again.' -ForegroundColor DarkRed
            }
        }
    }
}

Write-Head '=== 5. Free space ==='
Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue | Where-Object { $null -ne $_.Free } | ForEach-Object {
    Write-Host ("  {0}:  {1,8:N1} GB free" -f $_.Name, ($_.Free / 1GB))
}
Write-Host ''
Write-Host 'The installer needs about 1.7 GB free on C: for the download plus' -ForegroundColor Gray
Write-Host '3.4 GB on whichever drive you install to (about 5 GB if both are C:).' -ForegroundColor Gray
Write-Host ''
Write-Host 'Done. If everything above is green, run the Touchless installer now.' -ForegroundColor Cyan

# Author: Konstantin Markov

# SIG # Begin signature block
# MII25gYJKoZIhvcNAQcCoII21zCCNtMCAQExDzANBglghkgBZQMEAgEFADB5Bgor
# BgEEAYI3AgEEoGswaTA0BgorBgEEAYI3AgEeMCYCAwEAAAQQH8w7YFlLCE63JNLG
# KX7zUQIBAAIBAAIBAAIBAAIBADAxMA0GCWCGSAFlAwQCAQUABCB/nFfD7STMmXNB
# bCv6at/kZofdOwWR148qjFeB30DLZKCCG0wwggXMMIIDtKADAgECAhBUmNLR1FsZ
# lUgTecgRwIeZMA0GCSqGSIb3DQEBDAUAMHcxCzAJBgNVBAYTAlVTMR4wHAYDVQQK
# ExVNaWNyb3NvZnQgQ29ycG9yYXRpb24xSDBGBgNVBAMTP01pY3Jvc29mdCBJZGVu
# dGl0eSBWZXJpZmljYXRpb24gUm9vdCBDZXJ0aWZpY2F0ZSBBdXRob3JpdHkgMjAy
# MDAeFw0yMDA0MTYxODM2MTZaFw00NTA0MTYxODQ0NDBaMHcxCzAJBgNVBAYTAlVT
# MR4wHAYDVQQKExVNaWNyb3NvZnQgQ29ycG9yYXRpb24xSDBGBgNVBAMTP01pY3Jv
# c29mdCBJZGVudGl0eSBWZXJpZmljYXRpb24gUm9vdCBDZXJ0aWZpY2F0ZSBBdXRo
# b3JpdHkgMjAyMDCCAiIwDQYJKoZIhvcNAQEBBQADggIPADCCAgoCggIBALORKgeD
# Bmf9np3gx8C3pOZCBH8Ppttf+9Va10Wg+3cL8IDzpm1aTXlT2KCGhFdFIMeiVPvH
# or+Kx24186IVxC9O40qFlkkN/76Z2BT2vCcH7kKbK/ULkgbk/WkTZaiRcvKYhOuD
# PQ7k13ESSCHLDe32R0m3m/nJxxe2hE//uKya13NnSYXjhr03QNAlhtTetcJtYmrV
# qXi8LW9J+eVsFBT9FMfTZRY33stuvF4pjf1imxUs1gXmuYkyM6Nix9fWUmcIxC70
# ViueC4fM7Ke0pqrrBc0ZV6U6CwQnHJFnni1iLS8evtrAIMsEGcoz+4m+mOJyoHI1
# vnnhnINv5G0Xb5DzPQCGdTiO0OBJmrvb0/gwytVXiGhNctO/bX9x2P29Da6SZEi3
# W295JrXNm5UhhNHvDzI9e1eM80UHTHzgXhgONXaLbZ7LNnSrBfjgc10yVpRnlyUK
# xjU9lJfnwUSLgP3B+PR0GeUw9gb7IVc+BhyLaxWGJ0l7gpPKWeh1R+g/OPTHU3mg
# trTiXFHvvV84wRPmeAyVWi7FQFkozA8kwOy6CXcjmTimthzax7ogttc32H83rwjj
# O3HbbnMbfZlysOSGM1l0tRYAe1BtxoYT2v3EOYI9JACaYNq6lMAFUSw0rFCZE4e7
# swWAsk0wAly4JoNdtGNz764jlU9gKL431VulAgMBAAGjVDBSMA4GA1UdDwEB/wQE
# AwIBhjAPBgNVHRMBAf8EBTADAQH/MB0GA1UdDgQWBBTIftJqhSobyhmYBAcnz1AQ
# T2ioojAQBgkrBgEEAYI3FQEEAwIBADANBgkqhkiG9w0BAQwFAAOCAgEAr2rd5hnn
# LZRDGU7L6VCVZKUDkQKL4jaAOxWiUsIWGbZqWl10QzD0m/9gdAmxIR6QFm3FJI9c
# Zohj9E/MffISTEAQiwGf2qnIrvKVG8+dBetJPnSgaFvlVixlHIJ+U9pW2UYXeZJF
# xBA2CFIpF8svpvJ+1Gkkih6PsHMNzBxKq7Kq7aeRYwFkIqgyuH4yKLNncy2RtNwx
# AQv3Rwqm8ddK7VZgxCwIo3tAsLx0J1KH1r6I3TeKiW5niB31yV2g/rarOoDXGpc8
# FzYiQR6sTdWD5jw4vU8w6VSp07YEwzJ2YbuwGMUrGLPAgNW3lbBeUU0i/OxYqujY
# lLSlLu2S3ucYfCFX3VVj979tzR/SpncocMfiWzpbCNJbTsgAlrPhgzavhgplXHT2
# 6ux6anSg8Evu75SjrFDyh+3XOjCDyft9V77l4/hByuVkrrOj7FjshZrM77nq81YY
# uVxzmq/FdxeDWds3GhhyVKVB0rYjdaNDmuV3fJZ5t0GNv+zcgKCf0Xd1WF81E+Al
# GmcLfc4l+gcK5GEh2NQc5QfGNpn0ltDGFf5Ozdeui53bFv0ExpK91IjmqaOqu/dk
# ODtfzAzQNb50GQOmxapMomE2gj4d8yu8l13bS3g7LfU772Aj6PXsCyM2la+YZr9T
# 03u4aUoqlmZpxJTG9F9urJh4iIAGXKKy7aIwggaqMIIEkqADAgECAhMzAAb8ZAK5
# zBnUK0f5AAAABvxkMA0GCSqGSIb3DQEBDAUAMFoxCzAJBgNVBAYTAlVTMR4wHAYD
# VQQKExVNaWNyb3NvZnQgQ29ycG9yYXRpb24xKzApBgNVBAMTIk1pY3Jvc29mdCBJ
# RCBWZXJpZmllZCBDUyBBT0MgQ0EgMDMwHhcNMjYwOTIzMTA1MTMwWhcNMjYwOTI2
# MTA1MTMwWjBtMQswCQYDVQQGEwJVUzETMBEGA1UECBMKV2FzaGluZ3RvbjERMA8G
# A1UEBxMIQmVsbGV2dWUxGjAYBgNVBAoTEUtvbnN0YW50aW4gTWFya292MRowGAYD
# VQQDExFLb25zdGFudGluIE1hcmtvdjCCAaIwDQYJKoZIhvcNAQEBBQADggGPADCC
# AYoCggGBAKAPocxKpwgHZll3pjeMeHZzXV2D5Npmv+lCp7JkXVSJtxp+hzKy29cu
# KOSOKXYHlMcqMQJ9vY7a4z15QBnjVwdytSaCBPB3mdVQNOFLijCA5xcPwPsWTjrj
# +CkNmAL9ZvmIKONJjtpg/MqQ8hP2XPux10DUOKcLgvVhRbtVSlX30DuQDhu6NZUw
# aKqrs/8I6E4WZ0PrcJMFR1dBsrOfueJQfN29XHZizu69wa2DN493CgQMXw3QRcy0
# E+r7WddyboNBTnphXOz9Uj6I2ONvLp0ZnwLOOm8k/RjsKOw+oPQuwwR/zcGjntaJ
# NTBD2kV0Bn7nvatZRDqnxIU4ig6x+PM05re7+fUnuGaxrfLfZqBH6kg7QWx+Mud3
# u69I9JfR49y5KdvNrROBULfudKNOYPHkUpBbCg0uzN4g9oX/UimbzX1JaQeJ2dXl
# HteX/z6RGPUv//JbL2amzQQDnxQijlZA1hx6hotMdrplbAT9xWtH/jIeO/dTPyBo
# nw/HmR1tGQIDAQABo4IB1DCCAdAwDAYDVR0TAQH/BAIwADAOBgNVHQ8BAf8EBAMC
# B4AwOwYDVR0lBDQwMgYKKwYBBAGCN2EBAAYIKwYBBQUHAwMGGisGAQQBgjdhg6zx
# 926e+Lxegei68wqTk9F4MB0GA1UdDgQWBBT8iWczPE1oYa/zUARYcr6srUqZizAf
# BgNVHSMEGDAWgBSkQwx/dlqlhec+jSgPDBeiRWlwxjBnBgNVHR8EYDBeMFygWqBY
# hlZodHRwOi8vd3d3Lm1pY3Jvc29mdC5jb20vcGtpb3BzL2NybC9NaWNyb3NvZnQl
# MjBJRCUyMFZlcmlmaWVkJTIwQ1MlMjBBT0MlMjBDQSUyMDAzLmNybDB0BggrBgEF
# BQcBAQRoMGYwZAYIKwYBBQUHMAKGWGh0dHA6Ly93d3cubWljcm9zb2Z0LmNvbS9w
# a2lvcHMvY2VydHMvTWljcm9zb2Z0JTIwSUQlMjBWZXJpZmllZCUyMENTJTIwQU9D
# JTIwQ0ElMjAwMy5jcnQwVAYDVR0gBE0wSzBJBgRVHSAAMEEwPwYIKwYBBQUHAgEW
# M2h0dHA6Ly93d3cubWljcm9zb2Z0LmNvbS9wa2lvcHMvRG9jcy9SZXBvc2l0b3J5
# Lmh0bTANBgkqhkiG9w0BAQwFAAOCAgEAhKJnac3cnR5eCwW1+h40+OabpNR9IKhs
# lG7q4/NE1IN+H7HgLQfsgIT7l/ncrXR7bpQturr+3QqoZdjmf0ketAA4TM8yEdtF
# XxggvgUl+Bl+LRwFKOPN7z6wqEIfmtuv8ZrukAicmK60aaP1HUcon/v/j7QEEM1a
# tBXlCLpuQZSedRMYplaI4munUH5rNlw7LNY5bZCSuyPL8gQeiQduvAM1Yu78JNnO
# 3MOFQDX/A/fSDn0Nowb3C2PtHW459m+VM+FHNC8tJIStF1tX38vcSimCI+A94ND1
# cdPQcWkWB+PVRiVyeoyxV2+Wbcq9nHrobtF8B6tpL0XYnPmpbdLuHWj41oRaicXe
# 4rkBgpbjnS2OZoUz/1ByD0qHvYnfUQLdZpCTbxLWN4d037D427ehU8Sa6+os5ALp
# 3N95r8ut/19j+fwC1Z+alagBuUIYwrqj+uGfe5GhTcM0D7IN2EhdbH5ty6Ri9SZ2
# ZcgNXpJfydUhnc2xGk4AbL+l2ZZug/vT7u3AEDN7sPzb8h2HLo+vE84UBnX+NVOI
# hrBeVbczHPotpNcB+o+dHoWohm3IAAuGbz5viAmA6QxspgnPUPsEeenYerDvv/AW
# 8hP2w4CoFmidW7k3P4fy6F7dbLvtZCkB0RBaMs3wkCTX2JW5tcXePUI6bJFiIRJs
# UX8R5qvBf0wwggcoMIIFEKADAgECAhMzAAAAGA3rkVWpigCYAAAAAAAYMA0GCSqG
# SIb3DQEBDAUAMGMxCzAJBgNVBAYTAlVTMR4wHAYDVQQKExVNaWNyb3NvZnQgQ29y
# cG9yYXRpb24xNDAyBgNVBAMTK01pY3Jvc29mdCBJRCBWZXJpZmllZCBDb2RlIFNp
# Z25pbmcgUENBIDIwMjEwHhcNMjYwMzI2MTgxMTMyWhcNMzEwMzI2MTgxMTMyWjBa
# MQswCQYDVQQGEwJVUzEeMBwGA1UEChMVTWljcm9zb2Z0IENvcnBvcmF0aW9uMSsw
# KQYDVQQDEyJNaWNyb3NvZnQgSUQgVmVyaWZpZWQgQ1MgQU9DIENBIDAzMIICIjAN
# BgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAyIDaYDRWoon9lVnlj+SOj5xV8Sf5
# Qd+3yUeeRgr0exi2QTJAYo24ilcIKQSN8TOZ3+POM5x/6p3Cfjgqust44J0FvkfG
# Xe1Puy45a5nLJGpc0kNIITMRKZwVvPxx7NlfGSc0JOhz/kg7G77C+y3ZR/3jtpeJ
# pJ4QwcK9Gf0Peuk7xLYeW/JAsY9b6oleGDbYSxkamUfbtnyv8gTFrvN6ejuLqNhH
# YPvoBHsOSC+7555yhapkof0fbzyct1hdWHGXsAFMfLF2TVJ8d2YVYOfZdi6YrT4s
# MxOhTKiLKmhL1XtzM7hXdmv7lg2R+lWw8lIkSu/JiINQ0GAPcwxMsgRXDSPp8VUs
# 4Jby+ruz0bjaoHFd7H+hC8cPPcrEDP2eEdYURVl0acjliigCrXwR05NFJzYj3MZi
# zDGLPI3lIzonX1T40yK8v1FcJ8MXZZCvOXGXwRDGGfwwTTsHaJj+OfWNZ/IsypG4
# bGvqeJcPnEFcQEwRcfYIEe/R4a8k+xw5qTy75CbwWeMFuAlt9lE9kjMg3tvJyDlN
# 5voXx5VXinCwUHMpuVaEQ4yHAlSO7qoBltjzTBNHH3ovMwsAsuhwrLLCVhUu3oP2
# GxYZwEyXMlnzK5DbgGzHzDfDaYPHK0uo1VaMMg9Bhuc3YIvrkFXEiv+t/JgNcRGC
# t6ZyKEIDtPbrgwcCAwEAAaOCAdwwggHYMA4GA1UdDwEB/wQEAwIBhjAQBgkrBgEE
# AYI3FQEEAwIBADAdBgNVHQ4EFgQUpEMMf3ZapYXnPo0oDwwXokVpcMYwVAYDVR0g
# BE0wSzBJBgRVHSAAMEEwPwYIKwYBBQUHAgEWM2h0dHA6Ly93d3cubWljcm9zb2Z0
# LmNvbS9wa2lvcHMvRG9jcy9SZXBvc2l0b3J5Lmh0bTAZBgkrBgEEAYI3FAIEDB4K
# AFMAdQBiAEMAQTASBgNVHRMBAf8ECDAGAQH/AgEAMB8GA1UdIwQYMBaAFNlBKbAP
# D2Ns72nX9c0pnqRIajDmMHAGA1UdHwRpMGcwZaBjoGGGX2h0dHA6Ly93d3cubWlj
# cm9zb2Z0LmNvbS9wa2lvcHMvY3JsL01pY3Jvc29mdCUyMElEJTIwVmVyaWZpZWQl
# MjBDb2RlJTIwU2lnbmluZyUyMFBDQSUyMDIwMjEuY3JsMH0GCCsGAQUFBwEBBHEw
# bzBtBggrBgEFBQcwAoZhaHR0cDovL3d3dy5taWNyb3NvZnQuY29tL3BraW9wcy9j
# ZXJ0cy9NaWNyb3NvZnQlMjBJRCUyMFZlcmlmaWVkJTIwQ29kZSUyMFNpZ25pbmcl
# MjBQQ0ElMjAyMDIxLmNydDANBgkqhkiG9w0BAQwFAAOCAgEAcccgVvl+poXUYksA
# /TzDFnBlAJ8ef0FMJzb2XRRhF/uA0QyK/VgoeAvO8B7cPpYNQ97sytdA7LT19CxS
# wRQAt71jGF+CJl8KC4aEdMZTfJlHaKyd24J6QiVriNed9WdawsD7lK0pAcXziBg5
# N6dhAm9x6P8R4uT0UkfzlK1rkB8F4mlzE7l7tyES3s8FZGaRZjcGEQ+e0fTcdhf8
# jO7czmNB4dIRgmmBCt/P+ha0tEl2nV1sg1An5+VzhgAkY1Apx8fiUFBtH+Ehw/om
# 5aQCNIJfmR51ZnV18R02Xk2tAmAiIRcSj9vdtrNIOsy5nolddy1lJrbf1Be061l6
# TItv9FDZ4mg6B+65zxkVecVV/Ll8uLGYouGrMM6jzO2O/ps3K2p6mfBI2ZOYIy4U
# NwNrGWqa5TrvAmkZsn3CIlR+81X4AL5vNTFlxc4gH+5su0Dr58hBTxnXavDEnz7X
# 0csP1Kt7h+iqaGiTSHz2B+n3HmUoud0WrdQPYKxMat0To4YUqU3HIbgSLQDDVT8a
# CjW1Jvokf1915C/vVkIIp48h3voVy3JWPLwBlxQ9aeND6jCKQGLJhCQRSlvXX+P/
# 9TeaEA6/xWPSASZf6Ekve/Yua7U+zWc/Sr2K2gj0QRrNEAsvrFr4EGtHKDO9ECVS
# 3lcJksVDv9KHdMPUK8u20i68RqAwggeeMIIFhqADAgECAhMzAAAAB4ejNKN7pY4c
# AAAAAAAHMA0GCSqGSIb3DQEBDAUAMHcxCzAJBgNVBAYTAlVTMR4wHAYDVQQKExVN
# aWNyb3NvZnQgQ29ycG9yYXRpb24xSDBGBgNVBAMTP01pY3Jvc29mdCBJZGVudGl0
# eSBWZXJpZmljYXRpb24gUm9vdCBDZXJ0aWZpY2F0ZSBBdXRob3JpdHkgMjAyMDAe
# Fw0yMTA0MDEyMDA1MjBaFw0zNjA0MDEyMDE1MjBaMGMxCzAJBgNVBAYTAlVTMR4w
# HAYDVQQKExVNaWNyb3NvZnQgQ29ycG9yYXRpb24xNDAyBgNVBAMTK01pY3Jvc29m
# dCBJRCBWZXJpZmllZCBDb2RlIFNpZ25pbmcgUENBIDIwMjEwggIiMA0GCSqGSIb3
# DQEBAQUAA4ICDwAwggIKAoICAQCy8MCvGYgo4t1UekxJbGkIVQm0Uv96SvjB6yUo
# 92cXdylN65Xy96q2YpWCiTas7QPTkGnK9QMKDXB2ygS27EAIQZyAd+M8X+dmw6SD
# tzSZXyGkxP8a8Hi6EO9Zcwh5A+wOALNQbNO+iLvpgOnEM7GGB/wm5dYnMEOguua1
# OFfTUITVMIK8faxkP/4fPdEPCXYyy8NJ1fmskNhW5HduNqPZB/NkWbB9xxMqowAe
# WvPgHtpzyD3PLGVOmRO4ka0WcsEZqyg6efk3JiV/TEX39uNVGjgbODZhzspHvKFN
# U2K5MYfmHh4H1qObU4JKEjKGsqqA6RziybPqhvE74fEp4n1tiY9/ootdU0vPxRp4
# BGjQFq28nzawuvaCqUUF2PWxh+o5/TRCb/cHhcYU8Mr8fTiS15kRmwFFzdVPZ3+J
# V3s5MulIf3II5FXeghlAH9CvicPhhP+VaSFW3Da/azROdEm5sv+EUwhBrzqtxoYy
# E2wmuHKws00x4GGIx7NTWznOm6x/niqVi7a/mxnnMvQq8EMse0vwX2CfqM7Le/sm
# bRtsEeOtbnJBbtLfoAsC3TdAOnBbUkbUfG78VRclsE7YDDBUbgWt75lDk53yi7C3
# n0WkHFU4EZ83i83abd9nHWCqfnYa9qIHPqjOiuAgSOf4+FRcguEBXlD9mAInS7b6
# V0UaNwIDAQABo4ICNTCCAjEwDgYDVR0PAQH/BAQDAgGGMBAGCSsGAQQBgjcVAQQD
# AgEAMB0GA1UdDgQWBBTZQSmwDw9jbO9p1/XNKZ6kSGow5jBUBgNVHSAETTBLMEkG
# BFUdIAAwQTA/BggrBgEFBQcCARYzaHR0cDovL3d3dy5taWNyb3NvZnQuY29tL3Br
# aW9wcy9Eb2NzL1JlcG9zaXRvcnkuaHRtMBkGCSsGAQQBgjcUAgQMHgoAUwB1AGIA
# QwBBMA8GA1UdEwEB/wQFMAMBAf8wHwYDVR0jBBgwFoAUyH7SaoUqG8oZmAQHJ89Q
# EE9oqKIwgYQGA1UdHwR9MHsweaB3oHWGc2h0dHA6Ly93d3cubWljcm9zb2Z0LmNv
# bS9wa2lvcHMvY3JsL01pY3Jvc29mdCUyMElkZW50aXR5JTIwVmVyaWZpY2F0aW9u
# JTIwUm9vdCUyMENlcnRpZmljYXRlJTIwQXV0aG9yaXR5JTIwMjAyMC5jcmwwgcMG
# CCsGAQUFBwEBBIG2MIGzMIGBBggrBgEFBQcwAoZ1aHR0cDovL3d3dy5taWNyb3Nv
# ZnQuY29tL3BraW9wcy9jZXJ0cy9NaWNyb3NvZnQlMjBJZGVudGl0eSUyMFZlcmlm
# aWNhdGlvbiUyMFJvb3QlMjBDZXJ0aWZpY2F0ZSUyMEF1dGhvcml0eSUyMDIwMjAu
# Y3J0MC0GCCsGAQUFBzABhiFodHRwOi8vb25lb2NzcC5taWNyb3NvZnQuY29tL29j
# c3AwDQYJKoZIhvcNAQEMBQADggIBAH8lKp7+1Kvq3WYK21cjTLpebJDjW4ZbOX3H
# D5ZiG84vjsFXT0OB+eb+1TiJ55ns0BHluC6itMI2vnwc5wDW1ywdCq3TAmx0KWy7
# xulAP179qX6VSBNQkRXzReFyjvF2BGt6FvKFR/imR4CEESMAG8hSkPYso+GjlngM
# 8JPn/ROUrTaeU/BRu/1RFESFVgK2wMz7fU4VTd8NXwGZBe/mFPZG6tWwkdmA/jLb
# p0kNUX7elxu2+HtHo0QO5gdiKF+YTYd1BGrmNG8sTURvn09jAhIUJfYNotn7OlTh
# tfQjXqe0qrimgY4Vpoq2MgDW9ESUi1o4pzC1zTgIGtdJ/IvY6nqa80jFOTg5qzAi
# RNdsUvzVkoYP7bi4wLCj+ks2GftUct+fGUxXMdBUv5sdr0qFPLPB0b8vq516slCf
# RwaktAxK1S40MCvFbbAXXpAZnU20FaAoDwqq/jwzwd8Wo2J83r7O3onQbDO9TyDS
# tgaBNlHzMMQgl95nHBYMelLEHkUnVVVTUsgC0Huj09duNfMaJ9ogxhPNThgq3i8w
# 3DAGZ61AMeF0C1M+mU5eucj1Ijod5O2MMPeJQ3/vKBtqGZg4eTtUHt/BPjN74SsJ
# syHqAdXVS5c+ItyKWg3Eforhox9k3WgtWTpgV4gkSiS4+A09roSdOI4vrRw+p+fL
# 4WrxSK5nMYIa8DCCGuwCAQEwcTBaMQswCQYDVQQGEwJVUzEeMBwGA1UEChMVTWlj
# cm9zb2Z0IENvcnBvcmF0aW9uMSswKQYDVQQDEyJNaWNyb3NvZnQgSUQgVmVyaWZp
# ZWQgQ1MgQU9DIENBIDAzAhMzAAb8ZAK5zBnUK0f5AAAABvxkMA0GCWCGSAFlAwQC
# AQUAoIG8MBkGCSqGSIb3DQEJAzEMBgorBgEEAYI3AgEEMBwGCisGAQQBgjcCAQsx
# DjAMBgorBgEEAYI3AgEVMC8GCSqGSIb3DQEJBDEiBCDI654UyNmjNU/noh+B0K3u
# dZEKq8jPc98P9oO7/VBJGTBQBgorBgEEAYI3AgEMMUIwQKAkgCIAVABvAHUAYwBo
# AGwAZQBzAHMAIABDAGwAZQBhAG4AdQBwoRiAFmh0dHBzOi8vdG91Y2hsZXNzLmFw
# cC8wDQYJKoZIhvcNAQEBBQAEggGARdC/5idfCa5DHIVQmrgJLjdYYRF0+hroJQ5H
# sfZ8pdnMEfNmXtCFLZ+QvHr0LNwEpLLlp7p90sDEMsfcPTh0Bq9iyQbh3wh61V6U
# HKp0JFjABUKJ3cYCnh2o0tgEwKajJvZiQb6XZKWEp8w00atm85pgq10OziP+0D1Q
# rAZ4rAn1JlByipjxtOZc7hl6UHJU7hPxFDhH7TUAafNcGPPVh4Seq9ieZ6B1FFCp
# YtppySWbr92yH9vegSqwCYg+S7muoGbWDvv6W9OuiuimPe/CzECczaI5oGYbRPed
# FdGM7B3w8HuC+NRsytwj/EUSkdzTcgKt2c2t06Scqv5aNLHUm/EM8TEZrR/6Tj7x
# 1E25qBC9yGyX088r58BQkTI6XSXqyVsjXKq7l0WxqW36eus8TxLyd7AYg1qqxFBP
# qbCLAGvO87bos20kJSqMs3aId5vn0EzLSqHUgxUBcxZIUbs8HfmWPO3Eigxhs4Lb
# IBkXKpWGzn8tjB82I8OMmf72JHuSoYIYETCCGA0GCisGAQQBgjcDAwExghf9MIIX
# +QYJKoZIhvcNAQcCoIIX6jCCF+YCAQMxDzANBglghkgBZQMEAgEFADCCAWIGCyqG
# SIb3DQEJEAEEoIIBUQSCAU0wggFJAgEBBgorBgEEAYRZCgMBMDEwDQYJYIZIAWUD
# BAIBBQAEIHawyFwSwEPrcJxee08BxkpsZcewEnl11xc4W4sbcnLYAgZqqTOW63sY
# EzIwMjYwOTI1MDExNTQzLjgxMlowBIACAfSggeGkgd4wgdsxCzAJBgNVBAYTAlVT
# MRMwEQYDVQQIEwpXYXNoaW5ndG9uMRAwDgYDVQQHEwdSZWRtb25kMR4wHAYDVQQK
# ExVNaWNyb3NvZnQgQ29ycG9yYXRpb24xJTAjBgNVBAsTHE1pY3Jvc29mdCBBbWVy
# aWNhIE9wZXJhdGlvbnMxJzAlBgNVBAsTHm5TaGllbGQgVFNTIEVTTjpBNTAwLTA1
# RTAtRDk0NzE1MDMGA1UEAxMsTWljcm9zb2Z0IFB1YmxpYyBSU0EgVGltZSBTdGFt
# cGluZyBBdXRob3JpdHmggg8hMIIHgjCCBWqgAwIBAgITMwAAAAXlzw//Zi7JhwAA
# AAAABTANBgkqhkiG9w0BAQwFADB3MQswCQYDVQQGEwJVUzEeMBwGA1UEChMVTWlj
# cm9zb2Z0IENvcnBvcmF0aW9uMUgwRgYDVQQDEz9NaWNyb3NvZnQgSWRlbnRpdHkg
# VmVyaWZpY2F0aW9uIFJvb3QgQ2VydGlmaWNhdGUgQXV0aG9yaXR5IDIwMjAwHhcN
# MjAxMTE5MjAzMjMxWhcNMzUxMTE5MjA0MjMxWjBhMQswCQYDVQQGEwJVUzEeMBwG
# A1UEChMVTWljcm9zb2Z0IENvcnBvcmF0aW9uMTIwMAYDVQQDEylNaWNyb3NvZnQg
# UHVibGljIFJTQSBUaW1lc3RhbXBpbmcgQ0EgMjAyMDCCAiIwDQYJKoZIhvcNAQEB
# BQADggIPADCCAgoCggIBAJ5851Jj/eDFnwV9Y7UGIqMcHtfnlzPREwW9ZUZHd5HB
# XXBvf7KrQ5cMSqFSHGqg2/qJhYqOQxwuEQXG8kB41wsDJP5d0zmLYKAY8Zxv3lYk
# uLDsfMuIEqvGYOPURAH+Ybl4SJEESnt0MbPEoKdNihwM5xGv0rGofJ1qOYSTNcc5
# 5EbBT7uq3wx3mXhtVmtcCEr5ZKTkKKE1CxZvNPWdGWJUPC6e4uRfWHIhZcgCsJ+s
# ozf5EeH5KrlFnxpjKKTavwfFP6XaGZGWUG8TZaiTogRoAlqcevbiqioUz1Yt4FRK
# 53P6ovnUfANjIgM9JDdJ4e0qiDRm5sOTiEQtBLGd9Vhd1MadxoGcHrRCsS5rO9yh
# v2fjJHrmlQ0EIXmp4DhDBieKUGR+eZ4CNE3ctW4uvSDQVeSp9h1SaPV8UWEfyTxg
# GjOsRpeexIveR1MPTVf7gt8hY64XNPO6iyUGsEgt8c2PxF87E+CO7A28TpjNq5eL
# iiunhKbq0XbjkNoU5JhtYUrlmAbpxRjb9tSreDdtACpm3rkpxp7AQndnI0Shu/fk
# 1/rE3oWsDqMX3jjv40e8KN5YsJBnczyWB4JyeeFMW3JBfdeAKhzohFe8U5w9Wuvc
# P1E8cIxLoKSDzCCBOu0hWdjzKNu8Y5SwB1lt5dQhABYyzR3dxEO/T1K/BVF3rV69
# AgMBAAGjggIbMIICFzAOBgNVHQ8BAf8EBAMCAYYwEAYJKwYBBAGCNxUBBAMCAQAw
# HQYDVR0OBBYEFGtpKDo1L0hjQM972K9J6T7ZPdshMFQGA1UdIARNMEswSQYEVR0g
# ADBBMD8GCCsGAQUFBwIBFjNodHRwOi8vd3d3Lm1pY3Jvc29mdC5jb20vcGtpb3Bz
# L0RvY3MvUmVwb3NpdG9yeS5odG0wEwYDVR0lBAwwCgYIKwYBBQUHAwgwGQYJKwYB
# BAGCNxQCBAweCgBTAHUAYgBDAEEwDwYDVR0TAQH/BAUwAwEB/zAfBgNVHSMEGDAW
# gBTIftJqhSobyhmYBAcnz1AQT2ioojCBhAYDVR0fBH0wezB5oHegdYZzaHR0cDov
# L3d3dy5taWNyb3NvZnQuY29tL3BraW9wcy9jcmwvTWljcm9zb2Z0JTIwSWRlbnRp
# dHklMjBWZXJpZmljYXRpb24lMjBSb290JTIwQ2VydGlmaWNhdGUlMjBBdXRob3Jp
# dHklMjAyMDIwLmNybDCBlAYIKwYBBQUHAQEEgYcwgYQwgYEGCCsGAQUFBzAChnVo
# dHRwOi8vd3d3Lm1pY3Jvc29mdC5jb20vcGtpb3BzL2NlcnRzL01pY3Jvc29mdCUy
# MElkZW50aXR5JTIwVmVyaWZpY2F0aW9uJTIwUm9vdCUyMENlcnRpZmljYXRlJTIw
# QXV0aG9yaXR5JTIwMjAyMC5jcnQwDQYJKoZIhvcNAQEMBQADggIBAF+Idsd+bbVa
# FXXnTHho+k7h2ESZJRWluLE0Oa/pO+4ge/XEizXvhs0Y7+KVYyb4nHlugBesnFqB
# GEdC2IWmtKMyS1OWIviwpnK3aL5JedwzbeBF7POyg6IGG/XhhJ3UqWeWTO+Czb1c
# 2NP5zyEh89F72u9UIw+IfvM9lzDmc2O2END7MPnrcjWdQnrLn1Ntday7JSyrDvBd
# mgbNnCKNZPmhzoa8PccOiQljjTW6GePe5sGFuRHzdFt8y+bN2neF7Zu8hTO1I64X
# NGqst8S+w+RUdie8fXC1jKu3m9KGIqF4aldrYBamyh3g4nJPj/LR2CBaLyD+2BuG
# ZCVmoNR/dSpRCxlot0i79dKOChmoONqbMI8m04uLaEHAv4qwKHQ1vBzbV/nG89LD
# KbRSSvijmwJwxRxLLpMQ/u4xXxFfR4f/gksSkbJp7oqLwliDm/h+w0aJ/U5ccnYh
# Yb7vPKNMN+SZDWycU5ODIRfyoGl59BsXR/HpRGtiJquOYGmvA/pk5vC1lcnbeMrc
# WD/26ozePQ/TWfNXKBOmkFpvPE8CH+EeGGWzqTCjdAsno2jzTeNSxlx3glDGJgcd
# z5D/AAxw9Sdgq/+rY7jjgs7X6fqPTXPmaCAJKVHAP19oEjJIBwD1LyHbaEgBxFCo
# gYSOiUIr0Xqcr1nJfiWG2GwYe6ZoAF1bMIIHlzCCBX+gAwIBAgITMwAAAFZ+j51Y
# CI7pYAAAAAAAVjANBgkqhkiG9w0BAQwFADBhMQswCQYDVQQGEwJVUzEeMBwGA1UE
# ChMVTWljcm9zb2Z0IENvcnBvcmF0aW9uMTIwMAYDVQQDEylNaWNyb3NvZnQgUHVi
# bGljIFJTQSBUaW1lc3RhbXBpbmcgQ0EgMjAyMDAeFw0yNTEwMjMyMDQ2NTFaFw0y
# NjEwMjIyMDQ2NTFaMIHbMQswCQYDVQQGEwJVUzETMBEGA1UECBMKV2FzaGluZ3Rv
# bjEQMA4GA1UEBxMHUmVkbW9uZDEeMBwGA1UEChMVTWljcm9zb2Z0IENvcnBvcmF0
# aW9uMSUwIwYDVQQLExxNaWNyb3NvZnQgQW1lcmljYSBPcGVyYXRpb25zMScwJQYD
# VQQLEx5uU2hpZWxkIFRTUyBFU046QTUwMC0wNUUwLUQ5NDcxNTAzBgNVBAMTLE1p
# Y3Jvc29mdCBQdWJsaWMgUlNBIFRpbWUgU3RhbXBpbmcgQXV0aG9yaXR5MIICIjAN
# BgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAtKWfm/ul027/d8Rlb8Mn/g0QUvvL
# qY2Vsy3tI8U2tFSspTZomZOD3BHT8LkR+RrhMJgb1VjAKFNysaK9cLSXifPGSIBr
# PCgs9P4y24lrJEmrV6Q5z4BmqMhIPrZhEvZnWpCS4HO7jYSei/nxmC7/1Er+l5Lg
# 3PmSxb8d2IVcARxSw1B4mxB6XI0nkel9wa1dYb2wfGpofraFmxZOxT9eNht4LH0R
# BSVueba6ZNpjS/0gtfm7qiIiyP6p6PRzTTbMnVqsHnV/d/rW0zHx+Q+QNZ5wUqKm
# TZJB9hU853+2pX5rDfK32uNY9/WBOAmzbqgpEdQkbiMavUMyUDShmycIvgHdQnS2
# 07sTj8M+kJL3tOdahPuPqMwsaCCgdfwwQx0O9TKe7FSvbAEYs1AnldCl/KHGZCOV
# vUNqjyL10JLe0/+GD9/ynqXGWFpXOjaunvZ/cKROhjN4M5e6xx0b2miqcPii4/ii
# 2ZheKallJET7CKlpFShs3wyg6F/fojQxQvPnbWD4Nyx6lhjWjwmoLcx6w1FSCtav
# LCly33BLRSlTU4qKUxaa8d7YN7Eqpn9XO0SY0umOvKFXrWH7rxl+9iaicitdnTTk
# sAnRjvekdKT3lg7lRMfmfZU8vXNiN0UYJzT9EjqjRm0uN/h0oXxPhNfPYqeFbyPX
# GGxzaYUz6zx3qTcCAwEAAaOCAcswggHHMB0GA1UdDgQWBBS+tjPyu6tZ/h5GsyLv
# yz1H+FNIWjAfBgNVHSMEGDAWgBRraSg6NS9IY0DPe9ivSek+2T3bITBsBgNVHR8E
# ZTBjMGGgX6BdhltodHRwOi8vd3d3Lm1pY3Jvc29mdC5jb20vcGtpb3BzL2NybC9N
# aWNyb3NvZnQlMjBQdWJsaWMlMjBSU0ElMjBUaW1lc3RhbXBpbmclMjBDQSUyMDIw
# MjAuY3JsMHkGCCsGAQUFBwEBBG0wazBpBggrBgEFBQcwAoZdaHR0cDovL3d3dy5t
# aWNyb3NvZnQuY29tL3BraW9wcy9jZXJ0cy9NaWNyb3NvZnQlMjBQdWJsaWMlMjBS
# U0ElMjBUaW1lc3RhbXBpbmclMjBDQSUyMDIwMjAuY3J0MAwGA1UdEwEB/wQCMAAw
# FgYDVR0lAQH/BAwwCgYIKwYBBQUHAwgwDgYDVR0PAQH/BAQDAgeAMGYGA1UdIARf
# MF0wUQYMKwYBBAGCN0yDfQEBMEEwPwYIKwYBBQUHAgEWM2h0dHA6Ly93d3cubWlj
# cm9zb2Z0LmNvbS9wa2lvcHMvRG9jcy9SZXBvc2l0b3J5Lmh0bTAIBgZngQwBBAIw
# DQYJKoZIhvcNAQEMBQADggIBAA4DqAXEsO26j/La7Fgn/Qifit8xuZekqZ57+Ye+
# sH/hRTbEEjGYrZgsqwR/lUUfKCFpbZF8msaZPQJOR4YYUEU8XyjLrn8Y1jCSmoxh
# 9l7tWiSoc/JFBw356JAmzGGxeBA2EWSxRuTr1AuZe6nYaN8/wtFkiHcs8gMadxXB
# s6DxVhyu5YnhLPQkfumKm3lFftwE7pieV7f1lskmlgsC6AeSGCzGPZUgCvcH5Tv/
# Qe9z7bIImSD3SuzhOIwaP+eKQTYf67TifyJKkWQSdGfTA6Kcu41k8LB6oPK+MLk1
# jbxxK5wPqLSL62xjK04SBXHEJSEnsFt0zxWkxP/lgej1DxqUnmrYEdkxvzKSHIAq
# FWSZul/5hI+vJxvFPhsNQBEk4cSulDkJQpcdVi/gmf/mHFOYhDBjsa15s4L+2sBi
# l3XV/T8RiR66Q8xYvTLRWxd2dVsrOoCwnsU4WIeiC0JinCv1WLHEh7Qyzr9RSr4k
# KJLWdpNYLhgjkojTmEkAjFO774t3xB7enbvIF0GOsV19xnCUzq9EGKyt0gMuaphK
# lNjJ+aTpjWMZDGo+GOKsnp93Hmftml0Syp3F9+M3y+y6WJGUZoIZJq227jDjjEnd
# tpUrh9BdPdVIfVJD/Au81Rzh05UHAivorQ3Os8PELHIgiOd9TWzbdgmGzcILt/dd
# VQERMYIHQzCCBz8CAQEweDBhMQswCQYDVQQGEwJVUzEeMBwGA1UEChMVTWljcm9z
# b2Z0IENvcnBvcmF0aW9uMTIwMAYDVQQDEylNaWNyb3NvZnQgUHVibGljIFJTQSBU
# aW1lc3RhbXBpbmcgQ0EgMjAyMAITMwAAAFZ+j51YCI7pYAAAAAAAVjANBglghkgB
# ZQMEAgEFAKCCBJwwEQYLKoZIhvcNAQkQAg8xAgUAMBoGCSqGSIb3DQEJAzENBgsq
# hkiG9w0BCRABBDAcBgkqhkiG9w0BCQUxDxcNMjYwOTI1MDExNTQzWjAvBgkqhkiG
# 9w0BCQQxIgQgtbTIp9FI/Nnbnr1+T/8U3GTe+jyS2VRupNzWfZjLr7MwgbkGCyqG
# SIb3DQEJEAIvMYGpMIGmMIGjMIGgBCC2DDMlTaTj8JV3iTg5Xnpe4CSH60143Z+X
# 9o5NBgMMqDB8MGWkYzBhMQswCQYDVQQGEwJVUzEeMBwGA1UEChMVTWljcm9zb2Z0
# IENvcnBvcmF0aW9uMTIwMAYDVQQDEylNaWNyb3NvZnQgUHVibGljIFJTQSBUaW1l
# c3RhbXBpbmcgQ0EgMjAyMAITMwAAAFZ+j51YCI7pYAAAAAAAVjCCA14GCyqGSIb3
# DQEJEAISMYIDTTCCA0mhggNFMIIDQTCCAikCAQEwggEJoYHhpIHeMIHbMQswCQYD
# VQQGEwJVUzETMBEGA1UECBMKV2FzaGluZ3RvbjEQMA4GA1UEBxMHUmVkbW9uZDEe
# MBwGA1UEChMVTWljcm9zb2Z0IENvcnBvcmF0aW9uMSUwIwYDVQQLExxNaWNyb3Nv
# ZnQgQW1lcmljYSBPcGVyYXRpb25zMScwJQYDVQQLEx5uU2hpZWxkIFRTUyBFU046
# QTUwMC0wNUUwLUQ5NDcxNTAzBgNVBAMTLE1pY3Jvc29mdCBQdWJsaWMgUlNBIFRp
# bWUgU3RhbXBpbmcgQXV0aG9yaXR5oiMKAQEwBwYFKw4DAhoDFQD/c/cpFSqQWYBe
# XggyRJ2ZbvYEEaBnMGWkYzBhMQswCQYDVQQGEwJVUzEeMBwGA1UEChMVTWljcm9z
# b2Z0IENvcnBvcmF0aW9uMTIwMAYDVQQDEylNaWNyb3NvZnQgUHVibGljIFJTQSBU
# aW1lc3RhbXBpbmcgQ0EgMjAyMDANBgkqhkiG9w0BAQsFAAIFAO5gOEMwIhgPMjAy
# NjA5MjUwMDAxMDdaGA8yMDI2MDkyNjAwMDEwN1owdDA6BgorBgEEAYRZCgQBMSww
# KjAKAgUA7mA4QwIBADAHAgEAAgIDMTAHAgEAAgISLzAKAgUA7mGJwwIBADA2Bgor
# BgEEAYRZCgQCMSgwJjAMBgorBgEEAYRZCgMCoAowCAIBAAIDB6EgoQowCAIBAAID
# AYagMA0GCSqGSIb3DQEBCwUAA4IBAQALiXcZVdCo4Ummo91iSxofmuKLHm7yND0w
# E8BwyIpeYoDkAV/fYp2+7JDfmzaKDMgE0PWluauoe1wsCgHyQilxeT7V/gcQ+tDc
# EYqVkpo6tqdFUxlhZ/6ZQSga38iK96bFLdPLG+cQMG+4dxtCgGtPCbZDherzBfS8
# 4ozwwcoLVUBivvnhXor+foUy0Ps3P06mavE6fwTuxvyO3ZcNLNd2DI1nxxKpNbUh
# JhVMEuLCvyN2bfvJEUsouaRmICLruWqkJ6cKZEE2OJ3jjlajVJM6xKXQGf0jvfbN
# ITWU50vxbTVIl8fwbwy7fU7XE1kcDP3PeF/XSVrcY2Pgw7gCbKdvMA0GCSqGSIb3
# DQEBAQUABIICAARiTOpIv3X+HjcjSxb3AZylrTGn8ESaHi6lmYo/fEBqY0U9bLAb
# laDSVN5P3wf5dLtU8eQQdvxIdQMEuSPnWhCYuEXlY/34teEjzEFLwZGdlqwzR089
# xRZOx6i7ZZFtzO+z6HLyiLO+jDFxsiBRGSWZcT3VEVd6xFxEOUtY6kOhbXwiefA2
# 4cCDDaD0O6JnFVxfJW4C1VbDGrFM7/sdh1LSNR08s8b6GDIMudPB8p/nQzZsbpS4
# jv5lCvUQIL4f/nx7FuYerrZxvX/q4PgQcEL1yGETvwW+qmR9hRxLpED8rc6TqHeU
# 7guDPFdQkKtzFUFFikuYlYFPO8I/sTOtENslU296DLCPVME1XqKa9E1lQOJSD6H+
# w4GyiTkt1HSSdXRp6rZuMNweXNBVaQ1vKRi06cNaYApty4Nn80GOctipziwJHsKJ
# 7ioGFu7xKJzFfh572mrqUtvMFlPz84F85H+XJE5eHarqqlbcvGCnXuq9Wh0a+Xtd
# 8ZX1g2Rk/oJP6sPcrtYvzZR+FdVhZuSKHwkfP5jXBjW2mruk+VXdToJRikgn9y3g
# HO7YC6AVAcAFaBRVMNz0PXx/lo9SFtT0CdOFQX2s2r3h1nCMVws4um9C2dwpWNfN
# Sh7ikBKjXCsKKX7hOG2nksJbMC0kgHJLcWGdjZtxMDx0hv668KD0LGPs
# SIG # End signature block
