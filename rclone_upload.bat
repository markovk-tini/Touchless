@echo off
echo === Stub installer upload start: %date% %time% ===
rclone copyto "c:\HGR App v1.0.0\release\Touchless_Installer.exe" r2:hgr-downloads/windows/v1.1.0/Touchless_Installer.exe --s3-upload-cutoff=100M --s3-chunk-size=100M
if errorlevel 1 ( echo === STUB UPLOAD FAILED === & exit /b 1 )
echo === Stub done: %date% %time% ===
echo === Payload zip upload start: %date% %time% ===
rclone copyto "c:\HGR App v1.0.0\release\Touchless_Payload_v1.1.0.zip" r2:hgr-downloads/windows/v1.1.0/Touchless_Payload_v1.1.0.zip --s3-upload-cutoff=100M --s3-chunk-size=100M
if errorlevel 1 ( echo === PAYLOAD UPLOAD FAILED === & exit /b 1 )
echo === Payload done: %date% %time% ===
