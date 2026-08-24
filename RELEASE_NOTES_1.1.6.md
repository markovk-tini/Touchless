Hotfix release. Focused on clip recording reliability across hardware encoder vendors and a settings-page scroll-wheel bug.

## Clips

- **Auto-recovery to libx264** when the preferred hardware encoder (NVENC / AMF / QSV) silently fails to produce segments. After 13 seconds with no segment files, the cache restarts on CPU encoding so clips keep working on Intel-only / AMD-only PCs and on systems where the GPU encoder probe passed but the production encode silently stalled.
- **NVENC scale-to-fit** for capture regions wider than 4096 pixels (multi-monitor / ultrawide setups). NVENC stays active at the downscaled resolution instead of forcing the whole capture to CPU.
- **NVENC preset compatibility** for pre-Turing GPUs — falls back from modern p1-p7 presets to legacy preset names that older drivers understand.
- Clip-cache ffmpeg stderr is now captured so the watchdog can name the actual encoder failure in the log instead of "(none captured)".
- Fixed `_clip_cache_dir` os-import bug that was crashing the cache every start on some installs.
- `LOCALAPPDATA` fallback for the clip cache directory when the temp path isn't writable.

## Settings UI

- **Scroll wheel no longer changes slider / dropdown values** on any settings page. Previously, scrolling down a long settings panel would accidentally adjust whichever slider or combo box the cursor passed over. Wheel events on QSlider / QComboBox / QSpinBox are now ignored unless the widget has explicit keyboard focus.

## Installer

- Installer hardening: the stub now closes any running Touchless instance before extracting, and the extract path has five layers of fallback so antivirus or path-encoding edge cases don't break a fresh install.
- `_locate_ffmpeg_executable` now searches `_internal/` so the bundled ffmpeg is discoverable on the current PyInstaller layout. Fixes clip recording failing on installs without ffmpeg in PATH.

## Auto-update

If you're on v1.0.6 or later, Touchless will offer to update itself on next launch. Click "Update now" and the app swaps to v1.1.6 automatically.

If you're on an older build, download a fresh installer from [touchless-control.com](https://touchless-control.com).
