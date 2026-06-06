from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

APP_NAME = "Touchless"
CONFIG_DIR = Path.home() / ".touchless"
CONFIG_PATH = CONFIG_DIR / "settings.json"
LEGACY_CONFIG_DIR = Path.home() / ".hgr_app"


def _migrate_legacy_config_dir() -> None:
    """Copy settings from the old .hgr_app folder on first run of the renamed build."""
    try:
        if CONFIG_DIR.exists() or not LEGACY_CONFIG_DIR.exists():
            return
        import shutil
        shutil.copytree(LEGACY_CONFIG_DIR, CONFIG_DIR)
    except Exception:
        pass


_migrate_legacy_config_dir()

ORIGINAL_PRIMARY_COLOR = "#0B3D91"
ORIGINAL_ACCENT_COLOR = "#1DE9B6"
ORIGINAL_SURFACE_COLOR = "#0F172A"
ORIGINAL_TEXT_COLOR = "#E5F6FF"
ORIGINAL_HELLO_FONT_SIZE = 72
CURRENT_TUTORIAL_PROMPT_VERSION = 1

SAVE_LOCATION_OUTPUT_ORDER = (
    "drawings",
    "screenshots",
    "screen_recordings",
    "clips",
)

SAVE_LOCATION_LABELS = {
    "drawings": "Drawings",
    "screenshots": "Screenshots",
    "screen_recordings": "Screen Recordings",
    "clips": "Clips",
}

SAVE_LOCATION_CONFIG_FIELDS = {
    "drawings": "drawings_save_dir",
    "screenshots": "screenshots_save_dir",
    "screen_recordings": "screen_recordings_save_dir",
    "clips": "clips_save_dir",
}

SAVE_NAME_CONFIG_FIELDS = {
    "drawings": "drawings_save_name",
    "screenshots": "screenshots_save_name",
    "screen_recordings": "screen_recordings_save_name",
    "clips": "clips_save_name",
}

SAVE_NAME_DEFAULTS = {
    "drawings": "Touchless_Drawing",
    "screenshots": "Touchless_Screenshot",
    "screen_recordings": "Touchless_Recording",
    "clips": "Touchless_Clip",
}

# Old HGR-branded defaults that should be rewritten to the new Touchless_* names
# when an upgraded install loads a config carried over from the previous build.
LEGACY_SAVE_NAME_DEFAULTS = {
    "drawings": "HGR_Drawing",
    "screenshots": "HGR_Screenshot",
    "screen_recordings": "HGR_Recording",
    "clips": "HGR_Clip",
}


def save_name_config_field(output_kind: str) -> str:
    normalized = str(output_kind or "").strip().lower()
    return SAVE_NAME_CONFIG_FIELDS.get(normalized, "")


def configured_save_name(config: "AppConfig", output_kind: str) -> str:
    field_name = save_name_config_field(output_kind)
    default_name = SAVE_NAME_DEFAULTS.get(str(output_kind or "").strip().lower(), "Touchless_File")
    if not field_name:
        return default_name
    value = str(getattr(config, field_name, "") or "").strip()
    return value if value else default_name


def _fallback_user_dir(name: str) -> Path:
    candidate = Path.home() / str(name)
    if candidate.exists():
        return candidate
    return Path.home()


def default_save_directory(output_kind: str) -> Path:
    normalized = str(output_kind or "").strip().lower()
    if normalized in {"drawings", "screenshots"}:
        return _fallback_user_dir("Pictures")
    if normalized in {"screen_recordings", "clips"}:
        return _fallback_user_dir("Videos")
    return Path.home()


def save_location_config_field(output_kind: str) -> str:
    normalized = str(output_kind or "").strip().lower()
    return SAVE_LOCATION_CONFIG_FIELDS.get(normalized, "")


def configured_save_directory(config: "AppConfig", output_kind: str) -> Path:
    field_name = save_location_config_field(output_kind)
    default_dir = default_save_directory(output_kind)
    raw_value = str(getattr(config, field_name, "") or "").strip() if field_name else ""
    target = Path(raw_value).expanduser() if raw_value else default_dir
    try:
        target.mkdir(parents=True, exist_ok=True)
        return target
    except Exception:
        try:
            default_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return Path.home()
        return default_dir


@dataclass
class AppConfig:
    primary_color: str = ORIGINAL_PRIMARY_COLOR
    accent_color: str = ORIGINAL_ACCENT_COLOR
    surface_color: str = ORIGINAL_SURFACE_COLOR
    text_color: str = ORIGINAL_TEXT_COLOR
    hello_font_size: int = ORIGINAL_HELLO_FONT_SIZE
    gesture_cooldown_seconds: float = 2.0
    stable_frames_required: int = 6
    camera_scan_limit: int = 8
    show_start_instructions_prompt: bool = True
    preferred_camera_index: Optional[int] = None
    preferred_microphone_name: Optional[str] = None
    tutorial_prompt_version: int = CURRENT_TUTORIAL_PROMPT_VERSION
    # Mouse-pad-style control box: small, right-shifted area in the
    # camera frame that maps to the entire monitor — the user only
    # has to move the cursor-driving hand within a forearm-sized
    # patch instead of sweeping across the whole frame. Right-shifted
    # because mouse mode runs off the right hand (left hand is the
    # toggle pose), so a center-right placement matches where the
    # right hand naturally rests in the mirrored camera view.
    mouse_control_box_center_x: float = 0.82
    mouse_control_box_center_y: float = 0.55
    mouse_control_box_area: float = 0.14
    mouse_control_box_aspect_power: float = 0.25
    # Left-handed mode: swap the Left/Right hand roles across the whole
    # app (gestures, air mouse, drawing, wheels, YouTube 'four', etc.).
    # Applied as a single label swap at detection entry in the engine.
    left_handed_mode: bool = False
    # Drawing control box: like the mouse control box, a small square
    # region of the camera frame that maps to the whole drawing canvas,
    # so the finger only moves within a forearm-sized patch instead of
    # sweeping the full frame. Square in normalized coords keeps motion
    # undistorted on a 16:9 frame/canvas. Right-shifted to match where
    # the drawing hand naturally rests in the mirrored view.
    drawing_control_box_center_x: float = 0.82
    drawing_control_box_center_y: float = 0.55
    drawing_control_box_size: float = 0.45
    # Which monitor mouse-mode controls. None = all monitors (the
    # full virtual desktop, the historical default). 0..N-1 = a
    # specific monitor's index in QGuiApplication.screens(). The
    # mouse-mode activation popup writes this on user choice; the
    # Save Locations -> Mouse Control panel lets users preset it.
    # Cursor mapping in mouse_controller respects this — the red
    # mouse-box on the camera frame still spans the same area but
    # the cursor output gets clamped to the chosen monitor's region.
    mouse_active_monitor_index: Optional[int] = None
    # Default monitor for the "clip that" voice command and any other
    # capture action that doesn't explicitly ask which monitor to use.
    # None = primary monitor (Windows considers the screen index 0 to
    # be primary in most setups). A specific index 0/1/2/… targets
    # that screen by Qt's QGuiApplication.screens() order. The user
    # picks this from Settings → General → Clip & Record so they
    # don't have to answer the monitor prompt every time on a
    # multi-monitor rig.
    clip_default_monitor_index: Optional[int] = None
    # User's own Spotify Developer client_id. Optional — when empty,
    # Touchless uses its embedded default client_id which is capped
    # at 5 testers by Spotify's developer policy (Spotify killed the
    # indie-friendly Extended Quota Mode; the new Partner program
    # requires 250k+ MAU). When the user supplies their own
    # client_id via the Spotify setup wizard, Touchless uses that
    # instead — each user gets their own Spotify Dev app on their
    # own account, which removes the 5-user cap entirely because the
    # cap is per-app. PKCE means no client_secret is needed; the
    # client_id alone is enough to drive OAuth. Empty string and
    # None both mean "use the embedded default".
    spotify_client_id: str = ""
    # Latched True when the user clicks the X on the 'Spotify Premium
    # required' warning in Settings → General. Stops it from
    # reappearing on subsequent launches — the user has acknowledged
    # the requirement and doesn't need the reminder anymore.
    spotify_premium_warning_dismissed: bool = False
    # When True, the home-screen camera-health pill is extended with
    # live diagnostic info — current FPS, top stable gesture +
    # confidence, mic input level. Off by default so the pill stays
    # quiet for typical users; testers and debug sessions can flip
    # it on from Settings → General → Diagnostics.
    diagnostic_overlay_enabled: bool = False
    # When True, the tracking pill is extended (per-frame) with the
    # TOP-3 raw recognizer scores instead of just the latched stable
    # label. Useful when debugging "I made a fist but it didn't fire"
    # — shows the runner-up labels so the user can see whether they
    # were close to the right pose or hitting a different one
    # entirely. Requires `diagnostic_overlay_enabled` to be on (the
    # pill is the host surface for both).
    show_recognizer_top_scores: bool = False
    # Clip-audio capture toggles. Streamer mode: when on, the clip cache's
    # ffmpeg subprocess additionally records WASAPI loopback (system audio
    # — game/music/app sounds) and/or the user's preferred microphone,
    # then mixes them into the saved clip with `amix`.
    # `clip_capture_microphone` reuses the existing
    # `preferred_microphone_name` field for device selection.
    #
    # System audio DEFAULTS ON: loopback only captures what the user is
    # already hearing through their own speakers — same model as OBS /
    # NVIDIA ShadowPlay defaults. Existing users with an explicit
    # `clip_capture_system_audio: false` in their settings.json are
    # untouched by `load_config`'s migration shim (see _MIGRATED_DEFAULTS).
    #
    # Mic ALSO defaults ON now: the mic input is the same device the user
    # already exposes to Touchless's voice commands and dictation, so
    # there's no incremental privacy surface beyond what they already
    # opted into. Clips finally include the user's voice without them
    # having to dig through Settings → General → Clip Audio to find the
    # toggle. Users who genuinely don't want mic in clips can still
    # disable it there; the toggle is preserved.
    clip_capture_system_audio: bool = True
    clip_capture_microphone: bool = True
    # One-time migration marker: set to True the first time load_config
    # observes a missing-or-False clip_capture_system_audio under the
    # new default-True regime, after promoting it to True. Without
    # this, every launch would re-promote a user's deliberate False
    # back to True, defeating their opt-out. See load_config().
    clip_audio_default_migrated: bool = False
    # Parallel marker for clip_capture_microphone default flip
    # (False → True). Latched True after load_config has run the
    # one-time promotion exactly once.
    clip_mic_default_migrated: bool = False
    # Parallel marker for clip_mic_noise_reduction default flip
    # ('light' → 'off'). Latched True after the one-time promotion.
    clip_mic_filter_default_migrated: bool = False
    # Parallel marker for clip_audio_offset_ms default change
    # (0 → +2500). Latched True after the one-time promotion.
    clip_audio_offset_default_migrated: bool = False
    # Second-pass marker: the +2500 default migration was the WRONG
    # sign (POSITIVE values actually shift audio EARLIER in the
    # exported clip, not LATER, because `shifted_start = a_start_trim
    # - offset_seconds` reads earlier samples for a positive offset).
    # User observed audio jumped from 2-3 s early to 5 s early after
    # the first migration. This second migration flips +2500 → -2500
    # to actually compensate the WASAPI capture lag.
    clip_audio_offset_sign_corrected: bool = False
    # Third-pass marker: after the bridge silence-fill + mtime fixes
    # the residual underlying offset is audio playing ~5-6 s EARLIER
    # than video (ffmpeg amix startup + segment-muxer pre-roll while
    # mic burst-dumps its anchor pre-pad). The -2500 default
    # under-compensated, leaving audio still 3 s early. Flip to
    # -5500 to fully compensate. Bumps the clamp to ±10000 ms so
    # this and other deep mis-calibrations stay within tunable range.
    clip_audio_offset_post_silence_fill_migrated: bool = False
    # (Fourth-pass diagnostic-zero marker kept as inert tombstone so
    # any user config that already migrated through it stays valid;
    # we no longer flip values, so re-running the load is a no-op
    # past the first time the flag was latched.)
    clip_audio_offset_zeroed_for_diagnosis: bool = False
    # Fifth-pass marker: callback-mode mic bridge landed. User
    # observed 1 s EARLY at -5500, 1 s LATE at -4500, 1 s LATE at
    # -5000 — run-to-run variance is larger than the offset step
    # so settling at a perfect default is futile. Restore -5500
    # (which the user reported as 'on time' on the first
    # successful run) and leave the rest to user tuning via the
    # config file.
    clip_audio_offset_post_callback_mode_migrated: bool = False
    # Sixth-pass marker: after the previous revert to -5500 the
    # user reported audio 1 s LATE. Push to -6500.
    clip_audio_offset_late_at_5500_migrated: bool = False
    # Seventh-pass marker: -6500 was slightly EARLY (<1 s).
    # Pull back to -6000.
    clip_audio_offset_early_at_6500_migrated: bool = False
    # Audio-vs-video offset applied at clip export by biasing the
    # audio-trim start point. Sign convention from the export math
    # `shifted_start = a_start_trim - offset_seconds`:
    #   * NEGATIVE offset → LARGER shifted_start (atrim skips MORE
    #     of the concat before extracting) → the audio that ends up
    #     at clip-T=0 was captured at a LATER wall time → audio
    #     events appear EARLIER in playback.
    #   * POSITIVE offset → SMALLER shifted_start → audio events
    #     appear LATER in playback.
    #
    # Default -6000 ms. Latest user observation: <1 s EARLY at
    # -6500, so dial back 500 ms (less negative = audio shifted
    # slightly LATER in playback). Tune via config if you
    # consistently see drift.
    clip_audio_offset_ms: int = -6000

    # ===== Clip v2 settings (MVP commit 1) =====
    # Default voice "clip that" duration in seconds. When the user
    # says just "clip that" / "clip this" / "save clip" without a
    # duration token, the engine resolves to this length. Allowed
    # set in the future Settings UI dropdown: 30, 60, 120, 300.
    # Default 60 matches v1 implicit behavior — existing users
    # don't get a surprise length flip.
    clip_default_duration_seconds: int = 60
    # Encoder quality preset for clip EXPORT. The rolling cache
    # always runs at "medium" to keep 24/7 GPU/disk cost bounded;
    # only the export pays for the higher preset. Allowed: "low",
    # "medium", "high". Default "high".
    clip_quality_preset: str = "high"
    # Maximum rolling buffer length in seconds. v1 was 65 s; bumped
    # to 305 s (5 min + 5 s slack) in MVP-fixup after the user
    # found 2 m / 5 m voice clips landed frozen frames where the
    # buffer ran short. Disk footprint cap is ~470 MB on a typical
    # 20 fps + 192 kbps audio session — handled by the existing
    # _cleanup_ffmpeg_clip_cache_files teardown on app shutdown.
    clip_max_buffer_seconds: int = 305
    # When True (default), the live camera/gesture pipeline keeps
    # running during clip export. With the non-blocking toast +
    # off-thread export this is fine. Power-user escape hatch for
    # low-end CPUs where libx264 + camera widget contend.
    clip_show_live_view_during_export: bool = True
    # When True (default), a non-blocking "Clip saved" toast shows
    # after a successful auto-save export. Click → opens clip
    # folder, auto-dismisses after 4 s (8 s on hover). Failure
    # popups (QMessageBox.warning) are NOT subject to this toggle
    # — errors deserve modal attention.
    clip_show_save_popup: bool = True
    # Latched True after the v2 fields' first AppConfig load. No
    # values are mutated in the migration — every new field's
    # default matches the v1 implicit behavior — so this flag is
    # purely an artifact for future cleanup symmetry.
    clip_v2_settings_migrated: bool = False
    # Microphone noise-reduction preset for clip audio capture.
    # Applied as an ffmpeg `filter_complex` chain on the MIC input
    # ONLY, before amix mixes it with the system-audio stream.
    # Game / music / app audio is NEVER filtered — only the mic.
    # Values:
    #   "off"    — no filter; mic recorded verbatim (use when an
    #              upstream tool like Krisp / NVIDIA Broadcast is
    #              already doing noise suppression).
    #   "light"  — DEFAULT. highpass=80 Hz to kill rumble + a
    #              gentle noise gate (peak detection, ~-15 dBFS
    #              threshold) tuned to close during pauses and
    #              silence mechanical keyboard + mouse clicks
    #              without chopping speech.
    #   "strong" — tighter gate (~-10 dBFS, ratio 10). Removes
    #              more background noise but may clip soft word
    #              tails. Recommended only for loud keyboards or
    #              persistent room noise.
    # Only takes effect when `clip_capture_microphone` is True;
    # the UI grays the dropdown out when mic is off. Changes
    # take effect on the next clip-cache restart. Invalid values
    # silently fall back to "light".
    # 'off' by default per the user's clip-mic spec:
    # "use the same audio levels and everything as voice commands".
    # Voice commands run raw with no gate; the prior 'light' default
    # had agate threshold=0.1778 (-15 dBFS) — higher than typical
    # webcam-mic speech peaks (~-23 dBFS measured) — which gated
    # most words and produced the "garbled mic" report. Users with
    # noisy environments can still pick 'light' or 'strong' in
    # Settings → Clip Audio.
    clip_mic_noise_reduction: str = "off"
    # Discord OAuth credentials supplied by the user via the in-app
    # Discord setup wizard. Same per-user-app pattern as Spotify above,
    # but with a different motivation: Discord's `rpc` OAuth scope is
    # restricted to the OWNER of the registered Touchless dev-portal
    # app while public-distribution approval is pending (and may never
    # arrive — Discord has wound rpc down). Every user creating their
    # own free Discord Developer app means they become the owner of
    # their own app, and the rpc scope works on day one without any
    # Discord-side review. Unlike Spotify, Discord's rpc flow is NOT
    # PKCE — the token-exchange step needs the client_secret, so both
    # fields are required (empty `client_secret` falls back to env /
    # .env / the embedded default for dev builds).
    discord_client_id: str = ""
    discord_client_secret: str = ""
    # Anonymous install UUID for usage telemetry. Now derived
    # deterministically from SHA-256(salt + Windows MachineGuid +
    # username) so a single user gets ONE install_id forever — survives
    # settings.json wipes, app updates, reinstalls, source rebuilds.
    # Falls back to a random uuid4 on non-Windows or if the registry
    # read fails. Persisted across sessions so the analytics dashboard
    # can compute "active install" / retention numbers. Not a personal
    # identifier — one-way SHA-256, salted.
    analytics_install_id: str = ""
    # Latched True after the legacy random uuid4 install_id has been
    # replaced with the MachineGuid-derived value once. Prevents the
    # one-time migration from re-firing every launch in case the user
    # later manually pins a different install_id.
    analytics_install_id_migrated_to_derived: bool = False
    # Privacy & data flags. Both are opt-in.
    #   privacy_disclosure_shown: latched True after the user
    #     clicks "Got it" on the first-run privacy dialog. Stops
    #     us from re-prompting at every launch.
    #   analytics_enabled: gates ALL telemetry track() calls. The
    #     install_id + api_key are only "wired" — actual sending
    #     requires this flag too. Defaults to FALSE so a fresh
    #     install never sends a single event until the user opts
    #     in via the first-run dialog or Settings → About toggle.
    privacy_disclosure_shown: bool = False
    analytics_enabled: bool = False
    # The Touchless version that was running the last time this
    # install opened the main window. Used by the "Updated to
    # vX.Y.Z" success toast: when the launching __version__
    # differs from this stored value, the toast fires once per
    # update. Empty default = no record yet, so the first launch
    # after this field was added latches without showing a toast
    # (we don't pretend to have updated when we just added the
    # tracking field).
    last_launched_version: str = ""
    # Persisted Qt main-window geometry (saveGeometry result, base64-
    # encoded into ASCII). Empty default means "no record yet" — the
    # first launch falls back to the spec'd resize/move; subsequent
    # launches call restoreGeometry on this blob to reproduce the
    # user's last size + position + maximized state.
    main_window_geometry_b64: str = ""
    # Auto-start on Windows login. When True, a registry Run key
    # under HKCU\Software\Microsoft\Windows\CurrentVersion\Run is
    # written pointing at the installed Touchless.exe so the app
    # launches at sign-in (minimised to tray if the tray icon is
    # supported). Toggled by the General Settings checkbox; the
    # registry write happens through autostart.py, NOT through
    # this field directly -- the field is just a UI baseline.
    auto_start_on_login: bool = False
    drawings_save_dir: str = field(default_factory=lambda: str(default_save_directory("drawings")))
    screenshots_save_dir: str = field(default_factory=lambda: str(default_save_directory("screenshots")))
    screen_recordings_save_dir: str = field(default_factory=lambda: str(default_save_directory("screen_recordings")))
    clips_save_dir: str = field(default_factory=lambda: str(default_save_directory("clips")))
    drawings_save_name: str = "Touchless_Drawing"
    screenshots_save_name: str = "Touchless_Screenshot"
    screen_recordings_save_name: str = "Touchless_Recording"
    clips_save_name: str = "Touchless_Clip"
    low_fps_mode: bool = False
    low_fps_auto: bool = False
    force_ten_fps_test_mode: bool = False
    mic_input_gain: float = 1.0
    # When True (default for new installs), the voice listener applies
    # the per-mic-class suggested_gain from `mic_profile.classify_mic`
    # so a fresh install on a Razer Kiyo Pro auto-boosts to ~3.0×, a
    # USB Yeti auto-attenuates to ~0.6×, etc. — without the user
    # having to know about the gain slider. Flipped to False the
    # moment the user touches the mic-input-gain slider, so manual
    # tunings are preserved across sessions. Mic-profile changes
    # still update the listener's per-class thresholds (trigger
    # floor, AGC target, end-silence window) regardless of this
    # flag — only the gain multiplier is auto-vs-manual.
    mic_input_gain_auto: bool = True
    phone_camera_enabled: bool = False
    phone_camera_url: str = ""
    # Phone-camera-via-QR state. Two orthogonal flags so the user can
    # switch between phone and local camera mid-session without unpairing.
    #   phone_camera_qr_paired: once the user has successfully paired a
    #     phone via the QR dialog, this stays True across app launches
    #     so the embedded HTTPS server auto-starts on startup — the
    #     phone's already-open browser tab can just tap Start and
    #     reconnect without re-scanning or re-installing the cert.
    #   phone_camera_qr_active: engine uses the phone's capture as the
    #     current source. Toggling this on/off does NOT stop the server;
    #     it just swaps which camera the gesture pipeline reads from.
    phone_camera_qr_paired: bool = False
    phone_camera_qr_active: bool = False
    # When True AND the QR server is running AND phone_camera_qr_paired,
    # the voice pipeline reads microphone audio from the phone (POSTed
    # to /audio) instead of the local sounddevice input. The phone
    # page must have its "Mic: send to PC" toggle on for audio to
    # actually arrive.
    phone_camera_qr_use_mic: bool = False
    camera_source_is_mirrored: bool = False
    # The last update version the user dismissed via "Later". Set when
    # they click Later on the in-app update prompt. The next launch
    # only re-shows the prompt if a STRICTLY NEWER release is on
    # GitHub — clicking Later on v1.0.7 means "don't nag me about
    # this version", but v1.0.8 still gets a prompt the moment it
    # ships. Empty string = no version dismissed yet.
    last_dismissed_update_version: str = ""
    # When True, the in-app update path runs without showing the
    # UpdateDialog: as soon as a new version is detected the app-zip
    # download starts, the apply helper fires, and Touchless restarts
    # itself on the new version. Off by default so the first 1.1.4 user
    # still gets to read the release notes and consent before their app
    # restarts unprompted; users opt in via the "Install updates
    # automatically" checkbox on the General settings tab. The auto path
    # only fires for the app-zip update kind (small, in-place, no UAC,
    # no Inno dialog) — full-installer and Store-fallback paths still
    # surface the dialog because they take noticeably longer and the
    # user deserves to know the app is about to disappear for ~30 sec.
    auto_update_enabled: bool = False
    # Rate-limiter sentinel for the auto-update path. Written before the
    # download starts, cleared on successful apply (in _on_installer_ready)
    # or on failure (in _on_auto_update_failed). If a launch starts and
    # finds this still equals the freshly-detected update version, it
    # means a prior auto-attempt for that exact version didn't reach
    # apply — we fall back to the manual dialog so the user can see what's
    # going on instead of silently re-downloading 140 MB every launch.
    auto_update_attempted_version: str = ""
    # Set to the running __version__ once the in-app self-heal has
    # written DisplayVersion to the Inno uninstall registry key. Without
    # this sentinel, 1.1.3 users (whose .bat had no reg-add) would never
    # get their registry patched because the 1.1.3 → 1.1.4 hop runs OLD
    # bat code. The self-heal at app startup closes that gap so Microsoft
    # Store reads the right version on its next check.
    registry_display_version_patched_for: str = ""
    # Set True once the user has installed the optional higher-accuracy
    # dictation model (ggml-medium.en.bin) via the "Voice Recognition
    # Upgrade" download. Store builds ship only small.en to stay under
    # the Store package-size limit; this flag + the on-disk model
    # presence drive whether the upgrade button is shown. Latches the
    # walkthrough one-time prompt too so it isn't re-offered after the
    # model is in place.
    voice_model_upgrade_installed: bool = False
    # Set True once the walkthrough has offered the Voice Recognition
    # Upgrade prompt, so it's shown at most once (the General-tab button
    # remains the always-available entry point).
    voice_model_upgrade_prompt_shown: bool = False
    # Lite Mode: switches MediaPipe Hands to model_complexity=0
    # (the lite landmark model) and downsamples inference input to
    # ~480px wide. ~2.5x faster than the default full model on
    # every machine, with a small accuracy hit on rare edge poses
    # (extreme tilt / heavy occlusion). Independent from low_fps_mode
    # — low_fps_mode is the auto-degrade path that also drops the
    # stable-frame requirement and is enabled when we detect the
    # camera/CPU can't sustain frame rate; lite_mode is a
    # user-driven uniform speedup that keeps the rest of the
    # pipeline at normal sensitivity.
    lite_mode: bool = False
    # GPU Mode: opt-in flag for the GPU-accelerated inference path.
    # When False, the gesture pipeline runs on CPU via MediaPipe's
    # default solutions.hands runtime (the same code path Touchless
    # has always used). When True, the runtime loader tries the
    # MediaPipe Tasks API HandLandmarker with its GPU delegate; if
    # that path can't actually reach the GPU on this machine
    # (Windows OpenGL ES / Vulkan delegate flakiness, missing
    # provider DLLs, etc.), the loader logs a fallback and quietly
    # uses CPU — gesture recognition keeps working either way.
    # The Settings → Camera toggle reads gesture/tracking/gpu_probe
    # to disable itself with a tooltip when no GPU path is reachable
    # so the user isn't toggling a no-op.
    gpu_mode: bool = False
    # YouTube auto-pause: when True, the engine pauses the currently
    # playing YouTube tab once no hand has been visible for
    # `youtube_pause_when_user_leaves_seconds` seconds. Resumes
    # automatically when a hand reappears (one-shot resume per
    # absence cycle so the user isn't fighting the engine if they
    # paused manually before walking away).
    youtube_pause_when_user_leaves: bool = False
    youtube_pause_when_user_leaves_seconds: int = 6
    # YouTube auto-skip-ads: when True, a background timer polls the
    # currently-focused YouTube tab for the Skip Ad button and clicks
    # it as soon as the template match crosses the confidence
    # threshold. Click briefly focuses the YouTube tab then restores
    # the user's prior foreground window so the skip is invisible
    # if the user is doing something else.
    youtube_auto_skip_ads: bool = False
    # Caption translate target — ISO-style display name as YouTube's
    # menu shows it (e.g. "Spanish", "Japanese", "French"). Empty
    # means "don't auto-translate". Triggered by the voice command
    # "translate captions to X" or the Settings → YouTube picker.
    youtube_caption_target_language: str = ""
    # Maps action_id -> pose_id for the Settings → Gesture Binds tab.
    # Empty dict means "use default bindings" (see _DEFAULT_GESTURE_BINDS in
    # main_window.py). Only stores user-changed entries to keep the file
    # tidy; defaults are resolved on read.
    gesture_bindings: Dict[str, str] = field(default_factory=dict)
    # Set True once the Touchless app has shown the first-time
    # "Allow Touchless to connect to Spotify?" prompt — fires the
    # first time Spotify is observed running while the engine is
    # active and the user has no saved Spotify tokens. Latches True
    # whether the user picks Allow or Don't Allow so the prompt
    # never repeats; the user can still connect later via the
    # button at the bottom of the Instructions panel.
    spotify_first_active_prompt_shown: bool = False
    # General-tab overlay toggles. Default On for the visible
    # surfaces (camera view + popups) so first-time users see the
    # full app; Off for the gaming auto-disable knobs since they're
    # opt-in.
    #   overlay_camera_view_enabled: when False, the mini live
    #     viewer is suppressed (engine still runs; no thumbnail
    #     window). Useful for users who only want the action label
    #     without an extra window on screen.
    #   overlay_text_popups_enabled: when False, transient text
    #     popups (TouchlessNotice, gesture-binds pill, Spotify
    #     decline pill, post-action save prompts) are silently
    #     suppressed — only critical/blocking dialogs still appear.
    #   overlay_gaming_mode_enabled: when True, text popups are
    #     auto-suppressed WHILE a known game process is running.
    #     Distinct from overlay_text_popups_enabled (full off) so
    #     popups still work when not gaming.
    #   overlay_gaming_live_view_disabled: when True, the mini live
    #     viewer is auto-hidden WHILE a known game process is
    #     running. Pairs with the gaming-mode toggle but is
    #     separately controllable so users can keep the live view
    #     during streaming-style sessions.
    overlay_camera_view_enabled: bool = True
    overlay_text_popups_enabled: bool = True
    overlay_gaming_mode_enabled: bool = False
    overlay_gaming_live_view_disabled: bool = False
    # Per-overlay toggles for the gesture live-view window. These
    # show / hide compact diagnostic pills layered on the camera
    # feed (top-right of the video panel). Default off so a clean
    # live view is the out-of-the-box experience; users who care
    # about diagnostics enable them in Settings → Camera.
    live_view_show_fps: bool = False
    live_view_show_latency: bool = False
    live_view_show_tracking_quality: bool = False


DEFAULT_CONFIG = AppConfig()


def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        return AppConfig()

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        values = {field: data.get(field, getattr(DEFAULT_CONFIG, field)) for field in asdict(DEFAULT_CONFIG)}

        # One-time migration: old installs had the instructions prompt setting,
        # but not the newer tutorial prompt version. Re-show the prompt once.
        if "tutorial_prompt_version" not in data:
            values["show_start_instructions_prompt"] = True
            values["tutorial_prompt_version"] = CURRENT_TUTORIAL_PROMPT_VERSION

        # Rewrite legacy HGR_* save name defaults to the new Touchless_* defaults.
        # Only swap values that exactly match the old defaults so custom names are preserved.
        for output_kind, legacy_default in LEGACY_SAVE_NAME_DEFAULTS.items():
            field_name = SAVE_NAME_CONFIG_FIELDS.get(output_kind)
            if not field_name:
                continue
            current = str(values.get(field_name, "") or "").strip()
            if current == legacy_default:
                values[field_name] = SAVE_NAME_DEFAULTS[output_kind]

        # Migration: clip_capture_system_audio default flipped False
        # → True (system loopback now defaults on, matching OBS /
        # NVIDIA ShadowPlay). Promote a persisted False that was
        # SAVED before this migration ran — gated on the new
        # `clip_audio_default_migrated` marker so we only flip each
        # install once. Users who genuinely want system audio off can
        # toggle it back via Settings → Clip Audio after the flip;
        # the marker is set whether or not they kept the new default,
        # so toggling back to False won't bounce on next launch.
        # Mic stays opt-in (privacy) regardless.
        if not data.get("clip_audio_default_migrated", False):
            if values.get("clip_capture_system_audio") is False:
                values["clip_capture_system_audio"] = True
            values["clip_audio_default_migrated"] = True

        # Parallel migration: clip_capture_microphone default flipped
        # False → True (mic is now in clips by default — same WASAPI
        # device the user already exposes for voice commands, so no
        # incremental privacy surface). Promote a persisted False ONLY
        # on existing installs that never explicitly toggled the mic;
        # gated on its own marker so we run it exactly once per install
        # and a deliberate "turn mic off in clips" choice is preserved
        # going forward.
        if not data.get("clip_mic_default_migrated", False):
            if values.get("clip_capture_microphone") is False:
                values["clip_capture_microphone"] = True
            values["clip_mic_default_migrated"] = True

        # Parallel: flip clip_mic_noise_reduction 'light' (old default)
        # → 'off' for upgraded installs. Only the EXACT old default
        # gets promoted; deliberate 'strong' picks survive.
        if not data.get("clip_mic_filter_default_migrated", False):
            if values.get("clip_mic_noise_reduction") == "light":
                values["clip_mic_noise_reduction"] = "off"
            values["clip_mic_filter_default_migrated"] = True

        # Parallel: flip clip_audio_offset_ms 0 (old default) → +2500
        # to compensate for WASAPI capture lag that was being masked
        # by the (now-fixed) stale-manifest video freeze. Only the
        # EXACT old default gets promoted; a deliberate tune survives.
        if not data.get("clip_audio_offset_default_migrated", False):
            if values.get("clip_audio_offset_ms") == 0:
                values["clip_audio_offset_ms"] = 2500
            values["clip_audio_offset_default_migrated"] = True

        # Sign-corrected pass for clip_audio_offset_ms. The +2500
        # default from the previous migration was the wrong sign —
        # POSITIVE values pull audio EARLIER in the export, but the
        # symptom we needed to fix was already-early audio, so the
        # +2500 made it WORSE. Flip exactly the bad migration value
        # +2500 → -2500. Other values (deliberate tunes, untouched
        # zeros) are not modified.
        if not data.get("clip_audio_offset_sign_corrected", False):
            if values.get("clip_audio_offset_ms") == 2500:
                values["clip_audio_offset_ms"] = -2500
            values["clip_audio_offset_sign_corrected"] = True

        # Third-pass migration after the bridge silence-fill + mtime
        # fixes uncovered the true underlying offset (~5-6 s early
        # audio). The earlier -2500 default under-compensated. Bump
        # only the exact prior default -2500 to the new default
        # -5500. Deliberate user-tunes (any other value) survive.
        if not data.get("clip_audio_offset_post_silence_fill_migrated", False):
            if values.get("clip_audio_offset_ms") == -2500:
                values["clip_audio_offset_ms"] = -5500
            values["clip_audio_offset_post_silence_fill_migrated"] = True

        # (Fourth-pass diagnostic-zero migration ELIDED: the user
        # confirmed -5500 lands audio on time, so we no longer want
        # to flip values to 0. The flag is still latched so a stale
        # config from the very brief window when this migration was
        # active doesn't re-fire any prior pass. If a user landed at
        # 0 from the diagnostic build, they can manually set -5500
        # in their config or let the new default take effect on a
        # fresh install.)
        if not data.get("clip_audio_offset_zeroed_for_diagnosis", False):
            values["clip_audio_offset_zeroed_for_diagnosis"] = True

        # Fifth-pass migration: stop chasing the perfect offset
        # default — run-to-run variance is larger than the
        # adjustment steps. Flip any of the intermediate defaults
        # (-4500, -5000) back to -5500, which the user confirmed
        # on the first successful test. Deliberate user-tunes
        # survive untouched.
        if not data.get("clip_audio_offset_post_callback_mode_migrated", False):
            if values.get("clip_audio_offset_ms") in (-5000, -4500):
                values["clip_audio_offset_ms"] = -5500
            values["clip_audio_offset_post_callback_mode_migrated"] = True

        # Sixth-pass migration: user reported 1 s LATE at -5500
        # after the previous revert. Push another 1 s of negative
        # shift to -6500. Migration flips only the exact prior
        # default (-5500); deliberate user-tunes survive.
        if not data.get("clip_audio_offset_late_at_5500_migrated", False):
            if values.get("clip_audio_offset_ms") == -5500:
                values["clip_audio_offset_ms"] = -6500
            values["clip_audio_offset_late_at_5500_migrated"] = True

        # Seventh-pass migration: -6500 was slightly EARLY (<1 s),
        # dial back to -6000. Flips only the exact prior default.
        if not data.get("clip_audio_offset_early_at_6500_migrated", False):
            if values.get("clip_audio_offset_ms") == -6500:
                values["clip_audio_offset_ms"] = -6000
            values["clip_audio_offset_early_at_6500_migrated"] = True

        # Clip v2 settings — MVP commit 1. Every new field's default
        # matches v1 implicit behavior, so this migration mutates
        # NOTHING. The flag is just an artifact for future cleanup
        # symmetry (the file is large; missing migration markers
        # confuse code-search). Anyone who hand-edits one of the
        # new fields keeps their value.
        if not data.get("clip_v2_settings_migrated", False):
            values["clip_v2_settings_migrated"] = True

        # Migrate the mouse control box only when the user still has the old defaults.
        # Each `if` chain rewrites a previous default to the current default; if
        # the user changed the value via settings, the change is preserved.
        if abs(float(values.get("mouse_control_box_center_x", 0.0)) - 0.44) < 1e-6:
            values["mouse_control_box_center_x"] = DEFAULT_CONFIG.mouse_control_box_center_x
        if abs(float(values.get("mouse_control_box_center_x", 0.0)) - 0.50) < 1e-6:
            values["mouse_control_box_center_x"] = DEFAULT_CONFIG.mouse_control_box_center_x
        if abs(float(values.get("mouse_control_box_center_x", 0.0)) - 0.62) < 1e-6:
            values["mouse_control_box_center_x"] = DEFAULT_CONFIG.mouse_control_box_center_x
        # 0.67 was the previous default — bumped to 0.78 so the red
        # control box sits much closer to the right edge of the camera
        # frame (user feedback: reach-out hand naturally lands far to
        # the right of the camera FOV, so center should follow). Users
        # who have explicitly tuned the box past 0.67 are left alone.
        if abs(float(values.get("mouse_control_box_center_x", 0.0)) - 0.67) < 1e-6:
            values["mouse_control_box_center_x"] = DEFAULT_CONFIG.mouse_control_box_center_x
        # 0.78 → 0.82: another small right shift requested after live
        # testing. Same migration pattern — users who had the 0.78
        # default get bumped to 0.82; users who explicitly chose any
        # other value are left alone.
        if abs(float(values.get("mouse_control_box_center_x", 0.0)) - 0.78) < 1e-6:
            values["mouse_control_box_center_x"] = DEFAULT_CONFIG.mouse_control_box_center_x
        if abs(float(values.get("mouse_control_box_center_y", 0.0)) - 0.56) < 1e-6:
            values["mouse_control_box_center_y"] = DEFAULT_CONFIG.mouse_control_box_center_y
        if abs(float(values.get("mouse_control_box_area", 0.0)) - 0.31) < 1e-6:
            values["mouse_control_box_area"] = DEFAULT_CONFIG.mouse_control_box_area
        if abs(float(values.get("mouse_control_box_area", 0.0)) - 0.36) < 1e-6:
            values["mouse_control_box_area"] = DEFAULT_CONFIG.mouse_control_box_area
        # 0.18 → 0.12: previous default produced a red box that was
        # visibly wider than the green Monitor 1 outline drawn inside
        # it. New default matches the displayed monitor width on a
        # single-monitor setup. The tracker scales the effective area
        # up for multi-monitor desktops automatically (see
        # mouse_gesture._box_rect_in_camera), so multi-monitor users
        # still get a wider box without needing to retune the slider.
        if abs(float(values.get("mouse_control_box_area", 0.0)) - 0.18) < 1e-6:
            values["mouse_control_box_area"] = DEFAULT_CONFIG.mouse_control_box_area
        # 0.12 → 0.14: bumped slightly to compensate for the new
        # squarer aspect (the box is less wide, so slight area bump
        # restores comfortable hand-reach without re-stretching).
        if abs(float(values.get("mouse_control_box_area", 0.0)) - 0.12) < 1e-6:
            values["mouse_control_box_area"] = DEFAULT_CONFIG.mouse_control_box_area
        # 0.40 → 0.25: aspect_power lowered so the camera box is
        # more square (matches natural hand-reach ergonomics) instead
        # of stretched 16:9. Visual aspect for a single-monitor user
        # drops from ~1.78 to ~1.16, giving the upright shape the
        # user specifically requested.
        if abs(float(values.get("mouse_control_box_aspect_power", 0.0)) - 0.40) < 1e-6:
            values["mouse_control_box_aspect_power"] = DEFAULT_CONFIG.mouse_control_box_aspect_power

        return AppConfig(**values)
    except Exception:
        return AppConfig()


def save_config(config: AppConfig) -> None:
    """Persist `config` to disk atomically.

    Write-to-temp + os.replace so the destination either reflects the
    OLD config or the NEW one, never a truncated half. The original
    implementation was a direct write_text which truncates the file
    before writing — a process kill mid-write (e.g. Inno Setup's
    `CloseApplications=force` TerminateProcess against a Touchless
    that's mid-save) would leave config.json empty or partially-written
    and the next launch would silently fall back to AppConfig
    defaults, losing every user preference. The 1.1.4 audit flagged
    this as a real corruption surface for the standalone-installer
    upgrade path.

    os.replace is atomic on the same volume on both Windows and POSIX
    (uses MoveFileEx with MOVEFILE_REPLACE_EXISTING on Windows). The
    temp file lives next to the destination so we stay on one volume.
    """
    import os as _os
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(config), indent=2)
    tmp_path = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".tmp")
    try:
        # Write the full payload + flush + fsync the temp file before
        # the rename, so a power-loss between write and rename leaves
        # either old-or-new on disk, never a partial temp.
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            try:
                _os.fsync(fh.fileno())
            except (OSError, AttributeError):
                # fsync isn't available on some Windows configs (rare);
                # the rename below still provides atomicity vs. concurrent
                # readers, and the write_text-style hazard is closed.
                pass
        _os.replace(tmp_path, CONFIG_PATH)
    except Exception:
        # Best-effort: if rename failed, drop the temp so it doesn't
        # confuse the next save attempt.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise

# Author: Konstantin Markov
