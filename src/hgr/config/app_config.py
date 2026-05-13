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
    # Which monitor mouse-mode controls. None = all monitors (the
    # full virtual desktop, the historical default). 0..N-1 = a
    # specific monitor's index in QGuiApplication.screens(). The
    # mouse-mode activation popup writes this on user choice; the
    # Save Locations -> Mouse Control panel lets users preset it.
    # Cursor mapping in mouse_controller respects this — the red
    # mouse-box on the camera frame still spans the same area but
    # the cursor output gets clamped to the chosen monitor's region.
    mouse_active_monitor_index: Optional[int] = None
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
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

# Author: Konstantin Markov
