from __future__ import annotations

import platform
import time
from dataclasses import dataclass


@dataclass
class VolumeStatus:
    available: bool
    message: str
    level_scalar: float | None = None


class VolumeController:
    def __init__(self) -> None:
        self._available = False
        self._message = "Volume control unavailable."
        self._volume = None
        self._last_known_level: float | None = None
        self._last_known_muted: bool | None = None
        self._sync_window_seconds = 0.32
        self._level_write_until = 0.0
        self._mute_write_until = 0.0
        self._last_write_level: float | None = None
        self._last_write_time = 0.0
        self._min_write_step = 0.003
        self._endpoint_id: str | None = None
        self._last_endpoint_check_time = 0.0

        if platform.system() != "Windows":
            self._message = "Volume control is only supported on Windows."
            return

        try:
            self._rebind_endpoint()
        except Exception as exc:
            self._available = False
            self._volume = None
            self._message = f"Could not access system speakers: {type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._available and self._volume is not None

    @property
    def message(self) -> str:
        return self._message

    def get_level(self, *, prefer_cached: bool = True) -> float | None:
        self._refresh_default_endpoint_if_changed()
        for attempt in range(2):
            if not self.available:
                if attempt == 0 and self._recover_endpoint():
                    continue
                return self._last_known_level
            try:
                self._ensure_com_ready()
                level = float(self._volume.GetMasterVolumeLevelScalar())
            except Exception as exc:
                if attempt == 0 and self._recover_endpoint(exc):
                    continue
                self._message = f"Could not read system volume: {type(exc).__name__}"
                return self._last_known_level
            if prefer_cached and self._should_prefer_cached_level(level):
                return self._last_known_level
            self._last_known_level = level
            self._message = "Volume control ready."
            return level
        return self._last_known_level

    def nudge_system_volume_key(self, direction: int) -> bool:
        if platform.system() != "Windows" or direction == 0:
            return False
        try:
            import ctypes
            VK_VOLUME_DOWN = 0xAE
            VK_VOLUME_UP = 0xAF
            KEYEVENTF_KEYUP = 0x0002
            vk = VK_VOLUME_UP if direction > 0 else VK_VOLUME_DOWN
            user32 = ctypes.windll.user32
            user32.keybd_event(vk, 0, 0, 0)
            user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
            return True
        except Exception:
            return False

    def set_level(self, scalar: float) -> bool:
        self._refresh_default_endpoint_if_changed()
        scalar = max(0.0, min(1.0, float(scalar)))
        min_write_step = float(getattr(self, "_min_write_step", 0.003))
        last_write_time = float(getattr(self, "_last_write_time", 0.0))
        if (
            self._last_known_level is not None
            and abs(float(self._last_known_level) - scalar) < min_write_step
            and self._now() - last_write_time <= 0.08
        ):
            self._last_known_level = scalar
            return True

        for attempt in range(2):
            if not self.available:
                if attempt == 0 and self._recover_endpoint():
                    continue
                return False
            try:
                self._ensure_com_ready()
                self._volume.SetMasterVolumeLevelScalar(scalar, None)
                self._last_known_level = scalar
                self._last_write_level = scalar
                self._last_write_time = self._now()
                sync_window_seconds = float(getattr(self, "_sync_window_seconds", 0.32))
                self._level_write_until = self._last_write_time + sync_window_seconds
                self._message = "Volume control ready."
                return True
            except Exception as exc:
                if attempt == 0 and self._recover_endpoint(exc):
                    continue
                self._message = f"Could not change system volume: {type(exc).__name__}"
                return False
        return False

    def get_mute(self, *, prefer_cached: bool = True) -> bool | None:
        self._refresh_default_endpoint_if_changed()
        for attempt in range(2):
            if not self.available:
                if attempt == 0 and self._recover_endpoint():
                    continue
                return self._last_known_muted
            try:
                self._ensure_com_ready()
                muted = bool(self._volume.GetMute())
            except Exception as exc:
                if attempt == 0 and self._recover_endpoint(exc):
                    continue
                self._message = f"Could not read mute state: {type(exc).__name__}"
                return self._last_known_muted
            if prefer_cached and self._should_prefer_cached_mute(muted):
                return self._last_known_muted
            self._last_known_muted = muted
            self._message = "Volume control ready."
            return muted
        return self._last_known_muted

    def set_mute(self, muted: bool) -> bool:
        self._refresh_default_endpoint_if_changed()
        for attempt in range(2):
            if not self.available:
                if attempt == 0 and self._recover_endpoint():
                    continue
                return False
            try:
                self._ensure_com_ready()
                self._volume.SetMute(1 if muted else 0, None)
                self._last_known_muted = bool(muted)
                now = self._now()
                sync_window_seconds = float(getattr(self, "_sync_window_seconds", 0.32))
                self._mute_write_until = now + sync_window_seconds
                self._level_write_until = max(float(getattr(self, "_level_write_until", 0.0)), now + sync_window_seconds)
                self._message = "Volume control ready."
                return True
            except Exception as exc:
                if attempt == 0 and self._recover_endpoint(exc):
                    continue
                self._message = f"Could not change mute state: {type(exc).__name__}"
                return False
        return False

    def toggle_mute(self) -> bool | None:
        current = self.get_mute()
        if current is None:
            return None
        if not self.set_mute(not current):
            return None
        return not current

    def get_app_audio_info(self, process_names: list[str]) -> tuple[str | None, float | None]:
        if platform.system() != "Windows":
            return None, None
        try:
            self._ensure_com_ready()
            from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

            sessions = AudioUtilities.GetAllSessions()
            for session in sessions:
                proc = session.Process
                if proc is None:
                    continue
                proc_name = (proc.name() or "").lower()
                for target in process_names:
                    if target.lower() in proc_name:
                        try:
                            simple = session._ctl.QueryInterface(ISimpleAudioVolume)
                            level = float(simple.GetMasterVolume())
                            return target, level
                        except Exception:
                            return target, None
        except Exception:
            pass
        return None, None

    def get_active_app_audio_info(
        self,
        process_names: list[str],
        *,
        peak_threshold: float = 0.005,
    ) -> tuple[str | None, float | None]:
        """Like get_app_audio_info but only returns an app that is actually
        playing audio right now (meter peak > threshold). Picks the loudest
        matching session when several are active. Returns (None, None) if
        every matching session is silent — used by the dual-bar volume
        overlay so it doesn't show Chrome just because Chrome is open.

        Called once when the overlay appears (not on a hot tick loop), so
        the peak enumeration is safe for the Razer/Genshin driver stability
        concern that forced us to remove the YouTube auto-mode probe.
        """
        if platform.system() != "Windows":
            return None, None
        try:
            self._ensure_com_ready()
            from pycaw.pycaw import (
                AudioUtilities,
                IAudioMeterInformation,
                ISimpleAudioVolume,
            )

            sessions = AudioUtilities.GetAllSessions()
            best_peak = float(peak_threshold)
            best_target: str | None = None
            best_level: float | None = None
            for session in sessions:
                proc = session.Process
                if proc is None:
                    continue
                proc_name = (proc.name() or "").lower()
                matched_target: str | None = None
                for target in process_names:
                    if target.lower() in proc_name:
                        matched_target = target
                        break
                if matched_target is None:
                    continue
                try:
                    meter = session._ctl.QueryInterface(IAudioMeterInformation)
                    peak = float(meter.GetPeakValue())
                except Exception:
                    continue
                if peak <= best_peak:
                    continue
                try:
                    simple = session._ctl.QueryInterface(ISimpleAudioVolume)
                    level = float(simple.GetMasterVolume())
                except Exception:
                    level = None
                best_peak = peak
                best_target = matched_target
                best_level = level
            return best_target, best_level
        except Exception:
            return None, None

    def get_process_audio_peak(self, process_names: list[str]) -> float | None:
        if platform.system() != "Windows":
            return None
        try:
            self._ensure_com_ready()
            from pycaw.pycaw import AudioUtilities, IAudioMeterInformation

            sessions = AudioUtilities.GetAllSessions()
            peak: float | None = None
            for session in sessions:
                proc = session.Process
                if proc is None:
                    continue
                proc_name = (proc.name() or "").lower()
                if not any(target.lower() in proc_name for target in process_names):
                    continue
                try:
                    meter = session._ctl.QueryInterface(IAudioMeterInformation)
                    value = float(meter.GetPeakValue())
                except Exception:
                    continue
                peak = value if peak is None else max(peak, value)
            return peak
        except Exception:
            return None

    def set_app_audio_level(self, process_names: list[str], scalar: float) -> bool:
        if platform.system() != "Windows":
            return False
        scalar = max(0.0, min(1.0, float(scalar)))
        try:
            self._ensure_com_ready()
            from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

            sessions = AudioUtilities.GetAllSessions()
            for session in sessions:
                proc = session.Process
                if proc is None:
                    continue
                proc_name = (proc.name() or "").lower()
                for target in process_names:
                    if target.lower() in proc_name:
                        try:
                            simple = session._ctl.QueryInterface(ISimpleAudioVolume)
                            simple.SetMasterVolume(scalar, None)
                            return True
                        except Exception:
                            return False
        except Exception:
            pass
        return False

    def status(self) -> VolumeStatus:
        return VolumeStatus(
            available=self.available,
            message=self._message,
            level_scalar=self.get_level(),
        )

    def refresh_cache(self) -> VolumeStatus:
        self._refresh_default_endpoint_if_changed()
        if not self.available:
            return VolumeStatus(
                available=False,
                message=self._message,
                level_scalar=self._last_known_level,
            )
        self._level_write_until = 0.0
        self._mute_write_until = 0.0
        self._refresh_cache()
        return VolumeStatus(
            available=True,
            message=self._message,
            level_scalar=self._last_known_level,
        )

    def _refresh_cache(self) -> None:
        if not self.available:
            return
        try:
            self._last_known_level = float(self._volume.GetMasterVolumeLevelScalar())
        except Exception:
            pass
        try:
            self._last_known_muted = bool(self._volume.GetMute())
        except Exception:
            pass

    def sync_live_state(self) -> VolumeStatus:
        if not self.available:
            return VolumeStatus(
                available=False,
                message=self._message,
                level_scalar=self._last_known_level,
            )
        level = self.get_level(prefer_cached=False)
        muted = self.get_mute(prefer_cached=False)
        if muted is not None:
            self._last_known_muted = bool(muted)
        return VolumeStatus(
            available=True,
            message=self._message,
            level_scalar=level,
        )

    def _now(self) -> float:
        return time.monotonic()

    def _should_prefer_cached_level(self, live_level: float) -> bool:
        return (
            self._last_known_level is not None
            and self._now() < float(getattr(self, "_level_write_until", 0.0))
            and abs(float(live_level) - float(self._last_known_level)) >= 0.02
        )

    def _should_prefer_cached_mute(self, live_muted: bool) -> bool:
        return (
            self._last_known_muted is not None
            and self._now() < float(getattr(self, "_mute_write_until", 0.0))
            and bool(live_muted) != bool(self._last_known_muted)
        )

    def _ensure_com_ready(self) -> None:
        if platform.system() != "Windows":
            return
        try:
            from comtypes import CoInitialize

            CoInitialize()
        except Exception:
            pass

    def _rebind_endpoint(self, device=None) -> None:
        from pycaw.pycaw import AudioUtilities

        self._ensure_com_ready()
        endpoint_device = device if device is not None else AudioUtilities.GetSpeakers()
        # pycaw's GetSpeakers() returns an AudioDevice wrapper whose
        # endpoint id is exposed via the `.id` property. The previous
        # code called `.GetId()` which is the raw IMMDevice method —
        # AudioDevice doesn't proxy it, so the call always raised
        # AttributeError and the except clause silently set
        # endpoint_id = None. Combined with the same bug in
        # _refresh_default_endpoint_if_changed below, this meant
        # the comparison `None != None` was always False and the
        # controller NEVER rebound when the user swapped default
        # playback devices — all volume / mute calls kept hitting
        # whatever was default at app-startup time.
        endpoint_id = None
        try:
            endpoint_id = str(endpoint_device.id)
        except Exception:
            try:
                # Last-resort fallback: try .GetId() in case a future
                # pycaw version restores the raw IMMDevice method.
                endpoint_id = str(endpoint_device.GetId())
            except Exception:
                endpoint_id = None
        self._volume = endpoint_device.EndpointVolume
        self._endpoint_id = endpoint_id
        self._available = self._volume is not None
        self._message = "Volume control ready." if self._available else "Volume control unavailable."
        if self._available:
            self._refresh_cache()

    def _refresh_default_endpoint_if_changed(self) -> None:
        if platform.system() != "Windows":
            return
        now = self._now()
        if now - float(getattr(self, "_last_endpoint_check_time", 0.0)) < 0.35:
            return
        self._last_endpoint_check_time = now
        try:
            from pycaw.pycaw import AudioUtilities

            self._ensure_com_ready()
            device = AudioUtilities.GetSpeakers()
            # Same fix as _rebind_endpoint: AudioDevice exposes the
            # endpoint id via `.id` (property), not `.GetId()` (raw
            # IMMDevice method). The old code raised AttributeError
            # here every poll, set device_id = None, then compared
            # None != self._endpoint_id (also None from the same bug
            # at init) — the result was False so rebind never fired
            # even when Windows default playback changed.
            device_id = None
            try:
                device_id = str(device.id)
            except Exception:
                try:
                    device_id = str(device.GetId())
                except Exception:
                    device_id = None
            if self._volume is None or device_id != self._endpoint_id:
                self._rebind_endpoint(device=device)
        except Exception:
            pass

    def _recover_endpoint(self, exc: Exception | None = None) -> bool:
        if platform.system() != "Windows":
            return False
        try:
            self._rebind_endpoint()
            return self.available
        except Exception as recover_exc:
            reason = exc or recover_exc
            self._available = False
            self._volume = None
            self._message = f"Could not access system speakers: {type(reason).__name__}: {reason}"
            return False

# Author: Konstantin Markov
