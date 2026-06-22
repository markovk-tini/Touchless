"""Touchless cross-platform compatibility layer (Phase 0 of the macOS port).

This package is the seam the macOS port grows behind. Today it provides two
*additive*, low-risk pieces that any module can adopt incrementally:

  - `dirs`         — canonical per-platform application directories
  - `capabilities` — per-platform feature-availability flags

Design rules (see docs/MACOS_PORT.md):
  * Additive only — nothing is forced to migrate off `hgr.config.app_config.
    CONFIG_DIR` etc., so there is zero Windows-regression risk.
  * `app_config_dir()` (every OS) and `app_data_dir()` on Windows mirror the
    locations the app already uses. `app_logs_dir()`/`app_cache_dir()` are NEW
    canonical dirs that no existing code writes to yet — adopt incrementally.
  * Named `platform_compat` (not `platform`) so it can never shadow the stdlib
    `platform` module.
  * Future phases add input/window/accessibility/capture backends here under a
    `backend/` subpackage with `win/` + `mac/` implementations.

Author: Konstantin Markov (macOS port)
"""
from __future__ import annotations

from . import capabilities, dirs
from .capabilities import IS_LINUX, IS_MACOS, IS_WINDOWS

__all__ = ["dirs", "capabilities", "IS_WINDOWS", "IS_MACOS", "IS_LINUX"]
