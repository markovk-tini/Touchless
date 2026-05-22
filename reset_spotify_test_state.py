"""Reset Spotify connect-prompt state so the next launch of Touchless
treats the user as if it were their first run.

Clears two things:
  1) config.spotify_first_active_prompt_shown — the latch that says
     "user already saw the popup, don't ask again".
  2) every known auth_token.json on disk — the per-user OAuth tokens.
     Without these, the controller's has_authorization returns False,
     so the popup fires the next time Spotify is detected open.

After running this, launch the app from source:
    python run_app.py
…then open Spotify (or have it already open). The "Connect Spotify?"
popup should appear within ~3 seconds.

Run from the repo root:
    python reset_spotify_test_state.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure we can import the app's config helpers from src/.
_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hgr.config.app_config import load_config, save_config


def _reset_prompt_flag() -> None:
    cfg = load_config()
    if not bool(getattr(cfg, "spotify_first_active_prompt_shown", False)):
        print(
            "[reset] spotify_first_active_prompt_shown was already False "
            "— nothing to clear in config."
        )
        return
    cfg.spotify_first_active_prompt_shown = False
    save_config(cfg)
    print("[reset] spotify_first_active_prompt_shown -> False (saved to settings.json)")


def _token_paths() -> list[Path]:
    home = Path.home()
    return [
        _REPO_ROOT / "auth_token.json",
        home / "Documents" / "Touchless" / "auth_token.json",
        home / "Documents" / "HandGestureControl" / "HGRApp" / "auth_token.json",
        home / "Documents" / "HandAI" / "HandMeshLive" / "src" / "auth_token.json",
    ]


def _delete_token_files() -> None:
    any_deleted = False
    for path in _token_paths():
        try:
            if path.exists():
                path.unlink()
                print(f"[reset] deleted: {path}")
                any_deleted = True
        except Exception as exc:
            print(f"[reset] failed to delete {path}: {exc}")
    if not any_deleted:
        print("[reset] no auth_token.json files found — already clean.")


def main() -> int:
    print("[reset] clearing Spotify connect-prompt + auth state...")
    _reset_prompt_flag()
    _delete_token_files()
    print("[reset] done. Now launch: python run_app.py")
    print("[reset] Then open Spotify desktop — the popup should appear within ~3s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
