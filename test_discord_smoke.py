"""Interactive smoke-test for the Discord RPC controller.

Runs the full auth handshake against your *live* Discord desktop
client. Use this once after creating the dev-portal app to verify
that:

  * the IPC pipe is reachable (Discord must be running),
  * the AUTHORIZE modal pops up in Discord and your Allow returns
    a code,
  * the token-exchange POST to Discord's OAuth endpoint succeeds
    with the secret in `.env`,
  * AUTHENTICATE completes,
  * SET_VOICE_SETTINGS round-trips end-to-end (mute → un-mute).

Run from repo root:

    python test_discord_smoke.py

Author: Konstantin Markov
"""

from __future__ import annotations

import logging
import sys
import time

# Pull Discord controller from the src layout.
sys.path.insert(0, "src")
from hgr.debug.discord_controller import DiscordController  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ctrl = DiscordController()
    print(f"[smoke] platform-available: {ctrl.available}")
    print(f"[smoke] client_id loaded:    {bool(ctrl._client_id)}")
    print(f"[smoke] client_secret loaded: {bool(ctrl._client_secret)}")
    print(f"[smoke] redirect_uri:        {ctrl._redirect_uri}")
    print(f"[smoke] has_authorization:   {ctrl.has_authorization}")
    print()

    if not ctrl.available:
        print("[smoke] ABORT: controller not available on this platform.")
        return 1
    if not ctrl._client_secret:
        print("[smoke] ABORT: DISCORD_CLIENT_SECRET missing — set it in .env.")
        return 1

    if not ctrl.has_authorization:
        print("[smoke] No saved token. Running full authorize flow.")
        print("[smoke]   → Discord should pop up a modal: "
              "\"Allow Touchless to control your Discord?\".")
        print("[smoke]   → Click Allow.")
        ok = ctrl.authorize_full_scopes()
        print(f"[smoke] authorize_full_scopes -> {ok}")
        print(f"[smoke] controller message:     {ctrl.message}")
        if not ok:
            return 2
    else:
        print("[smoke] Saved token found; skipping authorize.")
        print("[smoke] Calling get_voice_settings to force handshake + AUTHENTICATE.")

    # Round-trip the voice state to confirm the authed pipe works.
    state = ctrl.get_voice_settings()
    print(f"[smoke] initial voice state: {state}")
    if state is None:
        print(f"[smoke] FAIL: could not read voice settings. Message: {ctrl.message}")
        return 3

    print("[smoke] muting self for 2s...")
    if not ctrl.set_self_mute(True):
        print(f"[smoke] FAIL: set_self_mute(True) — {ctrl.message}")
        return 4
    time.sleep(2.0)

    print("[smoke] un-muting...")
    if not ctrl.set_self_mute(False):
        print(f"[smoke] FAIL: set_self_mute(False) — {ctrl.message}")
        return 5

    final_state = ctrl.get_voice_settings()
    print(f"[smoke] final voice state:   {final_state}")
    print("[smoke] PASS — Discord RPC end-to-end working.")
    ctrl.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
