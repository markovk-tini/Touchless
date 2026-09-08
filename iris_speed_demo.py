"""Throwaway speed comparison: connector (API) path vs GUI path for iris.

IMPORTANT — what this does and does NOT measure:
  * It measures the MECHANICAL floor of each path:
      - API path:  one Spotify Web-API call via the connector.
      - GUI path:  one "look" (screenshot + JPEG encode) — what the model
                   must do on EACH round-trip before it can click.
  * It does NOT include the OpenAI Realtime model round-trip latency,
    which is the dominant cost and only exists in a live session. The GUI
    path needs SEVERAL look->reason->click->recheck round-trips; the API
    path needs ONE. So the real-world gap is much larger than the numbers
    here — this is a conservative lower bound on the API advantage.

Run:  .venv\\Scripts\\python.exe iris_speed_demo.py
Dev tool — not shipped, safe to delete.
"""
from __future__ import annotations

import io
import time

from src.hgr.live_api.connectors.spotify_connector import SpotifyConnector


def _time(fn, n):
    # one warmup, then N timed runs -> average milliseconds
    try:
        fn()
    except Exception:
        pass
    t = time.perf_counter()
    ok = 0
    for _ in range(n):
        try:
            fn()
            ok += 1
        except Exception:
            pass
    return (time.perf_counter() - t) / n * 1000.0, ok


def main() -> int:
    N = 5

    print("== API path (Spotify connector) ==")
    spotify = SpotifyConnector()
    if not spotify.available():
        print("  Spotify NOT authorized in this context — can't time the API path here.")
        print("  Run on a machine where Spotify is connected in Touchless to measure it.")
    else:
        # now_playing is a side-effect-free read (one Web-API call).
        ms, ok = _time(lambda: spotify.execute("spotify_now_playing", {}), N)
        print(f"  spotify_now_playing: {ms:.0f} ms/call  ({ok}/{N} ok)  <- one API round-trip")

    print("\n== GUI path 'look' cost (one screenshot the model needs per round-trip) ==")
    try:
        from PIL import ImageGrab

        def grab_and_encode():
            img = ImageGrab.grab(all_screens=True)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=60)
            return buf.getbuffer().nbytes

        ms, ok = _time(grab_and_encode, N)
        print(f"  screenshot + JPEG encode: {ms:.0f} ms/look  ({ok}/{N} ok)")
    except Exception as exc:
        print(f"  (couldn't time screenshot: {exc})")

    print("\n== Interpretation ==")
    print("  API task   ~= 1 API call  + 1 model round-trip")
    print("  GUI task   ~= (1 look + 1 click) x SEVERAL  + 1 model round-trip EACH")
    print("  The model round-trips (hundreds of ms to seconds each, plus Realtime")
    print("  audio time) dominate and aren't shown here. So in a LIVE session the")
    print("  API path's lead is larger than these mechanical numbers suggest.")
    print("\n  For the true end-to-end number, A/B the SAME command in a live iris")
    print("  session with the connector ON vs forced-GUI (see chat for setup).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
