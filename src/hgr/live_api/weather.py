"""weather_get — free, no-API-key current weather + short-term forecast.

Backed by wttr.in. Auto-detects the user's location from IP when no
location is passed; pass a city name / zip code / coordinates to query
elsewhere. Returns a compact, planner-friendly dict.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

_URL = "https://wttr.in/{loc}?format=j1"
_TIMEOUT = 8.0
_MAX_LOC_CHARS = 100


def get_weather(location: str = "", units: str = "imperial") -> Dict[str, Any]:
    """Current conditions + a 1-3 day forecast. `location` empty = auto from
    IP; pass anything wttr.in accepts (city, ZIP, "lat,lon", etc.). `units`
    decides which fields go into the human-readable summary ('imperial' or
    'metric'); both are present in the raw current/forecast blocks too."""
    loc = (location or "").strip()[:_MAX_LOC_CHARS]
    safe = urllib.parse.quote(loc) if loc else ""
    url = _URL.format(loc=safe)
    try:
        req = urllib.request.Request(
            url, method="GET",
            headers={"User-Agent": "TouchlessIris/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"status": "error", "error": f"HTTP {exc.code}",
                "code": "wttr_http_error"}
    except Exception as exc:
        return {"status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "code": "wttr_failed"}

    cur = (payload.get("current_condition") or [{}])[0]
    nearest = (payload.get("nearest_area") or [{}])[0]
    weather_days = payload.get("weather") or []

    use_imperial = (units or "").lower() != "metric"

    def _temp_with_unit(c) -> str:
        f = c.get("temp_F"); cc = c.get("temp_C")
        return f"{f}°F" if use_imperial and f else f"{cc}°C"

    def _resolved_location() -> str:
        # wttr returns rich location fields; pick the most user-friendly.
        for key in ("areaName", "region", "country"):
            val = (nearest.get(key) or [{}])[0].get("value")
            if val:
                return val
        return loc or "current location"

    place = _resolved_location()
    desc = ((cur.get("weatherDesc") or [{}])[0].get("value") or "").strip()
    temp_now = (f"{cur.get('temp_F')}°F" if use_imperial
                else f"{cur.get('temp_C')}°C")
    feels = (f"{cur.get('FeelsLikeF')}°F" if use_imperial
             else f"{cur.get('FeelsLikeC')}°C")
    wind_spd = (f"{cur.get('windspeedMiles')} mph" if use_imperial
                else f"{cur.get('windspeedKmph')} km/h")
    summary = (f"{place}: {desc.lower()}, {temp_now} (feels like {feels}). "
               f"Wind {wind_spd}, humidity {cur.get('humidity', '?')}%.")

    forecast: List[Dict[str, Any]] = []
    for day in weather_days[:3]:
        forecast.append({
            "date": day.get("date"),
            "min": (f"{day.get('mintempF')}°F" if use_imperial
                    else f"{day.get('mintempC')}°C"),
            "max": (f"{day.get('maxtempF')}°F" if use_imperial
                    else f"{day.get('maxtempC')}°C"),
            "description": ((day.get("hourly") or [{}])[len((day.get("hourly") or [])) // 2]
                            .get("weatherDesc") or [{}])[0].get("value", ""),
        })

    return {
        "status": "ok",
        "location": place,
        "description": desc,
        "temperature": temp_now,
        "feels_like": feels,
        "humidity_pct": cur.get("humidity"),
        "wind": wind_spd,
        "summary": summary,
        "forecast": forecast,
    }
