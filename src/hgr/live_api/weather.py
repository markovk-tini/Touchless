"""weather_get — free, no-API-key current weather + short-term forecast.

Two-source design: wttr.in is the rich primary; Open-Meteo + open-meteo
geocoding + ipapi.co for IP-based location is the fallback used when wttr
is slow, down, or rate-limiting. Both return the SAME planner-friendly
dict shape so callers can't tell which source answered.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import random
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple


# Rotating opener templates so the same place doesn't get the
# identical sentence every prompt.
_NOW_OPENERS = [
    "It's {desc} in {place}, {temp}",
    "Right now in {place} it's {desc}, {temp}",
    "{place} is {desc} at {temp}",
    "Looking at {place}, {desc} and {temp}",
]
_FORECAST_OPENERS = [
    "Currently it's {temp}",
    "Right now it's {temp}",
    "At the moment it's {temp}",
]


def _phrase_for_now(place: str, desc: str, temp: str, feels: str,
                    wind: str, humidity, time_hint: str = "") -> str:
    """Build a conversational sentence for current conditions."""
    opener = random.choice(_NOW_OPENERS).format(
        desc=desc.lower(), place=place, temp=temp)
    feels_clause = (f" (feels like {feels})"
                    if feels and feels != temp else "")
    extras: List[str] = []
    if wind and wind.strip() not in {"—", "0 mph", "0 km/h"}:
        extras.append(f"wind around {wind}")
    if humidity not in (None, "?"):
        extras.append(f"humidity at {humidity}%")
    extras_clause = ""
    if extras:
        extras_clause = ", with " + " and ".join(extras)
    suffix = f". {time_hint}" if time_hint else "."
    return f"{opener}{feels_clause}{extras_clause}{suffix}"


def _time_hint(temp_num: Optional[float], hi: Optional[float],
               lo: Optional[float]) -> str:
    """Add a time-of-day–aware phrase about where the temp is heading.
    Morning → 'heading for high'; evening → 'cooling down to low overnight'."""
    if temp_num is None:
        return ""
    try:
        hour = datetime.now().hour
    except Exception:
        return ""
    # Morning / early afternoon: still warming
    if 5 <= hour < 14:
        if hi is not None and hi - temp_num >= 3:
            return f"Heading for a high near {round(hi)}°."
    # Mid-afternoon: at peak
    if 14 <= hour < 18:
        if hi is not None and abs(hi - temp_num) < 3:
            return "Near the day's peak right now."
    # Evening / night: cooling
    if hour >= 18 or hour < 5:
        if lo is not None and temp_num - lo >= 3:
            return f"Cooling down to about {round(lo)}° overnight."
    return ""


def _advisories(desc: str, temp_num: Optional[float],
                wind_num: Optional[float],
                feels_num: Optional[float] = None) -> List[str]:
    """Casual recommendations for notable conditions. Keep it short
    and human — at most one or two lines."""
    out: List[str] = []
    d = (desc or "").lower()
    if any(w in d for w in ("rain", "shower", "drizzle", "thunder", "storm")):
        out.append("Might want a rain jacket or umbrella.")
    if any(w in d for w in ("snow", "sleet", "blizzard", "ice")):
        out.append("Grab a jacket — winter conditions out there.")
    if temp_num is not None and temp_num >= 92:
        out.append("Hot one — stay hydrated and don't forget sunscreen.")
    elif temp_num is not None and temp_num >= 80 and ("clear" in d or "sun" in d):
        out.append("Sunny and warm — don't forget sunscreen.")
    elif temp_num is not None and temp_num <= 38:
        msg = "Pretty cold — grab a jacket."
        if feels_num is not None and feels_num <= temp_num - 5:
            msg += " Feels cooler than the air temp suggests."
        out.append(msg)
    if wind_num is not None and wind_num >= 25:
        out.append("Quite windy too.")
    return out[:2]  # cap at two


_DAY_OPENERS = {
    "tomorrow": [
        "Tomorrow {desc} with a high of {hi} and a low of {lo}",
        "Tomorrow's looking {desc}, high {hi} / low {lo}",
        "For tomorrow, {desc} — high around {hi}, low near {lo}",
    ],
    "weekday": [
        "{day} {desc} with a high of {hi} and a low of {lo}",
        "On {day}, {desc} — high {hi}, low {lo}",
        "{day}'s {desc}, reaching {hi} with a low of {lo}",
    ],
}


def _phrase_for_day(day_label: str, desc: str, hi: str, lo: str) -> str:
    """Natural per-day sentence. day_label is 'tomorrow' or a weekday."""
    if day_label.lower() == "tomorrow":
        tpl = random.choice(_DAY_OPENERS["tomorrow"])
        return tpl.format(desc=desc.lower(), hi=hi, lo=lo)
    tpl = random.choice(_DAY_OPENERS["weekday"])
    return tpl.format(day=day_label, desc=desc.lower(), hi=hi, lo=lo)

_WTTR_URL = "https://wttr.in/{loc}?format=j1"
# wttr.in is notoriously slow / occasionally times out under load. Tight
# timeouts here so we fall through to Open-Meteo quickly instead of
# making the user wait 25s to fail.
_WTTR_TIMEOUT_FIRST = 8.0
_WTTR_TIMEOUT_RETRY = 5.0

# Open-Meteo (fallback): generous free tier, very reliable.
_OPEN_METEO_GEOCODE = (
    "https://geocoding-api.open-meteo.com/v1/search?"
    "name={q}&count=1&language=en&format=json")
_OPEN_METEO_FORECAST = (
    "https://api.open-meteo.com/v1/forecast?"
    "latitude={lat}&longitude={lon}"
    "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
    "weather_code,wind_speed_10m"
    "&daily=temperature_2m_min,temperature_2m_max,weather_code"
    "&forecast_days=3"
    "&temperature_unit={t_unit}&wind_speed_unit={w_unit}&timezone=auto")
# ipapi.co for IP geolocation (no key, ~1000 req/day free).
_IPAPI_URL = "https://ipapi.co/json/"
_GEO_TIMEOUT = 5.0

_MAX_LOC_CHARS = 100


# ---- shared HTTP -----------------------------------------------------------

def _fetch(url: str, timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, method="GET",
        headers={"User-Agent": "TouchlessIris/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---- WMO weather code -> human description --------------------------------
# Open-Meteo returns just a numeric code; we map the common ones so the
# `description` and `summary` strings read naturally.
_WMO_DESC = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def _wmo_desc(code: Optional[int]) -> str:
    if code is None:
        return ""
    return _WMO_DESC.get(int(code), f"weather code {code}")


# ---- Open-Meteo fallback ---------------------------------------------------

def _geocode_open_meteo(query: str) -> Optional[Tuple[float, float, str]]:
    """Resolve a place name to (lat, lon, label). None on failure."""
    if not query:
        return None
    try:
        data = _fetch(_OPEN_METEO_GEOCODE.format(q=urllib.parse.quote(query)),
                      timeout=_GEO_TIMEOUT)
    except Exception:
        return None
    results = data.get("results") or []
    if not results:
        return None
    r = results[0]
    name = r.get("name") or query
    country = r.get("country") or ""
    label = f"{name}, {country}" if country else name
    try:
        return float(r["latitude"]), float(r["longitude"]), label
    except (KeyError, TypeError, ValueError):
        return None


def _geo_from_ip() -> Optional[Tuple[float, float, str]]:
    """IP-based location when caller didn't pass one. Best-effort."""
    try:
        data = _fetch(_IPAPI_URL, timeout=_GEO_TIMEOUT)
    except Exception:
        return None
    try:
        lat = float(data["latitude"])
        lon = float(data["longitude"])
    except (KeyError, TypeError, ValueError):
        return None
    city = data.get("city") or ""
    region = data.get("region") or data.get("country_name") or ""
    label = f"{city}, {region}".strip(", ") or "current location"
    return lat, lon, label


def _get_weather_open_meteo(location: str,
                            use_imperial: bool) -> Optional[Dict[str, Any]]:
    """Fallback path. Returns the SAME dict shape as the wttr branch, or
    None if any step fails (caller surfaces the wttr error in that case)."""
    coords = _geocode_open_meteo(location) if location else _geo_from_ip()
    if coords is None:
        return None
    lat, lon, label = coords
    t_unit = "fahrenheit" if use_imperial else "celsius"
    w_unit = "mph" if use_imperial else "kmh"
    try:
        data = _fetch(
            _OPEN_METEO_FORECAST.format(lat=lat, lon=lon,
                                        t_unit=t_unit, w_unit=w_unit),
            timeout=_GEO_TIMEOUT + 5.0)
    except Exception:
        return None
    cur = data.get("current") or {}
    code = cur.get("weather_code")
    desc = _wmo_desc(code) or "current conditions"
    deg = "°F" if use_imperial else "°C"
    wind_unit = "mph" if use_imperial else "km/h"
    temp = cur.get("temperature_2m")
    feels = cur.get("apparent_temperature")
    humidity = cur.get("relative_humidity_2m")
    wind = cur.get("wind_speed_10m")
    temp_s = f"{round(temp)}{deg}" if temp is not None else "—"
    feels_s = f"{round(feels)}{deg}" if feels is not None else "—"
    wind_s = f"{round(wind)} {wind_unit}" if wind is not None else "—"
    # Pull today's hi/lo for time-of-day hint.
    forecast: List[Dict[str, Any]] = []
    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    mins = daily.get("temperature_2m_min") or []
    maxs = daily.get("temperature_2m_max") or []
    codes = daily.get("weather_code") or []
    today_hi = maxs[0] if maxs else None
    today_lo = mins[0] if mins else None
    temp_num = temp if isinstance(temp, (int, float)) else None
    wind_num = wind if isinstance(wind, (int, float)) else None
    feels_num = feels if isinstance(feels, (int, float)) else None
    time_hint = _time_hint(temp_num, today_hi, today_lo)
    summary = _phrase_for_now(label, desc, temp_s, feels_s, wind_s,
                              humidity, time_hint)
    for adv in _advisories(desc, temp_num, wind_num, feels_num):
        summary += " " + adv

    today_date = date.today()
    forecast_parts: List[str] = []
    for i in range(min(3, len(dates))):
        date_str = dates[i]
        try:
            day_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            day_date = today_date + timedelta(days=i)
        delta_days = (day_date - today_date).days
        if delta_days == 0:
            day_label = "today"
        elif delta_days == 1:
            day_label = "tomorrow"
        else:
            day_label = day_date.strftime("%A")
        d_desc = _wmo_desc(codes[i] if i < len(codes) else None) or ""
        d_min = mins[i] if i < len(mins) else None
        d_max = maxs[i] if i < len(maxs) else None
        if d_desc and d_min is not None and d_max is not None and delta_days >= 1:
            forecast_parts.append(_phrase_for_day(
                day_label, d_desc, f"{round(d_max)}{deg}",
                f"{round(d_min)}{deg}"))
        forecast.append({
            "date": date_str,
            "day_label": day_label,
            "min": (f"{round(d_min)}{deg}" if d_min is not None else "—"),
            "max": (f"{round(d_max)}{deg}" if d_max is not None else "—"),
            "description": d_desc,
        })
    if forecast_parts:
        summary += " " + ". ".join(forecast_parts) + "."
    summary += " Want more details?"
    return {
        "status": "ok",
        "location": label,
        "description": desc,
        "temperature": temp_s,
        "feels_like": feels_s,
        "humidity_pct": humidity,
        "wind": wind_s,
        "summary": summary,
        "forecast": forecast,
        "source": "open-meteo",
    }


# ---- wttr.in primary -------------------------------------------------------

def _get_weather_wttr(loc: str, use_imperial: bool) -> Optional[Dict[str, Any]]:
    """Primary path. Returns the planner-friendly dict, or None on any
    failure (caller falls back to Open-Meteo)."""
    safe = urllib.parse.quote(loc) if loc else ""
    url = _WTTR_URL.format(loc=safe)
    payload = None
    for timeout in (_WTTR_TIMEOUT_FIRST, _WTTR_TIMEOUT_RETRY):
        try:
            payload = _fetch(url, timeout)
            break
        except urllib.error.HTTPError:
            # 4xx/5xx aren't transient; bail to fallback.
            return None
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}".lower()
            if "timed out" not in msg and "timeout" not in msg:
                return None
    if payload is None:
        return None

    cur = (payload.get("current_condition") or [{}])[0]
    nearest = (payload.get("nearest_area") or [{}])[0]
    weather_days = payload.get("weather") or []

    def _resolved_location() -> str:
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
    # Build natural conversational summary.
    forecast: List[Dict[str, Any]] = []
    today_date = date.today()
    # Pull today's hi/lo for the time-of-day hint.
    today_hi = today_lo = None
    if weather_days:
        d0 = weather_days[0]
        try:
            today_hi = float(d0.get("maxtempF" if use_imperial else "maxtempC"))
        except (TypeError, ValueError):
            today_hi = None
        try:
            today_lo = float(d0.get("mintempF" if use_imperial else "mintempC"))
        except (TypeError, ValueError):
            today_lo = None
    try:
        temp_now_num = float(cur.get("temp_F" if use_imperial else "temp_C"))
    except (TypeError, ValueError):
        temp_now_num = None
    try:
        wind_num = float(cur.get("windspeedMiles" if use_imperial
                                  else "windspeedKmph"))
    except (TypeError, ValueError):
        wind_num = None
    try:
        feels_num = float(cur.get("FeelsLikeF" if use_imperial
                                   else "FeelsLikeC"))
    except (TypeError, ValueError):
        feels_num = None

    time_hint = _time_hint(temp_now_num, today_hi, today_lo)
    summary = _phrase_for_now(place, desc, temp_now, feels, wind_spd,
                              cur.get("humidity"), time_hint)
    # Advisories tacked on as a short sentence.
    for adv in _advisories(desc, temp_now_num, wind_num, feels_num):
        summary += " " + adv

    # Build per-day forecast in conversational form (skip 'today' since
    # the current-conditions sentence already covers it).
    forecast_parts: List[str] = []
    for i, day in enumerate(weather_days[:3]):
        date_str = day.get("date") or ""
        try:
            day_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            day_date = today_date + timedelta(days=i)
        delta_days = (day_date - today_date).days
        if delta_days == 0:
            day_label = "today"
        elif delta_days == 1:
            day_label = "tomorrow"
        else:
            day_label = day_date.strftime("%A")
        desc_mid = ((day.get("hourly") or [{}])[
            len((day.get("hourly") or [])) // 2]
            .get("weatherDesc") or [{}])[0].get("value", "").strip()
        min_t = day.get("mintempF") if use_imperial else day.get("mintempC")
        max_t = day.get("maxtempF") if use_imperial else day.get("maxtempC")
        unit = "°F" if use_imperial else "°C"
        if desc_mid and min_t and max_t and delta_days >= 1:
            forecast_parts.append(_phrase_for_day(
                day_label, desc_mid, f"{max_t}{unit}", f"{min_t}{unit}"))
        forecast.append({
            "date": date_str,
            "day_label": day_label,
            "min": (f"{min_t}{unit}" if min_t else ""),
            "max": (f"{max_t}{unit}" if max_t else ""),
            "description": desc_mid,
        })
    if forecast_parts:
        summary += " " + ". ".join(forecast_parts) + "."
    # Offer follow-up details instead of a raw URL.
    summary += " Want more details?"
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
        "source": "wttr",
    }


# ---- public entry point ----------------------------------------------------

def get_weather(location: str = "", units: str = "imperial") -> Dict[str, Any]:
    """Current conditions + a 1-3 day forecast. `location` empty = auto from
    IP; pass a city name / ZIP / "lat,lon" for elsewhere. Tries wttr.in
    first, falls back to Open-Meteo if wttr is slow or down."""
    loc = (location or "").strip()[:_MAX_LOC_CHARS]
    use_imperial = (units or "").lower() != "metric"
    result = _get_weather_wttr(loc, use_imperial)
    if result is not None:
        return result
    result = _get_weather_open_meteo(loc, use_imperial)
    if result is not None:
        return result
    return {
        "status": "error",
        "error": ("Both weather sources failed (wttr.in and Open-Meteo). "
                  "Network unreachable or both services down — try again "
                  "in a minute."),
        "code": "weather_unreachable",
    }
