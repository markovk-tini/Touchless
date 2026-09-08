"""Directions connector — driving/walking/transit directions via the Google
Maps Directions API.

IMPORTANT — this one is unlike the Workspace connectors:
  * It uses a **Maps Platform API key** (GOOGLE_MAPS_API_KEY), NOT the OAuth
    user login — there is no consent screen and no user data involved.
  * Google Maps Platform is **billable** (a monthly free credit, then
    pay-per-request). In a distributed build every user's requests bill to
    whoever's key is embedded — so gate/limit it or broker via a backend.

Dormant unless GOOGLE_MAPS_API_KEY is set, so it ships off by default.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from .base import Connector, connector_result, friendly_api_error


def _maps_api_key() -> str:
    return (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip()


class DirectionsConnector(Connector):
    id = "directions"
    description = "directions route navigation how far drive distance travel time map between places"

    def available(self) -> bool:
        return bool(_maps_api_key())

    def tools(self) -> List[Dict[str, Any]]:
        return [{
            "type": "function",
            "name": "directions_get",
            "description": ("Get directions between two places: distance, travel "
                            "time, and step summary. Use for 'how far is X', "
                            "'directions from A to B', 'how long to drive to X'."),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Start address/place."},
                    "destination": {"type": "string", "description": "End address/place."},
                    "mode": {"type": "string",
                             "enum": ["driving", "walking", "bicycling", "transit"],
                             "description": "Travel mode (default driving)."},
                },
                "required": ["origin", "destination"],
                "additionalProperties": False,
            },
        }]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name != "directions_get":
            return connector_result("error", error=f"unknown directions tool: {name}", code="no_handler")
        key = _maps_api_key()
        if not key:
            return connector_result("error", error="Maps API key not configured", code="not_ready")
        origin = str(args.get("origin") or "").strip()
        destination = str(args.get("destination") or "").strip()
        if not origin or not destination:
            return connector_result("error", error="origin and destination are required")
        mode = str(args.get("mode") or "driving").strip().lower()
        params = urllib.parse.urlencode({
            "origin": origin, "destination": destination, "mode": mode, "key": key,
        })
        url = "https://maps.googleapis.com/maps/api/directions/json?" + params
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return connector_result("error", error=friendly_api_error(exc, api_label="Google Maps"))
        status = data.get("status")
        if status != "OK" or not data.get("routes"):
            return connector_result("error", error=f"no route ({status})",
                                    detail=data.get("error_message"))
        leg = data["routes"][0]["legs"][0]
        steps = [s.get("html_instructions", "") for s in (leg.get("steps") or [])]
        # Strip the HTML tags Google embeds in step instructions.
        import re
        steps = [re.sub(r"<[^>]+>", " ", s).strip() for s in steps][:25]
        return connector_result(
            "ok", mode=mode,
            origin=leg.get("start_address"), destination=leg.get("end_address"),
            distance=(leg.get("distance") or {}).get("text"),
            duration=(leg.get("duration") or {}).get("text"),
            steps=steps,
        )
