"""Google identity connector — whoami + birthday self-lookup.

"Who am I signed in as", "what's my email", "when's my birthday" as
single API calls. Dormant until the shared GoogleClient is authorized.

Two distinct Google APIs are used here: OAuth2 v2 userinfo for the
cheap profile fields (name/email/picture/locale) and People API v1
for birthdays (oauth2 userinfo does not surface them). Both reuse the
same Credentials cached by GoogleClient.

Author: Konstantin Markov
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List

from .base import Connector, connector_result, friendly_api_error
from .google_client import GoogleClient

_IDENTITY_CALL_TIMEOUT_SEC = 25.0
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="identity")


class IdentityConnector(Connector):
    id = "identity"

    def __init__(self, client: GoogleClient | None = None) -> None:
        self._client = client or GoogleClient.shared()

    def _userinfo_svc(self):
        return self._client.service("oauth2", "v2")

    def _people_svc(self):
        return self._client.service("people", "v1")

    def available(self) -> bool:
        try:
            return self._client.ready()
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "google_whoami",
                "description": (
                    "Return the signed-in Google user's identity: full "
                    "name, given name, family name, email address, "
                    "profile picture URL, and locale if available. Use "
                    "for 'who am I signed in as', 'what's my email', "
                    "'what's my Google name'. Returns {ok, name, "
                    "given_name, family_name, email, picture, locale, "
                    "verified_email}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "google_my_birthday",
                "description": (
                    "Return the signed-in Google user's birthday from "
                    "their People profile (if they've set one). Use for "
                    "'when's my birthday', 'what's my date of birth', "
                    "'how old will I be'. Returns {ok, month, day, year "
                    "(nullable — many users hide the year), iso_date "
                    "(nullable, ISO 8601 if year present), text "
                    "(human-readable e.g. 'March 14')}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return _pool.submit(self._execute_locked, name, args).result(
                timeout=_IDENTITY_CALL_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            return connector_result(
                "error",
                error=f"{name} timed out after {_IDENTITY_CALL_TIMEOUT_SEC}s",
                code="timeout",
            )

    def _execute_locked(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "google_whoami":
            has_userinfo = self._client.has_scope(
                "https://www.googleapis.com/auth/userinfo.email")
            has_contacts = self._client.has_scope(
                "https://www.googleapis.com/auth/contacts")
            if has_userinfo:
                svc = self._userinfo_svc()
            else:
                svc = None
            if svc is None and not has_userinfo:
                drive = self._client.service("drive", "v3")
                if drive is None:
                    return connector_result(
                        "error", error="Identity not authorized",
                        code="not_ready")
                try:
                    about = drive.about().get(
                        fields=("user(displayName,emailAddress,"
                                "photoLink,permissionId)"),
                    ).execute() or {}
                    u = about.get("user") or {}
                    return connector_result(
                        "ok",
                        name=u.get("displayName"),
                        given_name=None,
                        family_name=None,
                        email=u.get("emailAddress"),
                        picture=u.get("photoLink"),
                        locale=None,
                        verified_email=None,
                        source="drive.about",
                    )
                except Exception as exc:
                    return connector_result(
                        "error", error=friendly_api_error(exc, api_label="Google identity"))
            if svc is None:
                return connector_result(
                    "error", error="Identity not authorized", code="not_ready")
            try:
                info = svc.userinfo().get().execute() or {}
                full_name = info.get("name")
                given = info.get("given_name")
                family = info.get("family_name")
                email = info.get("email")
                picture = info.get("picture")
                locale = info.get("locale")
                verified = bool(info.get("verified_email"))
                # Optional enrichment via People API when oauth2 userinfo
                # omits a field (rare, but Google sometimes drops locale
                # for accounts that never set a language preference).
                if has_contacts and not (full_name and given and family
                                         and locale):
                    try:
                        psvc = self._people_svc()
                        if psvc is not None:
                            profile = psvc.people().get(
                                resourceName="people/me",
                                personFields=("names,emailAddresses,"
                                              "photos,locales"),
                            ).execute() or {}
                            if not full_name:
                                names = profile.get("names") or []
                                if names:
                                    full_name = (names[0].get("displayName")
                                                 or full_name)
                                    given = (names[0].get("givenName")
                                             or given)
                                    family = (names[0].get("familyName")
                                              or family)
                            if not locale:
                                locales = profile.get("locales") or []
                                if locales:
                                    locale = (locales[0].get("value")
                                              or locale)
                            if not picture:
                                photos = profile.get("photos") or []
                                if photos:
                                    picture = (photos[0].get("url")
                                               or picture)
                    except Exception:
                        pass
                return connector_result(
                    "ok",
                    name=full_name,
                    given_name=given,
                    family_name=family,
                    email=email,
                    picture=picture,
                    locale=locale,
                    verified_email=verified,
                )
            except Exception as exc:
                return connector_result(
                    "error", error=friendly_api_error(exc, api_label="Google identity"))

        if name == "google_my_birthday":
            if not self._client.has_scope(
                    "https://www.googleapis.com/auth/user.birthday.read"):
                return connector_result(
                    "error",
                    code="needs_reconnect",
                    error=("Birthday lookup needs an updated Google grant. "
                           "Reconnect Google in Settings to enable."),
                )
            svc = self._people_svc()
            if svc is None:
                return connector_result(
                    "error", error="Identity not authorized", code="not_ready")
            try:
                profile = svc.people().get(
                    resourceName="people/me",
                    personFields="birthdays",
                ).execute() or {}
                entries = profile.get("birthdays") or []
                if not entries:
                    return connector_result(
                        "ok", set=False,
                        message="No birthday on your Google profile")
                # Prefer the primary entry; fall back to the first with a
                # structured `date`; fall back again to the first overall
                # so a text-only entry still surfaces something useful.
                chosen = None
                for entry in entries:
                    meta = entry.get("metadata") or {}
                    if meta.get("primary") and entry.get("date"):
                        chosen = entry
                        break
                if chosen is None:
                    for entry in entries:
                        if entry.get("date"):
                            chosen = entry
                            break
                if chosen is None:
                    chosen = entries[0]
                date = chosen.get("date") or {}
                month = date.get("month")
                day = date.get("day")
                year = date.get("year")
                text_raw = chosen.get("text")
                if not (month and day):
                    if text_raw:
                        return connector_result(
                            "ok", month=None, day=None, year=None,
                            iso_date=None, text=text_raw)
                    return connector_result(
                        "ok", set=False,
                        message="No birthday on your Google profile")
                month_names = ("January", "February", "March", "April",
                               "May", "June", "July", "August",
                               "September", "October", "November",
                               "December")
                month_i = int(month)
                day_i = int(day)
                if 1 <= month_i <= 12:
                    text = f"{month_names[month_i - 1]} {day_i}"
                else:
                    text = f"{month_i}/{day_i}"
                iso_date = None
                year_i = None
                if year:
                    year_i = int(year)
                    text = f"{text}, {year_i}"
                    iso_date = f"{year_i:04d}-{month_i:02d}-{day_i:02d}"
                return connector_result(
                    "ok", month=month_i, day=day_i, year=year_i,
                    iso_date=iso_date, text=text)
            except Exception as exc:
                return connector_result(
                    "error", error=friendly_api_error(exc, api_label="Google identity"))

        return connector_result(
            "error", error=f"unknown identity tool: {name}",
            code="no_handler")
