"""Google Contacts connector — People API search + create.

"Find Dani in my contacts", "add John to my contacts with email
john@example.com" as single API calls. Dormant until the shared
GoogleClient is authorized.

The People API has a cold-cache quirk on searchContacts: the first
query against a fresh session returns empty for ~30s. We mitigate by
issuing a throwaway warmup call lazily on the first search; the flag
lives on `self` so it costs nothing on subsequent calls.

Author: Konstantin Markov
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List

from .base import Connector, connector_result, friendly_api_error
from .google_client import GoogleClient

_CONTACTS_CALL_TIMEOUT_SEC = 25.0
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="contacts")


class ContactsConnector(Connector):
    id = "contacts"

    def __init__(self, client: GoogleClient | None = None) -> None:
        self._client = client or GoogleClient.shared()
        self._warm = False

    def _svc(self):
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
                "name": "contacts_search",
                "description": (
                    "Search the user's Google Contacts by name or email "
                    "substring and return matching people with their primary "
                    "email, phone, and resource name. Use this to resolve "
                    "'who is Dani' / 'find Dani's email' / 'what's my "
                    "contact's phone number' BEFORE composing email/text. "
                    "Returns up to `max` matches ranked by relevance."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Name or email substring (min 1 char)."},
                        "max": {"type": "integer",
                                "description": "Max matches (default 10, capped 1..30)."},
                        "read_mask_extra": {"type": "string",
                                            "description": "Comma-separated extra People API fields beyond the default names,emailAddresses,phoneNumbers,organizations (rarely needed)."},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "contacts_list",
                "description": (
                    "List the user's Google Contacts, newest-added first. "
                    "Use for 'list my contacts', 'show me my contacts', "
                    "'who's in my contacts', 'how many contacts do I have'. "
                    "Returns up to `max` people with the same shape as "
                    "contacts_search. No query needed. Pass optional "
                    "`starts_with` to filter to contacts whose display / "
                    "given name begins with that letter or short prefix "
                    "(case-insensitive) — use for 'list contacts starting "
                    "with A', 'my A contacts', 'anyone whose name starts "
                    "with C'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max": {"type": "integer",
                                "description": "Max entries (default 25, capped 1..100)."},
                        "starts_with": {"type": "string",
                                        "description": "Optional prefix (single letter or short string). Filters to contacts whose display/given name starts with this (case-insensitive)."},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "contacts_create",
                "description": (
                    "Create a new contact in the user's Google Contacts. "
                    "Use for 'add John to my contacts with email "
                    "john@example.com' / 'save Mariya's phone number as "
                    "555-1234'. At least one of email or phone is required "
                    "in addition to the name. Returns the new contact's "
                    "resourceName + a People-app link."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "given_name": {"type": "string"},
                        "family_name": {"type": "string"},
                        "email": {"type": "string",
                                  "description": "Primary email."},
                        "phone": {"type": "string",
                                  "description": "Primary phone, free-form e.g. '+1 555-123-4567'."},
                        "organization": {"type": "string",
                                         "description": "Company name."},
                        "notes": {"type": "string",
                                  "description": "Biography/notes field."},
                    },
                    "required": ["given_name"],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return _pool.submit(self._execute_locked, name, args).result(
                timeout=_CONTACTS_CALL_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            return connector_result(
                "error",
                error=f"{name} timed out after {_CONTACTS_CALL_TIMEOUT_SEC}s",
                code="timeout",
            )

    def _execute_locked(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Contacts not authorized",
                                    code="not_ready")

        if name == "contacts_search":
            query = str(args.get("query") or "").strip()
            if not query:
                return connector_result("error", error="query is required")
            max_n = max(1, min(30, int(args.get("max") or 10)))
            read_mask = "names,emailAddresses,phoneNumbers,organizations"
            extra = str(args.get("read_mask_extra") or "").strip()
            if extra:
                read_mask = f"{read_mask},{extra}"
            if not self._warm:
                try:
                    svc.people().searchContacts(
                        query="", pageSize=1, readMask="names").execute()
                except Exception:
                    pass
                self._warm = True
            try:
                resp = svc.people().searchContacts(
                    query=query, pageSize=max_n, readMask=read_mask).execute()
            except Exception as exc:
                return connector_result(
                    "error",
                    error=friendly_api_error(
                        exc, api_label="Google Contacts"))
            out: List[Dict[str, Any]] = []
            for result in resp.get("results", []) or []:
                person = result.get("person", {}) or {}
                names = person.get("names", []) or []
                emails = person.get("emailAddresses", []) or []
                phones = person.get("phoneNumbers", []) or []
                orgs = person.get("organizations", []) or []
                if not names and not emails and not phones:
                    continue
                primary_name = _primary(names) or {}
                out.append(_build_contact_entry(
                    person, primary_name, emails, phones, orgs))
            summary = _build_contacts_summary(
                out, total=len(out), context="search", query=query)
            return connector_result("ok", count=len(out), contacts=out,
                                    summary=summary)

        if name == "contacts_list":
            starts_with = str(args.get("starts_with") or "").strip()
            read_mask = "names,emailAddresses,phoneNumbers,organizations"
            if starts_with:
                # Prefix-filter path: fetch a larger page (People API caps
                # connections().list pageSize at 1000; 500 is a safe default
                # that covers most personal address books in one call
                # without inflating payload). Filter locally on
                # display_name (fallback: given_name). Note: users with
                # >500 contacts will silently miss matches beyond the
                # first page — acceptable for a single-letter filter on
                # typical address books; TODO paginate if this bites.
                page_size = 500
                # No sortOrder needed — we're filtering, not showing
                # newest-first — and connections().list rejects
                # LAST_MODIFIED_DESCENDING alongside some field masks
                # anyway on stale sessions. Default order is fine.
                try:
                    resp = svc.people().connections().list(
                        resourceName="people/me",
                        pageSize=page_size,
                        personFields=read_mask).execute()
                except Exception as exc:
                    return connector_result(
                        "error",
                        error=friendly_api_error(
                            exc, api_label="Google Contacts"))
            else:
                max_n = max(1, min(100, int(args.get("max") or 25)))
                try:
                    resp = svc.people().connections().list(
                        resourceName="people/me",
                        pageSize=max_n,
                        personFields=read_mask,
                        sortOrder="LAST_MODIFIED_DESCENDING").execute()
                except Exception as exc:
                    return connector_result(
                        "error",
                        error=friendly_api_error(
                            exc, api_label="Google Contacts"))
            out: List[Dict[str, Any]] = []
            for person in resp.get("connections", []) or []:
                names = person.get("names", []) or []
                emails = person.get("emailAddresses", []) or []
                phones = person.get("phoneNumbers", []) or []
                orgs = person.get("organizations", []) or []
                if not names and not emails and not phones:
                    continue
                primary_name = _primary(names) or {}
                out.append(_build_contact_entry(
                    person, primary_name, emails, phones, orgs))
            if starts_with:
                # Client-side prefix filter. Compare against display_name
                # (falls back to given_name if missing) case-insensitively.
                # `total` MUST be the FILTERED count so the orchestrator
                # summary says "You have 3 contacts starting with A"
                # honestly rather than parroting the full address-book
                # total from resp.totalItems (which would let the LLM
                # hallucinate "you have 47 contacts starting with A"
                # when only 3 were returned).
                sw_lower = starts_with.lower()
                filtered: List[Dict[str, Any]] = []
                for p in out:
                    nm = (p.get("display_name")
                          or p.get("given_name") or "").strip().lower()
                    if nm.startswith(sw_lower):
                        filtered.append(p)
                # Sort filtered results alphabetically for a stable UX.
                filtered.sort(
                    key=lambda p: (p.get("display_name")
                                   or p.get("given_name") or "").lower())
                # Cap post-filter using `max` (default 25, 1..100).
                max_n = max(1, min(100, int(args.get("max") or 25)))
                filtered_total = len(filtered)
                out = filtered[:max_n]
                total = filtered_total
                summary = _build_contacts_summary(
                    out, total=total, context="starts_with",
                    starts_with=starts_with)
                return connector_result(
                    "ok", count=len(out), total=total,
                    contacts=out, summary=summary,
                    filter_applied=starts_with)
            total = resp.get("totalItems", len(out))
            summary = _build_contacts_summary(
                out, total=total, context="list")
            return connector_result(
                "ok", count=len(out), total=total,
                contacts=out, summary=summary)

        if name == "contacts_create":
            given = str(args.get("given_name") or "").strip()
            if not given:
                return connector_result("error", error="given_name is required")
            family = str(args.get("family_name") or "").strip()
            email = str(args.get("email") or "").strip()
            phone = str(args.get("phone") or "").strip()
            org = str(args.get("organization") or "").strip()
            notes = str(args.get("notes") or "").strip()
            if not email and not phone:
                return connector_result(
                    "error",
                    error="at least one of email or phone is required")
            name_obj: Dict[str, Any] = {"givenName": given}
            if family:
                name_obj["familyName"] = family
            body: Dict[str, Any] = {"names": [name_obj]}
            if email:
                body["emailAddresses"] = [{"value": email, "type": "other"}]
            if phone:
                body["phoneNumbers"] = [{"value": phone, "type": "mobile"}]
            if org:
                body["organizations"] = [{"name": org}]
            if notes:
                body["biographies"] = [{"value": notes,
                                        "contentType": "TEXT_PLAIN"}]
            try:
                created = svc.people().createContact(body=body).execute()
            except Exception as exc:
                return connector_result(
                    "error",
                    error=friendly_api_error(
                        exc, api_label="Google Contacts"))
            resource_name = created.get("resourceName") or ""
            display = ((created.get("names") or [{}])[0].get("displayName")
                       or " ".join(p for p in [given, family] if p))
            id_portion = resource_name.split("people/", 1)[-1] if resource_name else ""
            link = (f"https://contacts.google.com/person/{id_portion}"
                    if id_portion else None)
            return connector_result("ok", created=True,
                                    resourceName=resource_name,
                                    display_name=display, link=link,
                                    summary=f"Added {display} to your contacts.")

        return connector_result("error",
                                error=f"unknown contacts tool: {name}",
                                code="no_handler")


def _primary(entries: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Return the entry flagged metadata.primary, else the first one."""
    if not entries:
        return None
    for entry in entries:
        meta = entry.get("metadata") or {}
        if meta.get("primary"):
            return entry
    return entries[0]


# Cap on names enumerated in the deterministic summary. Keeps the summary
# string bounded when the address book runs to hundreds of entries while
# still giving the LLM enough grounded, quotable content that it won't
# feel license to invent additional names to satisfy an "all A contacts"
# prompt. Truncation is announced explicitly ("showing 30 of 87") so the
# model knows more exist rather than reading the truncated list as
# authoritative.
_SUMMARY_NAME_CAP = 30


def _build_contact_entry(person: Dict[str, Any],
                         primary_name: Dict[str, Any],
                         emails: List[Dict[str, Any]],
                         phones: List[Dict[str, Any]],
                         orgs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the per-contact dict shipped to the LLM.

    Every entry carries explicit `has_email` / `has_phone` booleans and a
    `missing` list of field labels the address book has no value for. The
    `emails` / `phones` arrays remain the source of truth (empty list =
    no address on file), but the redundant boolean flags give the LLM a
    second grounded signal to prefer over pattern-completion when a user
    asks 'what's their email' and no email exists. See Investigation A
    (2026-08-04) for the fabricated-address root cause this defends.
    """
    flat_emails = [{"value": e.get("value"),
                    "type": e.get("type") or "other"}
                   for e in (emails or []) if e.get("value")]
    flat_phones = [{"value": p.get("value"),
                    "type": p.get("type") or "other"}
                   for p in (phones or []) if p.get("value")]
    primary_org = _primary(orgs) or {}
    missing: List[str] = []
    if not flat_emails:
        missing.append("email")
    if not flat_phones:
        missing.append("phone")
    return {
        "resourceName": person.get("resourceName"),
        "display_name": primary_name.get("displayName"),
        "given_name": primary_name.get("givenName"),
        "family_name": primary_name.get("familyName"),
        "emails": flat_emails,
        "phones": flat_phones,
        "has_email": bool(flat_emails),
        "has_phone": bool(flat_phones),
        "missing": missing,
        "organization": primary_org.get("name"),
    }


def _contact_line(entry: Dict[str, Any]) -> str:
    """Render one contact for the summary: name (email, phone).

    Missing fields become the literal strings 'no email' / 'no phone' so
    the model can quote the summary verbatim without ever needing to
    fabricate a placeholder address. Present fields use the primary
    (first) value on file — enough to answer 'what's their email' at a
    glance without ballooning the summary to every alias.
    """
    nm = (entry.get("display_name")
          or " ".join(p for p in [entry.get("given_name"),
                                  entry.get("family_name")] if p)
          or "(unnamed)")
    emails = entry.get("emails") or []
    phones = entry.get("phones") or []
    email_part = (emails[0].get("value") if emails else None) or "no email"
    phone_part = (phones[0].get("value") if phones else None) or "no phone"
    return f"{nm} ({email_part}, {phone_part})"


def _build_contacts_summary(entries: List[Dict[str, Any]], *,
                            total: int,
                            context: str,
                            starts_with: str = "",
                            query: str = "") -> str:
    """Compose the deterministic, self-grounding summary string.

    Names every contact returned (capped at `_SUMMARY_NAME_CAP`), each
    annotated with its primary email + phone or the literal
    'no email' / 'no phone' string when absent. When `total` exceeds
    what we enumerated, appends "(showing N of TOTAL)" so the model
    reports the honest window rather than treating the returned slice
    as exhaustive.
    """
    total = int(total or len(entries))
    if not entries:
        if context == "starts_with" and starts_with:
            return f"No contacts starting with '{starts_with}'."
        if context == "search" and query:
            return f"No contacts matching \"{query}\"."
        return "Your contacts list is empty."
    shown = entries[:_SUMMARY_NAME_CAP]
    lines = [_contact_line(e) for e in shown]
    names_str = "; ".join(lines)
    if total > len(shown):
        window_note = f" (showing {len(shown)} of {total})"
    else:
        window_note = ""
    if context == "starts_with" and starts_with:
        head = (f"You have {total} contact"
                f"{'' if total == 1 else 's'} starting with "
                f"'{starts_with}'")
    elif context == "search":
        head = (f"Found {total} contact"
                f"{'' if total == 1 else 's'}")
        if query:
            head = f"{head} matching \"{query}\""
    else:
        head = f"You have {total} contact{'' if total == 1 else 's'}"
    return f"{head}{window_note}: {names_str}."
