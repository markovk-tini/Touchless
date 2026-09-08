"""Self-grounding contract for the contacts_list / contacts_search
deterministic summary + payload.

Investigation A (2026-08-04) found the Realtime backend sends the FULL
contacts_list tool result (contacts[] array + summary) to the LLM, but
the LLM was still fabricating email addresses for contacts whose
`emails` array is empty. Root cause: the connector's summary field only
named the first 5 display names with no per-field grounding, so under
prompt pressure ("what's their email?") the model completed the pattern
with a plausible-looking placeholder.

These tests pin the fix:

1. Every entry in `contacts` MUST carry explicit `has_email` /
   `has_phone` booleans and a `missing` list of missing field labels.
2. The `summary` string MUST name every returned contact (up to 30)
   with their actual primary email/phone or the literal strings
   'no email' / 'no phone' — NEVER an invented placeholder.
3. The orchestrator's client-side fallback (used when the connector's
   summary is somehow missing) MUST emit the same 'no email' /
   'no phone' phrasing rather than degrading to a bare name list.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.connectors.contacts_connector import (  # noqa: E402
    _build_contact_entry,
    _build_contacts_summary,
)


def _mk_person(name: str, emails=(), phones=()):
    """Shape a fake People API `person` payload."""
    return {
        "resourceName": f"people/{name.replace(' ', '_').lower()}",
        "names": [{"displayName": name, "givenName": name.split()[0]}],
        "emailAddresses": [{"value": e, "type": "home"} for e in emails],
        "phoneNumbers": [{"value": p, "type": "mobile"} for p in phones],
        "organizations": [],
    }


def _build(person):
    return _build_contact_entry(
        person=person,
        primary_name=person["names"][0],
        emails=person.get("emailAddresses") or [],
        phones=person.get("phoneNumbers") or [],
        orgs=person.get("organizations") or [],
    )


# ---- per-contact self-describing flags -------------------------------

def test_entry_has_email_and_phone_flags_true():
    entry = _build(_mk_person("Alice Full",
                              emails=["alice@x.com"], phones=["555-0100"]))
    assert entry["has_email"] is True
    assert entry["has_phone"] is True
    assert entry["missing"] == []
    assert entry["emails"][0]["value"] == "alice@x.com"
    assert entry["phones"][0]["value"] == "555-0100"


def test_entry_email_only_flags_phone_missing():
    entry = _build(_mk_person("Bob Email", emails=["bob@x.com"]))
    assert entry["has_email"] is True
    assert entry["has_phone"] is False
    assert entry["emails"] and entry["phones"] == []
    assert "phone" in entry["missing"]
    assert "email" not in entry["missing"]


def test_entry_neither_field_flags_both_missing():
    entry = _build(_mk_person("Carol Empty"))
    assert entry["has_email"] is False
    assert entry["has_phone"] is False
    assert entry["emails"] == []
    assert entry["phones"] == []
    assert set(entry["missing"]) == {"email", "phone"}


# ---- summary self-grounding ------------------------------------------

def _build_three_summary(context="list", **extra):
    entries = [
        _build(_mk_person("Alice Full",
                          emails=["alice@x.com"], phones=["555-0100"])),
        _build(_mk_person("Bob Email", emails=["bob@x.com"])),
        _build(_mk_person("Carol Empty")),
    ]
    return entries, _build_contacts_summary(
        entries, total=len(entries), context=context, **extra)


def test_summary_names_every_contact_with_email_and_phone_grounding():
    entries, summary = _build_three_summary()
    # All three names present.
    assert "Alice Full" in summary
    assert "Bob Email" in summary
    assert "Carol Empty" in summary
    # Real values verbatim.
    assert "alice@x.com" in summary
    assert "555-0100" in summary
    assert "bob@x.com" in summary
    # Missing fields spelled out — the LLM must never need to
    # fabricate a placeholder to satisfy a "what's their email"
    # follow-up.
    assert "no phone" in summary, summary
    assert "no email" in summary, summary
    # Ensure the phrase "no email" appears at least once (Carol)
    # AND "no phone" appears at least twice (Bob + Carol).
    assert summary.count("no phone") >= 2
    assert summary.count("no email") >= 1
    # And critically: no fabricated placeholder patterns.
    assert "example.com" not in summary
    assert "@x.com, no phone" in summary or "bob@x.com, no phone" in summary


def test_summary_uses_count_of_3_and_correct_pluralization():
    _entries, summary = _build_three_summary()
    assert summary.startswith("You have 3 contacts")


def test_starts_with_summary_includes_prefix_label():
    entries = [
        _build(_mk_person("Alice One",
                          emails=["a1@x.com"], phones=["555-0001"])),
        _build(_mk_person("Alan Two", emails=["a2@x.com"])),
    ]
    summary = _build_contacts_summary(
        entries, total=2, context="starts_with", starts_with="A")
    assert "starting with 'A'" in summary
    assert "Alice One" in summary and "Alan Two" in summary
    assert "no phone" in summary  # Alan
    assert "a1@x.com" in summary


def test_summary_truncation_announces_window():
    # 32 contacts, cap is 30 → must say "(showing 30 of 32)".
    entries = [
        _build(_mk_person(f"Person{i:02d}", emails=[f"p{i}@x.com"]))
        for i in range(32)
    ]
    summary = _build_contacts_summary(
        entries, total=len(entries), context="list")
    assert "(showing 30 of 32)" in summary


def test_summary_empty_list_says_so_plainly():
    summary = _build_contacts_summary([], total=0, context="list")
    assert summary == "Your contacts list is empty."


# ---- orchestrator fallback ------------------------------------------

def test_orchestrator_fallback_uses_no_email_no_phone_phrasing():
    """When the connector's `summary` field is missing (defensive
    fallback), the orchestrator's own formatter still emits the
    grounded 'no email' / 'no phone' phrasing rather than a bare name
    list the model could dress up with fabricated fields.
    """
    from hgr.live_api.planner.orchestrator import IrisPlanner
    from hgr.live_api.planner.plan import Step

    entries = [
        _build(_mk_person("Alice Full",
                          emails=["alice@x.com"], phones=["555-0100"])),
        _build(_mk_person("Bob Email", emails=["bob@x.com"])),
        _build(_mk_person("Carol Empty")),
    ]
    result_no_summary = {
        "status": "ok",
        "count": 3,
        "total": 3,
        "contacts": entries,
        # NO "summary" field on purpose.
    }
    step = Step(tool="contacts_list", args={})
    text = IrisPlanner._format_message(step, result_no_summary)
    assert "Alice Full" in text
    assert "Bob Email" in text
    assert "Carol Empty" in text
    assert "no phone" in text
    assert "no email" in text
    assert "example.com" not in text
