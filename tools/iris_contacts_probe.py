"""Diagnostic probe for Microsoft contacts_search.

Calls the real Microsoft Graph through the live MS365 connector, prints
which accounts are connected, which scopes the token actually carries,
and what each strategy returns for a given query. Run this when 'find
Dani' inexplicably comes up empty — it'll tell you whether the
problem is auth, scope, or the contact really not being where you
think it is.

Usage:
    python tools/iris_contacts_probe.py "Dani"
    python tools/iris_contacts_probe.py "Dani" --account "edu"  # restrict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from hgr.live_api.connectors.ms_graph_client import MsGraphClient  # noqa: E402
from hgr.live_api.connectors.ms365_connector import Microsoft365Connector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="name or substring to search for")
    ap.add_argument("--account", default="",
                    help="restrict to one account (substring of username)")
    args = ap.parse_args()

    client = MsGraphClient()
    accts = client.all_accounts()
    print(f"Connected Microsoft accounts ({len(accts)}):")
    for a in accts:
        print(f"  - {a.get('username')}  (home_account_id={a.get('home_account_id')!s:.40})")
    if not accts:
        print("  (none — sign in to Microsoft from the app first)")
        return

    # Show what scopes the active token actually has. Decoding the JWT's
    # 'scp' claim tells us if Contacts.ReadWrite was actually granted.
    print()
    for a in accts:
        tok = client.token_for(a)
        if not tok:
            print(f"  {a.get('username')}: token refresh FAILED — disconnect & reconnect")
            continue
        scp = _decode_scp(tok)
        print(f"  {a.get('username')} token scopes: {scp or '(could not decode)'}")
        if scp and "Contacts" not in scp:
            print(f"    ! 'Contacts.ReadWrite' is NOT in this token. "
                  f"Disconnect Microsoft and reconnect to upgrade scopes.")

    # Raw diagnostic calls so we can see what Microsoft Graph ACTUALLY says.
    # Bypass the connector's strategy stack and hit the endpoints directly.
    conn = Microsoft365Connector(client)
    for acct in accts:
        tok = client.token_for(acct)
        if not tok:
            continue
        print()
        print(f"=== Raw Graph calls for {acct.get('username')} ===")
        for label, path, headers in [
            ("/me (sanity — auth works?)", "/me", None),
            ("/me/contacts?$top=5 (any contacts at all?)",
             "/me/contacts?$top=5&$select=displayName,emailAddresses", None),
            (f"/me/contacts?$search=\"{args.query}\" (search)",
             f"/me/contacts?$search=\"{args.query}\"&$top=10"
             "&$select=displayName,emailAddresses",
             {"ConsistencyLevel": "eventual"}),
            (f"/me/contacts?$filter=startswith(displayName,'{args.query}')",
             f"/me/contacts?$top=10&$select=displayName,emailAddresses"
             f"&$filter=startswith(displayName,'{args.query}')", None),
            ("/me/people?$top=5 (correspondents)",
             "/me/people?$top=5", None),
            (f"/me/messages?$search=\"{args.query}\" (mail-search fallback)",
             f"/me/messages?$search=\"{args.query}\"&$top=10"
             "&$select=from,toRecipients,subject",
             {"ConsistencyLevel": "eventual"}),
        ]:
            data, err = conn._graph("GET", path, token=tok,
                                    extra_headers=headers)
            print(f"\n  {label}")
            if err:
                print(f"    ERR: {err[:300]}")
                continue
            if not isinstance(data, dict):
                print(f"    (non-dict response: {type(data).__name__})")
                continue
            keys = list(data.keys())
            print(f"    keys: {keys}")
            value = data.get("value") if "value" in keys else None
            if isinstance(value, list):
                print(f"    value count: {len(value)}")
                for entry in value[:5]:
                    dn = entry.get("displayName") or entry.get("userPrincipalName") or "?"
                    em = entry.get("emailAddresses") or entry.get("scoredEmailAddresses") or []
                    em_str = ", ".join(e.get("address") or "" for e in em)
                    print(f"      - {dn}  [{em_str}]")
            else:
                # /me just returns user info, no value array.
                print(f"    sample: {json.dumps({k: data[k] for k in keys[:6]}, default=str)[:200]}")

    print()
    print("=== contacts_search aggregate ===")
    result = conn.execute("contacts_search",
                          {"query": args.query, "account": args.account})
    print(json.dumps(result, indent=2, default=str)[:2000])


def _decode_scp(jwt: str) -> str:
    """JWTs are base64url-encoded header.payload.signature. The 'scp' claim
    in the payload lists granted scopes."""
    import base64
    try:
        payload_b64 = jwt.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("scp") or " ".join(payload.get("roles") or [])
    except Exception:
        return ""


if __name__ == "__main__":
    main()
