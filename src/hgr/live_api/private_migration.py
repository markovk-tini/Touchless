"""Migration: lift private data from ~/Documents/Touchless/ → vault.

Phase-1 trust substrate. One-time, idempotent migration that:

  1. Locates well-known plaintext-token files under ~/Documents/
     (Notion token, Spotify OAuth cache, future MCP token files).
  2. Imports them into the DPAPI-encrypted SecretVault.
  3. Renames the original files to `.migrated` so we don't re-import
     and the user can see what was moved.

We DO NOT delete the originals — leaving them as `.migrated` is a
"breadcrumb trail" the user can verify before manually wiping. The
delete-everything kill switch (later in Batch 3) handles permanent
cleanup if the user wants it.

Per CLAUDE.md rule #7: end users should never see "edit your tokens
JSON" instructions. This migration runs silently at startup once; if
nothing was found, it's a no-op.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .secret_vault import SecretVault, global_vault


def _docs_root() -> Path:
    return Path.home() / "Documents" / "Touchless"


# Migration spec table. Each entry: (rel_path_under_docs, scope, key,
# extractor). The extractor takes the file contents (str) and returns
# the secret string (str), or None to skip (e.g. malformed file).
#
# Adding a new known-token location is a one-line addition.
def _extract_notion(text: str) -> Optional[str]:
    # The notion token file is either raw token text or JSON {"token": ...}.
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            data = json.loads(text)
            tok = data.get("token") or data.get("api_key")
            return tok if isinstance(tok, str) and tok.strip() else None
        except Exception:
            return None
    return text if (text.startswith("secret_")
                    or text.startswith("ntn_")) else text


def _extract_raw(text: str) -> Optional[str]:
    text = (text or "").strip()
    return text or None


def _extract_oauth_json(text: str) -> Optional[str]:
    """Spotify/Google OAuth cache: returns the whole JSON as one string,
    vault stores it as an opaque blob. The connector reads it back and
    re-parses."""
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, dict) or not data:
        return None
    return json.dumps(data, separators=(",", ":"))


_MIGRATIONS: List[Tuple[str, str, str, Callable[[str], Optional[str]]]] = [
    # rel_path,                          scope,          key,            extractor
    ("notion/notion_token.txt",          "notion",       "api_key",      _extract_notion),
    ("spotify/oauth.json",               "spotify",      "oauth_cache",  _extract_oauth_json),
    ("ms365/active_account.txt",         "ms365",        "active",       _extract_raw),
    ("gmail/token.json",                 "google",       "oauth_cache",  _extract_oauth_json),
    ("openai/api_key.txt",               "openai",       "api_key",      _extract_raw),
    ("brave/api_key.txt",                "brave",        "api_key",      _extract_raw),
]


def migrate_once(*, vault: Optional[SecretVault] = None,
                 docs_root: Optional[Path] = None,
                 dry_run: bool = False) -> Dict[str, Any]:
    """Run the one-time migration. Idempotent — already-migrated files
    end in `.migrated` and are skipped. Returns a summary dict for
    logging/debug:
        {"migrated": [(scope, key, source_path), ...],
         "skipped":  [(scope, key, reason), ...],
         "errors":   [(scope, key, message), ...]}

    Pass `dry_run=True` to inventory what WOULD migrate without
    touching disk or the vault — useful for the UI to show a preview."""
    if vault is None:
        vault = global_vault()
    if docs_root is None:
        docs_root = _docs_root()
    summary: Dict[str, Any] = {"migrated": [], "skipped": [], "errors": []}
    for rel, scope, key, extractor in _MIGRATIONS:
        src = docs_root / rel
        if not src.exists():
            summary["skipped"].append((scope, key, "no source file"))
            continue
        migrated_marker = src.with_suffix(src.suffix + ".migrated")
        if migrated_marker.exists() and not src.exists():
            # Shouldn't happen (we check src.exists above) but defensive.
            summary["skipped"].append((scope, key, "already migrated"))
            continue
        # Try to read + extract.
        try:
            text = src.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            summary["errors"].append(
                (scope, key, f"read failed: {type(exc).__name__}: {exc}"))
            continue
        value = extractor(text)
        if value is None:
            summary["errors"].append(
                (scope, key, "extractor returned None (malformed?)"))
            continue
        if dry_run:
            summary["migrated"].append((scope, key, str(src)))
            continue
        # Write into the vault.
        try:
            ok = vault.put(scope, key, value)
        except Exception as exc:
            summary["errors"].append(
                (scope, key, f"vault.put failed: {type(exc).__name__}: {exc}"))
            continue
        if not ok:
            summary["errors"].append((scope, key, "vault.put returned False"))
            continue
        # Rename the original so we don't re-import. We keep it on disk
        # as a breadcrumb the user can manually wipe — see CLAUDE.md
        # rule #7 reasoning.
        try:
            src.rename(migrated_marker)
        except Exception as exc:
            # The vault now has the secret but the source survives —
            # next run would re-import. Surface so the user can fix.
            summary["errors"].append(
                (scope, key,
                 f"vault stored but rename failed: {exc} "
                 f"(secret already in vault — manually rename "
                 f"{src} to prevent re-import)"))
            continue
        summary["migrated"].append((scope, key, str(src)))
    return summary
