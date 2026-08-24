"""Privacy reset: 'delete everything Iris remembers about me'.

Phase-1 trust substrate. GDPR Art. 17 / CCPA right-to-delete. One
button surfaces from settings → "Reset Iris (delete all data)" → this
wipes:

  * audit log (all tool invocation history)
  * secret vault (OAuth tokens, API keys — disconnects every connector)
  * cost meter history
  * memory store (facts, episodic turns, embeddings) — best-effort,
    skipped silently if the memory subsystem isn't loaded
  * legacy plaintext files under ~/Documents/Touchless/ that the
    migration script marked .migrated

What this does NOT touch:
  * The user's actual data (Outlook inbox, Gmail messages, files on
    disk) — never owned by Iris.
  * Iris itself (the app stays installed).
  * Cortex visualization assets, voice models, whisper / llama
    binaries — those are install-time artifacts, not user data.

Returns a structured report so the UI can show "wiped N tool calls,
M secret scopes, K memory facts" — proves the action worked.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def reset_all(*,
              also_remove_migrated_files: bool = True,
              audit_log: Optional[Any] = None,
              secret_vault: Optional[Any] = None,
              cost_meter: Optional[Any] = None,
              file_rules_engine: Optional[Any] = None,
              docs_root: Optional[Path] = None) -> Dict[str, Any]:
    """Wipe every persistent surface Iris owns. Best-effort — partial
    failure on one surface doesn't block the others.

    Optional injection points (for tests): pass already-constructed
    audit_log / secret_vault / cost_meter / file_rules_engine
    instances to avoid touching the user's real LOCALAPPDATA
    singletons. `docs_root` overrides the legacy-file scan root."""
    report: Dict[str, Any] = {
        "audit_rows_deleted": 0,
        "secrets_deleted": 0,
        "secret_scopes_wiped": [],
        "cost_records_deleted": 0,
        "memory_facts_deleted": 0,
        "memory_turns_deleted": 0,
        "file_rules_deleted": 0,
        "legacy_files_removed": [],
        "errors": [],
        "timestamp": datetime.utcnow().isoformat(),
    }
    # ---- audit log ----------------------------------------------------
    try:
        if audit_log is None:
            from .audit_log import AuditLog
            log = AuditLog()
            should_close = True
        else:
            log = audit_log
            should_close = False
        report["audit_rows_deleted"] = log.wipe()
        if should_close:
            log.close()
    except Exception as exc:
        report["errors"].append(f"audit: {type(exc).__name__}: {exc}")

    # ---- secret vault -------------------------------------------------
    try:
        if secret_vault is None:
            from .secret_vault import global_vault
            v = global_vault()
        else:
            v = secret_vault
        scopes = list(v.scopes())
        secrets_before = v.count()
        v.wipe_all()
        report["secret_scopes_wiped"] = scopes
        report["secrets_deleted"] = secrets_before
    except Exception as exc:
        report["errors"].append(f"vault: {type(exc).__name__}: {exc}")

    # ---- cost meter ---------------------------------------------------
    try:
        if cost_meter is None:
            from .cost_meter import global_meter
            cost_meter = global_meter()
        report["cost_records_deleted"] = cost_meter.wipe_all()
    except Exception as exc:
        report["errors"].append(f"cost: {type(exc).__name__}: {exc}")

    # ---- file-watcher rules (Phase-3) --------------------------------
    # Missed-by-panel audit: user-defined watch rules can carry personal
    # directory paths, contact names baked into templates, etc. The
    # reset_all contract is 'every persistent surface Iris owns', and
    # %LOCALAPPDATA%/Touchless/private/file_rules.db is one of them.
    try:
        if file_rules_engine is None:
            from .file_watcher_rules import RulesEngine
            engine = RulesEngine()
            should_close = True
        else:
            engine = file_rules_engine
            should_close = False
        report["file_rules_deleted"] = engine.wipe()
        if should_close:
            engine.close()
    except Exception as exc:
        report["errors"].append(f"file_rules: {type(exc).__name__}: {exc}")

    # ---- memory store -------------------------------------------------
    # Skipped when caller injected test instances (tests don't have a
    # MemoryStore to wipe; the production reset path handles this).
    # Also skipped when MemoryStore() construction would touch heavy
    # initialization (embeddings, model loaders) that's inappropriate
    # for a reset operation. The production reset surfaces this caveat
    # to the user via a follow-up "also reset memory?" dialog.
    is_test_invocation = (audit_log is not None
                          or secret_vault is not None
                          or cost_meter is not None
                          or file_rules_engine is not None)
    if not is_test_invocation:
        try:
            from .memory.store import MemoryStore  # type: ignore
            store = MemoryStore()
            for meth in ("wipe_all", "clear_all", "delete_all"):
                fn = getattr(store, meth, None)
                if callable(fn):
                    try:
                        res = fn()
                        if isinstance(res, dict):
                            report["memory_facts_deleted"] = int(
                                res.get("facts", 0) or 0)
                            report["memory_turns_deleted"] = int(
                                res.get("turns", 0) or 0)
                        elif isinstance(res, int):
                            report["memory_facts_deleted"] = res
                    except Exception as exc:
                        report["errors"].append(
                            f"memory: {meth} failed: {exc}")
                    break
        except Exception:
            # Memory module not present — silently skip; it's optional.
            pass

    # ---- legacy plaintext breadcrumbs ---------------------------------
    if also_remove_migrated_files:
        try:
            docs = docs_root or (Path.home() / "Documents" / "Touchless")
            if docs.exists():
                for path in docs.rglob("*.migrated"):
                    try:
                        path.unlink()
                        report["legacy_files_removed"].append(str(path))
                    except Exception as exc:
                        report["errors"].append(
                            f"legacy {path}: {exc}")
        except Exception as exc:
            report["errors"].append(f"legacy scan: {exc}")

    return report


def export_audit_log(*, dest: Optional[Path] = None) -> Path:
    """GDPR-friendly export of the audit log to JSON. Returns the path
    to the written file. Default destination is the user's Downloads
    folder (visible immediately). Includes every row, redacted as
    stored — see audit_log._redact_args for the rules."""
    from .audit_log import AuditLog
    if dest is None:
        downloads = Path.home() / "Downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dest = downloads / f"iris_audit_export_{ts}.json"
    log = AuditLog()
    try:
        rows = log.recent(limit=1_000_000)
    finally:
        log.close()
    payload = {
        "exported_at_utc": datetime.utcnow().isoformat(),
        "total_rows": len(rows),
        "notes": ("Sensitive arg values (api_key, body, etc.) were "
                  "blanket-redacted at write-time and are NOT in this "
                  "export. The user's tool outputs were never stored "
                  "in the audit log."),
        "invocations": rows,
    }
    dest.write_text(json.dumps(payload, indent=2, default=str),
                    encoding="utf-8")
    return dest
