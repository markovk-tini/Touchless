"""DPAPI-backed secret vault.

Phase-1 trust substrate. Stores OAuth refresh tokens, API keys, MCP
server credentials encrypted with the user's Windows DPAPI key
(scoped to CurrentUser — the same key Windows uses for stored
browser passwords). The encrypted blob is meaningless to anyone who
copies the file: only the same Windows user account can decrypt it.

Why this matters:
  * Today Iris keeps tokens in plaintext under ~/Documents/Touchless/.
    OneDrive auto-syncs that to the cloud (a leak per CLAUDE.md rule
    #7). Family-shared accounts can browse Documents. Backup software
    silently exfiltrates it.
  * DPAPI ties decryption to the user's Windows login. Tokens that
    leave the machine (copy/paste, USB-stolen folder) decrypt to
    nothing.

Storage shape: a single SQLite file at
`%LOCALAPPDATA%\\Touchless\\private\\secrets.db` with one table:
    secrets(scope TEXT, key TEXT, ciphertext BLOB, updated REAL,
            PRIMARY KEY(scope, key))

Scopes group related secrets ("google", "ms365", "mcp:github", etc.)
so a Connect-X flow stays self-contained: revoking one scope wipes
exactly that scope.

Graceful degrade:
  * Non-Windows (CI, macOS dev) — falls back to plaintext storage
    in the same SQLite file. Marked `unencrypted=1` per row so
    callers can warn. (We don't pretend to encrypt without DPAPI.)
  * pywin32 missing — same plaintext fallback.

Author: Konstantin Markov
"""
from __future__ import annotations

import base64
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_vault_path() -> Path:
    """Per CLAUDE.md rule #7: private data goes under %LOCALAPPDATA%,
    NOT Documents (which syncs to OneDrive)."""
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(local) / "Touchless" / "private" / "secrets.db"


def _dpapi_available() -> bool:
    try:
        import win32crypt  # noqa: F401  type: ignore
        return True
    except Exception:
        return False


def _dpapi_encrypt(plaintext: bytes, scope: str) -> Optional[bytes]:
    """Wrap with DPAPI CurrentUser. Returns ciphertext bytes or None
    on failure. The `scope` becomes an additional-entropy parameter so
    even an attacker who can call DPAPI on the same machine has to
    also know the scope name to decrypt a given blob."""
    try:
        import win32crypt  # type: ignore
        # CryptProtectData(plain, description, entropy, ...) → cipher
        ct = win32crypt.CryptProtectData(
            plaintext, "iris-secret",
            scope.encode("utf-8"), None, None, 0)
        return bytes(ct)
    except Exception:
        return None


def _dpapi_decrypt(ciphertext: bytes, scope: str) -> Optional[bytes]:
    try:
        import win32crypt  # type: ignore
        _, pt = win32crypt.CryptUnprotectData(
            ciphertext, scope.encode("utf-8"), None, None, 0)
        return bytes(pt)
    except Exception:
        return None


class SecretVault:
    """Per-process secret vault. Thread-safe (write/read under lock).
    Singleton-friendly: multiple constructions pointing at the same
    DB file will share the row state but each carries its own
    connection — fine because we serialize via lock per-instance and
    sqlite handles cross-process writes with its own file locks.

    API:
        vault = SecretVault()
        vault.put("google", "refresh_token", "...")
        token = vault.get("google", "refresh_token")  # → str | None
        vault.delete("google", "refresh_token")
        vault.delete_scope("google")  # wipes all google secrets
        vault.scopes()  # → ["google", "ms365", "mcp:github"]
        vault.is_encrypted_at_rest()  # → True on Windows w/ pywin32
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS secrets (
        scope       TEXT NOT NULL,
        key         TEXT NOT NULL,
        ciphertext  BLOB NOT NULL,
        encrypted   INTEGER NOT NULL DEFAULT 1,
        updated     REAL NOT NULL,
        PRIMARY KEY (scope, key)
    );
    CREATE INDEX IF NOT EXISTS idx_secrets_scope ON secrets(scope);
    """

    def __init__(self, *, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _default_vault_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._encrypted = _dpapi_available()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.executescript(self.SCHEMA)

    # ---- introspection ------------------------------------------------

    def is_encrypted_at_rest(self) -> bool:
        return self._encrypted

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ---- core API -----------------------------------------------------

    def put(self, scope: str, key: str, value: str) -> bool:
        """Store `value` (str) under (scope, key). Overwrites if exists.
        Returns True on success."""
        if not scope or not key:
            return False
        pt = value.encode("utf-8")
        if self._encrypted:
            ct = _dpapi_encrypt(pt, scope)
            if ct is None:
                # DPAPI errored — bias toward NOT silently storing
                # plaintext on a machine that has DPAPI. Caller can
                # retry with vault force-fallback if it wants.
                return False
            enc_flag = 1
            blob = ct
        else:
            # Non-Windows / DPAPI missing — store base64 to keep the
            # blob "I copy/pasted this and it's clearly not random
            # garbage" obvious to a curious user.
            enc_flag = 0
            blob = base64.b64encode(pt)
        with self._lock:
            self._conn.execute(
                "INSERT INTO secrets(scope, key, ciphertext, encrypted, updated) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(scope, key) DO UPDATE SET "
                "  ciphertext=excluded.ciphertext, "
                "  encrypted=excluded.encrypted, "
                "  updated=excluded.updated",
                (scope, key, blob, enc_flag, time.time()),
            )
        return True

    def get(self, scope: str, key: str) -> Optional[str]:
        """Retrieve a secret. Returns None if missing or decryption
        fails (caller treats as "not present")."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT ciphertext, encrypted FROM secrets "
                "WHERE scope=? AND key=?",
                (scope, key),
            )
            row = cur.fetchone()
        if row is None:
            return None
        blob, enc_flag = row
        if enc_flag:
            pt = _dpapi_decrypt(blob, scope)
            if pt is None:
                return None
            try:
                return pt.decode("utf-8")
            except Exception:
                return None
        try:
            return base64.b64decode(blob).decode("utf-8")
        except Exception:
            return None

    def delete(self, scope: str, key: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM secrets WHERE scope=? AND key=?",
                (scope, key))
            return cur.rowcount > 0

    def delete_scope(self, scope: str) -> int:
        """Wipe every secret in `scope`. Returns count deleted. Used by
        'Disconnect X' flows."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM secrets WHERE scope=?", (scope,))
            return cur.rowcount

    def keys_in_scope(self, scope: str) -> List[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT key FROM secrets WHERE scope=? ORDER BY key",
                (scope,))
            return [r[0] for r in cur.fetchall()]

    def scopes(self) -> List[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT DISTINCT scope FROM secrets ORDER BY scope")
            return [r[0] for r in cur.fetchall()]

    def count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM secrets")
            return int(cur.fetchone()[0])

    def wipe_all(self) -> int:
        """Delete every secret. Returns count. Used by the
        delete-everything kill switch."""
        with self._lock:
            before = self.count()
            self._conn.execute("DELETE FROM secrets")
            return before

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ---- module-level singleton ----------------------------------------------


_global_vault: Optional[SecretVault] = None
_vault_lock = threading.Lock()


def global_vault() -> SecretVault:
    global _global_vault
    if _global_vault is None:
        with _vault_lock:
            if _global_vault is None:
                _global_vault = SecretVault()
    return _global_vault
