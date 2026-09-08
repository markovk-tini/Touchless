"""MCP server picker dialog.

Renders the curated default catalog (`mcp_default_servers.DEFAULT_SERVERS`)
grouped by category. Each row has:

  * an enable/disable checkbox
  * the server's name, category, and one-line description
  * for servers that need env vars (tokens / API keys), an inline input
    for each — saved into the server entry's `env` block so they don't
    have to be exported globally
  * for servers that need command-line arguments (e.g. filesystem root
    path), an inline input
  * a small "🔗 setup" link button that opens the relevant
    api-tokens / install page

Save writes the full server list (including disabled entries — preserves
the user's saved credentials so re-enabling later is one click) to
~/Documents/Touchless/mcp_servers.json via mcp_bridge.write_server_configs.

Changes take effect on the NEXT session start (the manager rebuilds the
connector registry when it starts, picking up the new config).

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from ...live_api.connectors.mcp_default_servers import (
    DEFAULT_SERVERS, McpServerSpec, TIER_LABELS, by_id, categories,
    zero_setup_ids,
)
from ...live_api.connectors.mcp_bridge import (
    libs_available, load_raw_servers, write_server_configs,
)


def _is_frozen() -> bool:
    """True when running from the PyInstaller installer (vs source).
    Used to hide developer-only affordances like `pip install` buttons
    (pip can't write into a frozen install) and to apply zero-friction
    defaults in the picker UX."""
    return bool(getattr(sys, "frozen", False))


_DEFAULT_PALETTE = {
    "primary": "#0B3D91",
    "accent": "#1DE9B6",
    "surface": "#0F172A",
    "text": "#E5F6FF",
}


def _runtime_available(runtime: str) -> bool:
    """Cheap PATH lookup so we can tell the user 'this needs Node, install it'."""
    if runtime == "node":
        return shutil.which("npx") is not None
    if runtime == "python":
        return shutil.which("uvx") is not None or shutil.which("pipx") is not None
    return True


# ---- runtime install panel ------------------------------------------------


class _PipInstaller(QObject):
    """Background `pip install <pkg>` runner. Emits status updates so the
    UI can show progress without blocking the Qt event loop."""

    progress = Signal(str)         # human-readable line
    finished = Signal(bool, str)   # (success, final message)

    def install(self, package: str) -> None:
        t = threading.Thread(
            target=self._do_install, args=(package,),
            name=f"pip-install-{package}", daemon=True)
        t.start()

    def _do_install(self, package: str) -> None:
        self.progress.emit(f"pip install {package} …")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "--upgrade", package],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
            )
        except Exception as exc:
            self.finished.emit(False, f"failed to spawn pip: {exc}")
            return
        # Stream stdout so the user sees we're alive; we only forward
        # 'collecting', 'downloading', 'installing' lines to avoid spam.
        last = ""
        for line in proc.stdout or []:
            stripped = line.strip()
            low = stripped.lower()
            if any(k in low for k in ("collecting", "downloading",
                                       "installing", "successfully")):
                last = stripped
                self.progress.emit(stripped[:140])
        code = proc.wait()
        if code == 0:
            self.finished.emit(True, f"installed {package}.")
        else:
            self.finished.emit(False,
                               f"pip exited {code} — {last or 'see log'}")


class _RuntimePanel(QFrame):
    """Status + one-click install for the three runtimes MCP servers need.

      * Python `mcp` SDK   → required by the bridge itself
      * `uv` (gives uvx)   → needed for Python-based servers
      * Node.js (gives npx) → needed for Node-based servers

    The first two we can `pip install` directly. Node has to come from
    nodejs.org — we just open the download page and tell the user to
    reopen the picker after installing."""

    refreshed = Signal()

    def __init__(self, palette: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._installer = _PipInstaller(self)
        self._installer.progress.connect(self._on_progress)
        self._installer.finished.connect(self._on_finished)
        self._current_install: Optional[str] = None
        self._rows: Dict[str, Dict[str, QWidget]] = {}
        self._build()

    def _build(self) -> None:
        pal = self._palette
        self.setStyleSheet(
            "_RuntimePanel { background:rgba(88,227,255,0.04); "
            f"border:1px solid {pal['accent']}33; border-radius:8px; }}"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(6)

        title = QLabel("Runtimes")
        title.setStyleSheet(
            f"color:{pal['accent']}; font-weight:700; font-size:11px; "
            "letter-spacing:1px;")
        outer.addWidget(title)

        # Three rows.
        outer.addLayout(self._make_row(
            key="mcp_sdk",
            name="Python `mcp` SDK",
            why="Required by the MCP bridge itself.",
            checker=libs_available,
            action_label="Install via pip",
            action=lambda: self._installer.install("mcp"),
        ))
        outer.addLayout(self._make_row(
            key="uv",
            name="`uv` (uvx)",
            why="Needed for Python servers: Git, Sentry, SQLite, Time, Fetch.",
            checker=lambda: shutil.which("uvx") is not None,
            action_label="Install via pip",
            action=lambda: self._installer.install("uv"),
        ))
        outer.addLayout(self._make_row(
            key="node",
            name="Node.js (npx)",
            why="Needed for Node servers: GitHub, Slack, Filesystem, Memory, Puppeteer, etc.",
            checker=lambda: shutil.which("npx") is not None,
            action_label="Open download page",
            action=lambda: webbrowser.open(
                "https://nodejs.org/en/download/"),
            external=True,
        ))

        self._status = QLabel("")
        self._status.setStyleSheet("color:#94A3B8; font-size:10px;")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

    def _make_row(self, *, key: str, name: str, why: str,
                  checker: Callable[[], bool],
                  action_label: str,
                  action: Callable[[], None],
                  external: bool = False) -> QHBoxLayout:
        pal = self._palette
        row = QHBoxLayout()
        row.setSpacing(8)

        ok = bool(checker())
        dot = QLabel("●")
        dot.setFixedWidth(12)
        dot.setStyleSheet(
            f"color:{'#1DE9B6' if ok else '#94A3B8'}; font-size:12px;")
        row.addWidget(dot)

        name_lbl = QLabel(name)
        name_lbl.setFixedWidth(150)
        name_lbl.setStyleSheet(
            f"color:{pal['text']}; font-weight:700; font-size:11px;")
        row.addWidget(name_lbl)

        why_lbl = QLabel(why)
        why_lbl.setStyleSheet("color:#94A3B8; font-size:10px;")
        why_lbl.setWordWrap(True)
        row.addWidget(why_lbl, 1)

        btn = QPushButton("installed ✓" if ok else action_label)
        btn.setEnabled(not ok)
        btn.setCursor(QCursor(Qt.PointingHandCursor))
        btn.setStyleSheet(self._btn_style(ok))
        btn.clicked.connect(lambda: self._on_install_clicked(key, action, external))
        row.addWidget(btn)

        self._rows[key] = {"dot": dot, "btn": btn, "checker": checker,
                           "external": external}
        return row

    def _btn_style(self, ok: bool) -> str:
        pal = self._palette
        if ok:
            return (
                f"QPushButton {{ background:transparent; color:#1DE9B6; "
                f"border:1px solid #1DE9B655; border-radius:6px; "
                f"padding:4px 10px; font-size:10px; }}")
        return (
            f"QPushButton {{ background:{pal['primary']}; color:{pal['text']}; "
            f"border:1px solid {pal['accent']}55; border-radius:6px; "
            f"padding:4px 10px; font-size:10px; font-weight:700; }}"
            f"QPushButton:hover {{ border:1px solid {pal['accent']}; }}"
            f"QPushButton:disabled {{ color:#64748B; "
            f"background:transparent; border-color:#33415544; }}")

    def _on_install_clicked(self, key: str, action: Callable[[], None],
                            external: bool) -> None:
        if external:
            # External installer (Node) — open the page and tell the user
            # to come back after running the installer; we can't detect
            # PATH changes without a restart.
            try:
                action()
            except Exception:
                pass
            self._status.setText(
                "Node download page opened — after installing, RESTART "
                "Touchless so PATH picks up the new node/npx binaries.")
            return
        if _is_frozen() and not external:
            # In a packaged install, `pip install` can't write into the
            # frozen Python — pretend we're not offering it. Direct the
            # user to wait for an app update instead.
            self._status.setText(
                "This runtime ships with the app — wait for the next "
                "Touchless update or run from source to install it.")
            return
        if self._current_install is not None:
            return
        self._current_install = key
        # Disable all pip buttons while one runs.
        for r in self._rows.values():
            if not r["external"]:
                r["btn"].setEnabled(False)
        self._status.setText(f"Installing {key}…")
        try:
            action()
        except Exception as exc:
            self._on_finished(False, f"spawn failed: {exc}")

    def _on_progress(self, line: str) -> None:
        self._status.setText(line)

    def _on_finished(self, ok: bool, msg: str) -> None:
        self._status.setText(msg)
        self._current_install = None
        # Recheck and refresh all rows.
        for key, r in self._rows.items():
            installed = bool(r["checker"]())
            r["dot"].setStyleSheet(
                f"color:{'#1DE9B6' if installed else '#94A3B8'}; font-size:12px;")
            btn: QPushButton = r["btn"]  # type: ignore[assignment]
            btn.setEnabled(not installed)
            if installed:
                btn.setText("installed ✓")
                btn.setStyleSheet(self._btn_style(True))
            else:
                # Restore the original label by passing through the
                # row's action — we kept it on the button via closure, so
                # only the style needs refreshing here.
                btn.setStyleSheet(self._btn_style(False))
        if ok:
            self.refreshed.emit()


def _spec_to_initial_config(spec: McpServerSpec) -> Dict[str, object]:
    """Turn a catalog spec into a fresh JSON entry — disabled by default,
    with empty env / args slots ready for the user to fill in."""
    return {
        "id": spec.id,
        "name": spec.id,  # the bridge uses 'name' as the connector id segment
        "command": spec.command,
        "args": list(spec.args),
        "env": {k: "" for k in spec.env_required},
        "env_required": list(spec.env_required),
        "description": spec.description,
        "enabled": False,
        "needs_args_keys": list(spec.needs_args),
    }


def _merge_with_catalog(saved: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Combine the user's saved entries with the catalog so the picker
    always shows ALL defaults (any new catalog additions are surfaced
    without having to delete mcp_servers.json), while preserving the
    user's saved enabled / env / args values for entries they already have."""
    by_saved_id = {str(s.get("id") or s.get("name")): s for s in saved}
    merged: List[Dict[str, object]] = []
    for spec in DEFAULT_SERVERS:
        if spec.id in by_saved_id:
            entry = dict(by_saved_id[spec.id])
            # Refresh static fields from catalog (description, args
            # template, env_required) so future catalog updates flow
            # through without resetting the user's tokens.
            entry["description"] = spec.description
            entry["env_required"] = list(spec.env_required)
            entry["needs_args_keys"] = list(spec.needs_args)
            merged.append(entry)
        else:
            merged.append(_spec_to_initial_config(spec))
    # Preserve user's custom-added (non-catalog) entries too.
    for s in saved:
        if str(s.get("id") or s.get("name")) not in {sp.id for sp in DEFAULT_SERVERS}:
            merged.append(s)
    return merged


# ---- one row in the picker -------------------------------------------------


class _ServerRow(QFrame):
    """One server's enable/config widget."""

    changed = Signal()

    def __init__(self, spec: McpServerSpec, entry: Dict[str, object],
                 palette: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._spec = spec
        self._entry = entry
        self._palette = palette
        self._env_edits: Dict[str, QLineEdit] = {}
        self._arg_edits: List[QLineEdit] = []
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._build()

    def _build(self) -> None:
        pal = self._palette
        self.setStyleSheet(
            f"_ServerRow {{ background:rgba(255,255,255,0.02); "
            f"border:1px solid #1E293B; border-radius:8px; }}"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(6)

        # ---- top row: checkbox + name + setup link
        head = QHBoxLayout()
        self._chk = QCheckBox()
        self._chk.setChecked(bool(self._entry.get("enabled")))
        self._chk.stateChanged.connect(lambda _=None: self.changed.emit())
        head.addWidget(self._chk)

        name = QLabel(self._spec.name)
        name.setStyleSheet(
            f"color:{pal['text']}; font-weight:700; font-size:13px;"
        )
        head.addWidget(name)

        cat = QLabel(self._spec.category)
        cat.setStyleSheet(
            "color:#94A3B8; font-size:10px; padding:2px 6px; "
            "background:rgba(148,163,184,0.10); border-radius:4px;"
        )
        head.addWidget(cat)

        # Tier badge — instantly tells the user whether they need to do
        # anything before this works. Color codes:
        #   zero    -> green (just works)
        #   path    -> blue  (you supply a path)
        #   oauth   -> teal  (one-click browser auth)
        #   api_key -> amber (advanced / token paste)
        tier = getattr(self._spec, "tier", "api_key")
        tier_label, _ = TIER_LABELS.get(tier, ("Advanced", ""))
        tier_color = {
            "zero":    "#1DE9B6",
            "path":    "#58E3FF",
            "oauth":   "#34D399",
            "api_key": "#F59E0B",
        }.get(tier, "#94A3B8")
        tier_chip = QLabel(tier_label)
        tier_chip.setStyleSheet(
            f"color:{tier_color}; font-size:10px; font-weight:700; "
            f"padding:2px 6px; background:{tier_color}1A; "
            f"border:1px solid {tier_color}44; border-radius:4px;")
        head.addWidget(tier_chip)
        head.addStretch(1)

        # Runtime warning when the binary isn't on PATH.
        if not _runtime_available(self._spec.runtime):
            warn = QLabel(f"needs {self._spec.runtime}")
            warn.setStyleSheet(
                "color:#F59E0B; font-size:10px; padding:2px 6px;"
                "background:rgba(245,158,11,0.10); border-radius:4px;")
            head.addWidget(warn)

        if self._spec.setup_url:
            link = QPushButton("🔗 setup")
            link.setFixedWidth(74)
            link.setCursor(QCursor(Qt.PointingHandCursor))
            link.setStyleSheet(
                f"QPushButton {{ background:transparent; "
                f"color:{pal['accent']}; border:1px solid {pal['accent']}55; "
                "border-radius:6px; padding:3px 8px; font-size:10px; }}"
                f"QPushButton:hover {{ border-color:{pal['accent']}; }}"
            )
            link.clicked.connect(self._open_setup_url)
            head.addWidget(link)

        # Phase-1 MCP trust chip. Every MCP server starts as PENDING
        # (no tools dispatchable) until the user explicitly grants
        # trust. The chip shows current state + flips on click.
        try:
            from ...live_api.mcp_trust import (
                global_store as _trust_store, TrustLevel)
            server_id = "mcp_" + self._spec.id
            self._trust_btn = QPushButton(
                self._trust_btn_label(
                    _trust_store().trust_level(server_id)))
            self._trust_btn.setCursor(QCursor(Qt.PointingHandCursor))
            self._trust_btn.setToolTip("Grant/revoke this server's tool-dispatch permission.")
            self._trust_btn.clicked.connect(self._on_trust_clicked)
            self._restyle_trust_btn(_trust_store().trust_level(server_id))
            head.addWidget(self._trust_btn)
        except Exception:
            self._trust_btn = None
        outer.addLayout(head)

        # ---- description
        blurb = QLabel(self._spec.description)
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color:#94A3B8; font-size:11px;")
        outer.addWidget(blurb)

        # ---- env-var inputs
        env_block = self._entry.get("env") or {}
        for key in self._spec.env_required:
            row = QHBoxLayout()
            row.setSpacing(8)
            lbl = QLabel(key)
            lbl.setFixedWidth(180)
            lbl.setStyleSheet(
                "color:#CBD5E1; font-size:11px; "
                "font-family:Consolas,monospace;")
            edit = QLineEdit(str(env_block.get(key) or ""))
            edit.setEchoMode(QLineEdit.Password)
            edit.setPlaceholderText("paste token / api key here")
            edit.setStyleSheet(self._line_edit_style())
            edit.textChanged.connect(lambda _t, k=key: (
                self._env_edits[k].text() and self.changed.emit()))
            self._env_edits[key] = edit
            row.addWidget(lbl)
            row.addWidget(edit, 1)
            outer.addLayout(row)

        # ---- needs_args inputs (filesystem root, db connection, etc.)
        if self._spec.needs_args:
            existing_args = list(self._entry.get("args") or list(self._spec.args))
            # Anything past the catalog-defined static args is user-supplied.
            user_supplied = existing_args[len(self._spec.args):]
            for i, label in enumerate(self._spec.needs_args):
                row = QHBoxLayout()
                row.setSpacing(8)
                lbl = QLabel(label)
                lbl.setFixedWidth(180)
                lbl.setStyleSheet(
                    "color:#CBD5E1; font-size:11px; "
                    "font-family:Consolas,monospace;")
                edit = QLineEdit(user_supplied[i] if i < len(user_supplied) else "")
                edit.setPlaceholderText(self._placeholder_for(label))
                edit.setStyleSheet(self._line_edit_style())
                edit.textChanged.connect(lambda _t: self.changed.emit())
                self._arg_edits.append(edit)
                row.addWidget(lbl)
                row.addWidget(edit, 1)
                outer.addLayout(row)

    def _line_edit_style(self) -> str:
        pal = self._palette
        return (
            f"QLineEdit {{ background:#0B1220; color:{pal['text']}; "
            f"border:1px solid {pal['accent']}33; border-radius:6px; "
            f"padding:4px 8px; font-size:11px; "
            f"font-family:Consolas,monospace; }}"
            f"QLineEdit:focus {{ border-color:{pal['accent']}; }}"
        )

    def _placeholder_for(self, key: str) -> str:
        if "path" in key:
            return r"e.g. C:\Users\me\Projects"
        if "url" in key.lower():
            return "e.g. postgresql://user:pass@host:5432/db"
        return key

    def _open_setup_url(self) -> None:
        try:
            webbrowser.open(self._spec.setup_url)
        except Exception:
            pass

    # ---- MCP trust toggle ---------------------------------------------
    @staticmethod
    def _trust_btn_label(level) -> str:
        # Compact label per trust level. Click cycles.
        try:
            v = getattr(level, "value", str(level))
        except Exception:
            v = str(level)
        return {
            "pending":       "🔒 trust pending",
            "trusted_read":  "👁 read only",
            "trusted_write": "✏ read+write",
            "trusted_full":  "🔓 full trust",
            "revoked":       "⛔ revoked",
        }.get(v, "🔒 trust pending")

    def _restyle_trust_btn(self, level) -> None:
        if not getattr(self, "_trust_btn", None):
            return
        try:
            v = getattr(level, "value", str(level))
        except Exception:
            v = str(level)
        color = {
            "pending":       ("#94A3B8", "#94A3B833"),
            "trusted_read":  ("#58E3FF", "#58E3FF22"),
            "trusted_write": ("#1DE9B6", "#1DE9B622"),
            "trusted_full":  ("#34D399", "#34D39922"),
            "revoked":       ("#F87171", "#F8717122"),
        }.get(v, ("#94A3B8", "#94A3B833"))
        fg, bg = color
        self._trust_btn.setStyleSheet(
            f"QPushButton {{ background:{bg}; color:{fg}; "
            f"border:1px solid {fg}66; border-radius:6px; "
            f"padding:3px 8px; font-size:10px; }}"
            f"QPushButton:hover {{ border-color:{fg}; }}")

    def _on_trust_clicked(self) -> None:
        try:
            from ...live_api.mcp_trust import (
                global_store as _trust_store, TrustLevel)
        except Exception:
            return
        server_id = "mcp_" + self._spec.id
        store = _trust_store()
        # Cycle: pending → read → write → full → revoked → pending.
        order = [TrustLevel.PENDING, TrustLevel.TRUSTED_READ,
                 TrustLevel.TRUSTED_WRITE, TrustLevel.TRUSTED_FULL,
                 TrustLevel.REVOKED]
        try:
            current = store.trust_level(server_id)
        except Exception:
            current = TrustLevel.PENDING
        try:
            idx = order.index(current)
        except ValueError:
            idx = 0
        next_level = order[(idx + 1) % len(order)]
        # Ensure the server is registered so the grant takes effect.
        try:
            store.register(server_id, self._spec.name, tool_count=0)
        except Exception:
            pass
        try:
            if next_level == TrustLevel.REVOKED:
                store.revoke(server_id)
            else:
                store.grant(server_id, next_level)
        except Exception:
            return
        if self._trust_btn is not None:
            self._trust_btn.setText(self._trust_btn_label(next_level))
            self._restyle_trust_btn(next_level)

    def collect(self) -> Dict[str, object]:
        """Snapshot the row's current state back into the JSON entry shape."""
        env = dict(self._entry.get("env") or {})
        for k, edit in self._env_edits.items():
            env[k] = edit.text().strip()
        args = list(self._spec.args)
        for edit in self._arg_edits:
            t = edit.text().strip()
            if t:
                args.append(t)
        out = dict(self._entry)
        out["enabled"] = self._chk.isChecked()
        out["env"] = env
        out["args"] = args
        # Refresh catalog-derived static fields each save.
        out["command"] = self._spec.command
        out["env_required"] = list(self._spec.env_required)
        out["needs_args_keys"] = list(self._spec.needs_args)
        out["description"] = self._spec.description
        return out


# ---- main dialog -----------------------------------------------------------


class McpPickerDialog(QDialog):
    """Modeless dialog letting the user enable + configure MCP servers."""

    servers_saved = Signal()

    def __init__(self, *, palette: Optional[dict] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MCP servers")
        self.setModal(False)
        self.setMinimumSize(640, 720)
        self._palette = dict(_DEFAULT_PALETTE)
        if palette:
            self._palette.update(palette)
        self._rows: List[_ServerRow] = []
        self._dirty = False
        # r51: install indigo chrome (previously bare OS-default).
        from .window_chrome import install_indigo_chrome
        self._body = install_indigo_chrome(self, "MCP servers")
        self._build_ui()

    def _build_ui(self) -> None:
        pal = self._palette
        self.setStyleSheet(
            f"QDialog {{ background-color:{pal['surface']}; color:{pal['text']}; }}"
        )
        root = QVBoxLayout(self._body)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(10)

        # ---- header
        header = QLabel("MCP server connections")
        header.setStyleSheet(
            f"color:{pal['text']}; font-weight:800; font-size:16px;")
        root.addWidget(header)

        sub = QLabel(
            "Enable any server below — Iris loads its tools on demand "
            "via find_capability, so adding servers doesn't bloat the "
            "model context. Green chips = zero-setup. Amber = needs a "
            "token. Changes apply on the next Start.")
        sub.setStyleSheet("color:#94A3B8; font-size:11px;")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # ---- one-click row: enable all zero-setup servers, toggle adv.
        actions_row = QHBoxLayout()
        actions_row.setSpacing(8)
        self._enable_zero_btn = QPushButton("✨ Enable all zero-setup servers")
        self._enable_zero_btn.setToolTip(
            "Enable every zero-setup server (skips API-key ones).")
        self._enable_zero_btn.setStyleSheet(self._btn_style(subtle=False))
        self._enable_zero_btn.clicked.connect(self._on_enable_all_zero)
        actions_row.addWidget(self._enable_zero_btn)

        self._show_advanced_chk = QCheckBox("Show advanced (API-key) servers")
        # In a packaged build we hide advanced by default — most users
        # never paste tokens and seeing greyed-out rows looks broken.
        # In source / dev mode show them by default (you DO use them).
        self._show_advanced_chk.setChecked(not _is_frozen())
        self._show_advanced_chk.stateChanged.connect(self._apply_advanced_filter)
        self._show_advanced_chk.setStyleSheet(
            f"QCheckBox {{ color:{pal['text']}; font-size:11px; }}")
        actions_row.addStretch(1)
        actions_row.addWidget(self._show_advanced_chk)
        root.addLayout(actions_row)

        # ---- runtime install panel (interactive — one-click pip install
        # for the Python runtimes; opens the Node download page for Node).
        # In a frozen build, this panel still surfaces status (so the
        # user can see what's missing) but the install buttons are
        # gated — they can't run pip into a frozen Python.
        self._runtime_panel = _RuntimePanel(self._palette, self)
        root.addWidget(self._runtime_panel)

        # ---- scrolling list, grouped by category
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(2, 2, 2, 2)
        col.setSpacing(10)

        saved = load_raw_servers()
        merged = _merge_with_catalog(saved)
        by_eid = {str(e.get("id") or e.get("name")): e for e in merged}

        for cat in categories():
            cat_label = QLabel(cat)
            cat_label.setStyleSheet(
                f"color:{pal['accent']}; font-weight:700; "
                "font-size:12px; padding-top:8px; letter-spacing:1px;")
            col.addWidget(cat_label)
            for spec in DEFAULT_SERVERS:
                if spec.category != cat:
                    continue
                entry = by_eid.get(spec.id) or _spec_to_initial_config(spec)
                row = _ServerRow(spec, entry, self._palette, self)
                row.changed.connect(self._mark_dirty)
                self._rows.append(row)
                col.addWidget(row)
        col.addStretch(1)
        scroll.setWidget(host)
        root.addWidget(scroll, 1)

        # Apply the advanced-tier visibility filter (default state set
        # above via _show_advanced_chk.setChecked).
        self._apply_advanced_filter()

        # ---- footer
        footer = QHBoxLayout()
        self._footer_status = QLabel("")
        self._footer_status.setStyleSheet("color:#94A3B8; font-size:11px;")
        footer.addWidget(self._footer_status, 1)

        cancel = QPushButton("Close")
        cancel.setStyleSheet(self._btn_style(subtle=True))
        cancel.clicked.connect(self.reject)
        footer.addWidget(cancel)

        save = QPushButton("Save")
        save.setStyleSheet(self._btn_style(subtle=False))
        save.clicked.connect(self._on_save)
        footer.addWidget(save)
        root.addLayout(footer)

    def _btn_style(self, *, subtle: bool) -> str:
        pal = self._palette
        bg = "transparent" if subtle else pal["primary"]
        return (
            f"QPushButton {{ background:{bg}; color:{pal['text']}; "
            f"border:1px solid {pal['accent']}55; border-radius:8px; "
            f"padding:7px 16px; font-weight:700; font-size:12px; }}"
            f"QPushButton:hover {{ border:1px solid {pal['accent']}; }}"
        )

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._footer_status.setText("Unsaved changes")

    def _on_enable_all_zero(self) -> None:
        """One-click: flip every zero-setup row's checkbox on. Excludes
        servers that need API keys, paths, or external binaries the user
        hasn't installed yet (covered by zero_setup_ids() at the catalog
        level)."""
        target = set(zero_setup_ids())
        flipped = 0
        for row in self._rows:
            if row._spec.id in target and not row._chk.isChecked():
                row._chk.setChecked(True)
                flipped += 1
        if flipped:
            self._footer_status.setText(
                f"Enabled {flipped} zero-setup server"
                + ("s" if flipped != 1 else "")
                + " — click Save to apply.")
        else:
            self._footer_status.setText(
                "All zero-setup servers were already enabled.")

    def _apply_advanced_filter(self) -> None:
        """Hide / show api_key-tier rows based on the 'Show advanced'
        checkbox. The end-user default in shipped builds hides them so
        the picker doesn't look full of unusable rows."""
        show_advanced = self._show_advanced_chk.isChecked()
        for row in self._rows:
            tier = getattr(row._spec, "tier", "api_key")
            row.setVisible(show_advanced or tier != "api_key")

    def _on_save(self) -> None:
        servers = [row.collect() for row in self._rows]
        try:
            write_server_configs(servers)
        except Exception as exc:
            self._footer_status.setText(f"Save failed: {exc}")
            return
        self._dirty = False
        self._footer_status.setText(
            "Saved. Restart the session (Stop → Start) to load the new "
            "MCP tools.")
        self.servers_saved.emit()


# Standalone-runnable for quick UI iteration.
if __name__ == "__main__":  # pragma: no cover
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dlg = McpPickerDialog()
    dlg.show()
    sys.exit(app.exec())
