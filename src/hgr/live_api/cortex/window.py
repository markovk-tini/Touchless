"""CortexWindow — the modeless QMainWindow hosting the Cortex viz.

Hosts a QWebEngineView that loads ``web/cortex.html``. A QWebChannel
links Python ↔ JS via a single CortexBridge object registered as
``cortex`` on the JS side. The page is frameless (custom titlebar)
to match the cinematic vibe and Touchless's overall windowing style.

Lifecycle:
  - __init__: build the window + bridge + bus, load the page
  - bus.suspend() until JS calls jsReady(), then bus.resume()
  - closeEvent: stop the bus timer + clear web view

The window is intentionally lazy — created only when the user
clicks "Iris" in the assistant window header (see
live_assistant_window._open_cortex_window).

Author: Konstantin Markov
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, QPoint, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import cortex_emit
from .bridge import CortexBridge
from .event_bus import CortexEventBus
from .world_state import get_world


def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"[cortex-win {time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _resolve_web_root() -> Path:
    """Locate ``web/`` next to this module.

    Works in source runs. For bundled (PyInstaller) builds, the spec
    must add the package's ``web/`` tree under ``datas`` — see
    docs/IRIS_VISUALIZATION.md → Packaging notes.
    """
    here = Path(__file__).resolve().parent
    candidate = here / "web"
    if candidate.exists():
        return candidate
    # PyInstaller frozen fallback — _MEIPASS root + the relative path
    # the spec ought to bundle under.
    base = getattr(sys, "_MEIPASS", None)
    if base:
        bundled = Path(base) / "hgr" / "live_api" / "cortex" / "web"
        if bundled.exists():
            return bundled
    return candidate  # let downstream surface a clear error


class _BrandLabel(QLabel):
    """Clickable IRIS CORTEX brand text in the titlebar. Click emits
    ``clicked`` so the window can dispatch a 'return to overview' to
    the JS scene without exiting the window."""

    clicked = Signal()

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class _Titlebar(QWidget):
    """Frameless titlebar — drags the window, hosts close + minimize."""

    minimize = Signal()
    close = Signal()
    home = Signal()         # IRIS CORTEX brand clicked

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(34)
        self.setStyleSheet(
            "background-color: #06101e;"
            "color: #C8E6FF;"
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 0, 6, 0)
        row.setSpacing(6)
        brand = _BrandLabel("IRIS CORTEX")
        brand.setStyleSheet(
            "color: #58E3FF; letter-spacing: 3px; font-size: 11px; font-weight: 800;"
        )
        brand.setToolTip("Return to overview")
        brand.clicked.connect(self.home.emit)
        row.addWidget(brand)
        row.addStretch(1)

        btn_min = QPushButton("—")
        btn_close = QPushButton("✕")
        for b, color in ((btn_min, "#A0B6CC"), (btn_close, "#FF8FA3")):
            b.setFixedSize(28, 24)
            b.setCursor(QCursor(Qt.PointingHandCursor))
            b.setStyleSheet(
                f"QPushButton {{ background: transparent; color: {color}; "
                f"border: none; font-size: 13px; }}"
                f"QPushButton:hover {{ background: #102134; color: white; }}"
            )
        btn_min.clicked.connect(self.minimize.emit)
        btn_close.clicked.connect(self.close.emit)
        row.addWidget(btn_min)
        row.addWidget(btn_close)

        self._drag_origin: Optional[QPoint] = None

    # ---- window-dragging ----

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_origin = event.globalPosition().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_origin is None:
            return super().mouseMoveEvent(event)
        win = self.window()
        if win is None:
            return
        new_pos = event.globalPosition().toPoint()
        delta = new_pos - self._drag_origin
        self._drag_origin = new_pos
        win.move(win.pos() + delta)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_origin = None
        super().mouseReleaseEvent(event)


class CortexWindow(QMainWindow):
    """Modeless 3D Iris Cortex visualization window."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Iris Cortex")
        self.setMinimumSize(960, 640)
        self.resize(1280, 800)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        # Frameless + dark; we own the titlebar.
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setStyleSheet("background-color: #050a14;")

        # Lazy imports so PySide6 WebEngine isn't pulled in for users
        # who never open the visualization. Surface a clear error if
        # WebEngine isn't installed.
        try:
            from PySide6.QtWebChannel import QWebChannel
            from PySide6.QtWebEngineWidgets import QWebEngineView
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "PySide6 WebEngine is required for the Cortex window. "
                "Install with: pip install PySide6-Addons "
                f"(import error: {exc})"
            ) from exc

        # Build the chrome: titlebar + web view.
        container = QWidget(self)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._titlebar = _Titlebar(container)
        self._titlebar.close.connect(self.close)
        self._titlebar.minimize.connect(self.showMinimized)
        self._titlebar.home.connect(self._return_to_overview)
        outer.addWidget(self._titlebar)

        self._view = QWebEngineView(container)
        self._view.setMinimumSize(QSize(640, 480))
        # Paint the WebEngine page's underlying surface dark-blue BEFORE
        # the HTML/WebGL page loads, so we never see a white or black
        # flash through Qt's GPU compositor on focus-loss / focus-regain
        # cycles. WA_OpaquePaintEvent tells Qt not to clear behind us —
        # the QWebEnginePage's setBackgroundColor owns the surface.
        try:
            self._view.page().setBackgroundColor(QColor(5, 10, 20))
        except Exception:
            pass
        try:
            self._view.setAttribute(Qt.WA_OpaquePaintEvent, True)
        except Exception:
            pass
        outer.addWidget(self._view, 1)

        self.setCentralWidget(container)

        # Bridge + channel + bus.
        self._bridge = CortexBridge(self)
        self._channel = QWebChannel(self._view.page())
        self._channel.registerObject("cortex", self._bridge)
        self._view.page().setWebChannel(self._channel)

        # JS-injection fallback so emit_project_added /
        # emit_project_removed can hit the page directly even when the
        # QWebChannel signal drops. Same wiring as the embedded panel
        # in live_assistant_window._build_embedded_cortex.
        try:
            _page = self._view.page()
            def _run_js(src: str, _p=_page) -> None:
                try:
                    _p.runJavaScript(src)
                except Exception:
                    pass
            self._bridge.set_js_runner(_run_js)
        except Exception:
            pass

        self.bus = CortexEventBus(writer=self._bridge.emit_event, parent=self)
        # Don't push events until JS confirms it's wired up.
        self.bus.suspend()
        self._bridge.set_ready_callback(self._on_js_ready)
        self._bridge.set_node_focused_callback(self._on_node_focused)
        self._bridge.set_node_unfocused_callback(self._on_node_unfocused)
        self._bridge.set_open_file_callback(self._on_open_file)
        # Register the bus globally so Iris subsystems can push events
        # without needing a direct reference to this window. The shim
        # silently no-ops once we clear_bus() on close.
        cortex_emit.set_bus(self.bus)

        # Load the page.
        web_root = _resolve_web_root()
        html = web_root / "cortex.html"
        if not html.exists():
            _log(f"cortex.html not found at {html}")
            raise FileNotFoundError(f"Iris Cortex assets missing: {html}")
        self._view.load(QUrl.fromLocalFile(str(html)))

        # Seed a tiny idle scene so the view isn't blank for the user
        # on first open. Real subsystems wire in later (build plan v5).
        self._seed_idle_scene_on_ready = True

    # ---- JS ready handshake ----

    def _on_js_ready(self) -> None:
        """JS finished loading — flush any queued events and seed an idle scene."""
        try:
            self.bus.resume()
            if self._seed_idle_scene_on_ready:
                self._seed_idle_scene_on_ready = False
                self._seed_demo()
        except Exception as exc:
            _log(f"on_js_ready failed: {exc}")

    # Hardcoded preview sub-content per capability (Phase 2). Real
    # data wiring lands in Phase 5. Each entry is a (sub_id_suffix,
    # label) tuple — the full child id becomes "<capability_id>::<suffix>".
    _CAPABILITY_PREVIEW = {
        "cap-memory": [
            ("facts",     "Facts"),
            ("episodes",  "Episodes"),
            ("active",    "Active context"),
        ],
        "cap-voice": [
            ("backend",   "Backend"),
            ("recent",    "Recent transcripts"),
            ("devices",   "Devices"),
        ],
        "cap-tools": [
            ("apps",      "Apps & windows"),
            ("files",     "Files & folders"),
            ("typeclick", "Type & click"),
            ("screen",    "Read the screen"),
            ("web",       "Web"),
            ("code",      "Code & scripts"),
        ],
        "cap-realtime": [
            ("conn",      "Connection"),
            ("model",     "Model"),
            ("turns",     "Recent turns"),
        ],
    }

    # Known projects that are always shown in the cortex regardless
    # of what's in the persistent world state. Auto-detected projects
    # from world_state still appear alongside these.
    #
    # Each branch can declare its own leaves. Two formats supported:
    #   - list of {"label": str, "path": str?} dicts (static)
    #   - string "scan:<glob>" — expanded at seed time into one leaf
    #     per matching file
    #
    # Real .md files become double-clickable leaves that open via
    # os.startfile / xdg-open.
    _KNOWN_PROJECTS: List[Dict[str, Any]] = [
        {
            "id": "touchless-dev",
            "label": "Touchless dev",
            "weight": 0.92,
            "branches": [
                {
                    "id_suffix": "features",
                    "label": "Features",
                    "leaves": [
                        {"label": "Gesture system"},
                        {"label": "Voice commands"},
                        {"label": "Iris assistant"},
                        {"label": "Custom gestures"},
                        {"label": "Dictation"},
                    ],
                },
                {
                    "id_suffix": "bugs",
                    "label": "Bugs / planned",
                    "leaves": [
                        {"label": "OPEN_ISSUES.md",
                         "path": "c:/HGR App v1.0.0/OPEN_ISSUES.md"},
                        {"label": "Touchless to-do.md",
                         "path": "c:/HGR App v1.0.0/Touchless to-do.md"},
                    ],
                },
                {
                    "id_suffix": "docs",
                    "label": "Subsystem docs",
                    "leaves": "scan:c:/HGR App v1.0.0/docs/*.md",
                },
                {
                    "id_suffix": "code",
                    "label": "Code",
                    "leaves": [
                        {"label": "src/hgr/app"},
                        {"label": "src/hgr/live_api"},
                        {"label": "src/hgr/gesture"},
                        {"label": "src/hgr/voice"},
                    ],
                },
            ],
        },
        {
            "id": "marketing",
            "label": "Marketing",
            "weight": 0.55,
            "branches": [
                {
                    "id_suffix": "website",
                    "label": "Website",
                    "leaves": [
                        {"label": "touchless-control.com"},
                        {"label": "Pages"},
                        {"label": "Deploy"},
                    ],
                },
                {
                    "id_suffix": "instagram",
                    "label": "Instagram",
                    "leaves": [
                        {"label": "@touchlessapp"},
                        {"label": "INSTAGRAM_SETUP.md",
                         "path": "c:/touchless-marketing/docs/INSTAGRAM_SETUP.md"},
                        {"label": "Grid plan"},
                    ],
                },
                {
                    "id_suffix": "x",
                    "label": "X (Twitter)",
                    "leaves": [
                        {"label": "@touchlessapp"},
                        {"label": "Plan TBD"},
                    ],
                },
                {
                    "id_suffix": "plans",
                    "label": "Plans",
                    "leaves": "scan:c:/touchless-marketing/docs/*.md",
                },
            ],
        },
    ]

    def _seed_memory_contents(self, bus) -> None:
        """Populate the cap-memory::facts and cap-memory::episodes
        sub-sections with real rows from memory.db.

        Opens the DB read-only via SQLite so we never block writes
        from the live MemoryManager. Silent on every failure — the
        cortex viz must never break the assistant. Caps row counts
        so a huge memory store doesn't drown the scene; the user can
        drill into specific facts/episodes later via the focus mode.
        """
        try:
            from ..memory.manager import default_memory_path
            import sqlite3
        except Exception as exc:
            _log(f"_seed_memory_contents: imports failed: {exc}")
            return

        db_path = default_memory_path()
        try:
            db_exists = db_path.exists()
        except Exception:
            db_exists = False

        if not db_exists:
            # Drop a single explainer leaf so each sub-section reads
            # as "nothing here yet" rather than mysteriously empty.
            self._spawn_memory_explainer(
                bus, "cap-memory::facts", "(no facts yet — use Iris)"
            )
            self._spawn_memory_explainer(
                bus, "cap-memory::episodes", "(no episodes yet)"
            )
            self._spawn_memory_explainer(
                bus, "cap-memory::active",
                "Loaded each turn — relevant facts + recent episodes",
            )
            return

        # Read-only URI connect so we don't accidentally clobber the
        # live MemoryManager's writes.
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
            conn.row_factory = sqlite3.Row
        except Exception as exc:
            _log(f"_seed_memory_contents: open failed ({db_path}): {exc}")
            return

        try:
            self._seed_memory_facts(bus, conn)
            self._seed_memory_episodes(bus, conn)
            self._spawn_memory_explainer(
                bus, "cap-memory::active",
                "Loaded each turn — relevant facts + recent episodes",
            )
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _seed_memory_facts(self, bus, conn) -> None:
        """Spawn one leaf per semantic fact (capped at 60 newest)."""
        try:
            rows = conn.execute(
                "SELECT id, kind, key, value FROM semantic "
                "ORDER BY ts DESC LIMIT 60"
            ).fetchall()
        except Exception as exc:
            _log(f"_seed_memory_facts: query failed: {exc}")
            return

        if not rows:
            self._spawn_memory_explainer(
                bus, "cap-memory::facts", "(no facts yet — use Iris)"
            )
            return

        for row in rows:
            try:
                kind = (row["kind"] or "").strip()
                key = (row["key"] or "").strip()
                value = (row["value"] or "").strip()
                label = self._format_fact_label(kind, key, value)
                bus.node_spawn(
                    f"cap-memory::facts::{row['id']}",
                    label,
                    category="subnode",
                    weight=0.35,
                    parent_id="cap-memory::facts",
                    preview=True,
                )
            except Exception as exc:
                _log(f"_seed_memory_facts: spawn failed for row {row.get('id', '?')}: {exc}")

    def _seed_memory_episodes(self, bus, conn) -> None:
        """Spawn one leaf per recent episodic row (capped at 40)."""
        try:
            rows = conn.execute(
                "SELECT id, ts, user_text, outcome FROM episodic "
                "ORDER BY ts DESC LIMIT 40"
            ).fetchall()
        except Exception as exc:
            _log(f"_seed_memory_episodes: query failed: {exc}")
            return

        if not rows:
            self._spawn_memory_explainer(
                bus, "cap-memory::episodes", "(no episodes yet)"
            )
            return

        for row in rows:
            try:
                user_text = (row["user_text"] or "").strip()
                snippet = user_text[:52] + ("…" if len(user_text) > 52 else "")
                if not snippet:
                    snippet = f"(episode #{row['id']})"
                bus.node_spawn(
                    f"cap-memory::episodes::{row['id']}",
                    snippet,
                    category="subnode",
                    weight=0.32,
                    parent_id="cap-memory::episodes",
                    preview=True,
                )
            except Exception as exc:
                _log(f"_seed_memory_episodes: spawn failed for row {row.get('id', '?')}: {exc}")

    @staticmethod
    def _format_fact_label(kind: str, key: str, value: str) -> str:
        """Human-readable label for a fact leaf. Keeps both key and
        value visible when they fit; trims long values."""
        key = key or "?"
        value = (value or "").replace("\n", " ").strip()
        max_value = 32
        if len(value) > max_value:
            value = value[:max_value - 1] + "…"
        # Drop the 'kind' tag in the visible label unless the key alone
        # is ambiguous — most facts read cleanly as "key: value".
        return f"{key}: {value}" if value else key

    @staticmethod
    def _spawn_memory_explainer(bus, parent_id: str, text: str) -> None:
        """Helper for the explanatory placeholder leaves shown when
        a memory section is empty (or for the always-transient
        'Active context' section)."""
        try:
            bus.node_spawn(
                f"{parent_id}::__info",
                text,
                category="subnode",
                weight=0.30,
                parent_id=parent_id,
                preview=True,
            )
        except Exception as exc:
            _log(f"_spawn_memory_explainer failed for {parent_id}: {exc}")

    def _expand_branch_leaves(self, leaves) -> List[Dict[str, Any]]:
        """Materialize a branch's ``leaves`` field into a concrete list.

        Strings like ``"scan:<glob>"`` are expanded by globbing the
        filesystem (silently ignored if the glob hits nothing — e.g.
        a project folder doesn't exist on this machine).
        """
        if isinstance(leaves, str) and leaves.startswith("scan:"):
            pattern = leaves[len("scan:"):]
            out: List[Dict[str, Any]] = []
            try:
                for path in sorted(glob.glob(pattern)):
                    out.append({"label": Path(path).name, "path": path})
            except Exception as exc:
                _log(f"_expand_branch_leaves scan {pattern!r} failed: {exc}")
            return out
        if isinstance(leaves, list):
            return list(leaves)
        return []

    def _seed_known_project(self, bus, proj: Dict[str, Any]) -> None:
        """Spawn one known project + its branches + their leaves."""
        pid = proj["id"]
        # Project sphere.
        try:
            bus.node_spawn(
                pid,
                proj["label"],
                category="project",
                weight=float(proj.get("weight", 0.5)),
            )
        except Exception as exc:
            _log(f"_seed_known_project: spawn project {pid} failed: {exc}")
            return

        for branch in proj.get("branches", []):
            branch_id = f"{pid}::{branch['id_suffix']}"
            try:
                bus.node_spawn(
                    branch_id,
                    branch["label"],
                    category="subnode",
                    weight=0.50,
                    parent_id=pid,
                    preview=True,
                )
            except Exception as exc:
                _log(f"_seed_known_project: spawn branch {branch_id} failed: {exc}")
                continue

            leaves = self._expand_branch_leaves(branch.get("leaves", []))
            for i, leaf in enumerate(leaves):
                leaf_id = f"{branch_id}::{i}"
                try:
                    bus.node_spawn(
                        leaf_id,
                        leaf["label"],
                        category="subnode",
                        weight=0.35,
                        parent_id=branch_id,
                        preview=True,
                        path=leaf.get("path"),
                    )
                except Exception as exc:
                    _log(f"_seed_known_project: leaf {leaf_id} failed: {exc}")

    def _seed_demo(self) -> None:
        """Seed the cortex from the persistent world state.

        Spawns:
          - Each known project as a node, sized by computed weight
          - Preview-child sub-nodes around each project (recent files)
          - Preview-child sub-nodes around each capability (Memory,
            Voice, Tools, Realtime) — hardcoded for now; Phase 5
            wires real data

        Preview children render as tiny dots in the default overview.
        Click a parent to expand — children grow to full-size labelled
        nodes (handled JS-side via _on_node_focused).
        """
        bus = self.bus
        bus.core_state("idle", intensity=0.4)

        # ---- Capability preview children (always present) ----
        for cap_id, children in self._CAPABILITY_PREVIEW.items():
            for suffix, label in children:
                child_id = f"{cap_id}::{suffix}"
                try:
                    bus.node_spawn(
                        child_id,
                        label,
                        category="subnode",
                        weight=0.45,
                        parent_id=cap_id,
                        preview=True,
                    )
                except Exception as exc:
                    _log(f"_seed_demo: preview spawn failed for {child_id}: {exc}")

        # ---- Real Memory contents from memory.db ----
        # Facts + Episodes become leaves under their respective
        # cap-memory sub-sections; Active context gets a placeholder
        # explainer until live-session wiring lands (Phase 5).
        self._seed_memory_contents(bus)

        # ---- Known projects (always shown — Touchless dev, Marketing) ----
        seeded_known_ids = set()
        for proj in self._KNOWN_PROJECTS:
            self._seed_known_project(bus, proj)
            seeded_known_ids.add(proj["id"])

        # ---- Auto-detected projects from world state ----
        # These appear alongside the known ones. We skip any id we
        # already spawned as a known project to avoid duplicates.
        try:
            world = get_world()
        except Exception as exc:
            _log(f"_seed_demo: world unavailable: {exc}")
            return

        try:
            projects = world.list_projects()
        except Exception as exc:
            _log(f"_seed_demo: list_projects failed: {exc}")
            projects = []

        for proj in projects:
            pid = proj["id"]
            if pid in seeded_known_ids:
                continue
            try:
                bus.node_spawn(
                    pid,
                    proj["label"],
                    category="project",
                    weight=float(proj.get("weight", 0.5)),
                )
            except Exception as exc:
                _log(f"_seed_demo: project spawn failed for {pid}: {exc}")
                continue

            # Up to 5 preview file-leaves around this project.
            try:
                files = world.files_for_project(pid, limit=5)
            except Exception as exc:
                _log(f"_seed_demo: files_for_project failed for {pid}: {exc}")
                files = []

            for i, f in enumerate(files):
                child_id = f"{pid}::file::{i}"
                try:
                    bus.node_spawn(
                        child_id,
                        f["label"],
                        category="subnode",
                        weight=0.35,
                        parent_id=pid,
                        preview=True,
                        path=f.get("path"),
                    )
                except Exception as exc:
                    _log(f"_seed_demo: file preview failed for {f.get('path')}: {exc}")

        stats = world.stats() if projects else None
        if stats:
            _log(f"seeded from world: {stats}")

    # ---- focus callbacks (JS clicks / Esc) ----

    def _on_node_focused(self, node_id: str) -> None:
        """JS reported a focus event. Phase 2 doesn't need to do any
        extra spawning — children already exist in preview mode and
        will transition to full mode JS-side. Hook present so later
        phases can pull lazier (deeper) sub-graphs on demand."""
        _log(f"node focused: {node_id}")

    def _on_node_unfocused(self) -> None:
        """JS returned to overview. Phase 2 is a no-op for the same
        reason as _on_node_focused."""
        _log("node unfocused")

    def _return_to_overview(self) -> None:
        """Titlebar IRIS CORTEX click → tell the JS scene to unfocus.

        Reaches into the page via runJavaScript rather than adding
        another bridge slot, since the data flow is one-way and
        trivial here."""
        try:
            page = self._view.page()
            page.runJavaScript(
                "if (typeof unfocusNode === 'function') unfocusNode();"
            )
        except Exception as exc:
            _log(f"_return_to_overview failed: {exc}")

    def _on_open_file(self, path: str) -> None:
        """JS reported a double-click on a file-leaf. Open the file in
        the OS default handler. Silent on failure — the cortex viz is
        decorative, must never break the assistant."""
        if not path:
            return
        try:
            # os.startfile is Windows-only; on other platforms fall
            # back to xdg-open / open via subprocess.
            import os
            if hasattr(os, "startfile"):
                os.startfile(path)  # type: ignore[attr-defined]
                return
            # Cross-platform fallback (mac/linux).
            import shutil
            import subprocess
            opener = shutil.which("open") or shutil.which("xdg-open")
            if opener:
                subprocess.Popen([opener, path])
        except Exception as exc:
            _log(f"_on_open_file failed for {path!r}: {exc}")

    # ---- public Iris-side API ----

    def event_bus(self) -> CortexEventBus:
        """Return the CortexEventBus so Iris subsystems can push events."""
        return self.bus

    def bridge(self) -> CortexBridge:
        return self._bridge

    # ---- lifecycle ----

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Backup Esc handler at the Qt level. The JS scene also wires
        Esc → unfocus, but Qt only sees the key when the QWebEngineView
        doesn't have focus (e.g. user just clicked the titlebar). With
        both paths in place, Esc reliably returns to overview."""
        if event.key() == Qt.Key_Escape:
            self._return_to_overview()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        try:
            # Pass our specific bus so a stale close can't clobber a
            # window that was just reopened.
            cortex_emit.clear_bus(self.bus)
        except Exception:
            pass
        try:
            self.bus.shutdown()
        except Exception:
            pass
        try:
            # Detach the channel so the page can clean up cleanly.
            self._view.page().setWebChannel(None)
        except Exception:
            pass
        super().closeEvent(event)


def main() -> int:
    """Standalone launcher: opens the Cortex window with the demo seed."""
    import sys as _sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(_sys.argv)
    win = CortexWindow()
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
