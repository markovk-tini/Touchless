"""KiCAD CLI connector — wraps `kicad-cli` for batch operations.

KiCAD 7.0+ ships with a command-line tool that handles exports (gerbers,
PDF, SVG, BOM, STEP), DRC, format conversion, and project setup. This
connector exposes those as Tier-1 tools so 'export gerbers for my board'
becomes one fast call — no vision, no rate limits, no waiting.

Auto-discovers kicad-cli at startup by checking PATH then common
install locations on Windows / macOS / Linux. If found, the connector
is available; if not, `iris_setup_tool('kicad')` can re-run discovery
or accept an explicit path.

NOT a substitute for the IPC API (KiCAD 9.0+) — kicad-cli can't modify
schematics or place components. It's for batch ops and exports.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Connector, connector_result, friendly_api_error
from ...utils.subprocess_utils import hidden_subprocess_kwargs


# Common locations checked when kicad-cli isn't on PATH. We probe the
# newest KiCAD versions first so a user with both 8 and 10 installed
# uses 10 by default.
_WINDOWS_LOCATIONS = [
    r"C:\Program Files\KiCad\{ver}\bin\kicad-cli.exe",
    r"C:\Program Files (x86)\KiCad\{ver}\bin\kicad-cli.exe",
]
_MAC_LOCATIONS = [
    "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
    "/Applications/KiCad.app/Contents/MacOS/kicad-cli",
]
_LINUX_LOCATIONS = [
    "/usr/bin/kicad-cli", "/usr/local/bin/kicad-cli",
]
_KICAD_VERSIONS = ["10.0", "9.0", "8.99", "8.0", "7.99", "7.0"]


def _find_kicad_cli(override_path: str = "") -> Optional[str]:
    """Return the path to kicad-cli, or None. Honors an explicit override,
    then PATH, then OS-specific common install dirs (newest version first)."""
    if override_path:
        p = Path(override_path).expanduser()
        if p.exists() and p.is_file():
            return str(p)
    # PATH
    found = shutil.which("kicad-cli")
    if found:
        return found
    # OS-specific common locations
    if sys.platform.startswith("win"):
        for tmpl in _WINDOWS_LOCATIONS:
            for ver in _KICAD_VERSIONS:
                p = tmpl.format(ver=ver)
                if Path(p).exists():
                    return p
    elif sys.platform == "darwin":
        for p in _MAC_LOCATIONS:
            if Path(p).exists():
                return p
    else:
        for p in _LINUX_LOCATIONS:
            if Path(p).exists():
                return p
    return None


class KiCadCliConnector(Connector):
    """Exposes kicad-cli operations as deterministic tools.

    `setup_self(path='')` is callable by iris_setup_tool to (re)discover
    the binary — or accept an explicit path the user typed. After setup
    succeeds, the connector becomes available and its tools enter the
    next session's tool catalog.
    """

    id = "kicad_cli"
    description = ("KiCAD electronics CAD: export gerbers, PDF, SVG, BOM, "
                   "STEP 3D model, run DRC, project info. Uses kicad-cli, "
                   "available with any KiCAD 7.0+ install.")

    def __init__(self, cli_path: Optional[str] = None) -> None:
        self._cli_path = cli_path or _find_kicad_cli()

    def available(self) -> bool:
        # Re-probe on each call so the connector picks up a freshly
        # installed kicad-cli without restarting the app.
        if self._cli_path and Path(self._cli_path).exists():
            return True
        self._cli_path = _find_kicad_cli()
        return self._cli_path is not None

    def setup_self(self, path: str = "") -> Dict[str, Any]:
        """Re-discover kicad-cli, optionally accepting an explicit path.
        Returns {ok, cli_path, version} so the orchestrator can surface a
        useful reply to the user."""
        found = _find_kicad_cli(override_path=path)
        if not found:
            return {
                "ok": False,
                "error": ("kicad-cli not found. Install KiCAD from "
                          "https://www.kicad.org/download/ — kicad-cli "
                          "ships with the install. Or pass an explicit "
                          "path: 'set up kicad at C:\\path\\to\\kicad-cli.exe'."),
            }
        self._cli_path = found
        # Quick version probe to confirm it actually runs. hidden_subprocess_
        # kwargs suppresses the console-window flash on Windows --windowed
        # builds (kicad-cli is a console-mode binary).
        try:
            r = subprocess.run([found, "version"], capture_output=True,
                               text=True, timeout=5,
                               **hidden_subprocess_kwargs())
            version = (r.stdout or "").strip().splitlines()[0] if r.stdout else ""
        except Exception:
            version = ""
        return {"ok": True, "cli_path": found, "version": version}

    def tools(self) -> List[Dict[str, Any]]:
        def fn(name, desc, props=None, required=None):
            return {"type": "function", "name": name, "description": desc,
                    "parameters": {"type": "object",
                                   "properties": props or {},
                                   "required": required or [],
                                   "additionalProperties": False}}
        return [
            fn("kicad_export_gerbers",
               "Export Gerber files from a KiCAD PCB. `board_file` is a "
               "path to a .kicad_pcb file; outputs land in `output_dir` "
               "(default: alongside the board). Returns the list of "
               "generated files.",
               {"board_file": {"type": "string"},
                "output_dir": {"type": "string"}},
               ["board_file"]),
            fn("kicad_export_pdf",
               "Export a PDF from a KiCAD schematic (.kicad_sch) or "
               "board (.kicad_pcb). Output path is optional — defaults "
               "to the same directory.",
               {"source_file": {"type": "string"},
                "output_file": {"type": "string"}},
               ["source_file"]),
            fn("kicad_export_bom",
               "Export the Bill of Materials from a KiCAD schematic to "
               "CSV. Lists every component, value, footprint, and refs.",
               {"sch_file": {"type": "string"},
                "output_file": {"type": "string"}},
               ["sch_file"]),
            fn("kicad_run_drc",
               "Run Design Rule Check on a KiCAD board. Returns DRC "
               "violations + warnings. Use to validate a layout before "
               "fab.",
               {"board_file": {"type": "string"},
                "report_file": {"type": "string",
                                "description": "Optional report path."}},
               ["board_file"]),
            fn("kicad_export_step",
               "Export a STEP 3D model from a KiCAD board (for mechanical "
               "CAD integration / 3D viewing).",
               {"board_file": {"type": "string"},
                "output_file": {"type": "string"}},
               ["board_file"]),
            fn("kicad_version",
               "Report the installed KiCAD CLI version. Useful for "
               "confirming setup before running other tools.",
               {}),
        ]

    # ---- execution ---------------------------------------------------------
    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available():
            return connector_result(
                "error",
                error="kicad-cli not found. Say 'set up kicad' and I'll "
                      "look for it.",
                code="not_ready")

        try:
            if name == "kicad_export_gerbers":
                board = _abs(args.get("board_file"))
                if not board:
                    return connector_result("error", error="board_file is required")
                out_dir = _abs(args.get("output_dir")) or str(
                    Path(board).parent / "gerbers")
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                cmd = [self._cli_path, "pcb", "export", "gerbers",
                       "--output", out_dir, board]
                return _run(cmd, output_dir=out_dir)

            if name == "kicad_export_pdf":
                src = _abs(args.get("source_file"))
                if not src:
                    return connector_result("error", error="source_file is required")
                out = _abs(args.get("output_file")) or (
                    str(Path(src).with_suffix(".pdf")))
                subcmd = "sch" if src.endswith(".kicad_sch") else "pcb"
                cmd = [self._cli_path, subcmd, "export", "pdf",
                       "--output", out, src]
                return _run(cmd, output_file=out)

            if name == "kicad_export_bom":
                sch = _abs(args.get("sch_file"))
                if not sch:
                    return connector_result("error", error="sch_file is required")
                out = _abs(args.get("output_file")) or (
                    str(Path(sch).with_suffix(".bom.csv")))
                cmd = [self._cli_path, "sch", "export", "bom",
                       "--output", out, sch]
                return _run(cmd, output_file=out)

            if name == "kicad_run_drc":
                board = _abs(args.get("board_file"))
                if not board:
                    return connector_result("error", error="board_file is required")
                report = _abs(args.get("report_file")) or (
                    str(Path(board).with_suffix(".drc.json")))
                cmd = [self._cli_path, "pcb", "drc",
                       "--output", report, "--format", "json", board]
                result = _run(cmd, report_file=report)
                # Parse the report if it landed; surface violation count
                # so the LLM doesn't have to read the file separately.
                if result.get("status") == "ok" and Path(report).exists():
                    try:
                        with open(report, "r", encoding="utf-8") as f:
                            doc = json.load(f)
                        result["violation_count"] = len(doc.get("violations") or [])
                        result["unconnected_count"] = len(doc.get("unconnected_items") or [])
                    except Exception:
                        pass
                return result

            if name == "kicad_export_step":
                board = _abs(args.get("board_file"))
                if not board:
                    return connector_result("error", error="board_file is required")
                out = _abs(args.get("output_file")) or (
                    str(Path(board).with_suffix(".step")))
                cmd = [self._cli_path, "pcb", "export", "step",
                       "--output", out, board]
                return _run(cmd, output_file=out)

            if name == "kicad_version":
                cmd = [self._cli_path, "version"]
                return _run(cmd)

        except Exception as exc:
            return connector_result(
                "error", error=friendly_api_error(exc, api_label="KiCad CLI"))
        return connector_result(
            "error", error=f"unknown kicad tool: {name}", code="no_handler")


def _abs(p: Any) -> str:
    """Normalize a possibly-relative path, expand ~, return absolute str."""
    s = str(p or "").strip()
    if not s:
        return ""
    return str(Path(s).expanduser().resolve())


def _run(cmd: List[str], **extra) -> Dict[str, Any]:
    """Run kicad-cli, capture stdout/stderr, return a structured result.
    hidden_subprocess_kwargs keeps the console flash from appearing on
    Windows --windowed builds."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           **hidden_subprocess_kwargs())
    except subprocess.TimeoutExpired:
        return connector_result("error", error="kicad-cli timed out (>120s)")
    if r.returncode != 0:
        return connector_result(
            "error",
            error=(r.stderr or "").strip()[:600] or "kicad-cli failed",
            code=f"exit_{r.returncode}")
    return connector_result(
        "ok", stdout=(r.stdout or "").strip()[:2000], **extra)
