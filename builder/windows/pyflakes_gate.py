"""Build-time undefined-name gate (v1.1.9.2 r18).

Runs pyflakes over src/hgr and FAILS the build on any "undefined name"
that is not in builder/windows/pyflakes_allow.txt, and on any syntax
error. Three shipped 1.1.9.2 builds (r15 `os`, r16 `QRectF`, and the
`number` hit in test_window.py) were NameErrors that only surfaced in
the frozen exe because `sys.frozen` short-circuits hid them in source
runs. pyflakes catches all three statically in ~10 s.

Allowlist format: one entry per line, `path undefined name 'X'` with
forward slashes and NO line number (line numbers drift; the pair of
file + name is what identifies a known-benign string annotation).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOW_FILE = Path(__file__).with_name("pyflakes_allow.txt")
_HIT = re.compile(r"^(.+?):(\d+):(\d+):\s*(undefined name '.*')\s*$")


def main() -> int:
    allow: set[str] = set()
    if ALLOW_FILE.exists():
        for raw in ALLOW_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                allow.add(line.replace("\\", "/"))
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pyflakes", "src/hgr"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        print(f"[pyflakes-gate] FAIL: could not run pyflakes: {exc}")
        return 1
    output = (proc.stdout or "") + (proc.stderr or "")
    if "No module named pyflakes" in output:
        print("[pyflakes-gate] FAIL: pyflakes is not installed in the build venv (pip install pyflakes).")
        return 1
    bad: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "invalid syntax" in line or "SyntaxError" in line or "unexpected indent" in line:
            bad.append(line)
            continue
        if "undefined name" not in line:
            continue
        m = _HIT.match(line)
        if not m:
            bad.append(line)
            continue
        key = f"{m.group(1).replace(chr(92), '/')} {m.group(4)}"
        if key in allow:
            continue
        bad.append(line)
    if bad:
        print("[pyflakes-gate] FAIL: undefined names / syntax errors not in allowlist:")
        for b in bad:
            print("  " + b)
        print("  (string-annotation false positives go in builder/windows/pyflakes_allow.txt)")
        return 1
    print(f"[pyflakes-gate] OK: no undefined names ({len(allow)} allowlisted string-annotation hits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Author: Konstantin Markov
