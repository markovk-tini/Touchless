"""Tiny CLI for seeding semantic facts into iris memory.

Usage examples:
    # Default email-send tool preference:
    python tools/iris_memory_set.py preference default_send_via gmail_send

    # Person -> email mapping (the planner would normally learn this on its
    # own after the first time you emailed someone, but seeding is fine):
    python tools/iris_memory_set.py person dani dani@mangollc.org

    # Clear all memory:
    python tools/iris_memory_set.py --clear

    # List everything:
    python tools/iris_memory_set.py --list

The memory DB is at %LOCALAPPDATA%\\Touchless\\memory.db on Windows. Override
with TOUCHLESS_MEMORY_DB if you want to seed a different file.

Author: Konstantin Markov
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from hgr.live_api.memory import MemoryStore, default_memory_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", nargs="?", help='e.g. "person", "preference", "file"')
    ap.add_argument("key", nargs="?", help='e.g. "dani" or "default_send_via"')
    ap.add_argument("value", nargs="?", help='e.g. "dani@x.io" or "gmail_send"')
    ap.add_argument("--source", default="user-seeded")
    ap.add_argument("--list", action="store_true", help="show all facts")
    ap.add_argument("--clear", action="store_true", help="WIPE ALL MEMORY")
    args = ap.parse_args()

    store = MemoryStore(default_memory_path())

    if args.clear:
        store.clear()
        print(f"Cleared memory at {default_memory_path()}")
        return

    if args.list or not args.kind:
        rows = store.find_facts(limit=200)
        if not rows:
            print(f"(no facts in {default_memory_path()})")
            return
        for r in rows:
            print(f"  {r.kind:>12}  {r.key:>20} = {r.value}  ({r.source or '-'})")
        return

    if not (args.kind and args.key and args.value):
        ap.error("need kind key value")

    store.add_semantic(args.kind, args.key, args.value, args.source)
    print(f"Saved: {args.kind} {args.key} = {args.value}")
    print(f"  in {default_memory_path()}")


if __name__ == "__main__":
    main()
