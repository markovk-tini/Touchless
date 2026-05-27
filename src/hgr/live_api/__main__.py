"""Standalone typed runner for the Live API agent (Phase 1 test harness).

Run:
    set OPENAI_API_KEY=sk-...            (cloud backend, default)
    python -m hgr.live_api

or the fully-local backend (no key, needs llama.cpp + Qwen present):
    set TOUCHLESS_LIVE_API_BACKEND=local
    python -m hgr.live_api

Type commands at the prompt and watch the pipeline work end-to-end:

    open chrome                  -> Layer 0 router handles it locally (no
                                    LLM call, no tokens spent)
    create a folder called demo  -> router declines; the LLM agent loop
      with main.py inside           takes over and calls tools

This harness exists so the whole stack (local-first router + realtime
LLM + tool execution) can be exercised without the full Touchless GUI.
Risky tools are auto-approved here with a printed notice — the real
in-app UI will show a proper confirmation dialog instead.
"""
from __future__ import annotations

import sys
import threading

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from .live_api_manager import LiveApiManager


class _Console(QObject):
    """Marshals stdin lines onto the Qt main thread.

    Reading stdin happens on a worker thread; `send_user_text` and the
    signals it fires must run on the thread that owns the manager, so we
    hop across via this queued-connection signal.
    """

    submit = Signal(str)

    def __init__(self, manager: LiveApiManager) -> None:
        super().__init__()
        self._manager = manager
        self.submit.connect(self._on_submit)

    def _on_submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if not self._manager.send_user_text(text):
            print("  (not ready yet — still connecting? try again in a moment)")


def main() -> int:
    app = QApplication(sys.argv)
    manager = LiveApiManager(text_only=True)

    # CLI confirm: auto-approve risky tools but announce them. The
    # in-app UI replaces this with a real yes/no dialog. We can't prompt
    # on stdin here because the stdin reader thread already owns it.
    def _confirm(title: str, detail: str) -> bool:
        print(f"\n  [auto-confirm] {title} — {detail}")
        return True

    manager.set_confirm_callback(_confirm)
    manager.state_changed.connect(
        lambda state, status: print(f"  [state] {getattr(state, 'value', state)} — {status}")
    )
    manager.transcript_received.connect(lambda t: print(f"\nYOU: {t}"))
    manager.assistant_text.connect(lambda t: print(t, end="", flush=True))
    manager.tool_event.connect(
        lambda kind, info: print(
            f"\n  [tool {kind}] {info.get('name', '')} {info.get('status', '')}".rstrip()
        )
    )
    manager.error_occurred.connect(lambda m: print(f"\n  [error] {m}"))

    console = _Console(manager)

    def _stdin_loop() -> None:
        print("\nLive API agent — type a command, or 'quit' to exit.\n")
        for line in sys.stdin:
            line = line.strip()
            if line.lower() in {"quit", "exit"}:
                break
            console.submit.emit(line)
        QTimer.singleShot(0, app.quit)

    manager.start()
    threading.Thread(target=_stdin_loop, daemon=True, name="live-api-cli-stdin").start()
    try:
        return int(app.exec())
    finally:
        manager.stop()


if __name__ == "__main__":
    raise SystemExit(main())

# Author: Konstantin Markov
