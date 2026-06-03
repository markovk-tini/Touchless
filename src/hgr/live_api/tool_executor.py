"""Tool dispatcher for the Live API session.

Validates arguments, routes calls to existing Touchless controllers
where possible, and returns structured results that are sent back to
the model as a `function_call_output` event.

Important boundaries:
  * NO arbitrary Python/PowerShell/CMD execution. If we need that
    later, it must be a separate locked-down tool with explicit user
    confirmation.
  * Risky filesystem ops (overwrite, delete) require explicit
    `overwrite=true` AND ideally a prior `ask_user_confirmation`. The
    executor returns `{"status": "needs_confirmation", ...}` rather
    than silently doing the action.
  * All paths are constrained to the safe workspace dir unless an
    absolute path was explicitly provided.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import cortex_emit
from .action_classifier import classify_tool_action
from .config import LiveApiConfig
from .live_api_logger import LiveApiLogger
from .schemas import validate_args
from .screen_context import ScreenContext


# Module-level snapshot of the rich Cortex world payload
# (the same shape ``run_iris_simulator.load_world_payload`` returns —
# projects with nested ``branches`` and ``leaves``). Populated by
# ``set_rich_world_payload`` from the assistant window's background
# payload-preload worker so iris_query_node can answer with real
# folder/file structure WITHOUT calling load_world_payload synchronously
# (which would block tool dispatch on a multi-second FS scan).
#
# Read by ``_load_iris_world_cache`` as an opportunistic enrichment;
# absence is fine — the query path falls back to the cheap world_state +
# combined_project_roots merge it already used.
_INJECTED_RICH_PAYLOAD: Optional[Dict[str, Any]] = None


# Module-level cache for iris_describe_project. Maps absolute project
# root path → (summary_text, project_label, file_count, timestamp).
# 24h TTL — described in the tool schema; the model is told the summary
# may be stale. write_md=True forces a re-scan so the freshly-written
# CLAUDE.md reflects current content.
_PROJECT_DESCRIPTION_CACHE: Dict[str, Dict[str, Any]] = {}
_PROJECT_DESCRIPTION_TTL_SEC = 24 * 60 * 60  # 24 hours


# Module-level JS runner fallback for cortex visualization.
#
# The QWebChannel signal path (CortexBridge.emit_project_added /
# emit_project_removed) occasionally drops events — the JS-side handler
# may not have wired up its listener before the signal fires, or the
# bridge instance may have been torn down between emit and dispatch.
# When that happens, projects appear/disappear from the world state but
# the cortex viz never animates, leaving the user staring at a stale
# scene.
#
# The assistant window registers a runner here that schedules a
# ``page.runJavaScript(src)`` call on the GUI thread via QTimer; the
# iris_add_project / iris_remove_project handlers invoke it as a
# belt-and-braces guarantee that the visual transition always fires.
# Safe-by-default: a missing runner is a no-op, and the runner itself
# wraps every call in try/except so a torn-down view can't crash the
# handler.
_CORTEX_JS_RUNNER: Optional[Callable[[str], None]] = None


def set_cortex_js_runner(runner: Optional[Callable[[str], None]]) -> None:
    """Register (or clear) the GUI-thread JS runner used as a fallback
    for cortex add/remove visual transitions. Pass ``None`` to unregister
    when the cortex view is destroyed."""
    global _CORTEX_JS_RUNNER
    if runner is None or callable(runner):
        _CORTEX_JS_RUNNER = runner


def _run_cortex_js(src: str) -> None:
    """Invoke the registered GUI-thread JS runner if any, swallowing
    every failure. The cortex viz is decorative — a missing runner,
    torn-down view, or runtime exception must never break the tool."""
    runner = _CORTEX_JS_RUNNER
    if runner is None:
        return
    try:
        runner(src)
    except Exception:
        pass


def set_rich_world_payload(payload: Optional[Dict[str, Any]]) -> None:
    """Inject the rich Cortex payload (with branches/leaves) for iris_query_node.

    Called by ``live_assistant_window`` after its background preload of
    ``load_world_payload()`` completes. Any in-flight ``ToolExecutor``
    instances will see the rich data on their NEXT query (since each
    instance's per-call cache is invalidated when this is set).

    Safe to call with ``None`` to clear. Never raises."""
    global _INJECTED_RICH_PAYLOAD
    try:
        _INJECTED_RICH_PAYLOAD = payload if isinstance(payload, dict) else None
    except Exception:
        _INJECTED_RICH_PAYLOAD = None


# Confidence floor for memory-write side-effects from tool classification.
# Action_classifier's per-kind floors are already conservative; this is an
# extra gate so noisy near-misses never reach long-lived storage. Anything
# below this is logged-only (other pipelines already cover those kinds).
_CLASSIFY_MEMORY_THRESHOLD = 0.8


# Tool-name → cortex world-touch mapping. Used by _emit_world_touches
# below. Purely additive — touch failures never affect tool results.

# Tools whose primary path argument identifies a file Iris just acted on.
# Keys = candidate arg names tried in order; first present wins.
_FILE_PATH_KEYS = (
    "path", "file_path", "filepath",
    "dest_path", "destination", "new_path", "target_path",
    "source_path", "source",
)

# Tools where the operation produces a "new" file location (the
# destination matters more than the source for "where it lives now").
_FILE_TOOLS = {
    "create_file", "read_file", "write_file", "append_file",
    "open_in_editor", "delete_file", "move_file", "rename_file",
}


def _emit_world_touches(name: str, args: Dict[str, Any]) -> None:
    """Map (tool_name, args) → cortex_emit.touch() calls.

    Called once after a tool dispatched successfully. Everything is
    best-effort + silent on failure — the cortex viz must never break
    tool execution. Designed to be cheap when the cortex window isn't
    open (cortex_emit.touch is a near-no-op).
    """
    if name == "open_app":
        app_name = args.get("name") or args.get("app_name") or args.get("app")
        if app_name:
            cortex_emit.touch("app", str(app_name))
        return

    if name == "open_url":
        url = args.get("url")
        if url:
            url_s = str(url)
            display = url_s if len(url_s) <= 48 else (url_s[:45] + "...")
            cortex_emit.touch("url", display, path=url_s)
        return

    if name in _FILE_TOOLS:
        for key in _FILE_PATH_KEYS:
            v = args.get(key)
            if not v:
                continue
            path_s = str(v)
            label = os.path.basename(path_s) or path_s
            cortex_emit.touch("file", label, path=path_s)
            return
from ..debug.foreground_window import get_foreground_window_info
from ..debug.mouse_controller import MouseController
from ..debug.text_input_controller import TextInputController
from ..utils.subprocess_utils import launch_external


# Map of human/model-friendly key names -> Win32 virtual-key codes.
# Only the modifiers and the most common keys — anything not here is
# treated as a literal character via SendInput unicode path (handled by
# TextInputController for printable text).
_VK_MAP: Dict[str, int] = {
    "ctrl": 0x11, "control": 0x11,
    "shift": 0x10,
    "alt": 0x12, "menu": 0x12,
    "win": 0x5B, "windows": 0x5B, "meta": 0x5B, "lwin": 0x5B,
    "tab": 0x09,
    "enter": 0x0D, "return": 0x0D,
    "esc": 0x1B, "escape": 0x1B,
    "space": 0x20,
    "backspace": 0x08, "back": 0x08,
    "delete": 0x2E, "del": 0x2E,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
}
for _i in range(26):
    _VK_MAP[chr(ord("a") + _i)] = 0x41 + _i
for _i in range(10):
    _VK_MAP[str(_i)] = 0x30 + _i


# A confirmation callback can be injected by the manager so risky tool
# calls show the existing Touchless overlay/dialog. Signature:
#   callback(message: str, risk_level: str) -> bool   # True = approved
ConfirmCallback = Callable[[str, str], bool]


class ToolExecutor:
    def __init__(
        self,
        *,
        config: LiveApiConfig,
        logger: LiveApiLogger,
        screen_context: ScreenContext,
        confirm_callback: Optional[ConfirmCallback] = None,
        external_action_router: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
        notify_callback: Optional[Callable[[str], None]] = None,
        memory_manager: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._logger = logger
        self._screen = screen_context
        self._confirm = confirm_callback
        self._external_router = external_action_router
        # Called (from a background thread) to push a proactive message into
        # the session — e.g. when auto-approve detects the app paused.
        self._notify = notify_callback
        # Optional MemoryManager used by the action_classifier wiring. When
        # None (default — live_api_manager doesn't wire it in yet), the
        # classifier still routes file/project touches via world_state and
        # logs unknowns; only fact/preference writes are skipped because
        # they have nowhere to land. Wiring this is a one-line change in
        # live_api_manager.ToolExecutor() when ready.
        self._memory = memory_manager

        # Lazy-init heavy controllers — only created when first used so
        # the main app's startup cost is unaffected when Live API is OFF.
        self._mouse: Optional[MouseController] = None
        self._text: Optional[TextInputController] = None
        self._youtube = None  # YouTubeController, lazy
        self._chrome = None   # ChromeController, lazy
        self._web = None      # WebController (CDP), lazy
        self._desktop = None  # DesktopController (file search), lazy
        self._uia = None      # UiaController (accessibility tree), lazy
        self._ocr = None      # ScreenOcr (OCR → exact-pixel click), lazy
        self._zoom_bounds = None  # last zoom_screen crop bounds for click_zoom
        self._voice_proc = None   # VoiceCommandProcessor for run_quick_command, lazy

        self._safe_workspace = Path(self._config.safe_workspace_dir)
        try:
            self._safe_workspace.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._logger.exception("safe_workspace_init_failed", exc, path=str(self._safe_workspace))
        # Bounded MRU of paths the agent has created or written this
        # session. Used by list_recent_paths so the model can recover
        # context when it loses track of "where did I just put that".
        self._recent_paths: list[str] = []

        # Cached cortex world payload for iris_query_node. Populated lazily
        # on first query (and refreshed whenever iris_remove_project runs)
        # by ``_load_iris_world_cache`` — calling ``load_world_payload`` on
        # every query is too slow (filesystem scan + auto-discovery). The
        # cache is a snapshot, NOT a live view; the staleness is documented
        # in the tool schema description.
        self._iris_world_cache: Optional[Dict[str, Any]] = None
        # ``id()`` of the injected rich payload the per-instance cache was
        # last built against — lets us auto-invalidate when the assistant
        # window swaps in fresh data.
        self._iris_world_cache_rich_id: Optional[int] = None

    # ---- public ----

    def execute(self, name: str, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        ok, err, args = validate_args(name, raw_args or {})
        if not ok:
            self._logger.event(
                "tool_validation_failed", tool=name, error=err, raw_keys=list((raw_args or {}).keys())
            )
            return _result(status="error", error=err, code="invalid_arguments")

        self._logger.event(
            "tool_call",
            tool=name,
            args=_summarize_args(name, args, debug=self._config.debug_text_logging),
        )
        # Cortex viz: pulse core -> tools as the dispatch fires.
        try:
            cortex_emit.tool_call(name)
        except Exception:
            pass
        # Cortex viz: also light up the SPECIFIC tool node ('tool-<name>')
        # so the user can see which built-in fired. Best-effort — the JS
        # handler silently skips unknown node ids, so a tool whose slug
        # doesn't match a seeded node just gets the generic cap-tools
        # pulse above. Must NEVER break tool dispatch.
        try:
            cortex_emit.node_activity(f"tool-{name}",
                                      intensity=0.9, duration_ms=150)
        except Exception:
            pass
        # Persistent tool-call log used for cross-session co-occurrence
        # aggregation (powers the cortex tool_links cross-edges).
        # Best-effort — must NEVER break tool dispatch.
        try:
            from .cortex.tool_call_log import record_tool_call as _record_tc
            _record_tc(name)
        except Exception:
            pass

        try:
            handler = self._handlers().get(name)
            if handler is None:
                result = _result(status="error", error=f"no handler for {name}", code="no_handler")
            else:
                result = handler(args)
        except Exception as exc:
            self._logger.exception("tool_exception", exc, tool=name)
            result = _result(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                code="exception",
            )

        self._logger.latency(f"tool:{name}", started, status=result.get("status"))
        # Cortex viz: drop a short-lived leaf showing the tool that ran.
        try:
            status = str(result.get("status", "") or "")
            label = name if status == "ok" else f"{name} ✕"
            cortex_emit.leaf_add("cap-tools", label, ttl_ms=4500)
        except Exception:
            pass
        # Cortex world: record persistent touches (files, apps, urls)
        # so the cortex builds up Iris's world model across sessions.
        try:
            if result.get("status") == "ok":
                _emit_world_touches(name, args)
        except Exception:
            pass
        # Cortex bridge: fire the discrete toolUsed signal so the live
        # simulator JS can animate the tool firing without parsing the
        # generic `event` stream. No-op when the simulator isn't running
        # (get_active_bridge returns None). Decorative — must NEVER
        # break tool dispatch. NOTE: leaf/project events are NOT fired
        # here — world_state.touch_file already fires `file_added` for
        # file-touching tools, and double-firing would duplicate leaves
        # in the simulator.
        try:
            if result.get("status") == "ok":
                from .cortex.bridge import get_active_bridge
                bridge = get_active_bridge()
                if bridge is not None:
                    bridge.emit_tool_used({"tool": name, "label": name})
        except Exception:
            pass
        # Action classifier: deterministic rule-based categorisation of the
        # tool dispatch (file_touch / new_project / fact / preference / ...).
        # Routes high-confidence results to long-lived stores. Must NEVER
        # break tool dispatch — wrapped in a broad try/except and a no-op
        # when status != "ok" (the classifier itself also early-returns on
        # failed status, but the guard here keeps the cost off the hot
        # path for the failure case).
        try:
            if result.get("status") == "ok":
                self._route_tool_classification(name, args, result)
        except Exception:
            pass
        return result

    def _route_tool_classification(
        self,
        name: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Run the deterministic action_classifier on a completed tool call
        and route the result to the matching long-lived store.

        Routing matrix (only fires when confidence >= _CLASSIFY_MEMORY_THRESHOLD):

          * ``kind == 'fact'`` or ``'preference'`` → MemoryManager.set_fact()
            when a memory_manager was injected. Skipped silently when no
            memory_manager is wired in (live_api_manager doesn't pass one
            yet — see __init__). Note: the LLM extractor in
            MemoryManager.observe_conversation / extract_facts already mines
            facts from conversation turns; this classifier covers a DIFFERENT
            channel — tool-dispatch — so the two don't double-fire.

          * ``kind == 'new_project'`` → world_state.add_project() if the
            target looks like a real folder. The world_state singleton is
            module-global so no injection is needed.

          * ``kind in {'file_touch', 'tool_use', 'pattern_seed',
            'episode', 'unknown'}`` → no-op. ``file_touch`` and app/url
            touches are already persisted by ``_emit_world_touches`` +
            ``world_state.touch_file/app/url`` above; tool_use is covered
            by the persistent tool_call_log; pattern_seed / episode /
            unknown have no destination store today (logged only).

        Failures inside the classifier or the routing target are swallowed.
        """
        try:
            classification = classify_tool_action(name, args, result)
        except Exception as exc:
            # Classifier bug must NEVER break dispatch.
            self._logger.exception("classify_tool_action_failed", exc, tool=name)
            return

        kind = str(classification.get("kind") or "unknown")
        target = classification.get("target")
        payload = classification.get("payload") or {}
        confidence = float(classification.get("confidence") or 0.0)

        # Always log the classification so we can audit what would have
        # routed and what didn't. Cheap, single structured-event line.
        try:
            self._logger.event(
                "tool_classification",
                tool=name,
                kind=kind,
                target=(str(target)[:160] if target is not None else None),
                confidence=round(confidence, 3),
            )
        except Exception:
            pass

        if confidence < _CLASSIFY_MEMORY_THRESHOLD:
            return

        if kind in ("fact", "preference"):
            # MemoryManager wiring is optional — when absent, the
            # extractor pipeline still mines conversational facts; we
            # just don't add tool-derived ones until live_api_manager
            # passes the manager in.
            if self._memory is None:
                return
            try:
                key = str(target) if target else "unknown"
                if kind == "preference":
                    value = str(payload.get("polarity")
                                or payload.get("text")
                                or "unknown")
                else:
                    value = str(payload.get("value")
                                or payload.get("text")
                                or "")
                self._memory.set_fact(
                    kind=kind,
                    key=key,
                    value=value,
                    source="tool_classifier",
                    source_kind="tool_classifier",
                    source_id=name,
                )
            except Exception as exc:
                self._logger.exception("classify_memory_set_fact_failed",
                                       exc, tool=name, kind=kind)
            return

        if kind == "new_project":
            # Promote an editor-opened folder to a Cortex project node.
            # world_state.add_project is idempotent (no-op if the
            # project_id already exists), so re-firing on the same
            # folder is safe.
            if not target:
                return
            try:
                from .cortex.world_state import get_world
                world = get_world()
                folder_path = str(target)
                # Use the folder's basename as both id and label so the
                # node is human-readable in the Cortex view. world_state
                # de-dupes by id, so repeated opens collapse to one node.
                project_id = os.path.basename(folder_path.rstrip("\\/")) or folder_path
                world.add_project(
                    project_id=project_id,
                    label=project_id,
                    root_path=folder_path,
                )
            except Exception as exc:
                self._logger.exception("classify_add_project_failed",
                                       exc, tool=name, target=str(target))
            return

        # Remaining kinds (file_touch, tool_use, pattern_seed, episode,
        # unknown) are already handled by other pipelines or have no
        # destination store yet — log-only above is sufficient.
        return

    # ---- handlers ----

    def _handlers(self) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
        return {
            "get_screen_context": self._t_get_screen_context,
            "read_screen": self._t_read_screen,
            "click_screen": self._t_click_screen,
            "click_type": self._t_click_type,
            "type_text": self._t_type_text,
            "send_to_coding_agent": self._t_send_to_coding_agent,
            "follow_up_coding_agent": self._t_follow_up_coding_agent,
            "click_text_on_screen": self._t_click_text_on_screen,
            "wait_for_screen_text": self._t_wait_for_screen_text,
            "drag": self._t_drag,
            "draw_path": self._t_draw_path,
            "draw_shape": self._t_draw_shape,
            "move_window_to_monitor": self._t_move_window,
            "control_window": self._t_control_window,
            "close_window": self._t_close_window,
            "run_quick_command": self._t_run_quick_command,
            "run_matlab_script": self._t_run_matlab_script,
            "zoom_screen": self._t_zoom_screen,
            "click_zoom": self._t_click_zoom,
            "press_hotkey": self._t_press_hotkey,
            "open_app": self._t_open_app,
            "open_in_editor": self._t_open_in_editor,
            "open_url": self._t_open_url,
            "create_folder": self._t_create_folder,
            "create_file": self._t_create_file,
            "write_file": self._t_write_file,
            "append_file": self._t_append_file,
            "read_file": self._t_read_file,
            "move_file": self._t_move_file,
            "rename_file": self._t_rename_file,
            "delete_file": self._t_delete_file,
            "list_recent_paths": self._t_list_recent_paths,
            "list_files": self._t_list_files,
            "run_existing_touchless_action": self._t_run_existing_action,
            "run_python_script": self._t_run_python_script,
            "skip_youtube_ad": self._t_skip_youtube_ad,
            "ask_user_confirmation": self._t_ask_user_confirmation,
            "compose_text": self._t_compose_text,
            "weather_get": self._t_weather_get,
            "email_summary": self._t_email_summary,
            "web_search": self._t_web_search,
            "web_navigate": self._t_web_navigate,
            "web_get_links": self._t_web_get_links,
            "web_get_text": self._t_web_get_text,
            "web_click": self._t_web_click,
            "web_scroll": self._t_web_scroll,
            "web_fill": self._t_web_fill,
            "web_wait_for": self._t_web_wait_for,
            "web_eval": self._t_web_eval,
            "open_path": self._t_open_path,
            "read_ui": self._t_read_ui,
            "click_ui": self._t_click_ui,
            "set_field": self._t_set_field,
            "wait_and_click": self._t_wait_and_click,
            "wait_and_press": self._t_wait_and_press,
            "auto_approve": self._t_auto_approve,
            "stop_auto_approve": self._t_stop_auto_approve,
            "iris_add_project": self._t_iris_add_project,
            "iris_remove_project": self._t_iris_remove_project,
            "iris_query_node": self._t_iris_query_node,
            "iris_describe_project": self._t_iris_describe_project,
            # ---- ambient / power-tools pack ---------------------------
            "notify_toast": self._t_notify_toast,
            "get_active_window": self._t_get_active_window,
            "clipboard_read": self._t_clipboard_read,
            "clipboard_write": self._t_clipboard_write,
            "clipboard_transform": self._t_clipboard_transform,
        }

    def _t_get_screen_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        info = get_foreground_window_info()
        # The actual screen IMAGE is delivered to the model as a proper
        # input_image by LiveApiManager (send_screen_image) right before
        # this tool's result — NOT embedded here. A base64 JPEG in a tool
        # result is text the model can't see and costs thousands of
        # tokens, so we return only the lightweight window context.
        return {
            "status": "ok",
            "active_window_title": (info.title if info else ""),
            "active_window_process": (info.process_name if info else ""),
            "timestamp": time.time(),
            "note": "Current screen image was provided separately as an image.",
        }

    def _t_click_screen(self, args: Dict[str, Any]) -> Dict[str, Any]:
        mouse = self._ensure_mouse()
        if mouse is None or not mouse.available:
            return _result(status="error", error="mouse unavailable", code="no_mouse")
        x = float(args["x"])
        y = float(args["y"])
        space = str(args.get("coordinate_space", "normalized"))
        button = str(args.get("button", "left"))
        double = bool(args.get("double_click", False))
        if button != "left":
            # Only left+right are wired into MouseController; right used directly,
            # middle returns not-implemented to keep behaviour explicit.
            if button == "right":
                if not mouse.right_click():
                    return _result(status="error", error="right click failed", code="click_failed")
                return _result(status="ok", message="right_click", x=x, y=y, space=space)
            return _result(status="error", error="middle button not supported", code="unsupported")
        if space == "normalized":
            mouse.move_normalized(max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
        else:
            bounds = mouse.virtual_bounds()
            if bounds is None:
                return _result(status="error", error="virtual bounds unknown", code="no_bounds")
            left, top, width, height = bounds
            nx = (float(x) - left) / max(width - 1, 1)
            ny = (float(y) - top) / max(height - 1, 1)
            mouse.move_normalized(max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny)))
        time.sleep(0.05)
        if not mouse.left_click():
            return _result(status="error", error="left click failed", code="click_failed")
        if double:
            time.sleep(0.06)
            mouse.left_click()
        return _result(status="ok", message="clicked", x=x, y=y, space=space, double=double)

    def _t_type_text(self, args: Dict[str, Any]) -> Dict[str, Any]:
        text = str(args.get("text", ""))
        method = str(args.get("method", "clipboard_paste"))
        window_title = str(args.get("window_title", "") or "").strip()
        controller = self._ensure_text()
        if controller is None or not controller.available:
            return _result(status="error", error=controller.message if controller else "text input unavailable", code="no_text")
        # Pick the target window. If the model named one (e.g. "Notepad" right
        # after opening it), resolve it by title/process via UIA — this is far
        # more reliable than guessing the foreground, because the Iris overlay
        # is always-on-top and the just-opened app may not be the foreground
        # when type_text runs (that's why text 'typed' into nothing). Fall back
        # to foreground capture only when no window_title was given.
        targeted = False
        if window_title:
            uia = self._ensure_uia()
            if uia is not None:
                try:
                    hwnd = uia._hwnd_for_title(window_title)
                except Exception:
                    hwnd = 0
                if hwnd:
                    controller._target_hwnd = int(hwnd)
                    controller._last_external_hwnd = int(hwnd)
                    targeted = True
        if not targeted:
            try:
                controller.capture_target_window()
            except Exception as exc:
                self._logger.exception("type_text_capture_failed", exc)
        ok = controller.insert_text(text, prefer_paste=(method == "clipboard_paste"))
        if not ok:
            return _result(status="error", error=controller.message, code="insert_failed")
        self._logger.text("tool_type_text_payload", text)
        return _result(status="ok", message="text inserted", chars=len(text), method=method,
                       target=window_title or "foreground")

    def _vk_for(self, name: str):
        """Resolve a key name to a virtual-key code (table + layout fallback)."""
        k = str(name).strip().lower()
        vk = _VK_MAP.get(k)
        if vk is None and len(k) == 1:
            try:
                import ctypes
                r = ctypes.windll.user32.VkKeyScanW(ord(k))
                if r != -1:
                    vk = r & 0xFF
            except Exception:
                vk = None
        return vk

    def _t_send_to_coding_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Send a prompt to a coding agent (Claude Code / Codex) in VS Code.

        via='terminal' (default, RELIABLE): open the integrated terminal, run
        the CLI ('claude'/'codex'), then paste the prompt into it. Terminal
        focus + paste are deterministic.

        via='extension': focus the open chat panel with its shortcut (Ctrl+Esc
        for Claude Code) and paste the prompt. Only works if the panel is open
        with a focusable input; the webview can be finicky.

        All typing uses clipboard paste WITHOUT re-focusing a window, so it
        lands in the focused control (terminal/chat), never the editor."""
        prompt = str(args.get("prompt", "")).strip()
        # open_only: just open a fresh agent tab and stop — used when the user
        # asks to 'open a Claude tab' WITHOUT giving Claude a task. No prompt
        # is typed, so Claude isn't told to do anything.
        open_only = bool(args.get("open_only", False)) or not prompt
        agent = str(args.get("agent", "claude")).strip().lower()
        via = str(args.get("via", "extension")).strip().lower()
        controller = self._ensure_text()
        if controller is None or not controller.available:
            return _result(status="error", error="text input unavailable", code="no_text")
        # By default, start the background approve-watcher AS PART of sending
        # the task — so the very first Yes/Allow prompt is clicked without
        # needing a separate auto_approve tool call. The model can pass
        # auto_approve=false to suppress this (e.g. the user wants to approve
        # manually). Locked to the editor window so it works in the background.
        want_watcher = bool(args.get("auto_approve", True))
        watcher_window = str(args.get("window_title", "") or "").strip() or "Visual Studio Code"
        # background: the user asked to NOT watch it happen. Typing the prompt
        # needs VS Code focused briefly (unavoidable with key injection), so we
        # do the setup on-screen, then hand focus back to whatever the user was
        # using — VS Code drops behind it and the UIA watcher keeps approving
        # prompts WITHOUT focus. (Full-minimize is avoided: a minimized VS Code
        # can stop rendering its webview, which would break the clicking.)
        run_background = bool(args.get("background", False))
        prev_foreground = 0
        if run_background:
            try:
                prev_foreground = int(ctypes.windll.user32.GetForegroundWindow())
            except Exception:
                prev_foreground = 0

        def _maybe_start_watcher(result: Dict[str, Any]) -> Dict[str, Any]:
            if want_watcher:
                try:
                    w = self._begin_auto_approve(window_title=watcher_window)
                    result["auto_approve"] = w.get("status", "unknown")
                except Exception as exc:  # never let the watcher sink the send
                    self._logger.exception("send_to_coding_agent_watcher_failed", exc)
                    result["auto_approve"] = "error"
            # In background mode, restore the user's previous window so VS Code
            # falls behind it and they can keep working while it runs.
            if run_background and prev_foreground:
                try:
                    time.sleep(0.3)  # let the final Enter register first
                    ctypes.windll.user32.SetForegroundWindow(prev_foreground)
                    result["background"] = True
                except Exception as exc:
                    self._logger.exception("send_to_coding_agent_restore_focus_failed", exc)
                    result["background"] = "focus_restore_failed"
            return result

        if via == "extension":
            # assume_focused: the caller already focused the right input
            # (e.g. clicked the tab's chat box) — just paste + submit. Used as
            # the recovery path when the prompt landed in the wrong Claude
            # surface (a separate panel can hold keyboard focus after the tab
            # opens).
            assume_focused = bool(args.get("assume_focused", False))
            project = str(args.get("project_folder", "") or "").strip()
            opened = False
            if not assume_focused:
                # CRITICAL: the Command Palette keystrokes only work if VS Code
                # is open AND focused. Ensure that first, otherwise the shortcut
                # + paste land in whatever app the user last touched and
                # silently do nothing. If a project_folder was given, create it
                # and open it as the workspace so the agent's cwd is that
                # folder; otherwise just make sure SOME VS Code window is up.
                if project:
                    ensured = self._prepare_project_folder(project)
                else:
                    ensured = self._ensure_vscode_foreground()
                if ensured.get("status") != "ok":
                    return ensured
                # Open the agent's TAB via the Command Palette (reliable across
                # versions): Ctrl+Shift+P -> type the open command -> Enter.
                open_cmd = args.get("open_command")
                if open_cmd is None:
                    open_cmd = "Claude Code: Open in New Tab" if agent == "claude" else ""
                open_cmd = str(open_cmd).strip()
                # The new tab's webview needs time to load AND grab keyboard
                # focus. Typing too soon makes the prompt land in the already-
                # focused bottom Claude panel — the exact bug we're fixing — so
                # default to a generous 4.5s and let each step settle.
                open_wait = max(2.0, min(10.0, float(args.get("open_wait_sec", 4.5) or 4.5)))
                p = self._vk_for("p")
                if open_cmd and p is not None:
                    controller._send_shortcut(0x11, 0x10, p)  # Ctrl+Shift+P
                    time.sleep(0.8)                            # palette opens
                    controller._paste_text(open_cmd)          # filter to the command
                    time.sleep(0.9)                            # let it highlight the match
                    controller._send_shortcut(0x0D)           # run -> opens tab
                    time.sleep(open_wait)                     # tab loads + input focuses
                    opened = True
                else:
                    controller._send_shortcut(0x11, 0x1B)     # Ctrl+Esc focus
                    time.sleep(0.5)
            if open_only:
                # Just opened the tab — don't type any prompt (no task for the
                # agent). Don't start the approve-watcher either.
                return _result(status="ok", agent=agent, via="extension", opened=opened,
                               open_only=True, note="Opened a fresh Claude tab; no prompt sent.")
            if not controller._paste_text(prompt):  # noqa: SLF001
                return _result(status="error", error="could not type the prompt into the agent", code="paste_failed")
            time.sleep(0.25)
            controller._send_shortcut(0x0D)
            self._logger.text("send_to_coding_agent_prompt", prompt)
            note = (
                "Done: opened a fresh Claude tab via the Command Palette, typed "
                "the prompt into it, and started the background approve-watcher. "
                "The task is now sent — do NOT call this again for the same task "
                "and do NOT screenshot to verify; just tell the user it's "
                "working and wait for the watcher's update."
                if opened else
                "assume_focused was set, so I pasted into the already-focused "
                "input WITHOUT opening a new tab — this can land in the wrong "
                "Claude panel. For a new task, do NOT set assume_focused; let it "
                "open a fresh tab."
            )
            return _maybe_start_watcher(_result(
                status="ok", agent=agent, via="extension", opened=opened,
                submitted=prompt[:120], note=note,
            ))

        # via == "terminal" (default). Open + focus the integrated terminal,
        # start the CLI, wait for it to be ready, then paste the prompt.
        ctrl = 0x11
        bt = self._vk_for("`")
        if bt is None:
            return _result(status="error", error="cannot resolve backtick key", code="invalid_key")
        cli = {"claude": "claude", "codex": "codex"}.get(agent, agent)
        startup_wait = max(1.0, min(20.0, float(args.get("startup_wait_sec", 4) or 4)))
        controller._send_shortcut(ctrl, bt)  # Ctrl+` opens + focuses the terminal
        time.sleep(0.9)
        if not controller._paste_text(cli):  # noqa: SLF001
            return _result(status="error", error="could not type into the terminal", code="paste_failed")
        time.sleep(0.1)
        controller._send_shortcut(0x0D)       # run the CLI
        time.sleep(startup_wait)              # let the agent boot
        if not controller._paste_text(prompt):  # noqa: SLF001
            return _result(status="error", error="could not paste the prompt", code="paste_failed")
        time.sleep(0.25)
        controller._send_shortcut(0x0D)       # submit the prompt
        self._logger.text("send_to_coding_agent_prompt", prompt)
        return _maybe_start_watcher(_result(
            status="ok", agent=agent, via="terminal", submitted=prompt[:120],
            note=f"Ran '{cli}' in the terminal, sent the prompt, and started the background approve-watcher. If the CLI was slow to start, the prompt may have been missed — re-send if so.",
        ))

    def _t_follow_up_coding_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Send a FOLLOW-UP message into the EXISTING Claude tab (same
        conversation) — e.g. to paste back the program's output and ask Claude
        to confirm it's correct / fix it. Unlike send_to_coding_agent this does
        NOT open a new tab (that would start a fresh conversation); it focuses
        the current Claude Code input via Ctrl+Esc and types there. Keeps the
        approve-watcher running so Claude's fix-and-rerun gets auto-approved."""
        message = str(args.get("message", "")).strip()
        if not message:
            return _result(status="error", error="empty message", code="invalid_arguments")
        controller = self._ensure_text()
        if controller is None or not controller.available:
            return _result(status="error", error="text input unavailable", code="no_text")
        # Make sure VS Code (with the Claude tab) is the foreground window so
        # the focus shortcut + paste land in it.
        ensured = self._ensure_vscode_foreground()
        if ensured.get("status") != "ok":
            return ensured
        # Ctrl+Esc focuses the active Claude Code input without opening a new
        # tab/conversation.
        controller._send_shortcut(0x11, 0x1B)  # Ctrl+Esc
        time.sleep(0.7)
        if not controller._paste_text(message):  # noqa: SLF001
            return _result(status="error", error="could not type the follow-up", code="paste_failed")
        time.sleep(0.25)
        controller._send_shortcut(0x0D)  # submit
        self._logger.text("follow_up_coding_agent", message)
        # Re-arm the watcher (idempotent) so any new Yes/Allow prompts from
        # Claude's follow-up work get auto-approved.
        result = _result(
            status="ok", submitted=message[:160],
            note=("Sent the follow-up into the existing Claude tab and kept the "
                  "approve-watcher running. Wait for the watcher's update for "
                  "Claude's verdict/fix."),
        )
        if bool(args.get("auto_approve", True)):
            try:
                window = str(args.get("window_title", "") or "").strip() or "Visual Studio Code"
                w = self._begin_auto_approve(window_title=window)
                result["auto_approve"] = w.get("status", "unknown")
            except Exception as exc:
                self._logger.exception("follow_up_watcher_failed", exc)
                result["auto_approve"] = "error"
        return result

    def _t_press_hotkey(self, args: Dict[str, Any]) -> Dict[str, Any]:
        keys = list(args.get("keys", []) or [])
        if not keys:
            return _result(status="error", error="empty hotkey", code="invalid_arguments")
        normalised = [str(k).strip().lower() for k in keys]
        # Soft-confirm destructive shortcuts.
        if normalised in ([["alt", "f4"]], [["ctrl", "shift", "q"]]) or normalised == ["alt", "f4"]:
            # HARD SAFETY: Alt+F4 hits the FOCUSED window. If that's our own
            # process (the assistant), it'd close Iris — refuse and redirect to
            # close_window for targeting a specific window.
            if "f4" in normalised:
                try:
                    fg = ctypes.windll.user32.GetForegroundWindow()
                    pid = ctypes.c_ulong(0)
                    ctypes.windll.user32.GetWindowThreadProcessId(fg, ctypes.byref(pid))
                    if int(pid.value) == os.getpid():
                        return _result(
                            status="error",
                            error="Alt+F4 would close the assistant (it's the focused window). "
                                  "To close a specific window, use close_window(window_title).",
                            code="would_close_self",
                        )
                except Exception:
                    pass
            if self._confirm and not self._confirm(
                f"Press {'+'.join(normalised)} (this may close a window)?", "high"
            ):
                return _result(status="needs_confirmation", message="user declined hotkey", keys=normalised)
        vks: list[int] = []
        for k in normalised:
            vk = _VK_MAP.get(k)
            if vk is None and len(k) == 1:
                # Fall back to the layout map for single chars not in the
                # table (e.g. the backtick in Ctrl+` for VS Code's terminal).
                try:
                    import ctypes
                    r = ctypes.windll.user32.VkKeyScanW(ord(k))
                    if r != -1:
                        vk = r & 0xFF
                except Exception:
                    vk = None
            if vk is None:
                return _result(status="error", error=f"unknown key: {k}", code="invalid_key")
            vks.append(vk)
        controller = self._ensure_text()
        if controller is None or not controller.available:
            return _result(status="error", error="text input unavailable", code="no_text")
        # If a window was named (e.g. Ctrl+S into "Notepad"), focus it first so
        # the shortcut lands there and not on the always-on-top Iris overlay or
        # a stale foreground window.
        window_title = str(args.get("window_title", "") or "").strip()
        if window_title:
            uia = self._ensure_uia()
            if uia is not None:
                try:
                    hwnd = uia._hwnd_for_title(window_title)
                    if hwnd:
                        controller._target_hwnd = int(hwnd)
                        controller._last_external_hwnd = int(hwnd)
                        controller.focus_target_window()
                        time.sleep(0.12)
                except Exception:
                    pass
        ok = controller._send_shortcut(*vks)  # noqa: SLF001 — internal but stable in this codebase
        return _result(status="ok" if ok else "error", message="hotkey", keys=normalised,
                       target=window_title or "foreground")

    def _t_open_app(self, args: Dict[str, Any]) -> Dict[str, Any]:
        name = str(args.get("app_name", "")).strip()
        if not name:
            return _result(status="error", error="empty app_name", code="invalid_arguments")
        # Editor names route to open_in_editor instead — this prevents
        # the common failure mode where the model launches VS Code with
        # no folder and shows the user an empty window. Returning a
        # corrective error tells the model exactly which tool to call
        # next, so it self-corrects on the second hop.
        # NOTE: Notepad is deliberately NOT here — it's a simple text app, so
        # launching it empty via open_app IS the right behavior (open it, type,
        # save). Only project/folder editors (VS Code) are useless when opened
        # empty and must route through open_in_editor.
        editor_aliases = {
            "code", "vscode", "vs code", "visual studio code", "vs",
        }
        if name.strip().lower() in editor_aliases:
            return _result(
                status="error",
                error=(
                    f"Use open_in_editor for {name}, not open_app. "
                    f"open_in_editor takes a folder_path so the editor "
                    f"opens with the folder loaded; open_app gives an "
                    f"empty window which is never useful."
                ),
                code="wrong_tool_for_editor",
                correct_tool="open_in_editor",
            )
        # Optional CLI args to forward (e.g. `code "C:\path\to\folder"`).
        cli_args = args.get("arguments") or []
        if not isinstance(cli_args, list):
            cli_args = []
        cli_args = [str(a) for a in cli_args]
        # Riot games: launching the client bare just shows its (Chromium,
        # un-clickable) UI. The official Valorant/LoL shortcut passes
        # --launch-product/--launch-patchline to BOOT the game directly — same
        # as the desktop shortcut — so "open valorant" starts the game with no
        # PLAY click needed. Only inject when the caller didn't pass args.
        if not cli_args:
            cli_args = list(_GAME_LAUNCH_ARGS.get(name.strip().lower(), []))
        # Models often pass friendly names ("Visual Studio Code") that
        # are NOT what Windows App Paths expects ("code"). Build a
        # candidate list of synonyms + likely install paths so a single
        # tool call resolves the common cases without forcing the model
        # to retry.
        candidates = _open_app_candidates(name)
        # Only true Riot games take the slow Riot-Client-wait + vision-PLAY
        # path. Discord is in _GAME_LAUNCH_ARGS only for its launch arg, so it
        # must NOT be treated as a game (that skipped its monitor placement and
        # made it hijack an already-open Riot window).
        is_game = name.strip().lower() in _RIOT_GAME_KEYS
        tried: list[str] = []
        for candidate in candidates:
            tried.append(candidate)
            if launch_external(candidate, args=cli_args or None):
                result = _result(
                    status="ok",
                    message=f"launched via '{candidate}'",
                    app=name,
                    resolved=candidate,
                    args=cli_args,
                    tried=tried,
                )
                if is_game:
                    # The Riot Client is SLOW to appear. Block here until its
                    # window exists (so we don't report back before there's
                    # anything on screen), then give the page a moment to
                    # render the PLAY button. Then tell the model to vision-
                    # click PLAY — UIA can't see the Chromium UI.
                    hwnd = self._find_window_hwnd("Riot Client", timeout=18.0)
                    if hwnd:
                        time.sleep(4.0)  # let the game page + PLAY button paint
                        # Bring it to the PRIMARY monitor + foreground so the
                        # PLAY button is big, unobscured and consistently placed
                        # — makes the vision click far more accurate than a
                        # tiny/half-offscreen window on a second monitor.
                        try:
                            self._do_primary_move("Riot Client")
                            time.sleep(0.8)
                            ctypes.windll.user32.SetForegroundWindow(hwnd)
                            time.sleep(0.4)
                        except Exception:
                            pass
                        result["client_window"] = "ready"
                        result["note"] = (
                            "The Riot Client is open and brought to the front on "
                            "the main monitor. To start the match: get_screen_"
                            "context, then click_screen the big PLAY button — a "
                            "large bright button labeled PLAY in the LOWER-LEFT "
                            "quarter of the Riot window (roughly x=0.10, y=0.92 "
                            "of the screen if the window fills it; adjust to "
                            "where you actually see it). If PLAY isn't rendered "
                            "yet, wait ~3s and screenshot again — don't give up "
                            "after one look. read_ui/click_ui do NOT work here."
                        )
                    else:
                        result["client_window"] = "not_yet"
                        result["note"] = (
                            "Launched, but the Riot Client window hasn't "
                            "appeared yet (it's slow). Wait ~10s, then "
                            "get_screen_context and click_screen the PLAY "
                            "button (lower-left). Don't report it as failed."
                        )
                # 'Open X on my second monitor [maximized]' — launch + WAIT for
                # the app's window + place it, all in this one call (more
                # reliable than the model launching then moving before the
                # window exists).
                monitor = str(args.get("monitor", "") or "").strip()
                if monitor and not is_game:
                    placement = str(args.get("placement", "maximize")).strip().lower()
                    hwnd = 0
                    deadline = time.time() + 12.0  # apps can be slow to show
                    while time.time() < deadline:
                        wins = self._find_windows(name)
                        if wins:
                            hwnd = wins[0][0]
                            break
                        time.sleep(0.5)
                    if hwnd:
                        time.sleep(0.6)  # let it finish painting
                        placed = self._place_window(hwnd, monitor, placement)
                        result["placed"] = placed.get("status")
                        result["monitor"] = placed.get("monitor")
                        # Some apps (Discord, VS Code) re-apply their OWN saved
                        # window bounds a moment after launch, overriding our
                        # move. Re-assert the placement a couple times on a
                        # background thread so it sticks. Non-blocking.
                        self._reassert_placement(name, monitor, placement)
                    else:
                        result["placed"] = "window_not_found_yet"
                        result["note"] = (f"Opened {name}, but its window didn't appear in time to "
                                          f"move it — it may still be loading; try move_window_to_monitor shortly.")
                return result
        # Fallback for Store/UWP apps (new Microsoft Teams, Calculator, Photos,
        # Settings, etc.) which aren't .exe candidates: DesktopController
        # launches them from the Start-Apps catalog via shell:AppsFolder.
        try:
            desktop = self._ensure_desktop()
            if desktop is not None and desktop.open_named_application(name):
                return _result(
                    status="ok", app=name, via="start_apps",
                    message=f"launched '{name}' via the Start-Apps catalog "
                            f"(Store/UWP app)",
                    tried=tried,
                )
        except Exception as exc:
            self._logger.exception("open_app_start_apps_fallback_failed", exc)
        return _result(
            status="error",
            error=f"could not launch '{name}'",
            code="launch_failed",
            tried=tried,
        )

    def _t_open_in_editor(self, args: Dict[str, Any]) -> Dict[str, Any]:
        editor = str(args.get("editor", "code")).strip().lower()
        folder = str(args.get("folder_path", "")).strip()
        file_to_open = str(args.get("file_to_open", "") or "").strip()
        if not folder:
            return _result(status="error", error="empty folder_path", code="invalid_arguments")
        folder_path = Path(folder).expanduser().resolve()
        if not folder_path.exists():
            # The model often guesses the wrong path (e.g. ~/Documents/HGR App
            # when the real folder is C:\HGR App v1.0.0). Resolve by NAME via
            # file search — but only ACCEPT a match that contains ALL the
            # requested words, so 'HGR App' doesn't silently open 'src\hgr'.
            req = [w for w in Path(folder).name.lower().replace("-", " ").replace("_", " ").split() if len(w) >= 2]
            desktop = self._ensure_desktop()
            accepted = None
            if desktop is not None and req:
                r, _amb = desktop.resolve_named_folder(Path(folder).name)
                if r is not None and all(w in str(r).lower() for w in req):
                    accepted = Path(r)
            if accepted is not None:
                folder_path = accepted
            else:
                return _result(
                    status="error",
                    error=f"folder not found: {folder}",
                    code="folder_not_found",
                    hint="Use the project's EXACT name (don't shorten it), or call open_path with the name first to get the real path, then open_in_editor that path.",
                )
        cli_args: list[str] = [str(folder_path)]
        if file_to_open:
            file_path = Path(file_to_open).expanduser().resolve()
            if file_path.exists():
                cli_args.append(str(file_path))
            else:
                # Don't fail — open the folder anyway, just without the
                # specific file. Logged so we know the model passed a
                # bad file path.
                self._logger.event(
                    "open_in_editor_file_missing",
                    file=str(file_path),
                    folder=str(folder_path),
                )
        # Reuse open_app's resolver so editor synonyms ("code", "vscode",
        # "notepad", etc.) all map to the right binary.
        candidates = _open_app_candidates(editor)
        tried: list[str] = []
        for candidate in candidates:
            tried.append(candidate)
            if launch_external(candidate, args=cli_args):
                # Editors often restore spanning multiple monitors — settle
                # the new window onto the primary monitor (unless the model
                # is told otherwise). Best-effort; matches the project name.
                self._move_window_to_primary(folder_path.name)
                return _result(
                    status="ok",
                    message=f"opened {editor} on {folder_path.name}",
                    folder=str(folder_path),
                    file=file_to_open or None,
                    resolved=candidate,
                    args=cli_args,
                )
        return _result(
            status="error",
            error=f"could not launch editor '{editor}'",
            code="editor_launch_failed",
            tried=tried,
        )

    def _t_open_url(self, args: Dict[str, Any]) -> Dict[str, Any]:
        target = str(args.get("url_or_query", "")).strip()
        if not target:
            return _result(status="error", error="empty url_or_query", code="invalid_arguments")
        browser = str(args.get("browser", "")).strip().lower()
        if "://" not in target and not target.startswith(("www.", "http")):
            url = f"https://www.google.com/search?q={target.replace(' ', '+')}"
        else:
            url = target if "://" in target else f"https://{target}"
        if browser in {"chrome", "google chrome"}:
            chrome = self._ensure_chrome()
            if chrome is not None and chrome.available:
                if chrome.open_url(url):
                    return _result(status="ok", message="opened in chrome", url=url)
        ok = launch_external(url)
        return _result(status="ok" if ok else "error", message="opened url", url=url)

    def _t_create_folder(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            base = self._resolve_base_dir(args.get("base_dir"))
        except ValueError as exc:
            return _result(
                status="error", error=str(exc), code="invalid_base_dir",
                next_action="retry_with_a_real_path_under_user_home",
            )
        folder_name = str(args.get("folder_name", "")).strip()
        if not folder_name:
            return _result(status="error", error="empty folder_name", code="invalid_arguments")
        target = (base / folder_name).resolve()
        if not _is_within(target, base) and not args.get("base_dir"):
            return _result(status="error", error="path escapes safe workspace", code="path_escape")
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return _result(status="error", error=str(exc), code="mkdir_failed", path=str(target))
        self._remember_path(str(target))
        return _result(status="ok", message="folder created", path=str(target))

    def _t_create_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            base = self._resolve_base_dir(args.get("base_dir"))
        except ValueError as exc:
            return _result(
                status="error", error=str(exc), code="invalid_base_dir",
                next_action="retry_with_a_real_path_under_user_home",
            )
        rel = str(args.get("relative_path", "")).strip()
        if not rel:
            return _result(status="error", error="empty relative_path", code="invalid_arguments")
        target = (base / rel).resolve()
        if not _is_within(target, base) and not args.get("base_dir"):
            return _result(status="error", error="path escapes safe workspace", code="path_escape")
        if target.exists() and not bool(args.get("overwrite", False)):
            return _result(
                status="needs_confirmation",
                error=(
                    f"File already exists at {target}. To overwrite, "
                    f"either ask the user for permission first OR call "
                    f"this tool again with overwrite=true if they "
                    f"already approved."
                ),
                code="exists",
                path=str(target),
                next_action="ask_user_then_retry_with_overwrite_true",
            )
        if target.exists() and self._confirm and not self._confirm(
            f"Overwrite existing file {target}?", "medium"
        ):
            return _result(status="needs_confirmation", error="user declined overwrite", path=str(target))
        content = args.get("content")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            text = "" if content is None else str(content)
            target.write_text(text, encoding="utf-8")
        except Exception as exc:
            return _result(status="error", error=str(exc), code="write_failed", path=str(target))
        if content is not None:
            self._logger.text("tool_create_file_content", str(content), path=str(target))
        self._remember_path(str(target))
        return _result(status="ok", message="file created", path=str(target), bytes=target.stat().st_size)

    def _t_write_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            base = self._resolve_base_dir(args.get("base_dir"))
        except ValueError as exc:
            return _result(
                status="error", error=str(exc), code="invalid_base_dir",
                next_action="retry_with_a_real_path_under_user_home",
            )
        rel = str(args.get("relative_path", "")).strip()
        content = str(args.get("content", ""))
        overwrite = bool(args.get("overwrite", False))
        if not rel:
            return _result(status="error", error="empty relative_path", code="invalid_arguments")
        target = (base / rel).resolve()
        if not _is_within(target, base) and not args.get("base_dir"):
            return _result(status="error", error="path escapes safe workspace", code="path_escape")
        if target.exists() and not overwrite:
            return _result(status="needs_confirmation", error="file exists, set overwrite=true", code="exists", path=str(target))
        if target.exists():
            try:
                shutil.copy2(target, target.with_suffix(target.suffix + ".bak"))
            except Exception as exc:
                self._logger.exception("backup_failed", exc, path=str(target))
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except Exception as exc:
            return _result(status="error", error=str(exc), code="write_failed", path=str(target))
        self._logger.text("tool_write_file_content", content, path=str(target))
        self._remember_path(str(target))
        return _result(status="ok", message="file written", path=str(target), bytes=target.stat().st_size)

    def _t_read_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read a file's TEXT content so you can summarize/answer about it
        WITHOUT opening it. Accepts an absolute path OR a name to find via file
        search (like open_path). Handles text/code/csv/json/md, PDF, and docx.
        Returns the content (truncated) for you to summarize."""
        raw = str(args.get("path") or args.get("name") or args.get("query") or "").strip()
        if not raw:
            return _result(status="error", error="empty path/name", code="invalid_arguments")
        p = Path(raw).expanduser()
        if not (p.exists() and p.is_file()):
            # Resolve by name via Touchless file search.
            desktop = self._ensure_desktop()
            if desktop is not None:
                r, amb = desktop.resolve_named_file(raw)
                if r is not None:
                    p = Path(r)
                elif amb:
                    return _result(status="ambiguous", matches=[str(m) for m in amb[:10]],
                                   note="Several files match; ask which, then read_file that exact path.")
        if not (p.exists() and p.is_file()):
            return _result(status="error", error=f"file not found: {raw}", code="not_found",
                           hint="try the exact path, or open_path with deep=true to locate it")
        max_chars = int(args.get("max_chars", 8000) or 8000)
        text, meta = self._extract_text(p, max_chars)
        if text is None:
            return _result(status="error", error=meta.get("error", "could not read file"),
                           code="read_failed", path=str(p), ext=meta.get("ext"))
        return _result(
            status="ok",
            path=str(p),
            ext=meta.get("ext"),
            chars=len(text),
            truncated=meta.get("truncated", False),
            content=text,
            note="Summarize or answer the user's question from this content; don't read it aloud verbatim.",
        )

    def _extract_text(self, path: Path, max_chars: int):
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                parts = []
                total = 0
                for page in reader.pages:
                    t = page.extract_text() or ""
                    parts.append(t)
                    total += len(t)
                    if total > max_chars:
                        break
                text = "\n".join(parts)
            elif ext == ".docx":
                import docx
                d = docx.Document(str(path))
                text = "\n".join(par.text for par in d.paragraphs)
            elif ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico",
                         ".exe", ".dll", ".zip", ".7z", ".rar", ".mp3", ".mp4",
                         ".mov", ".wav", ".bin", ".so", ".dylib"}:
                return None, {"error": f"{ext} is a binary/non-text file — can't read as text", "ext": ext}
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
            return text[:max_chars], {"truncated": len(text) > max_chars, "ext": ext}
        except Exception as exc:
            self._logger.exception("read_file_failed", exc, path=str(path))
            return None, {"error": f"{type(exc).__name__}: {exc}", "ext": ext}

    def _t_append_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            base = self._resolve_base_dir(args.get("base_dir"))
        except ValueError as exc:
            return _result(
                status="error", error=str(exc), code="invalid_base_dir",
                next_action="retry_with_a_real_path_under_user_home",
            )
        rel = str(args.get("relative_path", "")).strip()
        content = str(args.get("content", ""))
        if not rel:
            return _result(status="error", error="empty relative_path", code="invalid_arguments")
        target = (base / rel).resolve()
        if not _is_within(target, base) and not args.get("base_dir"):
            return _result(status="error", error="path escapes safe workspace", code="path_escape")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(content)
        except Exception as exc:
            return _result(status="error", error=str(exc), code="append_failed", path=str(target))
        self._logger.text("tool_append_file_content", content, path=str(target))
        self._remember_path(str(target))
        return _result(status="ok", message="file appended", path=str(target))

    def _t_move_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        src_str = str(args.get("source_path", "")).strip()
        dst_str = str(args.get("destination_path", "")).strip()
        overwrite = bool(args.get("overwrite", False))
        if not src_str or not dst_str:
            return _result(status="error", error="source_path and destination_path are required", code="invalid_arguments")
        try:
            src = Path(src_str).expanduser().resolve()
            dst = Path(dst_str).expanduser().resolve()
        except Exception as exc:
            return _result(status="error", error=f"path resolve failed: {exc}", code="path_resolve")
        if not src.exists():
            return _result(status="error", error=f"source does not exist: {src}", code="not_found", source=str(src))
        if _is_protected_path(src) or _is_protected_path(dst):
            return _result(status="error", error="refusing to move into/from a system directory", code="protected_path")
        if not dst.parent.exists():
            return _result(
                status="error",
                error=f"destination parent does not exist: {dst.parent}",
                code="invalid_destination",
                next_action="create_folder_first_then_retry",
            )
        if dst.exists() and not overwrite:
            return _result(
                status="needs_confirmation",
                error=f"destination already exists at {dst}; pass overwrite=true to replace",
                code="exists",
                destination=str(dst),
            )
        try:
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.move(str(src), str(dst))
        except Exception as exc:
            return _result(status="error", error=str(exc), code="move_failed", source=str(src), destination=str(dst))
        self._remember_path(str(dst))
        self._logger.event("tool_move_file_ok", source=str(src), destination=str(dst))
        return _result(status="ok", message=f"moved to {dst}", source=str(src), destination=str(dst))

    def _t_rename_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path_str = str(args.get("path", "")).strip()
        new_name = str(args.get("new_name", "")).strip()
        if not path_str or not new_name:
            return _result(status="error", error="path and new_name are required", code="invalid_arguments")
        if "/" in new_name or "\\" in new_name:
            return _result(status="error", error="new_name must be a basename, not a path", code="invalid_arguments")
        try:
            src = Path(path_str).expanduser().resolve()
        except Exception as exc:
            return _result(status="error", error=f"path resolve failed: {exc}", code="path_resolve")
        if not src.exists():
            return _result(status="error", error=f"path does not exist: {src}", code="not_found")
        if _is_protected_path(src):
            return _result(status="error", error="refusing to rename a system path", code="protected_path")
        dst = src.parent / new_name
        if dst.exists():
            return _result(
                status="needs_confirmation",
                error=f"a file named {new_name} already exists in {src.parent}",
                code="exists",
                destination=str(dst),
            )
        try:
            src.rename(dst)
        except Exception as exc:
            return _result(status="error", error=str(exc), code="rename_failed")
        self._remember_path(str(dst))
        self._logger.event("tool_rename_file_ok", source=str(src), destination=str(dst))
        return _result(status="ok", message=f"renamed to {dst}", source=str(src), destination=str(dst))

    def _t_delete_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path_str = str(args.get("path", "")).strip()
        confirmed = bool(args.get("confirmed", False))
        if not path_str:
            return _result(status="error", error="path is required", code="invalid_arguments")
        try:
            target = Path(path_str).expanduser().resolve()
        except Exception as exc:
            return _result(status="error", error=f"path resolve failed: {exc}", code="path_resolve")
        if not target.exists():
            return _result(status="ok", message=f"already absent: {target}", path=str(target))
        if _is_protected_path(target):
            return _result(status="error", error=f"refusing to delete protected path: {target}", code="protected_path")
        if not confirmed:
            return _result(
                status="needs_confirmation",
                error="ask the user for explicit permission, then call again with confirmed=true",
                code="needs_explicit_confirm",
                path=str(target),
                next_action="ask_user_then_retry_with_confirmed_true",
            )
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except Exception as exc:
            return _result(status="error", error=str(exc), code="delete_failed", path=str(target))
        self._logger.event("tool_delete_file_ok", path=str(target))
        return _result(status="ok", message=f"deleted {target}", path=str(target))

    def _t_list_recent_paths(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return _result(
            status="ok",
            paths=list(self._recent_paths),
            count=len(self._recent_paths),
            note=(
                "Paths the agent created or wrote in this session, "
                "most-recent first. If you've lost track of where a "
                "user-created file is, the latest entry is almost "
                "certainly it."
            ),
        )

    # File-type groups for list_files' `file_type` filter. Lets the model ask
    # for "image"/"document"/etc. instead of guessing extensions.
    _FILE_TYPE_GROUPS = {
        "image": {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tif", ".tiff", ".heic"},
        "screenshot": {".png", ".jpg", ".jpeg"},
        "document": {".pdf", ".docx", ".doc", ".txt", ".md", ".rtf", ".odt", ".xlsx", ".xls", ".pptx", ".ppt", ".csv"},
        "pdf": {".pdf"},
        "video": {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".webm", ".m4v"},
        "audio": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"},
        "archive": {".zip", ".7z", ".rar", ".tar", ".gz"},
        "code": {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".rb", ".php", ".html", ".css", ".json", ".yaml", ".yml"},
    }

    def _resolve_listing_folder(self, folder: str):
        """Resolve a folder for list_files: friendly names ('downloads',
        'desktop', 'screenshots', ...) OR an absolute/`~` path. Handles
        OneDrive-redirected known folders. Returns a Path or None."""
        home = Path.home()
        onedrive = home / "OneDrive"

        def first_existing(*paths):
            for p in paths:
                try:
                    if p and p.exists() and p.is_dir():
                        return p
                except Exception:
                    pass
            return None

        known = {
            "desktop": first_existing(home / "Desktop", onedrive / "Desktop"),
            "documents": first_existing(home / "Documents", onedrive / "Documents"),
            "docs": first_existing(home / "Documents", onedrive / "Documents"),
            "downloads": first_existing(home / "Downloads", onedrive / "Downloads"),
            "music": first_existing(home / "Music"),
            "pictures": first_existing(home / "Pictures", onedrive / "Pictures"),
            "photos": first_existing(home / "Pictures", onedrive / "Pictures"),
            "videos": first_existing(home / "Videos"),
            "screenshots": first_existing(
                home / "Pictures" / "Screenshots", onedrive / "Pictures" / "Screenshots",
            ),
            "home": home,
        }
        s = " ".join(str(folder or "").lower().split()).strip()
        if not s:
            return None
        if s in known and known[s] is not None:
            return known[s]
        for key, path in known.items():
            if path is not None and key in s:
                return path
        try:
            cand = Path(folder).expanduser().resolve()
        except Exception:
            return None
        return cand if (cand.exists() and cand.is_dir()) else None

    # Directory names never worth descending into during a search-everywhere
    # scan (huge and never hold user documents). Lowercased.
    _SCAN_SKIP_DIRS = {
        "node_modules", ".git", ".svn", ".hg", "__pycache__", "venv", ".venv",
        "env", ".cache", ".next", "$recycle.bin", "appdata",
    }

    @staticmethod
    def _norm_filename(s: str) -> str:
        """Lowercase + treat _, -, . as spaces so 'touchless drawing' matches
        'Touchless_Drawing_9.png'. Used for separator-insensitive name search."""
        s = str(s or "").lower()
        for ch in ("_", "-", ".", "(", ")", "[", "]"):
            s = s.replace(ch, " ")
        return " ".join(s.split())

    def _t_list_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find files with REAL paths so the model can act on the right one
        (move/read/open) instead of guessing. Supports friendly folder names, a
        file_type filter, and a name substring whose spaces/underscores/hyphens
        are interchangeable. Sorts newest-first by default. If `folder` is
        omitted or 'all'/'anywhere', searches across the user's common folders
        (Desktop, Documents, Downloads, Pictures, Videos, Music) recursively so
        'the three latest touchless drawings' works without naming the folder."""
        folder = str(args.get("folder", "")).strip()
        search_all = (not folder) or folder.lower() in ("all", "any", "everywhere", "anywhere", "*")

        file_type = str(args.get("file_type", "") or "").strip().lower()
        name_contains = self._norm_filename(args.get("name_contains", "") or "")
        recursive = bool(args.get("recursive", False))
        sort = str(args.get("sort", "recent") or "recent").strip().lower()
        limit = max(1, min(100, int(args.get("limit", 20) or 20)))

        exts = None
        if file_type and file_type not in ("any", "all", "*"):
            exts = self._FILE_TYPE_GROUPS.get(file_type)
            if exts is None:
                exts = {file_type if file_type.startswith(".") else "." + file_type}

        # Resolve the set of root folders to scan.
        if search_all:
            roots = []
            for name in ("desktop", "documents", "downloads", "pictures", "videos", "music"):
                r = self._resolve_listing_folder(name)
                if r is not None and r not in roots:
                    roots.append(r)
            recursive = True  # searching everywhere implies looking in subfolders
        else:
            base = self._resolve_listing_folder(folder)
            if base is None:
                return _result(
                    status="error",
                    error=f"folder not found: {folder!r}",
                    code="folder_not_found",
                    hint="Pass a known name (downloads, desktop, documents, pictures, screenshots, videos, music), an absolute path, or 'all' to search everywhere.",
                )
            roots = [base]

        entries: list = []
        scanned = 0
        budget = 40000  # hard cap so a search-everywhere can't run away
        deadline = time.time() + 8.0
        for root in roots:
            if scanned > budget or time.time() > deadline:
                break
            try:
                walker = os.walk(root) if recursive else [(str(root), [], [p.name for p in root.iterdir()])]
                for dirpath, dirnames, filenames in walker:
                    if scanned > budget or time.time() > deadline:
                        break
                    # Prune heavy/hidden dirs in place so os.walk skips them.
                    if recursive:
                        dirnames[:] = [
                            d for d in dirnames
                            if d.lower() not in self._SCAN_SKIP_DIRS and not d.startswith(".")
                        ]
                    for fn in filenames:
                        scanned += 1
                        if scanned > budget:
                            break
                        if exts is not None and os.path.splitext(fn)[1].lower() not in exts:
                            continue
                        if name_contains and name_contains not in self._norm_filename(fn):
                            continue
                        try:
                            fp = Path(dirpath) / fn
                            st = fp.stat()
                            entries.append((fp, st.st_mtime, st.st_size))
                        except Exception:
                            continue
            except Exception:
                continue

        if sort in ("name", "alpha", "alphabetical"):
            entries.sort(key=lambda e: e[0].name.lower())
        elif sort in ("oldest",):
            entries.sort(key=lambda e: e[1])
        else:  # "recent" / default — newest first
            entries.sort(key=lambda e: e[1], reverse=True)

        total = len(entries)
        files = []
        for p, mtime, size in entries[:limit]:
            files.append({
                "name": p.name,
                "path": str(p),
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)),
                "size_kb": round(size / 1024, 1),
            })
        return _result(
            status="ok",
            searched=("all common folders" if search_all else str(roots[0])),
            count=total,
            returned=len(files),
            files=files,
            note=("Use the 'path' of the file you want directly in move_file / "
                  "read_file / open_path. Results are newest-first, so 'the N "
                  "latest X' = the first N. If empty, drop the folder (search "
                  "everywhere) or loosen name_contains."),
        )

    def _t_run_existing_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self._external_router is None:
            return _result(
                status="not_implemented_yet",
                error="no external action router wired up",
                code="no_router",
            )
        action_name = str(args.get("action_name", ""))
        params = args.get("parameters") or {}
        try:
            return self._external_router(action_name, dict(params))
        except Exception as exc:
            self._logger.exception("external_router_failed", exc, action=action_name)
            return _result(status="error", error=str(exc), code="router_exception")

    def _t_run_python_script(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path_str = str(args.get("script_path", "")).strip()
        if not path_str:
            return _result(status="error", error="empty script_path", code="invalid_arguments")
        script_path = Path(path_str).expanduser().resolve()
        if not script_path.exists():
            return _result(
                status="error", error=f"script not found: {script_path}",
                code="not_found", path=str(script_path),
            )
        if script_path.suffix.lower() != ".py":
            return _result(
                status="error",
                error=f"only .py files supported, got {script_path.suffix}",
                code="unsupported_extension",
                path=str(script_path),
            )
        wait = bool(args.get("wait_for_exit", False))
        timeout = float(args.get("timeout_sec", 15.0) or 15.0)
        from ..utils.subprocess_utils import hidden_subprocess_kwargs
        python_exe = _resolve_python_executable()
        if not python_exe:
            return _result(
                status="error",
                error=(
                    "no python interpreter found. Install Python and "
                    "ensure 'python' is on PATH, or set the "
                    "TOUCHLESS_PYTHON env var to its absolute path."
                ),
                code="no_python",
                path=str(script_path),
            )
        cmd = [python_exe, str(script_path)]
        try:
            if wait:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=max(1.0, timeout),
                    cwd=str(script_path.parent),
                    **hidden_subprocess_kwargs(),
                )
                return _result(
                    status="ok" if proc.returncode == 0 else "script_error",
                    returncode=proc.returncode,
                    stdout=(proc.stdout or "")[:4000],
                    stderr=(proc.stderr or "")[:2000],
                    python=python_exe,
                    path=str(script_path),
                )
            else:
                # Detached — for GUI scripts (tkinter / pygame / pyside).
                popen = subprocess.Popen(
                    cmd,
                    cwd=str(script_path.parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **hidden_subprocess_kwargs(),
                )
                return _result(
                    status="ok",
                    message="script launched (detached)",
                    pid=popen.pid,
                    python=python_exe,
                    path=str(script_path),
                )
        except subprocess.TimeoutExpired:
            return _result(
                status="error",
                error=f"script timed out after {timeout}s",
                code="timeout",
                path=str(script_path),
            )
        except Exception as exc:
            return _result(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                code="run_failed",
                path=str(script_path),
            )

    def _t_run_matlab_script(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run a MATLAB .m script: launches the MATLAB desktop in the script's
        folder and executes it (matlab -sd <folder> -r <script>), so figures/
        plots display and MATLAB stays open. Use this to run a script Claude
        wrote into a MATLAB project folder — no MATLAB UI clicking needed."""
        path_str = str(args.get("script_path") or args.get("path") or "").strip()
        if not path_str:
            return _result(status="error", error="empty script_path", code="invalid_arguments")
        p = Path(path_str).expanduser()
        if not (p.exists() and p.is_file()):
            # Resolve by name via file search.
            desktop = self._ensure_desktop()
            if desktop is not None:
                r, amb = desktop.resolve_named_file(path_str)
                if r is not None:
                    p = Path(r)
        if not (p.exists() and p.is_file()):
            return _result(status="error", error=f"script not found: {path_str}", code="not_found")
        if p.suffix.lower() != ".m":
            return _result(status="error", error=f"not a MATLAB .m file: {p.suffix}", code="unsupported_extension")
        matlab = _matlab_exe()
        if not matlab:
            return _result(status="error", error="MATLAB not found (no matlab.exe)", code="no_matlab")
        folder = str(p.parent)
        stem = p.stem  # script name without .m, run from its folder
        # -sd sets the start directory; -r runs the script then leaves MATLAB
        # open so the user can see the figures.
        ok = launch_external(matlab, args=["-sd", folder, "-r", stem])
        if not ok:
            return _result(status="error", error="could not launch MATLAB", code="launch_failed")
        return _result(status="ok", message=f"MATLAB launching and running {p.name}",
                       script=str(p), folder=folder,
                       note="MATLAB opens and runs the script; figures/plots will appear there.")

    def _t_skip_youtube_ad(self, args: Dict[str, Any]) -> Dict[str, Any]:
        controller = self._ensure_youtube()
        if controller is None:
            return _result(status="not_implemented_yet", error="youtube controller unavailable")
        try:
            ok = controller.skip_ad()
        except Exception as exc:
            self._logger.exception("skip_ad_exception", exc)
            return _result(status="error", error=str(exc), code="exception")
        return _result(
            status="ok" if ok else "no_skippable_ad",
            message=getattr(controller, "_message", ""),
        )

    def _t_ask_user_confirmation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        message = str(args.get("message", "Continue?"))
        risk = str(args.get("risk_level", "medium"))
        if self._confirm is None:
            return _result(status="needs_confirmation", approved=False, error="no confirm UI wired up")
        try:
            approved = bool(self._confirm(message, risk))
        except Exception as exc:
            self._logger.exception("confirm_callback_failed", exc)
            return _result(status="error", error=str(exc), code="confirm_exception")
        return _result(status="ok", approved=approved, risk_level=risk)

    # ---- helpers ----

    def _remember_path(self, path: str) -> None:
        """Push to the most-recent-first list, dropping duplicates and capping at 20."""
        s = str(path)
        try:
            self._recent_paths = [s] + [p for p in self._recent_paths if p != s]
            self._recent_paths = self._recent_paths[:20]
        except Exception:
            pass

    def _resolve_base_dir(self, base_dir: Optional[str]) -> Path:
        """Resolve a model-supplied base_dir to an absolute Path.

        Returns the safe workspace when nothing was provided. Otherwise
        validates that the path's PARENT exists so we don't silently
        create weird top-level dirs (e.g. the model passed
        "C:\\Documents\\Demo" — Documents at the C:\\ root doesn't
        normally exist, and parents=True would silently invent it).
        Raises ValueError if validation fails so the caller surfaces a
        corrective error to the model.
        """
        if not base_dir:
            return self._safe_workspace.resolve()

        p = Path(str(base_dir)).expanduser()
        try:
            resolved = p.resolve()
        except Exception as exc:
            raise ValueError(f"could not resolve path: {exc}")

        # If the directory already exists, we're done — no validation needed.
        if resolved.exists():
            return resolved

        # New directory: only allow creation if the PARENT already
        # exists. This catches cases like "C:\Documents\Demo" where
        # Documents isn't at the root — the model probably meant
        # "~/Documents/Demo" and we should reject so the model
        # corrects on the next hop.
        parent = resolved.parent
        if not parent.exists():
            raise ValueError(
                f"refusing to create '{resolved}' because the parent "
                f"'{parent}' does not exist. Did you mean a path under "
                f"~/Documents (which expands to "
                f"{Path.home() / 'Documents'})?"
            )
        try:
            resolved.mkdir(exist_ok=True)
        except Exception as exc:
            raise ValueError(f"mkdir failed: {exc}")
        return resolved

    def _ensure_mouse(self) -> Optional[MouseController]:
        if self._mouse is None:
            try:
                self._mouse = MouseController()
            except Exception as exc:
                self._logger.exception("mouse_init_failed", exc)
                self._mouse = None
        return self._mouse

    def _ensure_web(self):
        if self._web is None:
            try:
                from .web_controller import WebController
                self._web = WebController(self._config, self._logger)
            except Exception as exc:
                self._logger.exception("web_controller_init_failed", exc)
                self._web = None
        return self._web

    def _t_compose_text(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Mid-plan natural-language synthesis. Used by Tier 2 plans like
        # 'create doc and write a debrief about my weather + emails' —
        # gather data in earlier steps, compose the prose with this tool,
        # then feed {step:N.text} into gdocs_create's `text` arg.
        from .compose import compose_text
        return compose_text(
            prompt=str(args.get("prompt") or ""),
            inputs=args.get("inputs"),
            max_tokens=int(args.get("max_tokens") or 500),
        )

    def _t_email_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Cascade Gmail → Microsoft → screen-read fallback for 'summarize
        my unread emails' style requests. The LLM has been caught
        fabricating demo emails when it touches mail data directly, so
        Layer 1 runs this deterministic cascade and emits the
        connector's pre-rendered `summary` field verbatim.

        Returns the SAME shape as gmail_list / ms_mail_list so the
        orchestrator's existing _format_email_list path applies. Adds a
        `source` field indicating which connector answered ('gmail',
        'microsoft', or 'none')."""
        unread_only = bool(args.get("unread_only", True))
        max_n = max(1, min(50, int(args.get("max") or 50)))
        include_body = bool(args.get("include_body"))
        sub_args = {"unread_only": unread_only, "max": max_n,
                    "include_body": include_body}

        # _try returns a 3-state shape so the cascade can distinguish:
        #   ("missing",     None)   -> connector class isn't even wired up
        #   ("unavailable", None)   -> connector exists but available()=False
        #                             (user hasn't connected/authenticated)
        #   ("ok",          result) -> connector ran; result is the dict
        # Without this 3-state, the old code couldn't tell "Outlook wasn't
        # checked" from "Outlook has 0 unread" — which produced the
        # infamous 'No unread emails in your Gmail account' message even
        # when the user has tons of unread in Outlook desktop.
        def _try(connector_id: str, tool_name: str) -> tuple:
            try:
                connector = self._registry.find_connector(connector_id)
            except Exception:
                connector = None
            if connector is None:
                return ("missing", None)
            try:
                if not connector.available():
                    return ("unavailable", None)
            except Exception:
                return ("unavailable", None)
            try:
                return ("ok", connector.execute(tool_name, dict(sub_args)))
            except Exception as exc:
                return ("ok", _result(
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    code="connector_failed"))

        # Per-connector state for accurate cascade messaging below.
        gmail_state, gmail_res = _try("gmail", "gmail_list")
        if gmail_state == "ok" and gmail_res is not None:
            if (gmail_res.get("status") == "ok"
                    and int(gmail_res.get("count") or 0) > 0):
                gmail_res["source"] = "gmail"
                return gmail_res

        ms_state, ms_res = _try("ms365", "ms_mail_list")
        if ms_state == "ok" and ms_res is not None:
            if (ms_res.get("status") == "ok"
                    and int(ms_res.get("count") or 0) > 0):
                ms_res["source"] = "microsoft"
                return ms_res

        # Surface a connector-level hard error verbatim so the user sees the
        # real failure instead of a misleading "no unread" message.
        for state, res in ((gmail_state, gmail_res), (ms_state, ms_res)):
            if state == "ok" and res is not None and res.get("status") != "ok":
                return res

        gmail_ok_empty = (gmail_state == "ok" and gmail_res is not None
                          and gmail_res.get("status") == "ok"
                          and int(gmail_res.get("count") or 0) == 0)
        ms_ok_empty = (ms_state == "ok" and ms_res is not None
                       and ms_res.get("status") == "ok"
                       and int(ms_res.get("count") or 0) == 0)

        screen_hint = ("If you have an inbox open in Outlook desktop that "
                       "the connectors can't see (e.g. an IMAP/Exchange "
                       "account that's not connected to Touchless yet), "
                       "say 'read my Outlook screen' and I'll OCR the "
                       "inbox you have open.")

        # Both connectors checked, both empty — the user's main complaint
        # is that we said "no unread in Gmail" without mentioning Outlook
        # at all, so this branch explicitly names both.
        if gmail_ok_empty and ms_ok_empty:
            return _result(
                status="ok", count=0, messages=[], source="both",
                summary=("Both your Gmail and Outlook (Graph) are showing "
                         "no unread. " + screen_hint))

        # Exactly one connector returned 0 and the other isn't connected.
        # Name the one we checked AND say plainly that the other wasn't
        # checked — never claim "no unread emails" globally.
        if gmail_ok_empty and ms_state != "ok":
            return _result(
                status="ok", count=0, messages=[], source="gmail",
                summary=("Your Gmail has no unread. I didn't check Outlook "
                         "because your Microsoft account isn't connected "
                         "to Touchless yet — connect it in settings, or "
                         "say 'read my Outlook screen' and I'll OCR the "
                         "inbox you have open."))
        if ms_ok_empty and gmail_state != "ok":
            return _result(
                status="ok", count=0, messages=[], source="microsoft",
                summary=("Your Outlook (via Microsoft Graph) has no "
                         "unread. I didn't check Gmail because it isn't "
                         "connected to Touchless yet — connect it in "
                         "settings, or say 'read my Outlook screen' and "
                         "I'll OCR the inbox you have open."))

        # Neither connector is wired up / available at all. Be explicit
        # about both options (connect, or screen-read).
        return _result(
            status="ok", count=0, messages=[], source="none",
            summary=("I don't have Gmail or Outlook connected via the "
                     "connectors. To summarize your inbox I can either "
                     "(a) connect Gmail / Microsoft in Touchless "
                     "settings, or (b) read your Outlook desktop window "
                     "directly — say 'read my Outlook screen' and I'll "
                     "OCR it."))

    def _t_weather_get(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Free, no-key weather via wttr.in / Open-Meteo. Auto-detects
        # user location from IP when `location` is omitted; otherwise
        # pass any city / zip / 'lat,lon' string. `days` controls the
        # forecast horizon (1-10, default 3). Cheap (~100-200ms HTTP).
        from .weather import get_weather
        try:
            days = int(args.get("days") or 0)
        except (TypeError, ValueError):
            days = 0
        return get_weather(
            location=str(args.get("location") or "").strip(),
            units=str(args.get("units") or "imperial").strip().lower(),
            days=days or 3,
        )

    def _t_web_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Structured search results (Google CSE preferred, DuckDuckGo
        # fallback). The planner uses this + web_navigate to read a
        # specific result without spinning up the Chrome controller for
        # the SERP itself — much cheaper than navigate-then-scrape.
        from .web_search import web_search
        query = str(args.get("query", "")).strip()
        if not query:
            return _result(status="error", error="empty query", code="invalid_arguments")
        count = int(args.get("count", 5) or 5)
        site = str(args.get("site", "")).strip()
        recent_days = int(args.get("recent_days", 0) or 0)
        return web_search(query, count=count, site=site, recent_days=recent_days)

    def _t_web_navigate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        web = self._ensure_web()
        if web is None:
            return _result(status="error", error="web controller unavailable", code="no_web")
        target = str(args.get("url_or_query", "")).strip()
        if not target:
            return _result(status="error", error="empty url_or_query", code="invalid_arguments")
        return web.navigate(target)

    def _t_web_get_links(self, args: Dict[str, Any]) -> Dict[str, Any]:
        web = self._ensure_web()
        if web is None:
            return _result(status="error", error="web controller unavailable", code="no_web")
        contains = args.get("contains")
        limit = int(args.get("limit", 30) or 30)
        return web.get_links(contains=contains, limit=limit)

    def _t_web_get_text(self, args: Dict[str, Any]) -> Dict[str, Any]:
        web = self._ensure_web()
        if web is None:
            return _result(status="error", error="web controller unavailable", code="no_web")
        max_chars = int(args.get("max_chars", 4000) or 4000)
        return web.get_text(max_chars=max_chars)

    def _t_web_click(self, args: Dict[str, Any]) -> Dict[str, Any]:
        web = self._ensure_web()
        if web is None:
            return _result(status="error", error="web controller unavailable", code="no_web")
        text = str(args.get("text", "")).strip()
        if not text:
            return _result(status="error", error="empty text", code="invalid_arguments")
        return web.click_text(text)

    def _t_web_scroll(self, args: Dict[str, Any]) -> Dict[str, Any]:
        web = self._ensure_web()
        if web is None:
            return _result(status="error", error="web controller unavailable", code="no_web")
        return web.scroll(str(args.get("to", "bottom")))

    def _t_web_fill(self, args: Dict[str, Any]) -> Dict[str, Any]:
        web = self._ensure_web()
        if web is None:
            return _result(status="error", error="web controller unavailable", code="no_web")
        target = str(args.get("field", "")).strip()
        if not target:
            return _result(status="error", error="empty field", code="invalid_arguments")
        return web.fill(target, str(args.get("text", "")))

    def _t_web_wait_for(self, args: Dict[str, Any]) -> Dict[str, Any]:
        web = self._ensure_web()
        if web is None:
            return _result(status="error", error="web controller unavailable", code="no_web")
        query = str(args.get("query", "")).strip()
        if not query:
            return _result(status="error", error="empty query", code="invalid_arguments")
        return web.wait_for(query, float(args.get("timeout_sec", 15) or 15))

    def _t_web_eval(self, args: Dict[str, Any]) -> Dict[str, Any]:
        web = self._ensure_web()
        if web is None:
            return _result(status="error", error="web controller unavailable", code="no_web")
        expr = str(args.get("javascript", "")).strip()
        if not expr:
            return _result(status="error", error="empty javascript", code="invalid_arguments")
        return web.evaluate(expr)

    def _ensure_text(self) -> Optional[TextInputController]:
        if self._text is None:
            try:
                self._text = TextInputController()
            except Exception as exc:
                self._logger.exception("text_input_init_failed", exc)
                self._text = None
        return self._text

    def _ensure_youtube(self):
        if self._youtube is None:
            try:
                from ..debug.youtube_controller import YouTubeController
                self._youtube = YouTubeController()
            except Exception as exc:
                self._logger.exception("youtube_init_failed", exc)
                self._youtube = None
        return self._youtube

    def _ensure_chrome(self):
        if self._chrome is None:
            try:
                from ..debug.chrome_controller import ChromeController
                self._chrome = ChromeController()
            except Exception as exc:
                self._logger.exception("chrome_init_failed", exc)
                self._chrome = None
        return self._chrome

    def _ensure_desktop(self):
        if self._desktop is None:
            try:
                from ..debug.desktop_controller import DesktopController
                self._desktop = DesktopController()
            except Exception as exc:
                self._logger.exception("desktop_init_failed", exc)
                self._desktop = None
        return self._desktop

    def _resolve_one_window(self, title: str, verb: str):
        """Find exactly one window by title/process. Returns (hwnd, name) or an
        error/ambiguous _result dict (caller checks for 'status')."""
        if not title:
            return _result(status="error", error="empty window_title", code="invalid_arguments")
        matches = self._find_windows(title)
        if not matches:
            return _result(status="error", error=f"no open window matching '{title}'", code="no_window")
        if len(matches) > 1:
            return _result(status="ambiguous", matches=[t for _h, t in matches],
                           note=f"Several windows match. Ask which to {verb}, then call again with that exact name.")
        return matches[0]

    def _t_control_window(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Close / minimize / maximize / restore / focus a SPECIFIC window
        (by title or app/process name). Targets only that window — close uses
        WM_CLOSE, never Alt+F4. If several windows match, returns ambiguous so
        you ask which."""
        title = str(args.get("window_title", "")).strip()
        action = str(args.get("action", "")).strip().lower()
        valid = {"close", "minimize", "maximize", "restore", "focus"}
        if action not in valid:
            return _result(status="error", error=f"action must be one of {sorted(valid)}", code="invalid_arguments")
        resolved = self._resolve_one_window(title, action)
        if isinstance(resolved, dict):  # error or ambiguous
            return resolved
        hwnd, matched_title = resolved
        try:
            user32 = ctypes.windll.user32
            if action == "close":
                user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE — this window only
            elif action == "minimize":
                user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
            elif action == "maximize":
                user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
            elif action == "restore":
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            elif action == "focus":
                user32.ShowWindow(hwnd, 9)  # restore if minimized
                user32.SetForegroundWindow(hwnd)
            self._logger.event("control_window", action=action, title=matched_title)
            return _result(status="ok", action=action, window=matched_title)
        except Exception as exc:
            self._logger.exception("control_window_failed", exc, action=action)
            return _result(status="error", error=str(exc), code="control_failed")

    def _t_close_window(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience alias: close a specific window (delegates to
        control_window with action='close')."""
        return self._t_control_window({"window_title": args.get("window_title", ""), "action": "close"})

    def _t_move_window(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Move a window to a monitor. window_title matches by title OR process
        name. If MULTIPLE windows match (e.g. several File Explorer windows),
        returns status='ambiguous' with their names so you ask the user which.
        monitor: 'primary'/'main' (1), 'secondary'/'second' (2), or a 1-based
        index. placement: 'maximize' (default, fill the monitor) or 'center'
        (keep the window's size, centered on that monitor)."""
        title = str(args.get("window_title", "")).strip()
        if not title:
            return _result(status="error", error="empty window_title", code="invalid_arguments")
        target = str(args.get("monitor", "secondary")).strip().lower()
        placement = str(args.get("placement", "maximize")).strip().lower()

        # Poll briefly for matching windows.
        matches = []
        deadline = time.time() + 3.0
        while time.time() < deadline:
            matches = self._find_windows(title)
            if matches:
                break
            time.sleep(0.3)
        if not matches:
            self._logger.event("move_window_no_window", title=title)
            return _result(status="error", error=f"no open window matching '{title}'", code="no_window")
        if len(matches) > 1:
            # Ask the user which one — list their exact titles.
            names = [t for _h, t in matches]
            self._logger.event("move_window_ambiguous", count=len(names), names=names)
            return _result(
                status="ambiguous",
                matches=names,
                note=("Several windows match. Ask the user which one (list these "
                      "names), then call move_window_to_monitor again with that "
                      "exact name as window_title."),
            )
        hwnd, matched_title = matches[0]
        try:
            placed = self._place_window(hwnd, target, placement)
            if placed.get("status") != "ok":
                return placed
            return _result(status="ok", moved=matched_title, monitor=placed.get("monitor"),
                           of=placed.get("of"), placement=placement)
        except Exception as exc:
            self._logger.exception("move_window_failed", exc)
            return _result(status="error", error=str(exc), code="move_failed")

    def _place_window(self, hwnd: int, target: str, placement: str) -> Dict[str, Any]:
        """Move+size a window onto a monitor. target: primary/secondary/index.
        placement: 'maximize' (fill), 'center' (keep size, centered), or a HALF/
        QUADRANT snap: left, right, top, bottom, or top-left/top-right/
        bottom-left/bottom-right. Shared by move_window_to_monitor and open_app's
        monitor/placement options."""
        mons = self._list_monitors()
        self._logger.event("place_window_monitors", count=len(mons), requested=target)
        if not mons:
            return _result(status="error", error="no monitors found", code="no_monitor")
        t = str(target).strip().lower()
        if t in ("primary", "main", "1", "first"):
            idx = 0
        elif t in ("secondary", "second", "2", "other"):
            idx = 1
        else:
            try:
                idx = max(1, int(t)) - 1
            except Exception:
                idx = 1
        if idx >= len(mons):
            return _result(status="error", error=f"only {len(mons)} monitor(s) connected",
                           code="no_such_monitor", monitors=len(mons))
        left, top, right, bottom = mons[idx]
        mon_w, mon_h = right - left, bottom - top
        p = str(placement).strip().lower().replace("_", "-").replace(" ", "-")
        try:
            user32 = ctypes.windll.user32
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE so the move takes
            # Half / quadrant snaps: compute the sub-rect of the monitor.
            half = {
                "left": (left, top, mon_w // 2, mon_h),
                "right": (left + mon_w // 2, top, mon_w - mon_w // 2, mon_h),
                "top": (left, top, mon_w, mon_h // 2),
                "bottom": (left, top + mon_h // 2, mon_w, mon_h - mon_h // 2),
                "left-half": (left, top, mon_w // 2, mon_h),
                "right-half": (left + mon_w // 2, top, mon_w - mon_w // 2, mon_h),
                "top-left": (left, top, mon_w // 2, mon_h // 2),
                "top-right": (left + mon_w // 2, top, mon_w - mon_w // 2, mon_h // 2),
                "bottom-left": (left, top + mon_h // 2, mon_w // 2, mon_h - mon_h // 2),
                "bottom-right": (left + mon_w // 2, top + mon_h // 2, mon_w - mon_w // 2, mon_h - mon_h // 2),
            }
            if p in half:
                hx, hy, hw, hh = half[p]
                user32.SetWindowPos(hwnd, 0, hx, hy, hw, hh, 0x0004 | 0x0040)
            elif p in ("center", "centered", "centre"):
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                w = max(200, rect.right - rect.left)
                h = max(150, rect.bottom - rect.top)
                w = min(w, mon_w); h = min(h, mon_h)
                nx = left + (mon_w - w) // 2
                ny = top + (mon_h - h) // 2
                user32.SetWindowPos(hwnd, 0, nx, ny, w, h, 0x0040)  # SWP_SHOWWINDOW
            else:
                user32.SetWindowPos(hwnd, 0, left, top, mon_w, mon_h, 0x0004 | 0x0040)
                user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
            try:
                user32.SetForegroundWindow(hwnd)  # make it the active window there
            except Exception:
                pass
            return _result(status="ok", monitor=idx + 1, of=len(mons), placement=p)
        except Exception as exc:
            self._logger.exception("place_window_failed", exc)
            return _result(status="error", error=str(exc), code="place_failed")

    def _reassert_placement(self, name: str, target: str, placement: str) -> None:
        """Re-apply a monitor placement a couple times after a delay, on a
        background thread. Apps like Discord/VS Code restore their own saved
        bounds shortly after launch, which silently undoes a single placement.
        Best-effort and non-blocking; re-finds the window each time in case its
        hwnd changed (Discord swaps its splash window for the main one)."""
        import threading

        def _runner():
            for delay in (2.0, 4.5):
                time.sleep(delay)
                try:
                    wins = self._find_windows(name)
                    if not wins:
                        continue
                    self._place_window(wins[0][0], target, placement)
                except Exception:
                    pass

        try:
            threading.Thread(target=_runner, daemon=True).start()
        except Exception:
            pass

    def _list_monitors(self) -> list:
        """Return each monitor's full rect (left,top,right,bottom), primary
        first, via EnumDisplayMonitors."""
        # FRESH WinDLL — the shared windll.user32 may reject our WINFUNCTYPE
        # callback due to argtypes polluted by other modules (same issue as
        # _find_windows), which made EnumDisplayMonitors return no monitors.
        user32 = ctypes.WinDLL("user32")
        out: list = []
        # MonitorEnumProc is BOOL(HMONITOR, HDC, LPRECT, LPARAM). The last arg
        # is LPARAM (pointer-sized int), NOT c_double.
        MonEnumProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(wintypes.RECT), ctypes.c_void_p,
        )

        def _cb(hmon, hdc, lprc, _data):
            try:
                r = lprc.contents
                rect = (int(r.left), int(r.top), int(r.right), int(r.bottom))
                # Primary monitor has its top-left at (0,0) — list it first.
                if rect[0] == 0 and rect[1] == 0:
                    out.insert(0, rect)
                else:
                    out.append(rect)
            except Exception:
                pass
            return True

        try:
            user32.EnumDisplayMonitors(0, 0, MonEnumProc(_cb), 0)
        except Exception as exc:
            self._logger.exception("enum_monitors_failed", exc)
        return out

    def _move_window_to_primary(self, name_match: str) -> None:
        """Move a freshly-opened window onto the PRIMARY monitor + maximize.
        Runs on a BACKGROUND thread with delays: apps (VS Code) apply their
        saved, often multi-monitor bounds a moment AFTER launch, so a single
        immediate move gets overridden — we re-assert primary at ~2.5s and
        ~4.5s, after it settles. Non-blocking."""
        import threading

        def _runner():
            import time as _t
            for gap in (2.5, 2.0):  # passes at ~2.5s and ~4.5s post-launch
                _t.sleep(gap)
                try:
                    self._do_primary_move(name_match)
                except Exception:
                    pass

        threading.Thread(target=_runner, name="IrisWindowMove", daemon=True).start()

    def _proc_name_of(self, pid: int) -> str:
        """basename of a pid's exe (lowercased), e.g. 'explorer.exe'."""
        if not pid:
            return ""
        try:
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
            if not h:
                return ""
            try:
                buf = ctypes.create_unicode_buffer(260)
                size = ctypes.c_ulong(260)
                if ctypes.windll.kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                    return os.path.basename(buf.value).lower()
            finally:
                ctypes.windll.kernel32.CloseHandle(h)
        except Exception:
            pass
        return ""

    def _find_windows(self, name_match: str) -> list:
        """Return ALL visible top-level windows whose TITLE or PROCESS NAME
        contains name_match, as [(hwnd, title), ...], skipping our own process
        and de-duplicated by title. Delegates to the UIA controller's window
        enumeration, which is proven to work in the live app (the auto-approve
        watcher resolves windows through it) — our own EnumWindows callback
        misbehaved in this process. Matching the process name lets 'explorer'/
        'file explorer' find Explorer windows whose title is the folder name."""
        out: list = []
        needle = (name_match or "").strip().lower()
        if not needle:
            return out
        proc_aliases = {
            "file explorer": "explorer", "explorer": "explorer",
            "files": "explorer", "explorer window": "explorer",
            "this pc": "explorer", "folder": "explorer",
        }
        proc_needle = proc_aliases.get(needle, needle)
        try:
            # FRESH WinDLL instance — the shared ctypes.windll.user32 can have
            # polluted argtypes from other modules, which makes EnumWindows
            # REJECT our WINFUNCTYPE callback ("expected WinFunctionType
            # instead of WinFunctionType") and silently return nothing.
            user32 = ctypes.WinDLL("user32")
            own = os.getpid()
            seen = set()
            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

            def _cb(hwnd, _lp):
                try:
                    if not user32.IsWindowVisible(hwnd):
                        return True
                    pid = ctypes.c_ulong(0)
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    if int(pid.value) == own:
                        return True
                    ln = user32.GetWindowTextLengthW(hwnd)
                    if ln <= 0:
                        return True
                    buf = ctypes.create_unicode_buffer(ln + 1)
                    user32.GetWindowTextW(hwnd, buf, ln + 1)
                    raw = (buf.value or "").strip()
                    if not raw or raw == "Program Manager":
                        return True
                    rl = raw.lower()
                    proc = self._proc_name_of(int(pid.value))
                    # Match either direction (length-guarded): the model may
                    # pass "Applications - File Explorer" while the real title
                    # is just "Applications", or vice versa.
                    title_match = (needle in rl) or (len(rl) >= 4 and rl in needle)
                    if title_match or (proc and proc_needle in proc):
                        if raw not in seen:
                            seen.add(raw)
                            out.append((int(hwnd), raw))
                except Exception:
                    pass
                return True

            user32.EnumWindows(WNDENUMPROC(_cb), 0)
        except Exception as exc:
            self._logger.exception("find_windows_failed", exc)
        return out

    def _find_window_hwnd(self, name_match: str, timeout: float = 0.0) -> int:
        """Front-most window matching name_match (title or process), polling up
        to `timeout`s. Returns HWND int or 0. (Single-match convenience used by
        the launch/focus helpers; window-MOVE uses _find_windows for disambig.)"""
        deadline = time.time() + max(0.0, timeout)
        while True:
            matches = self._find_windows(name_match)
            if matches:
                return matches[0][0]
            if time.time() >= deadline:
                return 0
            time.sleep(0.4)

    def _ensure_vscode_foreground(self) -> Dict[str, Any]:
        """Make sure a VS Code window exists and is the foreground window so
        keystroke injection (Ctrl+Shift+P, paste) lands in it. Launches VS Code
        if none is open. Returns {'status':'ok','launched':bool} or an error."""
        user32 = ctypes.windll.user32
        hwnd = self._find_window_hwnd("Visual Studio Code")
        launched = False
        if not hwnd:
            # No VS Code window — launch it, then wait for the window.
            opened = False
            for candidate in _open_app_candidates("code"):
                if launch_external(candidate):
                    opened = True
                    break
            if not opened:
                return _result(status="error", error="could not launch VS Code", code="vscode_launch_failed")
            launched = True
            hwnd = self._find_window_hwnd("Visual Studio Code", timeout=8.0)
            if not hwnd:
                return _result(status="error", error="VS Code did not open in time", code="vscode_no_window")
            # New launches often span monitors — settle onto primary.
            try:
                self._do_primary_move("Visual Studio Code")
            except Exception:
                pass
            time.sleep(1.0)  # let it finish painting before we drive it
        # Bring it to the front so our keystrokes go to it, not whatever the
        # user last clicked.
        try:
            user32.ShowWindow(hwnd, 9)   # SW_RESTORE (in case minimized)
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.4)
        except Exception:
            pass
        return _result(status="ok", launched=launched, hwnd=int(hwnd))

    def _prepare_project_folder(self, name: str) -> Dict[str, Any]:
        """Create a project folder under the user's Documents and open it in
        VS Code as the active workspace — so the coding agent's working dir IS
        that folder and the files it writes land inside it. Then focus the
        window so we can drive the Command Palette. Returns ok+path or error."""
        name = (name or "").strip()
        if not name:
            return _result(status="error", error="empty project_folder", code="invalid_arguments")
        base = Path.home() / "Documents"
        if not base.exists():
            base = Path.home()
        target = (base / name).resolve()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return _result(status="error", error=str(exc), code="mkdir_failed", path=str(target))
        self._remember_path(str(target))
        opened = False
        for candidate in _open_app_candidates("code"):
            if launch_external(candidate, args=[str(target)]):
                opened = True
                break
        if not opened:
            return _result(status="error", error="created folder but could not open VS Code",
                           code="vscode_launch_failed", path=str(target))
        # Wait for the window whose title carries the folder name; fall back to
        # any VS Code window. Settle on the primary monitor, then focus it.
        hwnd = self._find_window_hwnd(target.name, timeout=8.0) or \
            self._find_window_hwnd("Visual Studio Code", timeout=4.0)
        if hwnd:
            try:
                self._do_primary_move(target.name)
            except Exception:
                pass
            time.sleep(1.2)  # let the workspace finish loading before driving it
            try:
                user32 = ctypes.windll.user32
                user32.ShowWindow(hwnd, 9)   # SW_RESTORE
                user32.SetForegroundWindow(hwnd)
                time.sleep(0.4)
            except Exception:
                pass
        return _result(status="ok", path=str(target))

    def _do_primary_move(self, name_match: str, timeout: float = 3.0) -> None:
        """One pass: find the window by title, then move+maximize it on the
        primary monitor."""
        try:
            import ctypes
            import os
            import time as _time
            from ctypes import wintypes
        except Exception:
            return
        try:
            user32 = ctypes.windll.user32
            h = self._find_window_hwnd(name_match, timeout=timeout)
            if not h:
                return
            rect = wintypes.RECT()
            # SPI_GETWORKAREA -> the PRIMARY monitor's usable area.
            if not user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
                return
            user32.ShowWindow(h, 9)  # SW_RESTORE (un-maximize so the move takes)
            user32.SetWindowPos(
                h, 0, rect.left, rect.top,
                rect.right - rect.left, rect.bottom - rect.top,
                0x0004 | 0x0040,  # SWP_NOZORDER | SWP_SHOWWINDOW
            )
            user32.ShowWindow(h, 3)  # SW_MAXIMIZE (on the primary now)
        except Exception as exc:
            self._logger.exception("move_window_primary_failed", exc)

    def _ensure_uia(self):
        if self._uia is None:
            try:
                from .uia_controller import UiaController
                self._uia = UiaController(self._logger)
            except Exception as exc:
                self._logger.exception("uia_controller_init_failed", exc)
                self._uia = None
        return self._uia

    def _ensure_voice_processor(self):
        if getattr(self, "_voice_proc", None) is None:
            try:
                from ..voice.command_processor import VoiceCommandProcessor
                self._voice_proc = VoiceCommandProcessor()
            except Exception as exc:
                self._logger.exception("voice_processor_init_failed", exc)
                self._voice_proc = None
        return self._voice_proc

    def _t_run_quick_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run a simple, KNOWN Touchless command through the deterministic
        command processor — Spotify/media control ('play X on spotify', 'next
        song', 'pause'), and other built-in voice intents. Use this inside a
        multi-step request when you need a media/app action the processor
        already handles, instead of trying to script the app's UI."""
        cmd = str(args.get("command", "")).strip()
        if not cmd:
            return _result(status="error", error="empty command", code="invalid_arguments")
        proc = self._ensure_voice_processor()
        if proc is None:
            return _result(status="error", error="command processor unavailable", code="no_processor")
        try:
            result = proc.execute(cmd)
        except Exception as exc:
            self._logger.exception("quick_command_failed", exc, command=cmd)
            return _result(status="error", error=str(exc), code="exception", command=cmd)
        success = bool(getattr(result, "success", False))
        return _result(
            status="ok" if success else "error",
            command=cmd,
            message=(getattr(result, "info_text", "") or "").strip(),
        )

    def _t_read_screen(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read on-screen TEXT accurately and cheaply: the active window's
        accessibility names (UIA) + local OCR of the screen. Returns text (no
        image), so the model can read/summarize emails, docs, chats, etc.
        without vision tokens or any mail-reading API.

        With scroll_passes > 0 it scrolls the active window down and
        accumulates unique lines across passes — so a long inbox / document
        below the fold is fully captured, not just the visible top."""
        import time as _time
        from ..debug.foreground_window import get_foreground_window_info
        info = get_foreground_window_info()
        title = (info.title if info else "") or ""
        proc = (info.process_name if info else "") or ""
        scroll_passes = max(0, min(10, int(args.get("scroll_passes") or 0)))

        # Clickable elements WITH absolute-pixel centers — so the model can
        # click directly via click_screen(coordinate_space='screen', x, y)
        # instead of OCR-guessing or re-screenshotting (that retry loop is what
        # was blowing the rate limit).
        elements: list = []
        elem_seen: set = set()

        def _add_elem(text_val: str, x, y, source: str):
            t = (text_val or "").strip()
            if not t or x is None or y is None:
                return
            key = (t.lower(), int(x) // 8, int(y) // 8)
            if key in elem_seen:
                return
            elem_seen.add(key)
            elements.append({"text": t[:60], "x": int(x), "y": int(y), "source": source})

        uia = self._ensure_uia()
        if uia is not None:
            try:
                res = uia.list_elements(limit=80)
                if isinstance(res, dict) and res.get("status") == "ok":
                    for e in (res.get("elements") or []):
                        _add_elem(e.get("name"), e.get("cx"), e.get("cy"), "uia")
            except Exception as exc:
                self._logger.exception("read_screen_uia_failed", exc)

        ocr = self._ensure_ocr()
        lines: list = []
        seen: set = set()

        def _grab() -> int:
            if ocr is None:
                return 0
            try:
                res = ocr.read_text_boxes()
            except Exception as exc:
                self._logger.exception("read_screen_ocr_failed", exc)
                return 0
            if not isinstance(res, dict) or res.get("status") != "ok":
                return 0
            added = 0
            for it in (res.get("items") or []):
                s = str(it.get("text") or "").strip()
                if s and s not in seen:
                    seen.add(s)
                    lines.append(s)
                    _add_elem(s, it.get("x"), it.get("y"), "ocr")
                    added += 1
            return added

        _grab()
        for _ in range(scroll_passes):
            if not self._scroll_active_window_down():
                break
            _time.sleep(0.45)  # let the list repaint before OCR
            if _grab() == 0:
                break  # nothing new -> reached the bottom
        text = "\n".join(lines)
        if not text and not elements:
            return _result(status="error",
                           error="couldn't read screen text (no OCR/UIA result)",
                           active_window=title, process=proc)
        cap = 16000 if scroll_passes else 6000
        return _result(status="ok", active_window=title, process=proc,
                       scrolled=scroll_passes,
                       clickable_elements=elements[:120],
                       text=text[:cap])

    def _t_click_type(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Click → type → (optional Enter) in ONE call, so the model fills a
        field and sends in a single turn instead of three (faster, far fewer
        round trips / rate-limit pauses)."""
        import time as _t
        x = args.get("x")
        y = args.get("y")
        if x is None or y is None:
            return _result(status="error", error="x and y are required", code="invalid_arguments")
        space = str(args.get("coordinate_space", "screen") or "screen")
        clicked = self._t_click_screen({"x": x, "y": y, "coordinate_space": space})
        if clicked.get("status") != "ok":
            return clicked
        _t.sleep(0.2)  # let focus land in the field
        text = str(args.get("text") or "")
        if text:
            typed = self._t_type_text({"text": text})
            if typed.get("status") != "ok":
                return typed
        submitted = False
        if args.get("submit"):
            _t.sleep(0.15)
            try:
                u = ctypes.windll.user32
                VK_RETURN, KEYUP = 0x0D, 0x0002
                u.keybd_event(VK_RETURN, 0, 0, 0)
                u.keybd_event(VK_RETURN, 0, KEYUP, 0)
                submitted = True
            except Exception as exc:
                self._logger.exception("click_type_enter_failed", exc)
        return _result(status="ok", clicked=[x, y], typed=bool(text), submitted=submitted)

    def _scroll_active_window_down(self) -> bool:
        """Mouse-wheel-scroll down over the active window's left-center (the
        list/content area). Returns False on failure."""
        try:
            import ctypes
            from ctypes import wintypes
            u = ctypes.windll.user32
            hwnd = u.GetForegroundWindow()
            if not hwnd:
                return False
            rect = wintypes.RECT()
            if not u.GetWindowRect(hwnd, ctypes.byref(rect)):
                return False
            cx = rect.left + int((rect.right - rect.left) * 0.30)
            cy = rect.top + int((rect.bottom - rect.top) * 0.50)
            u.SetCursorPos(int(cx), int(cy))
            MOUSEEVENTF_WHEEL = 0x0800
            u.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, ctypes.c_int(-480), 0)  # down
            return True
        except Exception as exc:
            self._logger.exception("read_screen_scroll_failed", exc)
            return False

    def _ensure_ocr(self):
        if self._ocr is None:
            try:
                from .screen_ocr import ScreenOcr
                self._ocr = ScreenOcr(self._logger)
            except Exception as exc:
                self._logger.exception("screen_ocr_init_failed", exc)
                self._ocr = None
        return self._ocr

    def _t_click_text_on_screen(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find on-screen TEXT via OCR and click the exact pixel center of its
        box — for buttons UIA can't see (Chromium/webview UIs like Valorant's
        PLAY, Claude Code's Yes). More precise than click_screen, which makes
        the model estimate a fraction on a downscaled image."""
        ocr = self._ensure_ocr()
        if ocr is None:
            return _result(status="error", error="OCR unavailable", code="no_ocr")
        text = str(args.get("text", "")).strip()
        if not text:
            return _result(status="error", error="empty text", code="invalid_arguments")
        occurrence = int(args.get("occurrence", 1) or 1)
        timeout = float(args.get("timeout_sec", 0) or 0)
        clicks = 2 if bool(args.get("double", False)) else int(args.get("clicks", 1) or 1)
        retries = int(args.get("retries", 0) or 0)
        return ocr.click_text(text, occurrence=occurrence, timeout_sec=timeout, clicks=clicks, retries=retries)

    def _t_drag(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Click-and-drag from one screen point to another. Coords are
        NORMALIZED 0-1 fractions of the full screen (like click_screen)."""
        ocr = self._ensure_ocr()
        if ocr is None:
            return _result(status="error", error="vision unavailable", code="no_ocr")
        try:
            fx = float(args.get("from_x")); fy = float(args.get("from_y"))
            tx = float(args.get("to_x")); ty = float(args.get("to_y"))
        except (TypeError, ValueError):
            return _result(status="error", error="from_x/from_y/to_x/to_y required (0-1)", code="invalid_arguments")
        vl, vt, vw, vh = ocr._virtual_rect()
        clamp = lambda v: max(0.0, min(1.0, v))
        ax1 = int(vl + clamp(fx) * vw); ay1 = int(vt + clamp(fy) * vh)
        ax2 = int(vl + clamp(tx) * vw); ay2 = int(vt + clamp(ty) * vh)
        return ocr.drag(ax1, ay1, ax2, ay2)

    def _t_draw_path(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Draw a CONNECTED stroke through a list of points — one continuous pen
        stroke. Use for SHAPES: a square is 5 points (4 corners + back to the
        first), a triangle is 4, a line is 2. Points are NORMALIZED 0-1 [x,y]
        fractions of the full screen (like click_screen/drag). Much more
        reliable than issuing many separate drags."""
        ocr = self._ensure_ocr()
        if ocr is None:
            return _result(status="error", error="vision unavailable", code="no_ocr")
        raw = args.get("points")
        if not isinstance(raw, list) or len(raw) < 2:
            return _result(status="error",
                           error="points must be a list of at least 2 [x,y] pairs (0-1)",
                           code="invalid_arguments")
        vl, vt, vw, vh = ocr._virtual_rect()
        clamp = lambda v: max(0.0, min(1.0, float(v)))
        pts = []
        try:
            for p in raw:
                px, py = p[0], p[1]
                pts.append((int(vl + clamp(px) * vw), int(vt + clamp(py) * vh)))
        except (TypeError, IndexError, ValueError):
            return _result(status="error", error="each point must be [x,y] with 0-1 values",
                           code="invalid_arguments")
        return ocr.draw_path(pts)

    def _t_draw_shape(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Draw a primitive SHAPE inside a bounding box — the tool computes the
        stroke, so the model never has to find the app's shape tools or compute
        circle points (which is what was burning vision/tokens). shape: square,
        rectangle, circle, oval/ellipse, triangle, line, diamond. Box corners
        are NORMALIZED 0-1: x1,y1 (top-left) and x2,y2 (bottom-right). Drawn as
        one continuous pen stroke."""
        ocr = self._ensure_ocr()
        if ocr is None:
            return _result(status="error", error="vision unavailable", code="no_ocr")
        shape = str(args.get("shape", "")).strip().lower()
        try:
            x1 = float(args.get("x1")); y1 = float(args.get("y1"))
            x2 = float(args.get("x2")); y2 = float(args.get("y2"))
        except (TypeError, ValueError):
            return _result(status="error", error="x1,y1,x2,y2 required (0-1)", code="invalid_arguments")
        # Normalize so x1<x2, y1<y2.
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        # Work in PIXELS, not normalized fractions: the virtual desktop is wide,
        # so equal normalized deltas are UNEQUAL pixels (that's why a "square"
        # came out a wide rectangle and a "circle" a flat oval). Map the box to
        # pixels first, then geometry is in true pixel space.
        vl, vt, vw, vh = ocr._virtual_rect()
        clampn = lambda v: max(0.0, min(1.0, float(v)))
        px1 = vl + clampn(x1) * vw; py1 = vt + clampn(y1) * vh
        px2 = vl + clampn(x2) * vw; py2 = vt + clampn(y2) * vh
        # Equal-sided shapes (square, circle): force a SQUARE pixel box centered
        # on the box centre, using the smaller side so it fits. rectangle/oval/
        # ellipse keep the box as given.
        if shape in ("square", "circle", "round"):
            ccx = (px1 + px2) / 2.0; ccy = (py1 + py2) / 2.0
            side = min(abs(px2 - px1), abs(py2 - py1))
            if side < 40:  # degenerate box → use the larger dim
                side = max(abs(px2 - px1), abs(py2 - py1), 120)
            h = side / 2.0
            px1, px2 = ccx - h, ccx + h
            py1, py2 = ccy - h, ccy + h
        cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        rx, ry = (px2 - px1) / 2.0, (py2 - py1) / 2.0
        pix_pts: list = []
        if shape in ("square", "rectangle", "rect", "box"):
            pix_pts = [(px1, py1), (px2, py1), (px2, py2), (px1, py2), (px1, py1)]
        elif shape in ("triangle",):
            pix_pts = [(cx, py1), (px2, py2), (px1, py2), (cx, py1)]
        elif shape in ("diamond", "rhombus"):
            pix_pts = [(cx, py1), (px2, cy), (cx, py2), (px1, cy), (cx, py1)]
        elif shape in ("line",):
            pix_pts = [(px1, py1), (px2, py2)]
        elif shape in ("circle", "oval", "ellipse", "round"):
            seg = 40
            for i in range(seg + 1):  # +1 closes the loop
                t = 2.0 * math.pi * i / seg
                pix_pts.append((cx + rx * math.cos(t), cy + ry * math.sin(t)))
        else:
            return _result(status="error",
                           error=f"unknown shape {shape!r}; use square/rectangle/circle/oval/triangle/diamond/line",
                           code="invalid_arguments")
        pts = [(int(px), int(py)) for px, py in pix_pts]
        result = ocr.draw_path(pts)
        if isinstance(result, dict):
            result["shape"] = shape
        return result

    def _t_wait_for_screen_text(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Readiness check: poll the screen until `text` is visible (OCR), up
        to timeout. Use to confirm a step finished — an app's page loaded, a
        button rendered — BEFORE the next step, so you don't act too early."""
        ocr = self._ensure_ocr()
        if ocr is None:
            return _result(status="error", error="vision unavailable", code="no_ocr")
        text = str(args.get("text", "")).strip()
        if not text:
            return _result(status="error", error="empty text", code="invalid_arguments")
        timeout = float(args.get("timeout_sec", 20) or 20)
        return ocr.wait_for_text(text, timeout_sec=timeout)

    def _t_zoom_screen(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Capture a high-res crop around a rough point so the model can SEE a
        small target (icon, unlabeled button, region) zoomed in, then click it
        precisely with click_zoom. The crop image is injected by LiveApiManager
        (it pops _zoom_image_b64 from this result and sends it as an image)."""
        ocr = self._ensure_ocr()
        if ocr is None:
            return _result(status="error", error="vision unavailable", code="no_ocr")
        cx = float(args.get("x", 0.5) or 0.5)
        cy = float(args.get("y", 0.5) or 0.5)
        size = float(args.get("size", 0.25) or 0.25)
        try:
            b64, bounds = ocr.capture_crop(cx, cy, size)
        except Exception as exc:
            self._logger.exception("zoom_screen_failed", exc)
            return _result(status="error", error=f"zoom failed: {exc}", code="zoom_failed")
        self._zoom_bounds = bounds
        return _result(
            status="ok",
            _zoom_image_b64=b64,
            note=("Zoomed crop sent — look at it, then call click_zoom(x, y) "
                  "with coords as fractions 0-1 RELATIVE TO THIS CROP (e.g. the "
                  "center is 0.5, 0.5). If the target isn't in view, call "
                  "zoom_screen again with a different x,y."),
        )

    def _t_click_zoom(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Click a point given as fractions of the LAST zoom_screen crop. Maps
        crop-relative coords back to absolute screen pixels exactly."""
        bounds = getattr(self, "_zoom_bounds", None)
        if not bounds:
            return _result(status="error", error="call zoom_screen first", code="no_zoom")
        ocr = self._ensure_ocr()
        if ocr is None:
            return _result(status="error", error="vision unavailable", code="no_ocr")
        zx = max(0.0, min(1.0, float(args.get("x", 0.5) or 0.5)))
        zy = max(0.0, min(1.0, float(args.get("y", 0.5) or 0.5)))
        left, top, cw, ch = bounds
        ax = int(left + zx * cw)
        ay = int(top + zy * ch)
        if ocr._click_xy(ax, ay):
            return _result(status="ok", clicked=True, x=ax, y=ay)
        return _result(status="error", error="click failed", code="click_failed", x=ax, y=ay)

    def _t_read_ui(self, args: Dict[str, Any]) -> Dict[str, Any]:
        uia = self._ensure_uia()
        if uia is None:
            return _result(status="error", error="UI automation unavailable", code="no_uia")
        window = str(args.get("window_title", "") or "").strip() or None
        limit = int(args.get("limit", 40) or 40)
        return uia.list_elements(window_title=window, limit=limit)

    def _t_click_ui(self, args: Dict[str, Any]) -> Dict[str, Any]:
        uia = self._ensure_uia()
        if uia is None:
            return _result(status="error", error="UI automation unavailable", code="no_uia")
        target = str(args.get("target", "")).strip()
        if not target:
            return _result(status="error", error="empty target", code="invalid_arguments")
        window = str(args.get("window_title", "") or "").strip() or None
        return uia.click(target, window_title=window)

    def _t_set_field(self, args: Dict[str, Any]) -> Dict[str, Any]:
        uia = self._ensure_uia()
        if uia is None:
            return _result(status="error", error="UI automation unavailable", code="no_uia")
        target = str(args.get("target", "")).strip()
        if not target:
            return _result(status="error", error="empty target", code="invalid_arguments")
        text = str(args.get("text", ""))
        window = str(args.get("window_title", "") or "").strip() or None
        return uia.set_value(target, text, window_title=window)

    def _t_wait_and_click(self, args: Dict[str, Any]) -> Dict[str, Any]:
        uia = self._ensure_uia()
        if uia is None:
            return _result(status="error", error="UI automation unavailable", code="no_uia")
        target = str(args.get("target", "")).strip()
        if not target:
            return _result(status="error", error="empty target", code="invalid_arguments")
        timeout = float(args.get("timeout_sec", 30) or 30)
        window = str(args.get("window_title", "") or "").strip() or None
        return uia.wait_and_click(target, timeout_sec=timeout, window_title=window)

    def _t_wait_and_press(self, args: Dict[str, Any]) -> Dict[str, Any]:
        uia = self._ensure_uia()
        if uia is None:
            return _result(status="error", error="UI automation unavailable", code="no_uia")
        text = str(args.get("text", "")).strip()
        keys = args.get("keys")
        if not text or not keys:
            return _result(status="error", error="text and keys required", code="invalid_arguments")
        timeout = float(args.get("timeout_sec", 30) or 30)
        window = str(args.get("window_title", "") or "").strip() or None
        return uia.wait_and_press(text, keys, timeout_sec=timeout, window_title=window)

    # Words that appear ONLY on real approval prompts. Deliberately excludes
    # "Run"/"Apply"/"Continue" — those also name VS Code's titlebar Run button,
    # editor actions, and IntelliSense items, so the watcher used to click them
    # every cycle even with no Claude prompt present (jumping the cursor to the
    # titlebar). Claude/Codex approval options all start with one of these.
    _DEFAULT_APPROVE_TARGETS = [
        "Yes", "Allow", "Allow always", "Keep", "Accept", "Approve",
    ]

    def _begin_auto_approve(
        self,
        *,
        targets=None,
        duration_sec: float = 600.0,
        idle_sec: float = 20.0,
        check_interval_sec: float = 5.0,
        window_title=None,
    ) -> Dict[str, Any]:
        """Shared entry point for starting the background approve-watcher. Used
        directly by the auto_approve tool AND folded into send_to_coding_agent
        so a coding task auto-starts approving without a second tool call."""
        uia = self._ensure_uia()
        if uia is None:
            return _result(status="error", error="UI automation unavailable", code="no_uia")
        # Pin the watcher to VS Code by default. Coding agents (Claude/Codex)
        # run inside VS Code, so this is the window whose Yes/Allow prompts we
        # approve. Without a title the watcher would re-resolve to whatever's
        # foreground each poll — and when the user is talking to the Iris chat,
        # that's NOT their VS Code (often on another monitor), so it watched the
        # wrong window. The send_to_coding_agent path already defaults this way;
        # standalone auto_approve must match so "watch my vscode and approve"
        # works exactly like an Iris-initiated coding task.
        if not (window_title and str(window_title).strip()):
            window_title = "Visual Studio Code"
        # Pre-warm OCR on THIS thread so the watcher's background thread doesn't
        # race on first-time engine init when it falls back to OCR.
        try:
            self._ensure_ocr()
        except Exception:
            pass
        return uia.start_auto_approve(
            list(targets) if targets else list(self._DEFAULT_APPROVE_TARGETS),
            duration_sec=float(duration_sec),
            idle_sec=float(idle_sec),
            check_interval_sec=float(check_interval_sec),
            window_title=window_title,
            on_conclude=self._auto_conclude,
            ocr_fallback=self._ocr_approve_once,
        )

    def _ocr_approve_once(self) -> bool:
        """OCR-based fallback for the auto-approve watcher: when UIA can't SEE
        the prompt (Claude Code renders Yes/No in a webview/terminal canvas that
        UIA can't read), OCR the screen, and if a Yes/No prompt is present,
        click its region to FOCUS that panel, then press the Yes key (1/y/
        Enter). This is what makes the Claude Code extension's prompt
        answerable. Returns True only if it actually pressed a key. Best-effort
        and non-blocking-safe (runs on the watcher thread)."""
        ocr = self._ensure_ocr()
        if ocr is None:
            return False
        try:
            from .uia_controller import _yes_keystroke, _CLAUDE_PROMPT_MARKERS
            text = ocr.read_all_text()
            if not text:
                return False
            key = _yes_keystroke(text)
            if not key:
                return False
            low = text.lower()
            # Click a NEUTRAL part of the prompt (the question / footer) to give
            # the panel keyboard focus — never the "No" option. Clicking the
            # question text or "Esc to cancel" footer focuses the terminal/webview
            # without selecting an answer; the keystroke does the answering.
            focused = False
            for marker in ("esc to cancel", "do you want to proceed",
                           "allow this", "do you want to", "bash command"):
                if marker in low:
                    r = ocr.find_text(marker)
                    if r.get("status") == "ok" and r.get("matches"):
                        m = r["matches"][0]
                        ocr._click_xy(int(m["x"]), int(m["y"]))
                        focused = True
                        time.sleep(0.14)
                        break
            pressed = ocr.press_key(key)
            self._logger.event("ocr_auto_approve", key=key, focused=focused, pressed=pressed)
            return bool(pressed)
        except Exception as exc:
            self._logger.exception("ocr_approve_once_failed", exc)
            return False

    def _t_auto_approve(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self._begin_auto_approve(
            targets=args.get("targets"),
            duration_sec=float(args.get("duration_sec", 600) or 600),
            idle_sec=float(args.get("idle_sec", 20) or 20),
            check_interval_sec=float(args.get("check_interval_sec", 5) or 5),
            window_title=str(args.get("window_title", "") or "").strip() or None,
        )

    def _auto_conclude(self, reason: str, clicks: int, text_tail: str) -> None:
        """Background-thread callback when auto-approve pauses/ends. Pushes a
        proactive note so Iris tells the user whether the app finished or is
        showing a prompt that needs them."""
        if self._notify is None:
            return
        if reason == "idle":
            msg = (
                f"[auto-approve update] The app paused after I approved {clicks} prompt(s). "
                f"It has either finished, or it's showing something that isn't a simple "
                f"approve and needs the user. Read the on-screen text below and tell the "
                f"user concisely: (1) if it ran the program, what the OUTPUT or result was "
                f"(quote the actual printed output / error / traceback you can see); "
                f"(2) otherwise, whether it looks DONE, or like a question/choice that "
                f"needs them and what it's asking:\n\n{text_tail}"
            )
        elif reason == "stopped":
            msg = f"[auto-approve update] Stopped at the user's request after approving {clicks} prompt(s)."
        elif reason == "no_uia":
            msg = "[auto-approve update] Couldn't start the watcher (accessibility unavailable)."
        else:  # duration / other
            msg = (
                f"[auto-approve update] The watch window ended after approving {clicks} prompt(s); "
                f"the app may still be working. Current on-screen text:\n\n{text_tail}"
            )
        try:
            self._notify(msg)
        except Exception as exc:
            self._logger.exception("auto_notify_failed", exc)

    def _t_stop_auto_approve(self, args: Dict[str, Any]) -> Dict[str, Any]:
        uia = self._ensure_uia()
        if uia is None:
            return _result(status="ok", stopped=True)
        return uia.stop_auto_approve()

    def _t_iris_add_project(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Bulletproof entrypoint for iris_add_project.

        Wraps the real implementation in a catch-all so ANY unhandled
        exception (subprocess hang/abort, pathological Unicode path,
        Windows UNC resolve failure, world_state import bomb, JSON
        serialization of a Path object, etc.) becomes a normal error
        response instead of propagating up the realtime/planner stack
        and killing the app. Catches BaseException so SystemExit /
        KeyboardInterrupt / fatal Qt callbacks can't take down the
        process either.
        """
        # Entry log — even if the handler crashes before _logger.event
        # can write, this stderr line proves we got called. Helps
        # distinguish "handler ran and threw" from "handler never
        # called because crash was upstream".
        try:
            print(
                "iris_add_project: entry args="
                + repr({k: (str(v)[:80] if isinstance(v, str) else v)
                        for k, v in (args or {}).items()})[:200],
                file=sys.stderr, flush=True,
            )
        except Exception:
            pass
        try:
            return self._iris_add_project_impl(args)
        except BaseException as exc:
            try:
                self._logger.exception("iris_add_project_handler_crashed", exc)
            except Exception:
                pass
            try:
                print(
                    f"iris_add_project: caught {type(exc).__name__}: {exc}",
                    file=sys.stderr, flush=True,
                )
            except Exception:
                pass
            return _result(
                status="error",
                error=f"iris_add_project crashed: {type(exc).__name__}: {exc}",
                code="handler_exception",
            )

    def _iris_add_project_impl(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Add a project to Iris's cortex / world model.

        Mirrors ``_t_iris_remove_project`` in spirit — same exclusion-
        list / cache / cortex-viz surface — but inverts every step:

          1. Resolve the folder: prefer ``project_path`` (absolute or
             folder name); fall back to ``project_label`` via the
             desktop file-search resolver.
          2. Validate: path must exist on disk, be a directory, and
             not sit under a protected system root.
          3. Derive the canonical project_id slug + label from the
             folder name (same regex world_state uses, so the id we
             register here is identical to what auto-discovery would
             have produced).
          4. Register in the cortex world via WorldState.add_project()
             (idempotent — no-op if the slug already exists). This
             fires the world-state observer chain which auto-emits
             ``project_added`` to the cortex bridge, so any live
             iris_simulator window splats the new node into its
             scene without us touching JS.
          5. Remove the project from project_exclusions.json (case-
             insensitive bi-directional substring match against both
             the slug and the label, mirroring the matching style
             iris_remove_project uses) so a previously-excluded
             project actually shows.
          6. Append the project to auto_discovered_projects.json so
             it persists across launches without waiting for the
             next filesystem sweep.
          7. Fire a defensive cortex_emit.project_added() — redundant
             with world_state's observer chain when the bridge is
             active, but harmless and gives older bridges a second
             path.
          8. Invalidate the iris_query_node cache so the next query
             reflects the post-add world.

        Idempotent — re-adding an existing project succeeds without
        error.
        """
        raw_path = str(args.get("project_path", "") or "").strip()
        raw_label = str(args.get("project_label", "") or "").strip()
        if not raw_path and not raw_label:
            return _result(
                status="error",
                error="project_path or project_label is required",
                code="invalid_arguments",
            )

        # ---- step 1: resolve the folder ----
        # If project_path looks like an absolute or relative path,
        # resolve it directly; otherwise treat it as a folder name and
        # search common roots. project_label is used as a fallback name.
        folder_path: Optional[Path] = None
        candidates_to_try: list = []
        if raw_path:
            candidates_to_try.append(raw_path)
        if raw_label and raw_label != raw_path:
            candidates_to_try.append(raw_label)

        for candidate in candidates_to_try:
            try:
                p = Path(candidate).expanduser().resolve()
            except Exception:
                p = None
            if p is not None and p.exists() and p.is_dir():
                folder_path = p
                break

        # Fall back to desktop file search by name if direct resolution
        # didn't land on an existing directory. We require every word in
        # the requested name to appear in the resolved path so 'HGR App'
        # doesn't silently match 'src\\hgr'.
        if folder_path is None:
            desktop = self._ensure_desktop()
            for candidate in candidates_to_try:
                name = Path(candidate).name or candidate
                req = [
                    w for w in name.lower().replace("-", " ").replace("_", " ").split()
                    if len(w) >= 2
                ]
                if desktop is None or not req:
                    continue
                try:
                    r, _amb = desktop.resolve_named_folder(name)
                except Exception:
                    r = None
                if r is not None:
                    rp = Path(r)
                    if rp.exists() and rp.is_dir() and all(w in str(rp).lower() for w in req):
                        folder_path = rp
                        break

        if folder_path is None:
            return _result(
                status="error",
                error=(
                    f"folder not found: {raw_path or raw_label}. "
                    "Provide the EXACT folder name or an absolute path."
                ),
                code="folder_not_found",
            )

        # ---- step 2: validate ----
        if not folder_path.exists() or not folder_path.is_dir():
            return _result(
                status="error",
                error=f"path is not a directory: {folder_path}",
                code="not_a_directory",
            )
        try:
            if _is_protected_path(folder_path):
                return _result(
                    status="error",
                    error=(
                        f"'{folder_path}' sits under a protected system "
                        "directory and cannot be added as a project."
                    ),
                    code="protected_path",
                )
        except Exception as exc:
            self._logger.exception("iris_add_project_protected_check_failed", exc)
            # If the protection check itself crashed, refuse rather
            # than silently accept — safer default.
            return _result(
                status="error",
                error="failed to validate folder safety",
                code="validation_failed",
            )

        # ---- step 3: derive canonical id + label ----
        try:
            from .cortex.world_state import (
                _DEFAULT_COLOR,
                derive_project_id,
                derive_project_label,
                get_world,
            )
        except Exception as exc:
            self._logger.exception("iris_add_project_world_import_failed", exc)
            return _result(
                status="error",
                error=f"world_state unavailable: {exc}",
                code="world_unavailable",
            )

        try:
            project_id = derive_project_id(folder_path)
            label = derive_project_label(folder_path)
        except Exception as exc:
            self._logger.exception("iris_add_project_derive_failed", exc)
            return _result(
                status="error",
                error=f"failed to derive project id/label: {exc}",
                code="derive_failed",
            )

        # ---- step 4: register in cortex world ----
        world_added = False
        try:
            world = get_world()
            # add_project is idempotent and fires the project_added
            # observer chain (which auto-routes to the cortex bridge
            # and from there to liveAddProject() in iris_simulator.html).
            world.add_project(
                project_id=project_id,
                label=label,
                root_path=str(folder_path),
                color=_DEFAULT_COLOR,
            )
            world_added = True
        except Exception as exc:
            self._logger.exception("iris_add_project_world_add_failed", exc)
            return _result(
                status="error",
                error=f"failed to add project to world: {exc}",
                code="world_add_failed",
            )

        # ---- step 5: un-exclude from project_exclusions.json ----
        # Mirror iris_remove_project's matching style: case-insensitive
        # bi-directional substring against both slug and label so a
        # previously-excluded project like 'HGR App v1.0.0 - jarvis test'
        # (whose exclusion entry might be the label, the slug, or even
        # the absolute path) is actually un-hidden.
        unexcluded: list = []
        try:
            from .project_autodetect import (
                load_project_exclusions,
                save_project_exclusions,
            )
            try:
                exclusions = load_project_exclusions()
            except Exception:
                exclusions = []
            pid_l = project_id.lower()
            lab_l = label.lower()
            path_l = str(folder_path).lower()
            kept: list = []
            for entry in exclusions:
                e_l = str(entry).strip().lower()
                if not e_l:
                    continue
                # Drop on any case-insensitive substring overlap.
                drop = False
                for needle in (pid_l, lab_l, path_l):
                    if not needle:
                        continue
                    if e_l == needle or needle in e_l or e_l in needle:
                        drop = True
                        break
                if drop:
                    unexcluded.append(entry)
                else:
                    kept.append(entry)
            if unexcluded:
                try:
                    save_project_exclusions(kept)
                except Exception as exc:
                    # Non-fatal: project still added to world, just
                    # may be re-hidden by exclusion on next launch.
                    self._logger.exception(
                        "iris_add_project_exclusion_save_failed", exc)
        except Exception as exc:
            self._logger.exception("iris_add_project_exclusion_failed", exc)

        # ---- step 6: append to auto-discovered cache ----
        cache_persisted = False
        try:
            from .project_autodetect import (
                load_cached_auto_discovered,
                save_cached_auto_discovered,
            )
            try:
                cached = load_cached_auto_discovered()
            except Exception:
                cached = []
            already_present = any(
                str(c.get("id", "")).strip().lower() == project_id.lower()
                for c in cached
            )
            if not already_present:
                # IMPORTANT: ``root`` MUST be a str — save_cached_auto_discovered
                # serializes via json.dumps() and a Path object would silently
                # fail (the swallowed exception used to leave the cache stale).
                cached.append({
                    "id": project_id,
                    "label": label,
                    "root": str(folder_path),
                })
                try:
                    cache_persisted = save_cached_auto_discovered(cached)
                except Exception as exc:
                    self._logger.exception(
                        "iris_add_project_cache_save_failed", exc)
            else:
                cache_persisted = True
        except Exception as exc:
            # Cache failures are non-fatal — discover_and_cache will
            # regenerate it next launch and pick up the world entry.
            self._logger.exception("iris_add_project_cache_failed", exc)

        # ---- step 7: cortex viz signal (defensive) ----
        # world.add_project() already fired the project_added observer,
        # which auto-routes via _fire_observers -> bridge.emit_project_added.
        # We additionally call cortex_emit.project_added() so older
        # bridges (or test setups with a non-WorldState bus) still see
        # the event. The bridge path is idempotent on the JS side
        # (liveAddProject no-ops on duplicate ids).
        try:
            project_added_emit = getattr(cortex_emit, "project_added", None)
            if callable(project_added_emit):
                project_added_emit({
                    "project_id": project_id,
                    "label": label,
                    "root_path": str(folder_path),
                    "color": _DEFAULT_COLOR,
                })
        except Exception:
            # Decorative — never break the tool.
            pass

        # Belt-and-braces JS fallback. The bridge signal path above
        # occasionally misses (jsReady race, torn-down bridge, etc.);
        # the GUI-thread runner registered by the assistant window
        # invokes liveAddProject(projectId, label) directly so the
        # visual fade-in is guaranteed to fire as long as the embedded
        # cortex view is alive. The JS function is idempotent — it
        # no-ops on duplicate ids — so a double-add (bridge + fallback
        # both landing) is safe. Mirrors cortex/bridge.py's
        # emit_project_added fallback exactly.
        # NOTE: direct runJavaScript fallback temporarily disabled while
        # debugging a hard crash users reported on iris_add_project.
        # The bridge signal path (cortex_emit.project_added above) +
        # the embedded view's QWebChannel handler still drive the
        # visual fade; this fallback is purely belt-and-suspenders.
        # Re-enable once the crash is root-caused. Guarded with broad
        # BaseException + entry log so re-enabling is safe to test.
        try:
            if False:  # disabled — see comment above
                import json as _json_cortex
                pid_js = _json_cortex.dumps(project_id)
                label_js = _json_cortex.dumps(label)
                _run_cortex_js(
                    "if (typeof window.liveAddProject === 'function') { "
                    f"window.liveAddProject({pid_js}, {label_js}); "
                    "} else if (typeof liveAddProject === 'function') { "
                    f"liveAddProject({pid_js}, {label_js}); }}"
                )
        except BaseException:
            pass

        # ---- step 8: invalidate query cache ----
        self._iris_world_cache = None

        return _result(
            status="ok",
            message=(
                f"Added project '{label}' to your cortex."
                + (f" Un-excluded {len(unexcluded)} matching entries."
                   if unexcluded else "")
            ),
            project_id=project_id,
            label=label,
            root_path=str(folder_path),
            world_added=world_added,
            unexcluded=unexcluded,
            cache_persisted=cache_persisted,
        )

    def _t_iris_remove_project(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a project from Iris's cortex / world model.

        Steps (all wrapped in try/except, partial failure tolerated):
          1. Resolve project_id from args (id or label).
          2. Guard: hardcoded KNOWN_PROJECT_ROOTS entries are NEVER
             removable via this tool — return error.
          3. Append to the user exclusion list (project_exclusions.json).
          4. Remove from cortex_world.json via WorldState.remove_project().
          5. Purge from auto_discovered_projects.json cache.
          6. Fire cortex_emit.project_remove() so the live viz fades
             the project node.

        Idempotent — re-running for an already-excluded project succeeds
        with a "no-op" message rather than failing.
        """
        raw_id = str(args.get("project_id", "") or "").strip()
        raw_label = str(args.get("project_label", "") or "").strip()
        if not raw_id and not raw_label:
            return _result(
                status="error",
                error="project_id or project_label is required",
                code="invalid_arguments",
            )

        # ---- step 2: guard hardcoded projects ----
        protected_labels: set = set()
        protected_ids: set = set()
        try:
            from .known_projects import KNOWN_PROJECT_ROOTS
            for kp in KNOWN_PROJECT_ROOTS:
                protected_ids.add(str(kp["id"]).strip().lower())
                if kp.get("label"):
                    protected_labels.add(str(kp["label"]).strip().lower())
        except Exception as exc:
            self._logger.exception("iris_remove_project_known_load_failed", exc)
            # Fall back to a hardcoded safety net so a known-list import
            # failure can't accidentally unprotect a real Touchless project.
            protected_ids = {
                "touchless-dev", "touchless-website",
                "touchless-marketing", "touchless-tracking",
            }

        id_lower = raw_id.lower()
        label_lower = raw_label.lower()
        if (id_lower and id_lower in protected_ids) or (
            label_lower and label_lower in protected_labels
        ):
            return _result(
                status="error",
                error=(
                    f"project '{raw_id or raw_label}' is a protected "
                    "Touchless project root and cannot be removed."
                ),
                code="protected_project",
            )

        # ---- step 3: append to the exclusion list ----
        added_identifiers: list = []
        try:
            from .project_autodetect import (
                add_project_exclusion,
                purge_from_cache,
                load_cached_auto_discovered,
            )
            if raw_id:
                add_project_exclusion(raw_id)
                added_identifiers.append(raw_id)
            if raw_label and raw_label.lower() != raw_id.lower():
                add_project_exclusion(raw_label)
                added_identifiers.append(raw_label)
        except Exception as exc:
            self._logger.exception("iris_remove_project_exclude_failed", exc)
            return _result(
                status="error",
                error=f"failed to update exclusion list: {exc}",
                code="exclusion_failed",
            )

        # ---- step 4: remove from cortex_world.json ----
        # Try the exact id first; if not present, look for any
        # auto-discovered cache entry whose id/label substring-matches
        # what the user typed (handles "remove vcpkg" when the slug
        # is actually "vcpkg-master" or similar).
        removed_world_ids: list = []
        try:
            from .cortex.world_state import get_world
            world = get_world()
            candidates: list = []
            # Direct hit.
            if raw_id:
                candidates.append(raw_id)
            # Substring matches across the cache (auto-discovered only).
            try:
                cached = load_cached_auto_discovered()
            except Exception:
                cached = []
            for entry in cached:
                pid = str(entry.get("id", "")).strip()
                lab = str(entry.get("label") or pid).strip()
                pid_l = pid.lower()
                lab_l = lab.lower()
                hit = False
                if id_lower and (id_lower in pid_l or pid_l in id_lower
                                 or id_lower in lab_l or lab_l in id_lower):
                    hit = True
                if label_lower and (label_lower in pid_l or pid_l in label_lower
                                    or label_lower in lab_l or lab_l in label_lower):
                    hit = True
                if hit and pid and pid_l not in protected_ids:
                    candidates.append(pid)
            # Dedup while preserving order.
            seen: set = set()
            uniq_candidates = []
            for c in candidates:
                c_l = c.lower()
                if c_l in seen:
                    continue
                seen.add(c_l)
                uniq_candidates.append(c)
            for cand in uniq_candidates:
                try:
                    if world.remove_project(cand):
                        removed_world_ids.append(cand)
                except Exception as exc:
                    self._logger.exception(
                        "iris_remove_project_world_remove_failed", exc, target=cand)
        except Exception as exc:
            # Don't fail the entire tool if world removal blew up —
            # the exclusion already landed, which is the durable
            # contract.
            self._logger.exception("iris_remove_project_world_failed", exc)

        # ---- step 5: purge from auto-discovered cache ----
        purged_cache_ids: list = []
        try:
            for cand in (removed_world_ids or ([raw_id] if raw_id else [])):
                try:
                    purge_from_cache(cand)
                    purged_cache_ids.append(cand)
                except Exception:
                    pass
        except Exception as exc:
            # Cache failures are non-fatal — discover_and_cache will
            # regenerate it next launch with the exclusion applied.
            self._logger.exception("iris_remove_project_cache_purge_failed", exc)

        # ---- step 6: cortex viz signal ----
        # Send BOTH the raw cand AND its 'proj-<slug>' form so the JS
        # handler can resolve regardless of which id convention the
        # world store happened to use. Duplicate emits are harmless —
        # handleProjectRemoved no-ops on the second hit (node already
        # gone). Without this, a cached id stored bare ('vcpkg-master')
        # could miss when JS expects 'proj-vcpkg-master' under NODES.
        #
        # Fallback chain: prefer the world-removed ids (the canonical
        # form the world store actually knew about), else the raw_id
        # the user gave us, else derive a slug from raw_label. The
        # label fallback is critical for runtime-injected fixtures
        # (e.g. Demo Project: keyed 'proj-demo-project' in NODES but
        # NOT in the auto-discovered cache, so substring matching on
        # the cache returns nothing → without this fallback, the fade
        # signal would never fire when the user said 'remove demo
        # project').
        emit_candidates: list = list(removed_world_ids)
        if not emit_candidates and raw_id:
            emit_candidates.append(raw_id)
        if not emit_candidates and raw_label:
            # Slugify the label the same way liveAddProject does on the
            # JS side (lower-case, non-[a-z0-9_-] → '-', collapse
            # double-dashes) so 'Demo Project' → 'demo-project' → JS
            # finds NODES['proj-demo-project'].
            import re as _re_slug
            _slug = _re_slug.sub(
                r"[^a-z0-9_-]+", "-", raw_label.lower()).strip("-")
            while "--" in _slug:
                _slug = _slug.replace("--", "-")
            if _slug:
                emit_candidates.append(_slug)
        try:
            sent: set = set()
            for cand in emit_candidates:
                if not cand:
                    continue
                cand_str = str(cand).strip()
                if not cand_str:
                    continue
                forms = [cand_str]
                if not cand_str.lower().startswith("proj-"):
                    forms.append(f"proj-{cand_str}")
                for form in forms:
                    if form in sent:
                        continue
                    sent.add(form)
                    try:
                        cortex_emit.project_remove(form, duration_ms=2000)
                    except Exception as exc:
                        # Per-form failure must not kill the loop — the
                        # other form may still land.
                        self._logger.exception(
                            "iris_remove_project_cortex_emit_failed",
                            exc, target=form,
                        )
                    # Belt-and-braces JS fallback: directly call
                    # handleProjectRemoved on the page in case the
                    # QWebChannel signal path drops the event (cold
                    # start, signal race during bridge re-registration,
                    # etc). The registered runner marshals onto the
                    # GUI thread via QTimer.singleShot(0) so calling
                    # this from the Realtime worker is safe. The JS
                    # handler is idempotent — it tracks in-flight
                    # removals in _projectRemovalInFlight and silently
                    # no-ops on duplicate emits, so a double-fire
                    # (bridge + fallback both landing) does NOT start
                    # a second tween.
                    try:
                        import json as _json_cortex
                        form_js = _json_cortex.dumps(form)
                        _run_cortex_js(
                            "if (typeof window.handleProjectRemoved === 'function') { "
                            f"window.handleProjectRemoved({form_js}); "
                            "} else if (typeof handleProjectRemoved === 'function') { "
                            f"handleProjectRemoved({form_js}); }}"
                        )
                    except BaseException:
                        pass
        except Exception as exc:
            # Decorative — never break the tool, but log so we know.
            self._logger.exception(
                "iris_remove_project_cortex_signal_failed", exc)

        # Invalidate the iris_query_node cache so the next query reflects
        # the post-removal world. Cheap (just drops the dict reference);
        # the next call to _t_iris_query_node will re-materialise it.
        self._iris_world_cache = None

        return _result(
            status="ok",
            message=(
                f"Removed project '{raw_id or raw_label}'. "
                f"Added to exclusion list ({len(added_identifiers)} entries); "
                f"removed {len(removed_world_ids)} from world; "
                f"purged {len(purged_cache_ids)} from cache."
            ),
            excluded=added_identifiers,
            world_removed=removed_world_ids,
            cache_purged=purged_cache_ids,
        )

    # ---- iris_query_node ------------------------------------------------

    def _load_iris_world_cache(self) -> Dict[str, Any]:
        """Return (and cache) a snapshot of the cortex world payload for
        iris_query_node lookups. Layered fallback so we never trip up the
        query path on a heavy filesystem scan:

          1. If ``self._iris_world_cache`` is already populated, reuse it.
          2. Otherwise try to read the live ``WorldState`` singleton —
             cheap, in-memory, and already authoritative for projects /
             touch_counts / labels. We synthesise the same shape
             ``load_world_payload`` exposes (projects / memory / tools)
             so the query handler doesn't care about the source.
          3. Merge in ``known_projects.combined_project_roots()`` —
             KNOWN_PROJECT_ROOTS (e.g. "Touchless dev") + the cached
             auto-discovered list. This is cheap (in-memory list +
             one small JSON read) and ensures projects that haven't
             been touched in cortex_world.json yet are still
             discoverable. Deduped by raw_id (case-insensitive).
          4. The expensive ``run_iris_simulator.load_world_payload`` path
             is INTENTIONALLY not called here — it triggers filesystem
             walks + auto-discovery and would make every query feel laggy.
             Callers wanting fresh disk data should refresh out-of-band.
          5. If ``_INJECTED_RICH_PAYLOAD`` is populated (the assistant
             window's background preload completed), we opportunistically
             enrich shaped_projects with the ``branches`` / nested ``leaves``
             from that rich payload — so queries like "Touchless dev"
             return real folder structure instead of an empty leaves list.

        Returns ``{}`` on total failure — the query handler treats that
        as "no matches" rather than an exception."""
        # If the injected rich payload changed since we built our cache,
        # drop the stale snapshot so the fresh branches make it through.
        rich = _INJECTED_RICH_PAYLOAD
        rich_id = id(rich) if rich is not None else None
        if rich_id != self._iris_world_cache_rich_id:
            self._iris_world_cache = None
            self._iris_world_cache_rich_id = rich_id
        if isinstance(self._iris_world_cache, dict) and self._iris_world_cache:
            return self._iris_world_cache

        # Build a lookup of rich-payload projects by their raw slug id
        # (e.g. "touchless-dev") AND by lowercased label, so we can merge
        # branches in regardless of which ID flavor our cheap source used.
        # Rich payload entries look like:
        #   {"id": "proj-touchless-dev", "label": "Touchless dev",
        #    "branches": [{"id": ..., "label": "src", "leaves": [...]}]}
        rich_by_key: Dict[str, Dict[str, Any]] = {}
        try:
            if isinstance(rich, dict):
                for rp in (rich.get("projects") or []):
                    rid = str((rp or {}).get("id") or "")
                    if rid.startswith("proj-"):
                        rid = rid[len("proj-"):]
                    rid = rid.strip().lower()
                    if rid:
                        rich_by_key[rid] = rp
                    rlabel = str((rp or {}).get("label") or "").strip().lower()
                    if rlabel and rlabel not in rich_by_key:
                        rich_by_key[rlabel] = rp
        except Exception:
            rich_by_key = {}

        def _branches_for(raw_id: str, label: str) -> list:
            """Return the rich-payload branches list (or []) for the
            given project, matched by raw_id then by label."""
            if not isinstance(rich_by_key, dict) or not rich_by_key:
                return []
            for key in (
                (raw_id or "").strip().lower(),
                (label or "").strip().lower(),
            ):
                if key and key in rich_by_key:
                    return list((rich_by_key[key] or {}).get("branches") or [])
            return []

        payload: Dict[str, Any] = {}
        try:
            from .cortex.world_state import get_world
            world = get_world()
            projects = []
            try:
                projects = list(world.list_projects() or [])
            except Exception as exc:
                self._logger.exception("iris_query_world_list_failed", exc)
                projects = []
            shaped_projects = []
            for p in projects:
                pid = str((p or {}).get("id") or "").strip()
                if not pid:
                    continue
                label = str((p or {}).get("label") or pid)
                # Leaves: recent files for this project (capped so a
                # noisy project doesn't bloat the query response).
                leaves = []
                try:
                    files = world.files_for_project(pid, limit=20) or []
                    for f in files:
                        leaves.append({
                            "id": f.get("path") or "",
                            "label": f.get("label") or "",
                            "path": f.get("path"),
                        })
                except Exception:
                    leaves = []
                shaped_projects.append({
                    "id": f"proj-{pid}" if not pid.startswith("proj-") else pid,
                    "raw_id": pid,
                    "label": label,
                    "root_path": (p or {}).get("root_path"),
                    "touch_count": int((p or {}).get("touch_count", 0)),
                    "last_touched_at": (p or {}).get("last_touched_at"),
                    "leaves": leaves,
                    "branches": _branches_for(pid, label),
                })
            payload["projects"] = shaped_projects
        except Exception as exc:
            self._logger.exception("iris_query_world_load_failed", exc)
            payload["projects"] = []

        # --- step 3: merge in KNOWN_PROJECT_ROOTS + auto-discovered ---
        # Without this, projects that exist on disk (e.g. "Touchless
        # dev") but haven't been touched in the current session are
        # invisible to iris_query_node. combined_project_roots() is
        # cheap — hardcoded list + one ~10KB JSON read — and never
        # raises (best-effort by contract). Dedup by lowercased raw_id
        # so cortex_world entries (already in shaped_projects) win.
        try:
            from .known_projects import combined_project_roots
            existing_ids = {
                str(p.get("raw_id") or "").strip().lower()
                for p in payload["projects"]
            }
            existing_ids.discard("")
            for entry in combined_project_roots() or []:
                pid = str((entry or {}).get("id") or "").strip()
                if not pid or pid.lower() in existing_ids:
                    continue
                label = str((entry or {}).get("label") or pid)
                root = (entry or {}).get("root")
                root_path_str: Optional[str] = None
                if root is not None:
                    try:
                        root_path_str = str(root)
                    except Exception:
                        root_path_str = None
                payload["projects"].append({
                    "id": f"proj-{pid}" if not pid.startswith("proj-") else pid,
                    "raw_id": pid,
                    "label": label,
                    "root_path": root_path_str,
                    "touch_count": 0,
                    "last_touched_at": None,
                    "leaves": [],
                    "branches": _branches_for(pid, label),
                })
                existing_ids.add(pid.lower())
        except Exception as exc:
            # Auto-discovery is best-effort — never let it block the
            # query path. Cortex_world projects (step 2) still work.
            self._logger.exception("iris_query_known_projects_merge_failed", exc)

        # Tools — populated from the static tool schema list so even
        # tools never invoked this session resolve. Connector tools
        # added best-effort.
        tools: list = []
        try:
            from .schemas import all_tool_schemas
            for schema in all_tool_schemas() or []:
                tid = (schema or {}).get("name") or ""
                if not tid:
                    continue
                tools.append({
                    "id": f"tool-{tid}",
                    "raw_id": tid,
                    "label": tid.replace("_", " ").title(),
                    "description": ((schema or {}).get("description") or "")[:280],
                })
        except Exception as exc:
            self._logger.exception("iris_query_tools_load_failed", exc)
        payload["tools"] = tools

        # Hardcoded capability + memory anchor nodes (mirror the four
        # cap-* / mem-* nodes seeded in iris_simulator.html). Lets the
        # model answer "what is cap-tools" without us having to import
        # the JS scaffold.
        payload["capabilities"] = [
            {"id": "cap-memory",   "label": "Memory",        "kind": "capability"},
            {"id": "cap-voice",    "label": "Voice",         "kind": "capability"},
            {"id": "cap-tools",    "label": "Tools",         "kind": "capability"},
            {"id": "cap-realtime", "label": "Realtime",      "kind": "capability"},
            {"id": "mem-facts",    "label": "Facts",         "kind": "memory"},
            {"id": "mem-patterns", "label": "Patterns",      "kind": "memory"},
            {"id": "mem-episodes", "label": "Episodes",      "kind": "memory"},
            {"id": "core",         "label": "Iris",          "kind": "core"},
        ]

        self._iris_world_cache = payload
        return payload

    def _t_iris_query_node(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Look up a node in the cached cortex world payload.

        Case-insensitive substring match against project ids, project
        labels, project slugs ("proj-<slug>"), tool ids ("tool-<name>")
        and capability/memory anchor ids. Returns the FIRST unambiguous
        hit; on multiple matches, returns the candidate list so the
        model can disambiguate.

        Never calls ``load_world_payload`` synchronously — that would
        re-walk the filesystem and auto-discovery. Reads from
        ``_load_iris_world_cache`` (built lazily, invalidated on
        iris_remove_project). Stale-after-changes is documented in the
        tool schema description.
        """
        raw_id = str(args.get("node_id", "") or "").strip()
        raw_label = str(args.get("node_label", "") or "").strip()
        if not raw_id and not raw_label:
            return _result(
                status="error",
                error="node_id or node_label is required",
                code="invalid_arguments",
            )

        needle_id = raw_id.lower()
        needle_label = raw_label.lower()

        payload = self._load_iris_world_cache()
        candidates: list = []  # list of (kind, dict)

        def _match(*haystacks: str) -> bool:
            for hay in haystacks:
                h = (hay or "").lower()
                if not h:
                    continue
                if needle_id and (needle_id in h or h in needle_id):
                    return True
                if needle_label and (needle_label in h or h in needle_label):
                    return True
            return False

        # Projects.
        for proj in payload.get("projects", []) or []:
            pid = str(proj.get("id") or "")
            raw_pid = str(proj.get("raw_id") or "")
            label = str(proj.get("label") or "")
            if _match(pid, raw_pid, label):
                candidates.append(("project", proj))

        # Tools.
        for tool in payload.get("tools", []) or []:
            tid = str(tool.get("id") or "")
            raw_tid = str(tool.get("raw_id") or "")
            label = str(tool.get("label") or "")
            if _match(tid, raw_tid, label):
                candidates.append(("tool", tool))

        # Capability + memory + core anchors.
        for cap in payload.get("capabilities", []) or []:
            cid = str(cap.get("id") or "")
            label = str(cap.get("label") or "")
            if _match(cid, label):
                candidates.append((cap.get("kind") or "capability", cap))

        if not candidates:
            # Best-effort "did you mean?" — rank all known labels/ids by
            # similarity to the needle and surface the top few so the
            # model can retry without a full re-listing. difflib is
            # stdlib + cheap on a few hundred strings.
            suggestions: list = []
            try:
                import difflib
                needle = (raw_label or raw_id).lower()
                pool: list = []
                for proj in payload.get("projects", []) or []:
                    for key in ("label", "raw_id", "id"):
                        v = str(proj.get(key) or "")
                        if v:
                            pool.append(v)
                for tool in payload.get("tools", []) or []:
                    for key in ("label", "raw_id", "id"):
                        v = str(tool.get(key) or "")
                        if v:
                            pool.append(v)
                for cap in payload.get("capabilities", []) or []:
                    for key in ("label", "id"):
                        v = str(cap.get(key) or "")
                        if v:
                            pool.append(v)
                # Dedup while preserving order; lowercased for matching.
                seen: set = set()
                uniq = []
                for v in pool:
                    k = v.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    uniq.append(v)
                suggestions = difflib.get_close_matches(needle, uniq, n=5, cutoff=0.4)
            except Exception:
                suggestions = []
            msg = (
                f"no node matched '{raw_id or raw_label}'. "
                "Try the full project label, the slug, or 'cap-tools' "
                "for the tool root."
            )
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions)}?"
            return _result(
                status="error",
                error=msg,
                code="not_found",
                suggestions=suggestions,
            )

        # If exactly one hit, return rich details; otherwise return the
        # disambiguation list so the model can ask the user to narrow.
        if len(candidates) > 1:
            return _result(
                status="ambiguous",
                error=(
                    f"'{raw_id or raw_label}' matched {len(candidates)} "
                    "nodes — please provide a more specific id or label."
                ),
                code="ambiguous_match",
                matches=[
                    {
                        "id": (item.get("id") or ""),
                        "label": (item.get("label") or ""),
                        "kind": kind,
                    }
                    for kind, item in candidates[:12]
                ],
            )

        kind, node = candidates[0]
        if kind == "project":
            leaves = node.get("leaves", []) or []
            branches_raw = node.get("branches", []) or []
            parent_id = "core"  # projects hang off the core anchor

            # When the rich payload is available, return the folder
            # hierarchy: up to 10 branches, 5 leaves per branch. This is
            # the answer to "what's inside <project>?" — way more useful
            # than a flat last-touched file list.
            branches_out: list = []
            for b in branches_raw[:10]:
                b_leaves = (b or {}).get("leaves") or []
                branches_out.append({
                    "id": (b or {}).get("id"),
                    "label": (b or {}).get("label"),
                    "leaves_count": len(b_leaves),
                    "leaves": [
                        {"label": (lf or {}).get("label"),
                         "path": (lf or {}).get("path")}
                        for lf in b_leaves[:5]
                    ],
                })

            # Sparse-data note when neither rich branches nor world_state
            # leaves were populated — tells the model "this project is
            # known but Iris hasn't indexed its contents yet" so it
            # doesn't hallucinate folder structure.
            note: Optional[str] = None
            if not branches_out and not leaves:
                note = (
                    "sparse data: project is known but its file tree "
                    "hasn't been indexed yet (open the assistant once "
                    "to trigger a background scan)"
                )

            result: Dict[str, Any] = {
                "kind": "project",
                "id": node.get("id"),
                "label": node.get("label"),
                "root_path": node.get("root_path"),
                "touch_count": node.get("touch_count", 0),
                "last_touched_at": node.get("last_touched_at"),
                "parent_id": parent_id,
                "children_count": len(branches_out) if branches_out else len(leaves),
                "branches": branches_out,
                # Preserve the flat leaves list for backward-compat with
                # callers that already use it; capped to 20 as before.
                "leaves_count": len(leaves),
                "leaves": [
                    {"id": l.get("id"), "label": l.get("label"), "path": l.get("path")}
                    for l in leaves[:20]
                ],
            }
            if note:
                result["note"] = note
            return _result(status="ok", **result)
        if kind == "tool":
            return _result(
                status="ok",
                kind="tool",
                id=node.get("id"),
                label=node.get("label"),
                description=node.get("description"),
                parent_id="cap-tools",
                children_count=0,
                leaves_count=0,
                leaves=[],
            )
        # Capability / memory / core anchor — minimal payload.
        return _result(
            status="ok",
            kind=kind,
            id=node.get("id"),
            label=node.get("label"),
            parent_id="core" if kind != "core" else None,
            children_count=0,
            leaves_count=0,
            leaves=[],
        )

    # ---------- iris_describe_project ----------

    def _resolve_project_root_for_describe(
        self,
        project_id: str,
        project_label: str,
    ) -> Optional[Path]:
        """Resolve a project id/label to an absolute folder path.

        Reuses the cached iris world payload (same source as
        ``iris_query_node``) so we don't hit disk: every project in
        the cache already carries a ``root_path`` field. Matching is
        case-insensitive substring against both the raw slug and the
        label, mirroring ``_t_iris_query_node``.
        """
        needle_id = (project_id or "").strip().lower()
        needle_label = (project_label or "").strip().lower()
        if not needle_id and not needle_label:
            return None

        try:
            payload = self._load_iris_world_cache()
        except Exception:
            payload = {}

        def _match(*haystacks: str) -> bool:
            for hay in haystacks:
                h = (hay or "").lower()
                if not h:
                    continue
                if needle_id and (needle_id in h or h in needle_id):
                    return True
                if needle_label and (needle_label in h or h in needle_label):
                    return True
            return False

        for proj in payload.get("projects", []) or []:
            pid = str(proj.get("id") or "")
            raw_pid = str(proj.get("raw_id") or "")
            label = str(proj.get("label") or "")
            root = proj.get("root_path")
            if not root:
                continue
            if _match(pid, raw_pid, label):
                try:
                    p = Path(str(root)).expanduser().resolve()
                except Exception:
                    p = None
                if p is not None and p.is_dir():
                    return p

        # Fallback: treat project_id/label as a literal path or folder
        # name and try the desktop file-search resolver (same pattern as
        # iris_add_project). Helps when the user names a project not yet
        # in the cortex world.
        for raw in (project_id, project_label):
            raw = (raw or "").strip()
            if not raw:
                continue
            try:
                p = Path(raw).expanduser().resolve()
            except Exception:
                p = None
            if p is not None and p.is_dir():
                return p
        try:
            desktop = self._ensure_desktop()
        except Exception:
            desktop = None
        if desktop is not None:
            for raw in (project_id, project_label):
                raw = (raw or "").strip()
                if not raw:
                    continue
                name = Path(raw).name or raw
                req = [
                    w for w in name.lower().replace("-", " ").replace("_", " ").split()
                    if len(w) >= 2
                ]
                if not req:
                    continue
                try:
                    r, _amb = desktop.resolve_named_folder(name)
                except Exception:
                    r = None
                if r is not None:
                    rp = Path(r)
                    if rp.is_dir() and all(w in str(rp).lower() for w in req):
                        return rp.resolve()
        return None

    def _scan_project_for_description(
        self,
        root: Path,
    ) -> Dict[str, Any]:
        """Read up to ~50KB of context (metadata + sampled source files)
        from ``root``. Returns ``{label, file_count, context}``.

        Context is a single string composed of:
          * Metadata files first (CLAUDE.md, README.md, package.json
            description, pyproject.toml description), each truncated.
          * Then up to 10 sampled source files (.md, .py, .js, .ts,
            .html, .json, .yaml, .toml), prioritized by file
            extension weight and presence in src/docs/pages.

        Skips binaries (via ``_safe_read_text``) and noisy directories
        (``__pycache__``, ``node_modules``, ``.git``, ``.venv``, ``dist``,
        ``build``). Honors the 100KB-per-file cap from
        ``project_memory._safe_read_text`` and an overall 50KB context
        cap (the prompt + content is comfortably inside compose_text's
        32K input budget after metadata is added).
        """
        try:
            from .memory.project_memory import _safe_read_text
        except Exception:
            _safe_read_text = None

        TOTAL_CAP = 50 * 1024  # 50KB total context cap per spec
        used = 0
        parts: list = []

        # Metadata files — read first, truncated to ~5KB each.
        metadata_files = ["CLAUDE.md", "README.md", "package.json", "pyproject.toml"]
        for fname in metadata_files:
            if used >= TOTAL_CAP:
                break
            fpath = root / fname
            if not fpath.is_file():
                continue
            text = None
            if _safe_read_text is not None:
                try:
                    text = _safe_read_text(fpath)
                except Exception:
                    text = None
            if text is None:
                # Last-resort plain read with size guard.
                try:
                    if fpath.stat().st_size > 100 * 1024:
                        continue
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
            # For package.json / pyproject.toml, extract just the
            # description field rather than the whole manifest (saves
            # context budget for source files).
            if fname == "package.json":
                try:
                    import json as _json
                    obj = _json.loads(text)
                    desc = str(obj.get("description") or "").strip()
                    name = str(obj.get("name") or "").strip()
                    text = f"name: {name}\ndescription: {desc}" if (desc or name) else text[:1000]
                except Exception:
                    text = text[:1000]
            elif fname == "pyproject.toml":
                # Cheap line-scan; avoids the tomllib import for a one-
                # field lookup.
                lines = []
                for line in text.splitlines():
                    s = line.strip()
                    if s.startswith(("name", "description", "version")):
                        lines.append(line)
                text = "\n".join(lines) if lines else text[:1000]
            chunk = f"# {fname}\n{text}"
            remaining = TOTAL_CAP - used
            chunk = chunk[:min(5000, remaining)]
            parts.append(chunk)
            used += len(chunk)

        # Sampled source files — prioritized by extension and folder.
        ext_weight = {
            ".md": 10, ".rst": 9,
            ".py": 8, ".ts": 8, ".tsx": 8, ".jsx": 8, ".js": 7,
            ".html": 7, ".yaml": 6, ".yml": 6, ".toml": 6, ".json": 5,
        }
        skip_dirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
            "dist", "build", ".next", "target", ".cache", ".idea",
            ".vscode", "site-packages", ".pytest_cache", ".mypy_cache",
            ".tox",
        }
        priority_folders = {"src", "docs", "pages", "lib", "app"}

        # Walk shallowly first, then deeper — bounded total iterations
        # so a giant project doesn't take seconds to enumerate.
        candidates: list = []
        try:
            files_seen = 0
            total_files = 0
            for dirpath, dirnames, filenames in os.walk(str(root)):
                # Mutate dirnames in place so os.walk doesn't descend.
                dirnames[:] = [d for d in dirnames if d not in skip_dirs]
                total_files += len(filenames)
                for fn in filenames:
                    files_seen += 1
                    if files_seen > 5000:
                        break
                    fp = Path(dirpath) / fn
                    ext = fp.suffix.lower()
                    if ext not in ext_weight:
                        continue
                    # Skip metadata files we already read (avoid dups).
                    if fp.parent == root and fn in metadata_files:
                        continue
                    # Score: extension weight + folder bonus + depth penalty.
                    score = ext_weight[ext]
                    try:
                        rel_parts = fp.relative_to(root).parts
                    except Exception:
                        rel_parts = (fn,)
                    for part in rel_parts[:-1]:
                        if part.lower() in priority_folders:
                            score += 3
                            break
                    score -= max(0, len(rel_parts) - 3)
                    candidates.append((score, fp))
                if files_seen > 5000:
                    break
        except Exception:
            candidates = []
            total_files = 0

        candidates.sort(key=lambda t: t[0], reverse=True)
        samples_taken = 0
        for _, fp in candidates:
            if samples_taken >= 10 or used >= TOTAL_CAP:
                break
            text = None
            if _safe_read_text is not None:
                try:
                    text = _safe_read_text(fp)
                except Exception:
                    text = None
            if text is None:
                continue
            try:
                rel = fp.relative_to(root)
            except Exception:
                rel = fp
            remaining = TOTAL_CAP - used
            # Per-file cap: keep it small so we fit several samples.
            chunk = f"\n# {rel}\n{text}"
            chunk = chunk[:min(4000, remaining)]
            if not chunk.strip():
                continue
            parts.append(chunk)
            used += len(chunk)
            samples_taken += 1

        context = "\n".join(parts)
        return {
            "label": root.name,
            "file_count": total_files,
            "samples_count": samples_taken,
            "context": context,
            "context_bytes": used,
        }

    def _describe_project_via_llm(
        self,
        label: str,
        context: str,
    ) -> Dict[str, Any]:
        """Call the cheap LLM to synthesize a 1-2 paragraph description.

        Uses urllib directly (30-second timeout, no temperature) so we
        don't depend on the openai SDK and can keep this tool
        independent of compose.py's defaults. The 30s timeout matches
        compose.py — LLM synthesis of ~50KB of project files routinely
        takes 5-15s depending on model latency; the prior 5s ceiling
        was guaranteed to time out on every call.
        """
        import json as _json
        import urllib.error as _ue
        import urllib.request as _ur

        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            return {"status": "error", "error": "OPENAI_API_KEY not set",
                    "code": "not_configured"}

        # gpt-4o-mini is the correct current small/cheap model. The
        # prior default ('gpt-5-mini') does not exist in OpenAI's API
        # and caused every describe call to hang until the request
        # timed out. Env overrides still win if the user has a model
        # they prefer.
        model = (os.environ.get("TOUCHLESS_COMPOSE_MODEL")
                 or os.environ.get("TOUCHLESS_PLANNER_MODEL")
                 or "gpt-4o-mini")

        system = (
            "You're describing a project. Output 1-2 paragraphs "
            "covering purpose, audience, tech stack (if code), and "
            "current focus. Be concrete and practical. Works for any "
            "project type — code, school, writing, research. "
            "Plain text only — no markdown headers, no bullet lists, "
            "no preamble like 'This project is...'. Just the "
            "description prose."
        )
        user = f"Project: {label}\n\nFiles:\n{context}"

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": 800,
        }
        req = _ur.Request(
            "https://api.openai.com/v1/chat/completions",
            data=_json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _ur.urlopen(req, timeout=30) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
        except _ue.HTTPError as exc:
            # Surface the HTTP body when present — invalid model names
            # (e.g. a future env override pointing at a model OpenAI
            # doesn't serve) return 404 with a JSON body that explains
            # exactly what's wrong, which is far more useful than a
            # bare 'HTTP 404'.
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            return {"status": "error",
                    "error": f"HTTP {exc.code}{(' — ' + detail) if detail else ''}",
                    "code": "http_error"}
        except Exception as exc:
            # Includes socket.timeout / URLError — friendly error so the
            # caller can show a sensible bubble instead of crashing.
            return {"status": "error",
                    "error": f"describe failed: {type(exc).__name__}: {exc}",
                    "code": "describe_failed"}

        text = ((payload.get("choices") or [{}])[0]
                .get("message", {}).get("content") or "").strip()
        if not text:
            return {"status": "error",
                    "error": "empty LLM response",
                    "code": "empty_response"}
        return {"status": "ok", "text": text, "model": model}

    def _get_project_description_cache(self, key: str) -> Optional[Dict[str, Any]]:
        entry = _PROJECT_DESCRIPTION_CACHE.get(key)
        if not entry:
            return None
        ts = float(entry.get("ts") or 0.0)
        if (time.time() - ts) > _PROJECT_DESCRIPTION_TTL_SEC:
            try:
                del _PROJECT_DESCRIPTION_CACHE[key]
            except KeyError:
                pass
            return None
        return entry

    def _set_project_description_cache(
        self,
        key: str,
        summary: str,
        label: str,
        file_count: int,
    ) -> None:
        _PROJECT_DESCRIPTION_CACHE[key] = {
            "summary": summary,
            "label": label,
            "file_count": int(file_count),
            "ts": time.time(),
        }

    def _t_iris_describe_project(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Scan a project folder and produce a 1-2 paragraph description.

        Pipeline:
          1. Resolve project root from project_id / project_label
             (cached iris world payload first, then path fallback,
             then desktop file-search).
          2. Check the 24h in-memory cache (keyed by resolved absolute
             path). Skip on cache hit unless write_md=True (which
             forces re-scan so the freshly-written CLAUDE.md reflects
             current state).
          3. Scan: read metadata files + sample 5-10 source files,
             capped to 50KB total context.
          4. Call gpt-5-mini (5-second timeout, max_completion_tokens
             only — newer GPT-5 models reject temperature/max_tokens).
          5. Cache the result.
          6. If write_md=True AND no CLAUDE.md exists in the project
             root AND the folder is user-writable, write CLAUDE.md
             with the summary.

        All steps after argument validation are wrapped in try/except
        so a bug in any phase never blocks tool dispatch.
        """
        try:
            project_id = str(args.get("project_id", "") or "").strip()
            project_label = str(args.get("project_label", "") or "").strip()
            write_md = bool(args.get("write_md", False))

            if not project_id and not project_label:
                return _result(
                    status="error",
                    error="project_id or project_label is required",
                    code="invalid_arguments",
                )

            # ---- step 1: resolve project root ----
            root = self._resolve_project_root_for_describe(project_id, project_label)
            if root is None:
                return _result(
                    status="error",
                    error=(
                        f"project not found: {project_id or project_label}. "
                        "Try iris_query_node first to see available "
                        "projects, or pass an absolute path as project_id."
                    ),
                    code="not_found",
                )

            try:
                if _is_protected_path(root):
                    return _result(
                        status="error",
                        error=(
                            f"'{root}' sits under a protected system "
                            "directory and cannot be scanned."
                        ),
                        code="protected_path",
                    )
            except Exception:
                pass

            cache_key = str(root)

            # ---- step 2: cache lookup ----
            if not write_md:
                cached = self._get_project_description_cache(cache_key)
                if cached:
                    return _result(
                        status="ok",
                        summary=cached["summary"],
                        project_label=cached["label"],
                        root_path=cache_key,
                        file_count=cached["file_count"],
                        cached=True,
                    )

            # ---- step 3: scan files ----
            try:
                scan = self._scan_project_for_description(root)
            except Exception as exc:
                self._logger.exception("iris_describe_scan_failed", exc, root=str(root))
                return _result(
                    status="error",
                    error=f"failed to scan project: {exc}",
                    code="scan_failed",
                )

            label = scan.get("label") or root.name
            context = scan.get("context") or ""
            if not context.strip():
                return _result(
                    status="error",
                    error=(
                        f"no readable text files found in '{root}'. "
                        "The folder may be empty or contain only binary files."
                    ),
                    code="no_content",
                )

            # ---- step 4: LLM synthesis ----
            try:
                llm = self._describe_project_via_llm(label, context)
            except Exception as exc:
                self._logger.exception("iris_describe_llm_failed", exc)
                return _result(
                    status="error",
                    error=f"LLM call failed: {exc}",
                    code="describe_failed",
                )
            if llm.get("status") != "ok":
                return _result(
                    status="error",
                    error=str(llm.get("error") or "LLM call failed"),
                    code=str(llm.get("code") or "describe_failed"),
                )
            summary = str(llm.get("text") or "").strip()

            # ---- step 5: cache ----
            try:
                self._set_project_description_cache(
                    cache_key,
                    summary=summary,
                    label=label,
                    file_count=int(scan.get("file_count") or 0),
                )
            except Exception:
                # Caching is best-effort.
                pass

            # ---- step 6: optional CLAUDE.md write ----
            generated_md_path: Optional[str] = None
            md_skipped_reason: Optional[str] = None
            if write_md:
                claude_md = root / "CLAUDE.md"
                if claude_md.exists():
                    md_skipped_reason = "CLAUDE.md already exists"
                else:
                    try:
                        from datetime import datetime
                        stamp = datetime.now().strftime("%Y-%m-%d")
                        header = (
                            f"# {label}\n\n"
                            f"<!-- Auto-generated by Iris iris_describe_project on {stamp}. "
                            f"Edit freely — Iris won't overwrite a CLAUDE.md that already exists. -->\n\n"
                        )
                        claude_md.write_text(header + summary + "\n",
                                             encoding="utf-8")
                        generated_md_path = str(claude_md)
                    except Exception as exc:
                        md_skipped_reason = f"write failed: {exc}"

            return _result(
                status="ok",
                summary=summary,
                project_label=label,
                root_path=cache_key,
                file_count=int(scan.get("file_count") or 0),
                samples_count=int(scan.get("samples_count") or 0),
                context_bytes=int(scan.get("context_bytes") or 0),
                model=str(llm.get("model") or ""),
                cached=False,
                generated_md_path=generated_md_path,
                md_skipped_reason=md_skipped_reason,
            )
        except Exception as exc:
            # Outer guard — must NEVER let this tool break dispatch.
            self._logger.exception("iris_describe_project_uncaught", exc)
            return _result(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                code="exception",
            )

    def close(self) -> None:
        """Tear down background workers. Called when the Iris session stops so
        the auto-approve watcher thread (a daemon that polls + clicks on a
        timer) doesn't keep running — and moving the cursor — after the
        session ends. Best-effort; never raises."""
        if self._uia is not None:
            try:
                self._uia.stop_auto_approve()
            except Exception as exc:
                self._logger.exception("executor_close_stop_watcher_failed", exc)

    def _t_open_path(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find file(s)/folder(s) by name with Touchless's own file search and
        open them. Accepts a single `query` OR a `queries` list (open many in
        ONE call, so a multi-item request can't stall part-way). Reuses
        DesktopController so behavior matches voice."""
        kind = str(args.get("kind", "any")).strip().lower()
        deep = bool(args.get("deep", False))
        queries = args.get("queries")
        if isinstance(queries, list) and queries:
            targets = [str(q).strip() for q in queries if str(q).strip()]
        else:
            single = str(args.get("query", "")).strip()
            targets = [single] if single else []
        if not targets:
            return _result(status="error", error="empty query", code="invalid_arguments")
        if len(targets) == 1:
            return self._open_one(targets[0], kind, deep)
        # Multiple: open every one in this single call.
        results = [self._open_one(t, kind, deep) for t in targets]
        opened = sum(1 for r in results if r.get("status") == "ok")
        overall = "ok" if opened == len(results) else ("partial" if opened else "error")
        return _result(
            status=overall,
            opened_count=opened,
            total=len(results),
            results=[
                {"query": t, "status": r.get("status"), "opened": r.get("opened"),
                 "matches": r.get("matches"), "error": r.get("error")}
                for t, r in zip(targets, results)
            ],
        )

    def _open_one(self, query: str, kind: str, deep: bool = False) -> Dict[str, Any]:
        # If the model already has an exact, existing path, just open it.
        try:
            p = Path(query).expanduser()
            if p.exists():
                desktop = self._ensure_desktop()
                if desktop is not None and desktop.open_resolved_path(p):
                    return _result(status="ok", opened=str(p), message=getattr(desktop, "message", ""))
        except Exception:
            pass
        desktop = self._ensure_desktop()
        if desktop is None:
            return _result(status="error", error="file search unavailable", code="no_desktop")
        # Try folder and/or file resolution per the requested kind.
        resolved = None
        candidates: list = []
        order = ("folder", "file") if kind in ("any", "folder") else ("file", "folder")
        for k in order:
            if k == "folder":
                r, amb = desktop.resolve_named_folder(query)
            else:
                r, amb = desktop.resolve_named_file(query)
            if r is not None:
                resolved = r
                candidates = []
                break
            if amb:
                candidates = amb  # remember ambiguous set; keep trying other kind
            if kind != "any":
                break
        if resolved is not None:
            ok = desktop.open_resolved_path(resolved)
            return _result(
                status="ok" if ok else "error",
                opened=str(resolved),
                message=getattr(desktop, "message", ""),
                error=None if ok else "open failed",
            )
        if candidates:
            return _result(
                status="ambiguous",
                matches=[str(p) for p in candidates[:10]],
                note=(
                    "Multiple matches found. Ask the user which one (list them "
                    "by name), then call open_path again with the chosen exact path."
                ),
            )
        if deep:
            # On-demand THOROUGH search: walk all drives + home deeply with a
            # wall-clock time budget (never hangs). Used when the fast search
            # misses and the user wants it found.
            best, others = desktop.deep_find(query, kind=kind)
            if best is not None:
                ok = desktop.open_resolved_path(best)
                return _result(
                    status="ok" if ok else "error",
                    opened=str(best),
                    message=getattr(desktop, "message", ""),
                    error=None if ok else "open failed",
                    deep=True,
                )
            if others:
                return _result(
                    status="ambiguous",
                    matches=[str(p) for p in others[:10]],
                    note="Deep search found several matches. Ask which one, then open_path that exact path.",
                    deep=True,
                )
        return _result(
            status="error",
            error=f"could not find a file/folder matching '{query}'",
            code="not_found",
            hint=getattr(desktop, "message", "")
            or ("not found by deep search" if deep else "retry with deep=true for a thorough all-drives search"),
        )


    # ====================================================================
    # Ambient / power-tools pack — added after the "everywhere" roadmap.
    # Each tool is small, deterministic, OS-level. All have graceful
    # fallbacks so a missing dep (winsdk, pywin32, pycaw) returns a
    # clear error instead of crashing the session.
    # ====================================================================

    def _t_notify_toast(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Show a Windows toast. Tries `win11toast` first (zero ALU-side
        deps), then `winsdk.windows.ui.notifications` (newer). Both are
        non-blocking and return immediately."""
        title = str(args.get("title") or "").strip()
        if not title:
            return _result(status="error", error="title is required")
        body = str(args.get("body") or "").strip()
        urgency = str(args.get("urgency") or "normal").lower()
        # win11toast is a thin wrapper that handles AUMID registration
        # for us — preferred when installed because it Just Works on
        # Win10/11 across packaging modes.
        try:
            from win11toast import toast as _toast  # type: ignore
            duration = "short" if urgency == "low" else (
                "long" if urgency == "high" else "short")
            try:
                _toast(title, body or None, duration=duration,
                       app_id="Touchless.Iris", on_click=None)
                return _result(status="ok", surface="win11toast",
                               title=title)
            except TypeError:
                # Older win11toast: missing some kwargs.
                _toast(title, body or None)
                return _result(status="ok", surface="win11toast",
                               title=title)
        except Exception:
            pass
        # Fallback: raw winsdk path. AUMID is set globally by main.py.
        try:
            from winsdk.windows.ui.notifications import (  # type: ignore
                ToastNotification, ToastNotificationManager,
            )
            from winsdk.windows.data.xml.dom import XmlDocument  # type: ignore
            xml_template = (
                '<toast><visual><binding template="ToastGeneric">'
                f'<text>{_xml_escape(title)}</text>'
                + (f'<text>{_xml_escape(body)}</text>' if body else '')
                + '</binding></visual></toast>'
            )
            doc = XmlDocument()
            doc.load_xml(xml_template)
            ToastNotificationManager.create_toast_notifier(
                "Touchless.Iris").show(ToastNotification(doc))
            return _result(status="ok", surface="winsdk", title=title)
        except Exception as exc:
            return _result(
                status="error",
                error=("no toast backend available — pip install "
                       "win11toast (or winsdk). Detail: "
                       f"{type(exc).__name__}: {exc}"))

    def _t_get_active_window(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return focused window context. Combines the existing
        foreground_window probe with UIA TextPattern for selected text."""
        info = get_foreground_window_info()
        title = info.title if info else ""
        app_name = (getattr(info, "process_name", "") or "") if info else ""
        exe_path = (getattr(info, "process_path", "") or "") if info else ""
        hwnd = (getattr(info, "hwnd", 0) or 0) if info else 0
        window_class = (getattr(info, "class_name", "") or "") if info else ""
        selected_text = ""
        selection_app = ""
        if bool(args.get("include_selection", True)) and hwnd:
            try:
                selected_text, selection_app = _read_uia_selection(hwnd)
            except Exception:
                selected_text, selection_app = "", ""
        # Truncate selection so a 5MB document open in an editor doesn't
        # blow up the model context.
        if len(selected_text) > 4000:
            selected_text = selected_text[:4000] + "…[truncated]"
        return _result(
            status="ok",
            title=title,
            app_name=app_name,
            exe_path=exe_path,
            hwnd=int(hwnd) if hwnd else 0,
            window_class=window_class,
            selected_text=selected_text,
            selection_app=selection_app,
            has_selection=bool(selected_text),
        )

    def _t_clipboard_read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        text, source_app = _read_clipboard_text()
        if text is None:
            return _result(
                status="error",
                error="clipboard is empty or contains non-text content")
        return _result(status="ok",
                       text=text,
                       length=len(text),
                       source_app=source_app)

    def _t_clipboard_write(self, args: Dict[str, Any]) -> Dict[str, Any]:
        text = args.get("text")
        if text is None:
            return _result(status="error", error="text is required")
        text = str(text)
        ok, err = _write_clipboard_text(text)
        if not ok:
            return _result(status="error", error=err or "clipboard write failed")
        if bool(args.get("notify", True)):
            preview = text.replace("\n", " ")[:60]
            if len(text) > 60:
                preview += "…"
            try:
                self._t_notify_toast({
                    "title": f"Copied {len(text)} chars to clipboard",
                    "body": preview, "urgency": "low",
                })
            except Exception:
                pass
        return _result(status="ok", chars=len(text))

    def _t_clipboard_transform(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read clipboard → run instruction via Ollama → replace
        clipboard. Falls back to an error message when Ollama isn't
        configured so the realtime path can take over."""
        instruction = str(args.get("instruction") or "").strip()
        if not instruction:
            return _result(status="error",
                           error="instruction is required")
        text, source_app = _read_clipboard_text()
        if text is None:
            return _result(
                status="error",
                error="clipboard is empty or contains non-text content")
        if len(text) > 8000:
            return _result(
                status="error",
                error=(f"clipboard text is {len(text)} chars — too "
                       "long for in-process transform. Read it, "
                       "summarize first, then transform."))
        try:
            from .connectors.ollama_connector import OllamaConnector
            ollama = OllamaConnector()
            if not ollama.available():
                return _result(
                    status="error",
                    error=("Ollama isn't running — install + "
                           "start it (the local model powers free "
                           "clipboard transforms), or do this via "
                           "realtime instead."))
            prompt = (
                f"{instruction}\n\n"
                "Apply the instruction above to the text below. "
                "Return ONLY the transformed text — no preamble, "
                "no explanation, no quotes around it.\n\n"
                f"---\n{text}\n---"
            )
            out = ollama.execute("ollama_generate", {"prompt": prompt})
        except Exception as exc:
            return _result(
                status="error",
                error=f"ollama call failed: {type(exc).__name__}: {exc}")
        if (out or {}).get("status") != "ok":
            return _result(
                status="error",
                error=str((out or {}).get("error") or "ollama returned no text"))
        new_text = str(out.get("text") or "").strip()
        if not new_text:
            return _result(status="error",
                           error="ollama returned an empty string")
        ok, err = _write_clipboard_text(new_text)
        if not ok:
            return _result(status="error",
                           error=err or "clipboard write failed")
        try:
            self._t_notify_toast({
                "title": "Clipboard transformed",
                "body": (instruction[:48]
                         + ("…" if len(instruction) > 48 else "")),
                "urgency": "low",
            })
        except Exception:
            pass
        return _result(
            status="ok",
            original_chars=len(text),
            new_chars=len(new_text),
            preview=new_text[:200],
            source_app=source_app,
        )


def _result(**fields: Any) -> Dict[str, Any]:
    return dict(fields)


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


# ---- clipboard helpers -----------------------------------------------------

def _read_clipboard_text() -> tuple:
    """Returns (text|None, source_app_name). Tolerant of:
      * empty clipboard → (None, "")
      * non-text content (image, file list) → (None, "")
      * other process holding the clipboard → retries once
      * pywin32 missing → (None, "") (graceful degrade)"""
    try:
        import win32clipboard  # type: ignore
        import win32con  # type: ignore
    except Exception:
        return None, ""
    # Source app = whoever last set the clipboard. Best-effort.
    source_app = ""
    text = None
    for attempt in (1, 2):
        try:
            win32clipboard.OpenClipboard()
        except Exception:
            import time
            time.sleep(0.05)
            continue
        try:
            try:
                owner_hwnd = win32clipboard.GetClipboardOwner()
                if owner_hwnd:
                    import win32process  # type: ignore
                    import win32api  # type: ignore
                    _, pid = win32process.GetWindowThreadProcessId(owner_hwnd)
                    try:
                        h = win32api.OpenProcess(0x0400 | 0x0010, False, pid)
                        source_app = win32process.GetModuleFileNameEx(h, 0)
                        source_app = source_app.rsplit("\\", 1)[-1]
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                text = str(data) if data is not None else None
            except Exception:
                # Not text content (image, file list, etc.).
                text = None
        finally:
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass
        break
    return text, source_app


def _write_clipboard_text(text: str) -> tuple:
    """Returns (ok, err). Retries once on race with other clipboard
    consumers."""
    try:
        import win32clipboard  # type: ignore
        import win32con  # type: ignore
    except Exception:
        return False, "pywin32 not available"
    last_err = ""
    for attempt in (1, 2):
        try:
            win32clipboard.OpenClipboard()
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            import time
            time.sleep(0.05)
            continue
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
            return True, ""
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass
    return False, last_err or "unknown clipboard write error"


# ---- UIA selection helper --------------------------------------------------

def _read_uia_selection(hwnd: int) -> tuple:
    """Best-effort: probe the focused UIA element for selected text. Returns
    (text, owner_app_name). On any failure returns ("", "")."""
    try:
        import uiautomation as uia  # type: ignore
    except Exception:
        return "", ""
    try:
        focused = uia.GetFocusedControl()
        if not focused:
            return "", ""
        # Owner app name from the focused element's process.
        owner = ""
        try:
            owner = (focused.ProcessId  # type: ignore[attr-defined]
                     and uia.GetRootControl().Name) or ""
        except Exception:
            owner = ""
        # Prefer the TextPattern selection (most editors, browsers,
        # Office apps). Fall back to the ValuePattern (input fields)
        # if no selection range exists.
        try:
            tp = focused.GetTextPattern()
            if tp:
                try:
                    sels = tp.GetSelection()
                    if sels:
                        parts = []
                        for r in sels:
                            try:
                                parts.append(r.GetText(-1) or "")
                            except Exception:
                                pass
                        joined = "".join(parts).strip()
                        if joined:
                            return joined, owner
                except Exception:
                    pass
        except Exception:
            pass
        # ValuePattern fallback (single-line inputs).
        try:
            vp = focused.GetValuePattern()
            if vp:
                v = (vp.Value or "").strip()
                if v:
                    return v, owner
        except Exception:
            pass
    except Exception:
        pass
    return "", ""


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_protected_path(path: Path) -> bool:
    """True if `path` falls under a system / install directory we should
    never touch via move/delete/rename tools.

    Errs on the side of caution: anything under C:\\Windows,
    Program Files, Program Files (x86), ProgramData, the system root,
    or directly at a drive root is protected.
    """
    try:
        s = str(path).lower()
    except Exception:
        return True
    protected_roots = []
    for env_var in ("WINDIR", "ProgramFiles", "ProgramFiles(x86)", "ProgramData", "SystemRoot"):
        v = os.environ.get(env_var, "")
        if v:
            protected_roots.append(v.lower())
    # Drive roots themselves (e.g. "c:\\") — never let move/delete touch.
    try:
        if str(path) == str(path.anchor):
            return True
    except Exception:
        pass
    for root in protected_roots:
        if not root:
            continue
        # Match if path equals or sits under one of the protected roots.
        if s == root or s.startswith(root + os.sep) or s.startswith(root + "/"):
            return True
    return False


# Friendly-name → list of candidates to try in order. The first one that
# succeeds via ShellExecuteW wins. Order matters: registry names first
# (cheap), explicit install paths last (covers user-installs not in
# HKLM App Paths).
_APP_SYNONYMS: Dict[str, list[str]] = {
    "vs code": ["code", "code.cmd"],
    "vscode": ["code", "code.cmd"],
    "visual studio code": ["code", "code.cmd"],
    "vs": ["code", "code.cmd"],
    "google chrome": ["chrome"],
    "chrome": ["chrome"],
    "spotify": ["spotify"],
    "notepad": ["notepad"],
    # Classic Paint launches via mspaint (there's no "paint" App Path, and
    # "Paint 3D" is a different app) — map it so 'open paint' opens Paint, no
    # disambiguation.
    "paint": ["mspaint"],
    "ms paint": ["mspaint"],
    "mspaint": ["mspaint"],
    "explorer": ["explorer"],
    "file explorer": ["explorer"],
    "files": ["explorer"],
    "calculator": ["calc"],
    "terminal": ["wt", "powershell", "cmd"],
    "powershell": ["powershell"],
    "command prompt": ["cmd"],
    "edge": ["msedge"],
    "microsoft edge": ["msedge"],
    # Games / launchers. The Riot Client lives at a fixed default path and
    # isn't in App Paths, so ShellExecuteW("valorant") fails — point at the
    # launcher exe (opening it shows the game's PLAY button for the agent to
    # click). Falls back to the literal name if the default path is missing.
    "valorant": [r"C:\Riot Games\Riot Client\RiotClientServices.exe", "valorant"],
    "riot": [r"C:\Riot Games\Riot Client\RiotClientServices.exe"],
    "riot games": [r"C:\Riot Games\Riot Client\RiotClientServices.exe"],
    "riot client": [r"C:\Riot Games\Riot Client\RiotClientServices.exe"],
    # HoYoPlay launcher (Genshin / Star Rail / ZZZ all launch through it). The
    # launcher itself isn't the game — open it, then click the game's Start
    # button via OCR. Genshin/etc. map to the same launcher.
    "hoyoplay": [r"C:\Program Files\HoYoPlay\launcher.exe", "hoyoplay"],
    # NOTE: launching GenshinImpact.exe directly just bounces back to HoYoPlay
    # (the game requires the launcher's anti-cheat init), so there's NO working
    # bypass — open the launcher, then the user/agent must click Start Game.
    "genshin": [r"C:\Program Files\HoYoPlay\launcher.exe", "genshin impact"],
    "genshin impact": [r"C:\Program Files\HoYoPlay\launcher.exe"],
    "star rail": [r"C:\Program Files\HoYoPlay\launcher.exe"],
    "honkai star rail": [r"C:\Program Files\HoYoPlay\launcher.exe"],
    "zenless": [r"C:\Program Files\HoYoPlay\launcher.exe"],
    "zzz": [r"C:\Program Files\HoYoPlay\launcher.exe"],
}


# Launch args that make the Riot Client BOOT a specific game directly (the
# same args the game's desktop shortcut uses) instead of just showing the
# launcher UI — so opening these starts the game with no manual PLAY click.
_GAME_LAUNCH_ARGS: Dict[str, list[str]] = {
    "valorant": ["--launch-product=valorant", "--launch-patchline=live"],
    "league of legends": ["--launch-product=league_of_legends", "--launch-patchline=live"],
    "lol": ["--launch-product=league_of_legends", "--launch-patchline=live"],
    # Discord's Update.exe launches the app version-independently via this arg.
    "discord": ["--processStart", "Discord.exe"],
}

# The subset of _GAME_LAUNCH_ARGS that are ACTUAL Riot games — these need the
# slow Riot-Client-window wait + vision PLAY click. Discord shares the launch-
# args dict purely for its --processStart arg, but is NOT a game: it must keep
# the normal monitor/placement handling, so it's deliberately excluded here.
_RIOT_GAME_KEYS = frozenset({"valorant", "league of legends", "lol"})


def _open_app_candidates(name: str) -> list[str]:
    """Build the ordered list of launch targets for a friendly app name."""
    seen: list[str] = []

    def _add(candidate: str) -> None:
        candidate = (candidate or "").strip()
        if candidate and candidate not in seen:
            seen.append(candidate)

    lookup_key = name.strip().lower()
    _VSCODE_KEYS = {"vs code", "vscode", "visual studio code", "vs", "code"}

    # For VS Code, prefer the concrete Code.exe BEFORE the bare "code"
    # name. launch_external uses ShellExecuteW, which resolves apps via
    # the App Paths registry, NOT PATH — so the `code` CLI shim
    # (code.cmd on PATH) can return a ">32 success" without actually
    # opening the editor. Launching an absolute Code.exe is unambiguous.
    if lookup_key in _VSCODE_KEYS:
        local_app = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        exe_paths = []
        if local_app:
            exe_paths.append(Path(local_app) / "Programs" / "Microsoft VS Code" / "Code.exe")
        exe_paths.append(Path(program_files) / "Microsoft VS Code" / "Code.exe")
        for exe in exe_paths:
            try:
                if exe.exists():
                    _add(str(exe))
            except Exception:
                pass

    # Discord installs per-user under LOCALAPPDATA with a versioned folder; the
    # version-independent launcher is Update.exe (--processStart Discord.exe is
    # added by _GAME_LAUNCH_ARGS). Fall back to the newest versioned exe.
    if lookup_key == "discord":
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            disc = Path(local_app) / "Discord"
            upd = disc / "Update.exe"
            try:
                if upd.exists():
                    _add(str(upd))
                versioned = sorted(disc.glob("app-*/Discord.exe"), reverse=True)
                if versioned:
                    _add(str(versioned[0]))
            except Exception:
                pass

    # MATLAB installs under Program Files\MATLAB\<version>\bin\matlab.exe; glob
    # for the newest version so we don't hard-code R2025b.
    if lookup_key in {"matlab", "mat lab"}:
        m = _matlab_exe()
        if m:
            _add(m)

    # Then the literal name — might be a real registry key or a path
    # the model resolved correctly.
    _add(name)

    for synonym in _APP_SYNONYMS.get(lookup_key, ()):
        _add(synonym)

    # Unconditional VS Code fallbacks (in case the exe wasn't found at
    # the checked path above, or env vars were empty).
    if lookup_key in _VSCODE_KEYS:
        local_app = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        if local_app:
            _add(str(Path(local_app) / "Programs" / "Microsoft VS Code" / "Code.exe"))
        _add(str(Path(program_files) / "Microsoft VS Code" / "Code.exe"))

    return seen


def _matlab_exe() -> Optional[str]:
    """Find matlab.exe — newest Program Files\\MATLAB\\<ver>\\bin install, or PATH."""
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
        try:
            root = Path(base) / "MATLAB"
            if root.exists():
                exes = sorted(root.glob("*/bin/matlab.exe"), reverse=True)
                if exes:
                    return str(exes[0])
        except Exception:
            pass
    found = shutil.which("matlab")
    return found or None


def _resolve_python_executable() -> Optional[str]:
    """Locate a Python interpreter we can use to run user scripts.

    Resolution order:
      1. TOUCHLESS_PYTHON env var (escape hatch for unusual setups).
      2. sys.executable — but ONLY when not running as a PyInstaller
         frozen binary, because in frozen mode sys.executable is
         Touchless.exe itself, which obviously can't run a .py file.
      3. shutil.which("python") / "python3" / "py" — first match.
      4. Common Windows install locations as a last-resort fallback.

    Returns the absolute path string, or None if nothing is found.
    """
    env_override = os.environ.get("TOUCHLESS_PYTHON", "").strip()
    if env_override and Path(env_override).exists():
        return env_override

    if not getattr(sys, "frozen", False):
        # Source mode — sys.executable is the user's actual python.
        # Prefer it because it's the same env they ran the app from
        # (so deps like tkinter, pygame, etc. are guaranteed there).
        if sys.executable and Path(sys.executable).exists():
            return sys.executable

    for candidate in ("python", "python3", "py"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    # Common Windows fallbacks for installed-mode users who don't have
    # python on PATH but do have it installed in the standard locations.
    local_app = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    fallback_globs = []
    if local_app:
        fallback_globs.append(Path(local_app) / "Programs" / "Python")
    fallback_globs.append(Path(program_files) / "Python313")
    fallback_globs.append(Path(program_files) / "Python312")
    fallback_globs.append(Path(program_files) / "Python311")
    for base in fallback_globs:
        if not base.exists():
            continue
        # Direct python.exe in the dir, OR nested Python3xx/python.exe.
        direct = base / "python.exe"
        if direct.exists():
            return str(direct)
        for child in base.glob("Python3*/python.exe"):
            return str(child)
    return None


def _summarize_args(name: str, args: Dict[str, Any], *, debug: bool) -> Dict[str, Any]:
    """Produce a privacy-friendly summary of args for the log."""
    if debug:
        return args
    out: Dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 64 and key in {"text", "content"}:
            out[key] = {"length": len(value), "preview": value[:64]}
        else:
            out[key] = value
    return out

# Author: Konstantin Markov
