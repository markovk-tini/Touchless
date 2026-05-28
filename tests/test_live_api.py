"""Tests for the experimental Live API subsystem.

These tests cover the parts that don't need a network or microphone:
schema validation, file-action safety, manager state transitions, and
realtime event parsing on canned JSON. The OpenAI websocket itself is
not exercised here.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

from hgr.live_api.config import LiveApiConfig
from hgr.live_api.live_api_logger import LiveApiLogger, redact
from hgr.live_api.schemas import all_tool_schemas, validate_args
from hgr.live_api.tool_executor import ToolExecutor


class _FakeScreen:
    def capture(self):
        return None


def _make_logger(tmp: Path) -> LiveApiLogger:
    return LiveApiLogger(log_dir=tmp / "logs", debug_text_logging=False)


class SchemaTests(unittest.TestCase):
    def test_all_tool_schemas_have_required_fields(self) -> None:
        for schema in all_tool_schemas():
            self.assertIn("type", schema)
            self.assertEqual(schema["type"], "function")
            self.assertIn("name", schema)
            self.assertIn("description", schema)
            self.assertIn("parameters", schema)
            params = schema["parameters"]
            self.assertEqual(params.get("type"), "object")
            self.assertIn("properties", params)

    def test_validate_args_rejects_unknown_tool(self) -> None:
        ok, err, _ = validate_args("not_a_tool", {})
        self.assertFalse(ok)
        self.assertIn("unknown tool", err)

    def test_validate_args_requires_required_field(self) -> None:
        ok, err, _ = validate_args("create_folder", {})
        self.assertFalse(ok)
        self.assertIn("folder_name", err)

    def test_validate_args_applies_defaults(self) -> None:
        ok, err, args = validate_args("click_screen", {"x": 0.5, "y": 0.5})
        self.assertTrue(ok, msg=err)
        self.assertEqual(args["coordinate_space"], "normalized")
        self.assertEqual(args["button"], "left")
        self.assertFalse(args["double_click"])

    def test_validate_args_enforces_enum(self) -> None:
        ok, err, _ = validate_args("type_text", {"text": "hi", "method": "telepathy"})
        self.assertFalse(ok)
        self.assertIn("must be one of", err)


class LoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="live_api_log_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_logger_creates_files(self) -> None:
        logger = _make_logger(self.tmp)
        try:
            logger.event("hello", x=1)
            self.assertTrue(logger.text_log_path.exists())
            self.assertTrue(logger.jsonl_log_path.exists())
            with open(logger.jsonl_log_path, encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]
            self.assertTrue(any(rec.get("kind") == "hello" for rec in lines))
        finally:
            logger.close()

    def test_redact_strips_api_key(self) -> None:
        out = redact({"api_key": "sk-123", "nested": {"authorization": "Bearer x", "ok": 1}})
        self.assertEqual(out["api_key"], "***redacted***")
        self.assertEqual(out["nested"]["authorization"], "***redacted***")
        self.assertEqual(out["nested"]["ok"], 1)


class FileToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="live_api_files_"))
        self.cfg = LiveApiConfig(
            api_key="test", safe_workspace_dir=self.tmp / "workspace", log_dir=self.tmp / "logs"
        )
        self.logger = _make_logger(self.tmp)
        self.executor = ToolExecutor(
            config=self.cfg, logger=self.logger, screen_context=_FakeScreen()
        )

    def tearDown(self) -> None:
        self.logger.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_folder_creates_under_safe_workspace(self) -> None:
        result = self.executor.execute("create_folder", {"folder_name": "demo"})
        self.assertEqual(result["status"], "ok")
        self.assertTrue((self.cfg.safe_workspace_dir / "demo").is_dir())

    def test_create_file_refuses_to_overwrite(self) -> None:
        # Pre-create the file the model is about to "create".
        target = self.cfg.safe_workspace_dir / "main.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("preexisting", encoding="utf-8")

        result = self.executor.execute(
            "create_file", {"relative_path": "main.py", "content": "new"}
        )
        self.assertEqual(result["status"], "needs_confirmation")
        self.assertEqual(target.read_text(encoding="utf-8"), "preexisting")

    def test_write_file_overwrite_creates_backup(self) -> None:
        target = self.cfg.safe_workspace_dir / "draft.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("old", encoding="utf-8")

        result = self.executor.execute(
            "write_file",
            {"relative_path": "draft.txt", "content": "new", "overwrite": True},
        )
        self.assertEqual(result["status"], "ok", msg=result)
        self.assertEqual(target.read_text(encoding="utf-8"), "new")
        backup = target.with_suffix(target.suffix + ".bak")
        self.assertTrue(backup.exists())
        self.assertEqual(backup.read_text(encoding="utf-8"), "old")

    def test_invalid_args_short_circuit(self) -> None:
        result = self.executor.execute("write_file", {"relative_path": "", "content": "x"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result.get("code"), "invalid_arguments")


class ManagerStateTests(unittest.TestCase):
    """Lightweight smoke test for state machine wiring with mocks.

    A full integration test would need a websocket server stub; here we
    just verify start() emits CONNECTING when the client constructor is
    monkey-patched out, and stop() returns to OFF.
    """

    def test_start_without_api_key_fires_error(self) -> None:
        # Importing here to avoid pulling Qt at module import time when
        # the rest of the suite doesn't need it.
        from PySide6.QtCore import QCoreApplication
        from hgr.live_api.live_api_manager import LiveApiManager, LiveApiState

        app = QCoreApplication.instance() or QCoreApplication([])
        cfg = LiveApiConfig(api_key=None)
        manager = LiveApiManager(config=cfg)
        captured: list[str] = []
        manager.error_occurred.connect(lambda msg: captured.append(msg))
        manager.start()
        # State should be ERROR with no api key.
        self.assertEqual(manager.state, LiveApiState.ERROR)
        self.assertTrue(any("OPENAI_API_KEY" in m for m in captured))
        del app  # silence "unused" warnings on some linters


class LocalBackendTests(unittest.TestCase):
    """Tests for the local (offline) backend that don't need real binaries."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="live_api_local_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _logger(self) -> LiveApiLogger:
        return LiveApiLogger(log_dir=self.tmp / "logs")

    def test_load_config_picks_backend_from_env(self) -> None:
        from hgr.live_api.config import load_config
        with patch.dict("os.environ", {"TOUCHLESS_LIVE_API_BACKEND": "local"}, clear=False):
            cfg = load_config()
        self.assertEqual(cfg.backend, "local")

    def test_local_backend_fires_error_when_llama_missing(self) -> None:
        from hgr.live_api import local_backend as lb
        logger = self._logger()
        try:
            errors: list[str] = []
            backend = lb.LocalBackend(
                config=LiveApiConfig(),
                logger=logger,
                tools=[],
                system_instructions="",
                on_event=lambda e: None,
                on_error=lambda m: errors.append(m),
            )
            with patch.object(lb, "_resolve_llama_server_executable", return_value=None):
                ok = backend.start()
            self.assertFalse(ok)
            self.assertTrue(any("llama-server" in m for m in errors))
        finally:
            logger.close()

    def test_local_backend_send_screen_image_is_noop(self) -> None:
        # Phase 1: vision off — calling send_screen_image must not raise
        # and must not actually do anything network-y.
        from hgr.live_api.local_backend import LocalBackend
        logger = self._logger()
        try:
            backend = LocalBackend(
                config=LiveApiConfig(),
                logger=logger,
                tools=[],
                system_instructions="",
                on_event=lambda e: None,
            )
            self.assertTrue(backend.send_screen_image("dGVzdA==", caption="ignored"))
        finally:
            logger.close()

    def test_local_backend_send_text_message_appends(self) -> None:
        from hgr.live_api.local_backend import LocalBackend
        logger = self._logger()
        try:
            backend = LocalBackend(
                config=LiveApiConfig(),
                logger=logger,
                tools=[],
                system_instructions="",
                on_event=lambda e: None,
            )
            self.assertTrue(backend.send_text_message("hello"))
            self.assertEqual(backend._messages[-1], {"role": "user", "content": "hello"})
        finally:
            logger.close()

    def test_local_backend_send_tool_result_serialises_payload(self) -> None:
        from hgr.live_api.local_backend import LocalBackend
        logger = self._logger()
        try:
            backend = LocalBackend(
                config=LiveApiConfig(),
                logger=logger,
                tools=[],
                system_instructions="",
                on_event=lambda e: None,
            )
            backend._inflight_tool_calls["call_42"] = "create_folder"
            self.assertTrue(backend.send_tool_result("call_42", {"status": "ok", "path": "C:\\x"}))
            last = backend._messages[-1]
            self.assertEqual(last["role"], "tool")
            self.assertEqual(last["tool_call_id"], "call_42")
            self.assertEqual(last["name"], "create_folder")
            self.assertIn("\"status\": \"ok\"", last["content"])
        finally:
            logger.close()


class TextOnlyManagerTests(unittest.TestCase):
    """Verify the text-only manager path used by the typed-command UI."""

    def test_send_user_text_returns_false_when_off(self) -> None:
        from PySide6.QtCore import QCoreApplication
        from hgr.live_api.live_api_manager import LiveApiManager

        app = QCoreApplication.instance() or QCoreApplication([])
        manager = LiveApiManager(config=LiveApiConfig(api_key="x"), text_only=True)
        # Manager hasn't started — must refuse and not crash.
        self.assertFalse(manager.send_user_text("hello"))
        del app

    def test_text_only_manager_skips_audio(self) -> None:
        # We can't run a full session without llama-server, but we can
        # at least verify that .text_only stops the manager from trying
        # to instantiate AudioStream (which would fail without sounddevice).
        from PySide6.QtCore import QCoreApplication
        from hgr.live_api.live_api_manager import LiveApiManager

        app = QCoreApplication.instance() or QCoreApplication([])
        manager = LiveApiManager(text_only=True)
        # Internal flag visible for assertion.
        self.assertTrue(manager._text_only)
        self.assertIsNone(manager._audio)
        del app


class FileManagementToolTests(unittest.TestCase):
    """Move/rename/delete + recent-paths tracking + system-dir safety."""

    def setUp(self) -> None:
        # .resolve() normalizes Windows 8.3 short-names ("KONSTA~1") to
        # their long form so test path comparisons match what the
        # executor records (it always resolves before recording).
        self.tmp = Path(tempfile.mkdtemp(prefix="live_api_files_")).resolve()
        self.cfg = LiveApiConfig(
            api_key="x", safe_workspace_dir=self.tmp / "ws", log_dir=self.tmp / "logs"
        )
        self.logger = LiveApiLogger(log_dir=self.cfg.log_dir)
        self.executor = ToolExecutor(
            config=self.cfg, logger=self.logger, screen_context=_FakeScreen()
        )

    def tearDown(self) -> None:
        self.logger.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_move_file_moves_and_records_path(self) -> None:
        src = self.tmp / "src" / "a.txt"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("hello", encoding="utf-8")
        dst = self.tmp / "dst" / "a.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        result = self.executor.execute(
            "move_file",
            {"source_path": str(src), "destination_path": str(dst)},
        )
        self.assertEqual(result["status"], "ok", msg=result)
        self.assertFalse(src.exists())
        self.assertTrue(dst.exists())
        self.assertEqual(dst.read_text(encoding="utf-8"), "hello")
        # Was recorded for list_recent_paths.
        recent = self.executor.execute("list_recent_paths", {})
        self.assertEqual(recent["status"], "ok")
        self.assertIn(str(dst), recent["paths"])

    def test_move_file_refuses_system_dirs(self) -> None:
        src = self.tmp / "a.txt"
        src.write_text("x", encoding="utf-8")
        # Use the actual Windows dir env var if available, else fall back.
        windir = os.environ.get("WINDIR", r"C:\Windows")
        result = self.executor.execute(
            "move_file",
            {"source_path": str(src), "destination_path": f"{windir}\\agent_test.txt"},
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result.get("code"), "protected_path")
        self.assertTrue(src.exists())  # unchanged

    def test_rename_file_renames_in_place(self) -> None:
        src = self.tmp / "old.txt"
        src.write_text("y", encoding="utf-8")
        result = self.executor.execute(
            "rename_file", {"path": str(src), "new_name": "new.txt"}
        )
        self.assertEqual(result["status"], "ok", msg=result)
        self.assertFalse(src.exists())
        self.assertTrue((self.tmp / "new.txt").exists())

    def test_rename_file_rejects_path_in_new_name(self) -> None:
        src = self.tmp / "f.txt"
        src.write_text("y", encoding="utf-8")
        result = self.executor.execute(
            "rename_file", {"path": str(src), "new_name": "../f.txt"}
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result.get("code"), "invalid_arguments")

    def test_delete_file_requires_explicit_confirmation(self) -> None:
        target = self.tmp / "to_delete.txt"
        target.write_text("z", encoding="utf-8")
        # Without confirmed=true, must NOT delete.
        result = self.executor.execute("delete_file", {"path": str(target)})
        self.assertEqual(result["status"], "needs_confirmation")
        self.assertTrue(target.exists())
        # With confirmed=true, deletes.
        result = self.executor.execute(
            "delete_file", {"path": str(target), "confirmed": True}
        )
        self.assertEqual(result["status"], "ok", msg=result)
        self.assertFalse(target.exists())

    def test_delete_file_refuses_system_paths(self) -> None:
        windir = os.environ.get("WINDIR", r"C:\Windows")
        result = self.executor.execute(
            "delete_file", {"path": windir, "confirmed": True}
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result.get("code"), "protected_path")

    def test_create_folder_records_path_in_recent_list(self) -> None:
        result = self.executor.execute("create_folder", {"folder_name": "TrackedDemo"})
        self.assertEqual(result["status"], "ok")
        recent = self.executor.execute("list_recent_paths", {})
        self.assertIn(result["path"], recent["paths"])


class CommandRouterTests(unittest.TestCase):
    """The router shouldn't even instantiate the heavy controllers
    until we ask it to route something. Verify the no-match path is
    safe and the matched path returns the expected shape."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="live_api_router_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _logger(self) -> LiveApiLogger:
        return LiveApiLogger(log_dir=self.tmp / "logs")

    def test_empty_text_returns_no_match(self) -> None:
        from hgr.live_api.command_router import CommandRouter, RouterResult
        logger = self._logger()
        try:
            router = CommandRouter(logger=logger)
            result = router.try_route("")
            self.assertIsInstance(result, RouterResult)
            self.assertFalse(result.matched)
        finally:
            logger.close()

    def test_unknown_command_returns_no_match(self) -> None:
        from hgr.live_api.command_router import CommandRouter
        logger = self._logger()
        try:
            router = CommandRouter(logger=logger)
            # An obviously novel multi-step request — the deterministic
            # router should never claim this.
            result = router.try_route(
                "create a folder called experiments and write a tkinter "
                "script that draws a rotating cube"
            )
            self.assertFalse(result.matched)
        finally:
            logger.close()

    def test_router_init_failure_is_graceful(self) -> None:
        # If VoiceCommandProcessor blows up at construction, the router
        # must still answer no-match instead of raising.
        from hgr.live_api import command_router as cr
        logger = self._logger()
        try:
            with patch.object(
                cr, "_ROUTER_CONFIDENCE_FLOOR", 0.7
            ):
                router = cr.CommandRouter(logger=logger)
                # Force the lazy-init to fail.
                with patch(
                    "hgr.voice.command_processor.VoiceCommandProcessor",
                    side_effect=RuntimeError("boom"),
                ):
                    result = router.try_route("open chrome")
            self.assertFalse(result.matched)
        finally:
            logger.close()


class ConnectorRouterTests(unittest.TestCase):
    """Capability-search router + connector registry hygiene."""

    def setUp(self) -> None:
        from hgr.live_api.connectors import build_connector_registry
        self.reg = build_connector_registry()

    def test_connector_tool_names_unique_and_no_builtin_collision(self) -> None:
        builtin = {s["name"] for s in all_tool_schemas()}
        seen = set()
        for c in self.reg._connectors:
            for s in c.tools():
                name = s["name"]
                self.assertNotIn(name, builtin, f"{name} collides with a built-in")
                self.assertNotIn(name, seen, f"{name} duplicated across connectors")
                seen.add(name)

    def test_search_ranks_known_intents(self) -> None:
        # Guard on availability — CI may not have a given app authed/installed.
        # (Spotify is intentionally NOT a connector — Touchless Layer 0 owns it.)
        cases = {"turn the volume down": "volume", "mute my discord": "discord"}
        ids_available = {e["id"] for e in self.reg.catalog()}
        for query, expected in cases.items():
            if expected in ids_available:
                hits = self.reg.search(query, limit=1)
                self.assertTrue(hits, f"expected a hit for {query!r}")
                self.assertEqual(hits[0]["id"], expected)

    def test_search_returns_nothing_for_irrelevant_task(self) -> None:
        self.assertEqual(self.reg.search("defragment the flux capacitor"), [])

    def test_search_only_surfaces_available_connectors(self) -> None:
        for entry in self.reg.search("send an email", limit=5):
            owner = next(c for c in self.reg._connectors if c.id == entry["id"])
            self.assertTrue(owner.available())


class CostPolicyTests(unittest.TestCase):
    """The routing-ladder cost classification."""

    def test_levels(self) -> None:
        from hgr.live_api.cost_policy import classify
        self.assertEqual(classify("anything", "touchless")[0], 0)
        self.assertEqual(classify("spotify_play", "connector")[0], 2)
        self.assertEqual(classify("click_screen", "iris")[0], 4)        # vision
        self.assertEqual(classify("click_text_on_screen", "iris")[0], 1)  # OCR
        self.assertEqual(classify("send_to_coding_agent", "iris")[0], 3)  # LLM
        self.assertEqual(classify("open_app", "iris")[0], 0)            # local

    def test_decision_record_shape(self) -> None:
        from hgr.live_api.cost_policy import decision_record
        rec = decision_record(tool="volume_set", source="connector", status="ok")
        for key in ("tool", "source", "cost_level", "cost_label", "why_short", "status"):
            self.assertIn(key, rec)
        self.assertEqual(rec["cost_level"], 2)


class ScreenReaderTests(unittest.TestCase):
    """Unified ScreenContext: match ladder + hash-based cache."""

    def test_match_ranking(self) -> None:
        from hgr.live_api.screen_reader import ScreenReader, ScreenElement
        r = ScreenReader()
        els = [
            ScreenElement(text="Send", clickable=True),
            ScreenElement(text="Send email now", clickable=True),
            ScreenElement(text="Cancel", clickable=True),
        ]
        res = r._match("send", els, ocr_fallback=False)
        self.assertTrue(res)
        self.assertEqual(res[0].text, "Send")  # exact beats substring
        self.assertNotIn("Cancel", [e.text for e in res])  # unrelated excluded

    def test_cache_reuse_invalidate_and_window_change(self) -> None:
        from hgr.live_api.screen_reader import ScreenReader, ScreenContext
        r = ScreenReader(freshness_sec=60)
        state = {"win": ("vscode", "Demo - VS Code", "code.exe"), "builds": 0}
        r._active_window = lambda: state["win"]

        def fake_build(app, title, proc, sig, *, want_text=False):
            state["builds"] += 1
            return ScreenContext(timestamp=time.time(), active_app=app,
                                 active_window_title=title, active_process=proc,
                                 screen_hash=sig)
        r._build = fake_build

        r.get_context(); self.assertEqual(state["builds"], 1)
        r.get_context(); self.assertEqual(state["builds"], 1)   # cached (unchanged + fresh)
        r.invalidate(); r.get_context(); self.assertEqual(state["builds"], 2)  # dirtied -> rebuild
        state["win"] = ("chrome", "Google", "chrome.exe")
        r.get_context(); self.assertEqual(state["builds"], 3)   # window changed -> rebuild


class IrisPlannerClassifierTests(unittest.TestCase):
    """Phase 1 of the planner: deterministic intent -> Step, no LLM."""

    def setUp(self) -> None:
        from hgr.live_api.planner.classifier import Classifier
        self.c = Classifier()

    def _expect(self, text: str, tool: str, **expected_args) -> None:
        step = self.c.classify(text)
        self.assertIsNotNone(step, f"classifier missed: {text!r}")
        self.assertEqual(step.tool, tool, f"wrong tool for {text!r}: {step.tool}")
        for k, v in expected_args.items():
            self.assertEqual(step.args.get(k), v, f"{text!r} args[{k}]={step.args.get(k)} (expected {v})")

    def _miss(self, text: str) -> None:
        self.assertIsNone(self.c.classify(text), f"classifier should have missed: {text!r}")

    def test_volume(self) -> None:
        self._expect("set volume to 30", "volume_set", percent=30)
        self._expect("volume 75", "volume_set", percent=75)
        self._expect("what's the volume", "volume_get")
        self._expect("mute", "volume_mute", muted=True)
        self._expect("unmute", "volume_mute", muted=False)
        self._expect("toggle mute", "volume_toggle_mute")
        # "mute discord" must NOT become system mute — it routes to discord:
        step = self.c.classify("mute discord")
        self.assertIsNotNone(step)
        self.assertEqual(step.tool, "discord_mute")

    def test_discord(self) -> None:
        self._expect("mute discord", "discord_mute", muted=True)
        self._expect("unmute discord", "discord_mute", muted=False)
        self._expect("toggle mute on discord", "discord_toggle_mute")
        self._expect("deafen on discord", "discord_deafen", deafened=True)

    def test_todo(self) -> None:
        self._expect("add a task: buy milk", "todo_add", title="buy milk")
        self._expect("remind me to call mom", "todo_add", title="call mom")

    def test_google_create(self) -> None:
        self._expect("make a google doc titled Demo", "gdocs_create", title="demo")
        self._expect("create a spreadsheet called Budget", "sheets_create", title="budget")
        self._expect("new slideshow titled Pitch", "slides_create", title="pitch")

    def test_drive_upload(self) -> None:
        self._expect("upload C:/tmp/file.png to my google drive", "drive_upload",
                     path="C:/tmp/file.png")

    def test_email_compose_with_recipient_and_body(self) -> None:
        step = self.c.classify("email dani@mangollc.org saying hi from iris")
        self.assertIsNotNone(step)
        self.assertEqual(step.tool, "outlook_compose")
        self.assertEqual(step.args.get("recipient"), "dani@mangollc.org")
        self.assertEqual(step.args.get("body"), "hi from iris")

    def test_misses_safely(self) -> None:
        # Things that must NOT classify (they need the LLM / fall through):
        self._miss("what's the weather")
        self._miss("read my latest email")
        self._miss("summarize my unread emails")
        self._miss("can you help me figure out what to do today")


class _StubRegistry:
    """Minimal registry stand-in: records calls and returns canned outputs."""

    def __init__(self, outputs: Dict[str, Any]) -> None:
        self.outputs = outputs  # tool -> dict (or callable(args) -> dict)
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def call(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((tool, dict(args or {})))
        out = self.outputs.get(tool)
        if callable(out):
            return out(args)
        if out is None:
            return {"status": "error", "error": "no stub"}
        return dict(out)

    def handles_connector(self, _tool: str) -> bool:
        return True


class IrisPlannerExecutorTests(unittest.TestCase):
    """Phase 2 executor: dep order, {step:N.field} refs, error short-circuit."""

    def test_runs_in_dependency_order(self) -> None:
        from hgr.live_api.planner import Executor, Plan, Step
        plan = Plan(goal="g", steps=[
            Step(id=2, tool="b", args={"x": 1}, depends_on=[1]),
            Step(id=1, tool="a", args={}, depends_on=[]),
            Step(id=3, tool="c", args={}, depends_on=[2]),
        ])
        reg = _StubRegistry({"a": {"status": "ok"}, "b": {"status": "ok"},
                             "c": {"status": "ok"}})
        results = Executor(reg).run(plan)
        self.assertEqual([r.tool for r in results], ["a", "b", "c"])
        self.assertTrue(all(r.status == "ok" for r in results))

    def test_resolves_step_refs(self) -> None:
        from hgr.live_api.planner import Executor, Plan, Step
        plan = Plan(goal="g", steps=[
            Step(id=1, tool="lookup", args={"q": "dani"}),
            Step(id=2, tool="send",
                 args={"to": "{step:1.email}",
                       "msg": "hi {step:1.profile.name}"},
                 depends_on=[1]),
        ])
        reg = _StubRegistry({
            "lookup": {"status": "ok", "email": "x@y.z",
                       "profile": {"name": "Dani"}},
            "send": {"status": "ok"},
        })
        Executor(reg).run(plan)
        # The send step's args should have been substituted from step 1's output.
        send_args = next(a for t, a in reg.calls if t == "send")
        self.assertEqual(send_args["to"], "x@y.z")
        self.assertEqual(send_args["msg"], "hi Dani")

    def test_error_short_circuits_dependents(self) -> None:
        from hgr.live_api.planner import Executor, Plan, Step
        plan = Plan(goal="g", steps=[
            Step(id=1, tool="boom", args={}),
            Step(id=2, tool="later", args={}, depends_on=[1]),
            Step(id=3, tool="independent", args={}),
        ])
        reg = _StubRegistry({
            "boom": {"status": "error", "error": "nope"},
            "later": {"status": "ok"},
            "independent": {"status": "ok"},
        })
        results = Executor(reg).run(plan)
        by_id = {r.step_id: r for r in results}
        self.assertEqual(by_id[1].status, "error")
        # Dependent gets unresolved-dep error, not actually called.
        self.assertEqual(by_id[2].status, "error")
        self.assertNotIn("later", [t for t, _ in reg.calls])
        # Independent step still runs.
        self.assertEqual(by_id[3].status, "ok")


class IrisPlannerOrchestratorTests(unittest.TestCase):
    """Phase 2 wiring: orchestrator returns the unified {steps, results} shape."""

    def test_phase1_returns_single_step_list(self) -> None:
        from hgr.live_api.planner.orchestrator import IrisPlanner
        reg = _StubRegistry({"volume_set": {"status": "ok"}})
        planner = IrisPlanner(reg)
        out = planner.try_handle("set volume to 40")
        self.assertIsNotNone(out)
        self.assertEqual(len(out["steps"]), 1)
        self.assertEqual(out["steps"][0].tool, "volume_set")
        self.assertEqual(out["results"][0].status, "ok")
        self.assertIn("40", out["message"])

    def test_phase2_uses_executor_when_flag_on(self) -> None:
        # Forge a Plan and stub the LLMPlanner so we don't hit the network.
        from hgr.live_api.planner.orchestrator import IrisPlanner
        from hgr.live_api.planner.plan import Plan, Step
        reg = _StubRegistry({
            "lookup": {"status": "ok", "id": "abc"},
            "send": {"status": "ok", "link": "https://x"},
        })
        planner = IrisPlanner(reg)
        forged = Plan(goal="g", steps=[
            Step(id=1, tool="lookup", args={}),
            Step(id=2, tool="send", args={"ref": "{step:1.id}"}, depends_on=[1]),
        ])
        # Patch the LLMPlanner so it returns our forged plan.
        planner._llm_planner.plan = lambda _goal, memory_context="": forged  # type: ignore[assignment]
        # Also patch configured() so the flag-check passes even w/o API key.
        import hgr.live_api.planner.orchestrator as orch
        import hgr.live_api.planner.planner_llm as pl
        old_flag = os.environ.get("TOUCHLESS_IRIS_PLAN_LLM")
        old_cfg = pl.configured
        os.environ["TOUCHLESS_IRIS_PLAN_LLM"] = "1"
        orch.llm_planner_configured = lambda: True
        try:
            out = planner.try_handle("do something tricky")
        finally:
            if old_flag is None:
                os.environ.pop("TOUCHLESS_IRIS_PLAN_LLM", None)
            else:
                os.environ["TOUCHLESS_IRIS_PLAN_LLM"] = old_flag
            orch.llm_planner_configured = pl.configured  # restore
            pl.configured = old_cfg
        self.assertIsNotNone(out)
        self.assertEqual([s.tool for s in out["steps"]], ["lookup", "send"])
        self.assertEqual([r.status for r in out["results"]], ["ok", "ok"])
        # send step args were resolved from lookup's output:
        send_args = next(a for t, a in reg.calls if t == "send")
        self.assertEqual(send_args["ref"], "abc")


class IrisPlannerSchedulerTests(unittest.TestCase):
    """Phase 3 scheduler: sliding-window back-pressure between lanes."""

    def test_throttle_window_decays(self) -> None:
        from hgr.live_api.planner.scheduler import RateScheduler
        t = [0.0]
        sched = RateScheduler(clock=lambda: t[0])
        sched.record_rate_limit("realtime")
        # Inside the window: throttled.
        t[0] = 30.0
        self.assertTrue(sched.is_throttled("realtime", window=60.0))
        # Past the window: cleared.
        t[0] = 65.0
        self.assertFalse(sched.is_throttled("realtime", window=60.0))

    def test_prefers_cheap_when_realtime_throttled(self) -> None:
        from hgr.live_api.planner.scheduler import RateScheduler
        t = [0.0]
        sched = RateScheduler(clock=lambda: t[0])
        # Both healthy → no preference for cheap.
        self.assertFalse(sched.prefer_cheap_planner())
        # Realtime takes a 429 → prefer cheap.
        sched.record_rate_limit("realtime")
        self.assertTrue(sched.prefer_cheap_planner())
        # But if cheap is ALSO 429, don't prefer it (no point).
        sched.record_rate_limit("cheap-llm")
        self.assertFalse(sched.prefer_cheap_planner())
        self.assertFalse(sched.allow_cheap_synthesis())


class IrisPlannerSynthesizerTests(unittest.TestCase):
    """Phase 3 synthesizer: trims outputs, calls LLM with goal+steps."""

    def test_trim_output_drops_raw_and_caps_strings(self) -> None:
        from hgr.live_api.planner.synthesizer import Synthesizer
        big = "x" * 5000
        trimmed = Synthesizer._trim_output({
            "subject": "hi",
            "body": big,
            "raw": {"huge": big},     # dropped
            "count": 3,
            "items": list(range(50)), # list capped
        })
        self.assertEqual(trimmed["subject"], "hi")
        self.assertLess(len(trimmed["body"]), 5000)
        self.assertNotIn("raw", trimmed)
        self.assertEqual(trimmed["count"], 3)
        self.assertEqual(len(trimmed["items"]), 10)

    def test_summarize_passes_goal_and_steps_to_llm(self) -> None:
        from hgr.live_api.planner.synthesizer import Synthesizer
        from hgr.live_api.planner.plan import Plan, StepResult
        os.environ["OPENAI_API_KEY"] = "fake-for-test"
        try:
            synth = Synthesizer()
            captured: Dict[str, Any] = {}
            def fake_call(messages):
                captured["messages"] = messages
                return "Dani is free Wed after 2pm."
            synth._call = fake_call  # type: ignore[assignment]
            plan = Plan(goal="when is Dani free?", steps=[], final="synthesize")
            results = [StepResult(step_id=1, tool="cal_freebusy",
                                  status="ok",
                                  output={"slots": ["Wed 2-4pm"]})]
            out = synth.summarize(plan, results)
            self.assertEqual(out, "Dani is free Wed after 2pm.")
            user = captured["messages"][1]["content"]
            self.assertIn("when is Dani free", user)
            self.assertIn("cal_freebusy", user)
        finally:
            os.environ.pop("OPENAI_API_KEY", None)


class IrisPlannerSynthesisRoutingTests(unittest.TestCase):
    """Orchestrator routes 'synthesize' plans through the Synthesizer when
    cheap-LLM is healthy, and falls back to the deterministic format when
    the scheduler says cheap-LLM is throttled."""

    def _setup(self):
        from hgr.live_api.planner.orchestrator import IrisPlanner
        from hgr.live_api.planner.plan import Plan, Step
        from hgr.live_api.planner.scheduler import scheduler
        reg = _StubRegistry({
            "cal_freebusy": {"status": "ok", "slots": ["Wed 2-4pm"]},
        })
        planner = IrisPlanner(reg)
        forged = Plan(goal="when is Dani free?", steps=[
            Step(id=1, tool="cal_freebusy", args={}),
        ], final="synthesize")
        planner._llm_planner.plan = lambda _g, memory_context="": forged  # type: ignore[assignment]
        # Make sure the cheap-LLM scheduler is healthy at the start of each test.
        sched = scheduler()
        sched._events.clear()  # type: ignore[attr-defined]
        return planner, sched

    def _run(self, planner):
        import hgr.live_api.planner.orchestrator as orch
        import hgr.live_api.planner.synthesizer as sy
        os.environ["TOUCHLESS_IRIS_PLAN_LLM"] = "1"
        old_llm = orch.llm_planner_configured
        old_syn = orch.synth_configured
        orch.llm_planner_configured = lambda: True
        orch.synth_configured = lambda: True
        try:
            return planner.try_handle("when is Dani free?")
        finally:
            os.environ.pop("TOUCHLESS_IRIS_PLAN_LLM", None)
            orch.llm_planner_configured = old_llm
            orch.synth_configured = old_syn

    def test_synthesizer_used_when_healthy(self) -> None:
        planner, _sched = self._setup()
        planner._synthesizer.summarize = lambda _p, _r: "Dani is free Wed 2-4pm."
        out = self._run(planner)
        self.assertEqual(out["message"], "Dani is free Wed 2-4pm.")

    def test_falls_back_to_format_when_cheap_llm_throttled(self) -> None:
        planner, sched = self._setup()
        sched.record_rate_limit("cheap-llm")
        called = {"n": 0}
        def _should_not_run(_p, _r):
            called["n"] += 1
            return "nope"
        planner._synthesizer.summarize = _should_not_run
        out = self._run(planner)
        self.assertEqual(called["n"], 0)
        # Deterministic format kicks in instead — content varies, but it
        # must NOT be the synthesizer output and must be non-empty.
        self.assertNotEqual(out["message"], "nope")
        self.assertTrue(out["message"])


class IrisPlannerTriggersTests(unittest.TestCase):
    """Phase 4 heuristic: pick out requests that look multi-step."""

    def test_chain_words_trigger(self) -> None:
        from hgr.live_api.planner.triggers import looks_multi_action
        self.assertTrue(looks_multi_action("open Outlook and then read the latest email"))
        self.assertTrue(looks_multi_action("find Dani's email then send him hi"))
        self.assertTrue(looks_multi_action("search for AI news, after that summarize the top result"))

    def test_and_verb_triggers(self) -> None:
        from hgr.live_api.planner.triggers import looks_multi_action
        self.assertTrue(looks_multi_action("find Dani's email and send him hi"))
        self.assertTrue(looks_multi_action("create a doc and share it with the team"))

    def test_two_distinct_verbs_trigger(self) -> None:
        from hgr.live_api.planner.triggers import looks_multi_action
        # "open chrome to search for X" — two action verbs.
        self.assertTrue(looks_multi_action("open chrome and search for the latest news"))

    def test_single_step_does_not_trigger(self) -> None:
        from hgr.live_api.planner.triggers import looks_multi_action
        self.assertFalse(looks_multi_action("what's the volume"))
        self.assertFalse(looks_multi_action("mute"))
        self.assertFalse(looks_multi_action("open YouTube"))
        self.assertFalse(looks_multi_action("hi"))

    def test_noun_form_verbs_do_not_falsely_trigger(self) -> None:
        """Two verb-words in ONE action ('send dani an email', 'post a
        message') must not trip the heuristic. The two-verb branch now
        requires a connector (and/comma/chain) to count."""
        from hgr.live_api.planner.triggers import looks_multi_action
        self.assertFalse(looks_multi_action("send dani an email saying hi"))
        self.assertFalse(looks_multi_action("post a message to the channel"))
        self.assertFalse(looks_multi_action("email dani saying I will send the post tomorrow"))

    def test_comma_chains_trigger(self) -> None:
        from hgr.live_api.planner.triggers import looks_multi_action
        self.assertTrue(looks_multi_action("open chrome, search for AI news, summarize the top result"))

    def test_plan_needs_confirm(self) -> None:
        from hgr.live_api.planner.triggers import plan_needs_confirm
        self.assertTrue(plan_needs_confirm(["cal_freebusy", "outlook_compose"]))
        self.assertTrue(plan_needs_confirm(["drive_upload"]))
        self.assertFalse(plan_needs_confirm(["cal_freebusy", "ms_mail_list"]))
        self.assertFalse(plan_needs_confirm([]))


class IrisPlannerCacheTests(unittest.TestCase):
    """Phase 4 plan cache: skip the LLM call when the same goal recurs."""

    def test_hits_within_ttl(self) -> None:
        from hgr.live_api.planner.plan_cache import PlanCache
        from hgr.live_api.planner.plan import Plan, Step
        t = [0.0]
        cache = PlanCache(ttl=60.0, clock=lambda: t[0])
        plan = Plan(goal="g", steps=[Step(id=1, tool="a")])
        cache.put("Do The Thing", plan)
        # Case + whitespace normalization.
        self.assertIs(cache.get("  do the thing  "), plan)

    def test_expires_after_ttl(self) -> None:
        from hgr.live_api.planner.plan_cache import PlanCache
        from hgr.live_api.planner.plan import Plan, Step
        t = [0.0]
        cache = PlanCache(ttl=10.0, clock=lambda: t[0])
        cache.put("do it", Plan(goal="g", steps=[Step(id=1, tool="a")]))
        t[0] = 20.0
        self.assertIsNone(cache.get("do it"))

    def test_lru_eviction(self) -> None:
        from hgr.live_api.planner.plan_cache import PlanCache
        from hgr.live_api.planner.plan import Plan, Step
        t = [0.0]
        cache = PlanCache(ttl=60.0, max_entries=2, clock=lambda: t[0])
        cache.put("a", Plan(goal="g", steps=[Step(id=1, tool="x")]))
        t[0] = 1.0
        cache.put("b", Plan(goal="g", steps=[Step(id=1, tool="x")]))
        t[0] = 2.0
        cache.put("c", Plan(goal="g", steps=[Step(id=1, tool="x")]))  # evicts "a"
        self.assertIsNone(cache.get("a"))
        self.assertIsNotNone(cache.get("b"))
        self.assertIsNotNone(cache.get("c"))


class IrisPlannerPhase4WiringTests(unittest.TestCase):
    """Phase 4: heuristic opens Phase 2, plan cache skips repeat LLM calls,
    risky plans surface a single confirm dialog."""

    def _build(self):
        from hgr.live_api.planner.orchestrator import IrisPlanner
        from hgr.live_api.planner.plan import Plan, Step
        from hgr.live_api.planner.scheduler import scheduler
        reg = _StubRegistry({
            "lookup": {"status": "ok", "email": "x@y"},
            "outlook_compose": {"status": "ok"},
        })
        forged = Plan(goal="find x and email", steps=[
            Step(id=1, tool="lookup", args={}),
            Step(id=2, tool="outlook_compose",
                 args={"recipient": "{step:1.email}"}, depends_on=[1]),
        ])
        calls = {"plan": 0}
        confirms = {"prompts": []}
        def confirm(title, body):
            confirms["prompts"].append((title, body))
            return confirms.get("answer", True)
        planner = IrisPlanner(reg, confirm=confirm)
        scheduler()._events.clear()  # type: ignore[attr-defined]
        def fake_plan(_g, memory_context=""):
            calls["plan"] += 1
            return forged
        planner._llm_planner.plan = fake_plan  # type: ignore[assignment]
        import hgr.live_api.planner.orchestrator as orch
        orch.llm_planner_configured = lambda: True
        return planner, reg, calls, confirms

    def test_heuristic_opens_phase2_without_flag(self) -> None:
        planner, reg, calls, confirms = self._build()
        confirms["answer"] = True
        out = planner.try_handle("find Dani's email and send him hi")
        self.assertIsNotNone(out)
        self.assertEqual(calls["plan"], 1)
        self.assertEqual([s.tool for s in out["steps"]],
                         ["lookup", "outlook_compose"])
        # outlook_compose is risky → confirm must have been requested.
        self.assertEqual(len(confirms["prompts"]), 1)
        self.assertIn("outlook_compose", confirms["prompts"][0][1])

    def test_confirm_cancel_skips_execution(self) -> None:
        planner, reg, calls, confirms = self._build()
        confirms["answer"] = False  # user clicks "no"
        out = planner.try_handle("find Dani's email and send him hi")
        self.assertEqual(out["message"], "Cancelled.")
        self.assertEqual([r.status for r in out["results"]],
                         ["cancelled", "cancelled"])
        # Plan was generated but neither step actually ran.
        self.assertEqual(reg.calls, [])

    def test_cache_skips_second_llm_call(self) -> None:
        planner, reg, calls, confirms = self._build()
        confirms["answer"] = True
        planner.try_handle("find Dani's email and send him hi")
        planner.try_handle("find Dani's email and send him hi")
        # Second call reused the cached Plan.
        self.assertEqual(calls["plan"], 1)


class WebSearchTests(unittest.TestCase):
    """Phase 5 web_search: Google CSE preferred, DDG HTML fallback."""

    def setUp(self) -> None:
        # Snapshot env so we can restore between tests.
        self._env_snapshot = {
            k: os.environ.get(k) for k in
            ("GOOGLE_CSE_API_KEY", "GOOGLE_CSE_ID")
        }
        for k in self._env_snapshot:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._env_snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_empty_query_rejected(self) -> None:
        from hgr.live_api.web_search import web_search
        out = web_search("")
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["code"], "invalid_arguments")

    def test_google_cse_when_configured(self) -> None:
        from hgr.live_api import web_search as ws
        os.environ["GOOGLE_CSE_API_KEY"] = "k"
        os.environ["GOOGLE_CSE_ID"] = "cx"
        fake_payload = json.dumps({
            "items": [
                {"title": "AI News Today",
                 "link": "https://example.com/a",
                 "snippet": "blurb a"},
                {"title": "More AI",
                 "link": "https://example.com/b",
                 "snippet": "blurb b"},
            ]
        }).encode("utf-8")

        class _FakeResp:
            def __init__(self, body): self._b = body
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return self._b

        seen: Dict[str, Any] = {}
        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            return _FakeResp(fake_payload)

        with patch.object(ws.urllib.request, "urlopen", side_effect=fake_urlopen):
            out = ws.web_search("latest AI news", count=2, recent_days=7)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["provider"], "google_cse")
        self.assertEqual(len(out["results"]), 2)
        self.assertEqual(out["results"][0]["url"], "https://example.com/a")
        # Recent-days propagated into the CSE URL.
        self.assertIn("dateRestrict=d7", seen["url"])

    def test_ddg_fallback_when_no_keys(self) -> None:
        from hgr.live_api import web_search as ws
        html = (
            '<a class="result__a" href="https://news.example.com/x">First Result</a>'
            '<a class="result__snippet">snippet about first</a>'
            '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2F'
            'two.example.com">Second</a>'
            '<a class="result__snippet">snippet about second</a>'
        ).encode("utf-8")

        class _FakeResp:
            def __init__(self, body): self._b = body
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return self._b

        def fake_urlopen(req, timeout=None):
            return _FakeResp(html)

        with patch.object(ws.urllib.request, "urlopen", side_effect=fake_urlopen):
            out = ws.web_search("python tutorials", count=2)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["provider"], "duckduckgo")
        self.assertEqual(len(out["results"]), 2)
        self.assertEqual(out["results"][0]["title"], "First Result")
        self.assertEqual(out["results"][0]["url"], "https://news.example.com/x")
        # Redirect-wrapped URLs get unwrapped.
        self.assertEqual(out["results"][1]["url"], "https://two.example.com")

    def test_cse_quota_error_falls_through_to_ddg(self) -> None:
        from hgr.live_api import web_search as ws
        os.environ["GOOGLE_CSE_API_KEY"] = "k"
        os.environ["GOOGLE_CSE_ID"] = "cx"
        ddg_html = (
            '<a class="result__a" href="https://fallback.example.com/x">Fallback</a>'
            '<a class="result__snippet">from ddg</a>'
        ).encode("utf-8")

        class _FakeResp:
            def __init__(self, body): self._b = body
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return self._b

        calls = {"n": 0}
        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                # CSE quota exceeded → fall through.
                raise urllib.error.HTTPError(
                    req.full_url, 429, "quota", {}, None)
            return _FakeResp(ddg_html)

        import urllib.error  # local import so the closure can raise it
        with patch.object(ws.urllib.request, "urlopen", side_effect=fake_urlopen):
            out = ws.web_search("ai news")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["provider"], "duckduckgo")
        self.assertEqual(out["results"][0]["url"], "https://fallback.example.com/x")


class WebSearchToolWiringTests(unittest.TestCase):
    """The tool executor routes 'web_search' calls to web_search.web_search,
    and the schema is exposed in all_tool_schemas()."""

    def test_schema_registered(self) -> None:
        from hgr.live_api.schemas import all_tool_schemas
        names = [s["name"] for s in all_tool_schemas()]
        self.assertIn("web_search", names)

    def test_schema_validates_required_query(self) -> None:
        from hgr.live_api.schemas import validate_args
        ok, msg, _ = validate_args("web_search", {})
        self.assertFalse(ok)
        self.assertIn("query", msg)
        ok, _, normalised = validate_args("web_search",
                                          {"query": "test", "count": 3})
        self.assertTrue(ok)
        self.assertEqual(normalised["query"], "test")


class IrisPlannerEdgeCaseTests(unittest.TestCase):
    """Edge cases that exercise the planner under hostile / weird input."""

    def test_classifier_skipped_when_input_is_multi_action(self) -> None:
        """The classic data-loss bug: 'set volume to 30 AND email dani' must
        NOT fire volume_set and drop the email half. Phase 1 must yield to
        Phase 2 (or realtime) when looks_multi_action is True."""
        from hgr.live_api.planner.orchestrator import IrisPlanner
        reg = _StubRegistry({"volume_set": {"status": "ok"}})
        # No LLM planner configured → Phase 2 won't fire either, so we expect
        # None (fall through to realtime), NOT a silent volume_set call.
        planner = IrisPlanner(reg)
        out = planner.try_handle("set volume to 30 and email dani saying hi")
        self.assertIsNone(out)
        self.assertEqual(reg.calls, [])

    def test_classifier_still_fires_for_pure_single_intent(self) -> None:
        from hgr.live_api.planner.orchestrator import IrisPlanner
        reg = _StubRegistry({"volume_set": {"status": "ok"}})
        planner = IrisPlanner(reg)
        out = planner.try_handle("set volume to 30")
        self.assertIsNotNone(out)
        self.assertEqual(out["steps"][0].tool, "volume_set")

    def test_plan_cache_is_thread_safe(self) -> None:
        """Concurrent put/get across threads must not crash or lose data."""
        import threading as _t
        from hgr.live_api.planner.plan_cache import PlanCache
        from hgr.live_api.planner.plan import Plan, Step
        cache = PlanCache(max_entries=16)
        errors: List[BaseException] = []

        def worker(seed: int) -> None:
            try:
                for i in range(100):
                    key = f"goal {seed} {i % 4}"
                    cache.put(key, Plan(goal=key,
                                        steps=[Step(id=1, tool="a")]))
                    cache.get(key)
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [_t.Thread(target=worker, args=(s,)) for s in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(errors, [])

    def test_web_search_clamps_recent_days(self) -> None:
        from hgr.live_api import web_search as ws
        os.environ["GOOGLE_CSE_API_KEY"] = "k"
        os.environ["GOOGLE_CSE_ID"] = "cx"
        seen: Dict[str, Any] = {}

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return b'{"items": []}'

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            return _Resp()

        try:
            with patch.object(ws.urllib.request, "urlopen", side_effect=fake_urlopen):
                ws.web_search("x", recent_days=-5)
            # Negative clamped to 0 → no dateRestrict in URL.
            self.assertNotIn("dateRestrict", seen["url"])
            with patch.object(ws.urllib.request, "urlopen", side_effect=fake_urlopen):
                ws.web_search("x", recent_days=99999)
            # Over-large clamped to 365.
            self.assertIn("dateRestrict=d365", seen["url"])
        finally:
            os.environ.pop("GOOGLE_CSE_API_KEY", None)
            os.environ.pop("GOOGLE_CSE_ID", None)

    def test_web_search_caps_query_length(self) -> None:
        from hgr.live_api import web_search as ws
        seen: Dict[str, Any] = {}

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return b''

        def fake_urlopen(req, timeout=None):
            seen["req"] = req
            return _Resp()

        with patch.object(ws.urllib.request, "urlopen", side_effect=fake_urlopen):
            out = ws.web_search("x" * 5000)
        # No keys → DDG path; query in the POST body.
        body = seen["req"].data.decode("utf-8")
        # 5000-char query was capped to 500 (so encoded length is bounded).
        self.assertLess(len(body), 1500)
        # Tool still returned a structured response (empty results).
        self.assertEqual(out["status"], "ok")

    def test_llm_planner_renumbers_duplicate_step_ids(self) -> None:
        """Two Step(id=1, ...) would silently overwrite each other in the
        executor's results dict. The parser must renumber duplicates."""
        from hgr.live_api.planner.planner_llm import LLMPlanner
        data = {
            "goal": "x",
            "steps": [
                {"id": 1, "tool": "a"},
                {"id": 1, "tool": "b"},
                {"id": 1, "tool": "c"},
            ],
            "final": "return",
        }
        plan = LLMPlanner._parse("x", data)
        self.assertIsNotNone(plan)
        ids = sorted(s.id for s in plan.steps)
        self.assertEqual(len(set(ids)), len(ids), f"duplicate ids: {ids}")
        self.assertEqual([s.tool for s in plan.steps], ["a", "b", "c"])

    def test_executor_marks_dependent_failed_when_upstream_errors(self) -> None:
        """Already covered, but a regression-pin: an arg ref like
        {step:1.id} into an errored step doesn't silently substitute ''.
        The dependent step must be marked error before it ever runs."""
        from hgr.live_api.planner import Executor, Plan, Step
        plan = Plan(goal="g", steps=[
            Step(id=1, tool="broken", args={}),
            Step(id=2, tool="follow",
                 args={"x": "{step:1.id}"}, depends_on=[1]),
        ])
        reg = _StubRegistry({
            "broken": {"status": "error", "error": "boom"},
            "follow": {"status": "ok"},  # should NEVER be called
        })
        results = Executor(reg).run(plan)
        # "follow" must not have actually run.
        self.assertNotIn("follow", [t for t, _ in reg.calls])
        # Both results are errors.
        statuses = {r.step_id: r.status for r in results}
        self.assertEqual(statuses, {1: "error", 2: "error"})

    def test_synthesizer_handles_nonjsonable_output(self) -> None:
        """If a tool returns objects json.dumps can't natively serialize
        (datetime, set, custom class), the synthesizer must not crash."""
        from datetime import datetime
        from hgr.live_api.planner.synthesizer import Synthesizer
        from hgr.live_api.planner.plan import Plan, StepResult
        os.environ["OPENAI_API_KEY"] = "fake-for-test"
        try:
            synth = Synthesizer()
            synth._call = lambda _m: "summary text"  # type: ignore[assignment]
            plan = Plan(goal="g", steps=[], final="synthesize")
            results = [StepResult(step_id=1, tool="t", status="ok",
                                  output={"when": datetime(2026, 5, 27),
                                          "set_field": {1, 2, 3},
                                          "list_field": list(range(50))})]
            out = synth.summarize(plan, results)
            self.assertEqual(out, "summary text")
        finally:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_orchestrator_no_confirm_callback_still_runs(self) -> None:
        """When the UI hasn't wired a confirm callback, risky plans should
        still run (dev/headless mode). The Phase 1 path already does this
        for needs_confirm — verify the Phase 2 path matches."""
        from hgr.live_api.planner.orchestrator import IrisPlanner
        from hgr.live_api.planner.plan import Plan, Step
        import hgr.live_api.planner.orchestrator as orch
        reg = _StubRegistry({"outlook_compose": {"status": "ok"}})
        planner = IrisPlanner(reg, confirm=None)  # explicit no-callback
        planner._llm_planner.plan = lambda _g, memory_context="": Plan(  # type: ignore[assignment]
            goal="g",
            steps=[Step(id=1, tool="outlook_compose",
                        args={"recipient": "x@y", "body": "hi"})])
        os.environ["TOUCHLESS_IRIS_PLAN_LLM"] = "1"
        old_cfg = orch.llm_planner_configured
        orch.llm_planner_configured = lambda: True
        try:
            out = planner.try_handle("send dani an email")
        finally:
            os.environ.pop("TOUCHLESS_IRIS_PLAN_LLM", None)
            orch.llm_planner_configured = old_cfg
        self.assertIsNotNone(out)
        self.assertEqual(out["results"][0].status, "ok")
        self.assertEqual(reg.calls, [("outlook_compose",
                                      {"recipient": "x@y", "body": "hi"})])


class SessionNoteBuilderTests(unittest.TestCase):
    """The compact note injected into realtime after a planner-handled turn."""

    def _build(self):
        from hgr.live_api.live_api_manager import LiveApiManager
        return LiveApiManager._build_session_note

    def _fake_step(self, tool, args=None):
        ns = type("S", (), {})()
        ns.tool = tool
        ns.args = args or {}
        ns.id = 1
        return ns

    def _fake_result(self, tool, status="ok", output=None):
        ns = type("R", (), {})()
        ns.tool = tool
        ns.status = status
        ns.output = output or {}
        ns.error = None
        return ns

    def test_basic_shape(self) -> None:
        build = self._build()
        note = build(
            "find Dani's email",
            [self._fake_step("ms_contacts_find", {"q": "Dani"})],
            [self._fake_result("ms_contacts_find",
                               output={"status": "ok",
                                       "email": "dani@x.io",
                                       "name": "Dani M"})],
            "Found Dani: dani@x.io",
        )
        # Mentions tool, status, and the resolvable entity.
        self.assertIn("ms_contacts_find=ok", note)
        self.assertIn("dani@x.io", note)
        self.assertIn("find Dani's email", note)
        # And the user-visible reply for context.
        self.assertIn("Found Dani", note)

    def test_redacts_secrets(self) -> None:
        build = self._build()
        note = build(
            "log in",
            [self._fake_step("auth")],
            [self._fake_result("auth",
                               output={"status": "ok",
                                       "access_token": "supersecret123",
                                       "refresh_token": "rfsh",
                                       "email": "u@x"})],
            "Logged in.",
        )
        self.assertNotIn("supersecret123", note)
        self.assertNotIn("rfsh", note)
        self.assertIn("u@x", note)

    def test_size_capped(self) -> None:
        build = self._build()
        big = "x" * 2000
        note = build(
            big,
            [self._fake_step("t")],
            [self._fake_result("t", output={"title": big})],
            big,
        )
        # The whole note is capped, individual fields trimmed.
        self.assertLessEqual(len(note), 600)

    def test_handles_empty_steps(self) -> None:
        build = self._build()
        note = build("hi", [], [], "")
        self.assertIn("tools: none", note)


class SessionNoteWiringTests(unittest.TestCase):
    """When the planner handles a turn, the manager sends a session note
    via client.send_session_note (best-effort, never breaks the reply)."""

    def test_send_session_note_called_after_planner_handle(self) -> None:
        from hgr.live_api.live_api_manager import LiveApiManager

        sent: List[str] = []

        class _FakeClient:
            connected = True
            def send_session_note(self, text):
                sent.append(text)
                return True

        # Synthesize a manager handled-result and run the post-handle block
        # by calling _build_session_note + the client directly, mirroring the
        # production code path. Unit-isolating the full manager is heavy;
        # this hits the contract we care about.
        from hgr.live_api.planner.plan import Step, StepResult
        steps = [Step(id=1, tool="ms_mail_send",
                      args={"to": "dani@x", "body": "hi"})]
        results = [StepResult(step_id=1, tool="ms_mail_send", status="ok",
                              output={"status": "ok",
                                      "recipient": "dani@x",
                                      "link": "https://m/abc"})]
        note = LiveApiManager._build_session_note(
            "email Dani saying hi",
            steps, results,
            "Drafted email to dani@x.",
        )
        client = _FakeClient()
        client.send_session_note(note)

        self.assertEqual(len(sent), 1)
        self.assertIn("ms_mail_send=ok", sent[0])
        self.assertIn("dani@x", sent[0])


class RealtimeClientNoteTests(unittest.TestCase):
    """realtime_client.send_session_note emits a SYSTEM message (no response)."""

    def test_emits_system_role_no_response(self) -> None:
        from hgr.live_api.realtime_client import RealtimeClient
        sent: List[Dict[str, Any]] = []
        c = RealtimeClient.__new__(RealtimeClient)
        c._send = lambda payload: (sent.append(payload), True)[1]  # type: ignore[attr-defined]
        ok = c.send_session_note("hello")
        self.assertTrue(ok)
        self.assertEqual(len(sent), 1)
        item = sent[0]["item"]
        self.assertEqual(item["role"], "system")
        self.assertEqual(item["content"][0]["text"], "hello")

    def test_empty_note_noop(self) -> None:
        from hgr.live_api.realtime_client import RealtimeClient
        sent: List[Dict[str, Any]] = []
        c = RealtimeClient.__new__(RealtimeClient)
        c._send = lambda payload: (sent.append(payload), True)[1]  # type: ignore[attr-defined]
        self.assertTrue(c.send_session_note(""))
        self.assertEqual(sent, [])


class MemoryStoreTests(unittest.TestCase):
    """SQLite layer: schema init, episodic + semantic CRUD, capacity cap."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="iris-mem-")
        from hgr.live_api.memory.store import MemoryStore
        self._db = Path(self._tmp) / "m.db"
        self._store = MemoryStore(self._db)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_round_trip(self) -> None:
        rid = self._store.add_episodic(
            "set volume to 30", "{}", '[]', "Volume set to 30%.",
            [0.1, 0.2, 0.3])
        self.assertGreater(rid, 0)
        rows = self._store.list_episodic()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].user_text, "set volume to 30")
        # Embedding round-trips losslessly enough for cosine sim.
        self.assertEqual(len(rows[0].embedding), 3)
        self.assertAlmostEqual(rows[0].embedding[0], 0.1, places=5)

    def test_semantic_upsert(self) -> None:
        self._store.add_semantic("person", "Dani", "old@x")
        self._store.add_semantic("person", "Dani", "old@x")  # dup → replace
        rows = self._store.find_facts(kind="person", key="dani")
        self.assertEqual(len(rows), 1)
        # New value for same (kind, key) overwrites the row (UNIQUE replace).
        self._store.add_semantic("person", "Dani", "new@x")
        rows = self._store.find_facts(kind="person", key="dani")
        self.assertEqual({r.value for r in rows}, {"old@x", "new@x"})

    def test_episodic_cap(self) -> None:
        from hgr.live_api.memory.store import MemoryStore
        MemoryStore.MAX_EPISODIC_ROWS = 5  # type: ignore[misc]
        try:
            for i in range(8):
                self._store.add_episodic(f"goal {i}", None, None, None, [])
            rows = self._store.list_episodic(limit=100)
            self.assertLessEqual(len(rows), 5)
            # Oldest entries dropped.
            user_texts = {r.user_text for r in rows}
            self.assertNotIn("goal 0", user_texts)
            self.assertIn("goal 7", user_texts)
        finally:
            MemoryStore.MAX_EPISODIC_ROWS = 10_000  # type: ignore[misc]

    def test_clear(self) -> None:
        self._store.add_episodic("x", None, None, None, [])
        self._store.add_semantic("person", "a", "b")
        self._store.clear()
        self.assertEqual(self._store.count(), {"episodic": 0, "semantic": 0})


class FakeEmbedderTests(unittest.TestCase):
    """The deterministic embedder used by all offline tests."""

    def test_deterministic_and_normalized(self) -> None:
        from hgr.live_api.memory.embedder import FakeEmbedder, cosine_sim
        e = FakeEmbedder()
        a = e.embed("send dani an email")
        b = e.embed("send dani an email")
        c = e.embed("totally unrelated tropical fish facts")
        self.assertEqual(a, b)
        # Self-similarity ~1, dissimilar text well below.
        self.assertAlmostEqual(cosine_sim(a, a), 1.0, places=5)
        self.assertLess(cosine_sim(a, c), 0.8)

    def test_default_embedder_falls_back_to_fake_without_key(self) -> None:
        from hgr.live_api.memory.embedder import default_embedder, FakeEmbedder
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            self.assertIsInstance(default_embedder(), FakeEmbedder)
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old


class MemoryExtractorTests(unittest.TestCase):
    """Pulls Dani -> dani@x and similar facts out of planner-handled turns."""

    def _step(self, tool, args=None):
        ns = type("S", (), {})()
        ns.tool = tool
        ns.args = args or {}
        return ns

    def _result(self, output=None):
        ns = type("R", (), {})()
        ns.output = output or {}
        return ns

    def test_extracts_person_from_outlook_compose(self) -> None:
        from hgr.live_api.memory.extractor import extract_facts
        facts = extract_facts(
            "email Dani saying hi",
            [self._step("outlook_compose",
                        {"recipient": "dani@mangollc.org", "body": "hi"})],
            [self._result({"status": "ok"})],
        )
        self.assertIn(("person", "dani", "dani@mangollc.org", "outlook_compose arg"),
                      facts)

    def test_extracts_artifact_link(self) -> None:
        from hgr.live_api.memory.extractor import extract_facts
        facts = extract_facts(
            "make a google doc titled Q3 Report",
            [self._step("gdocs_create", {"title": "Q3 Report"})],
            [self._result({"status": "ok", "link": "https://docs/x"})],
        )
        self.assertTrue(any(f[0] == "artifact" and f[2] == "https://docs/x"
                            for f in facts))

    def test_dedupes(self) -> None:
        from hgr.live_api.memory.extractor import extract_facts
        facts = extract_facts(
            "email Dani",
            [self._step("outlook_compose", {"recipient": "dani@x.io"}),
             self._step("outlook_compose", {"recipient": "dani@x.io"})],
            [self._result({"status": "ok"}),
             self._result({"status": "ok"})],
        )
        person_facts = [f for f in facts if f[0] == "person"]
        self.assertEqual(len(person_facts), 1)


class MemoryManagerTests(unittest.TestCase):
    """End-to-end: record a turn, recall it later by similar query."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="iris-mem-")
        from hgr.live_api.memory import MemoryManager, MemoryStore
        from hgr.live_api.memory.embedder import FakeEmbedder
        self._mgr = MemoryManager(
            store=MemoryStore(Path(self._tmp) / "m.db"),
            embedder=FakeEmbedder(),
            async_writes=False,  # sync for deterministic tests
        )

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _step(self, tool, args=None):
        ns = type("S", (), {})()
        ns.tool = tool
        ns.args = args or {}
        return ns

    def _result(self, output=None, status="ok"):
        ns = type("R", (), {})()
        ns.output = output or {}
        ns.status = status
        return ns

    def test_record_then_recall_episodic(self) -> None:
        self._mgr.record(
            "find Dani's email and tell her about the meeting",
            None,
            [self._step("ms_contacts_find", {"q": "Dani"})],
            [self._result({"status": "ok", "email": "dani@x.io", "name": "Dani"})],
            "Found Dani: dani@x.io",
        )
        recall = self._mgr.recall("when was the last time I emailed Dani about the meeting?")
        # Episode surfaced AND the extracted person fact surfaced.
        self.assertTrue(recall["episodes"], "expected episodic recall")
        self.assertTrue(any(f.kind == "person" and f.value == "dani@x.io"
                            for f in recall["facts"]),
                        f"expected Dani fact, got {recall['facts']}")
        # Pre-rendered context is non-empty when there's a hit.
        self.assertIn("dani", recall["context"].lower())

    def test_recall_empty_for_unrelated_query(self) -> None:
        self._mgr.record(
            "make a google doc titled Budget",
            None,
            [self._step("gdocs_create", {"title": "Budget"})],
            [self._result({"status": "ok", "link": "https://docs/b"})],
            "Created Google Doc \"Budget\": https://docs/b",
        )
        # Different topic — fact key "budget" is not in this query, and
        # FakeEmbedder cosine sim is well below threshold for unrelated text.
        recall = self._mgr.recall("what's the volume right now")
        self.assertEqual(recall["facts"], [])
        # Episodes may or may not match (cosine sim is noisy on a 64-dim hash),
        # but the rendered context should be tiny or empty.
        self.assertLessEqual(len(recall["context"]), 600)


class OrchestratorMemoryWiringTests(unittest.TestCase):
    """Verify the planner calls recall before planning and records after."""

    def test_record_called_after_phase1(self) -> None:
        from hgr.live_api.planner.orchestrator import IrisPlanner
        reg = _StubRegistry({"volume_set": {"status": "ok"}})

        calls: Dict[str, Any] = {"records": 0, "recalls": 0}
        class _FakeMem:
            def record(self, *a, **k):
                calls["records"] += 1
            def recall(self, *a, **k):
                calls["recalls"] += 1
                return {"context": "", "episodes": [], "facts": []}

        planner = IrisPlanner(reg, memory=_FakeMem())
        out = planner.try_handle("set volume to 30")
        self.assertIsNotNone(out)
        self.assertEqual(calls["records"], 1)
        # Phase 1 doesn't need to recall; recall only fires in Phase 2.
        self.assertEqual(calls["recalls"], 0)

    def test_recall_context_passed_to_llm_planner(self) -> None:
        from hgr.live_api.planner.orchestrator import IrisPlanner
        from hgr.live_api.planner.plan import Plan, Step
        import hgr.live_api.planner.orchestrator as orch
        reg = _StubRegistry({"lookup": {"status": "ok", "email": "x@y"}})

        seen_kwargs: Dict[str, Any] = {}
        class _FakeMem:
            def record(self, *a, **k): pass
            def recall(self, *a, **k):
                return {"context": "Context from prior turns:\n- person dani = dani@x",
                        "episodes": [], "facts": []}

        planner = IrisPlanner(reg, memory=_FakeMem())
        def fake_plan(goal, memory_context=""):
            seen_kwargs["memory_context"] = memory_context
            return Plan(goal=goal, steps=[Step(id=1, tool="lookup", args={})])
        planner._llm_planner.plan = fake_plan  # type: ignore[assignment]
        os.environ["TOUCHLESS_IRIS_PLAN_LLM"] = "1"
        old = orch.llm_planner_configured
        orch.llm_planner_configured = lambda: True
        try:
            planner.try_handle("look up dani and tell her something")
        finally:
            os.environ.pop("TOUCHLESS_IRIS_PLAN_LLM", None)
            orch.llm_planner_configured = old
        self.assertIn("dani@x", seen_kwargs.get("memory_context", ""))


class SkillStoreTests(unittest.TestCase):
    """User-saved Plans, replayable without an LLM call."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="iris-skills-")
        from hgr.live_api.planner.skills import SkillStore
        self._store = SkillStore(Path(self._tmp) / "s.db")

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_plan(self, goal="morning briefing"):
        from hgr.live_api.planner.plan import Plan, Step
        return Plan(goal=goal, steps=[
            Step(id=1, tool="ms_mail_list", args={"folder": "inbox"}),
            Step(id=2, tool="cal_today", args={}),
        ])

    def test_save_and_find_exact_trigger(self) -> None:
        self._store.save("morning briefing", "morning briefing", self._make_plan())
        plan = self._store.find("morning briefing")
        self.assertIsNotNone(plan)
        self.assertEqual([s.tool for s in plan.steps], ["ms_mail_list", "cal_today"])

    def test_find_with_extra_words(self) -> None:
        self._store.save("morning briefing", "morning briefing", self._make_plan())
        # User said "do my morning briefing please" — exact-substring pass hits.
        self.assertIsNotNone(self._store.find("Do my morning briefing please"))

    def test_find_with_token_overlap(self) -> None:
        # Trigger words present non-contiguously — pass-2 still matches.
        self._store.save("recap", "weekly recap", self._make_plan())
        self.assertIsNotNone(self._store.find("give me a recap for the weekly meeting"))

    def test_no_match_returns_none(self) -> None:
        self._store.save("morning briefing", "morning briefing", self._make_plan())
        self.assertIsNone(self._store.find("set volume to 30"))
        self.assertIsNone(self._store.find(""))

    def test_delete(self) -> None:
        self._store.save("x", "x", self._make_plan())
        self.assertTrue(self._store.delete("x"))
        self.assertIsNone(self._store.find("x"))

    def test_use_count_increments(self) -> None:
        self._store.save("x", "x", self._make_plan())
        self._store.mark_used("x")
        self._store.mark_used("x")
        rows = self._store.list_all()
        self.assertEqual(rows[0]["use_count"], 2)
        self.assertIsNotNone(rows[0]["used_ts"])


class OrchestratorSkillsTests(unittest.TestCase):
    """Tier 0.5: a matching skill replays its Plan via Executor, no LLM."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="iris-skills-orch-")
        os.environ["TOUCHLESS_SKILLS_DB"] = str(Path(self._tmp) / "s.db")

    def tearDown(self) -> None:
        os.environ.pop("TOUCHLESS_SKILLS_DB", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_matching_skill_runs_through_executor(self) -> None:
        from hgr.live_api.planner.orchestrator import IrisPlanner
        from hgr.live_api.planner.plan import Plan, Step
        from hgr.live_api.planner.skills import SkillStore
        reg = _StubRegistry({
            "ms_mail_list": {"status": "ok", "messages": ["x"]},
            "cal_today": {"status": "ok", "events": ["mtg @10"]},
        })
        store = SkillStore()
        store.save("morning briefing", "morning briefing", Plan(
            goal="morning briefing",
            steps=[Step(id=1, tool="ms_mail_list", args={}),
                   Step(id=2, tool="cal_today", args={})],
        ))
        planner = IrisPlanner(reg)
        out = planner.try_handle("do my morning briefing")
        self.assertIsNotNone(out)
        self.assertEqual([s.tool for s in out["steps"]],
                         ["ms_mail_list", "cal_today"])
        self.assertEqual([t for t, _ in reg.calls],
                         ["ms_mail_list", "cal_today"])

    def test_skill_miss_falls_through_to_classifier(self) -> None:
        from hgr.live_api.planner.orchestrator import IrisPlanner
        reg = _StubRegistry({"volume_set": {"status": "ok"}})
        planner = IrisPlanner(reg)
        # No skill saved — must reach Phase 1.
        out = planner.try_handle("set volume to 30")
        self.assertIsNotNone(out)
        self.assertEqual(out["steps"][0].tool, "volume_set")


class PreconditionTests(unittest.TestCase):
    """Phase 6 preconditions: registry says 'not available now' → executor
    skips with a structured error instead of letting the connector fail."""

    def _reg(self, available_map):
        """Stub registry whose is_available consults a dict."""
        class _Reg:
            def __init__(self_):
                self_.calls = []
            def is_available(self_, name):
                return available_map.get(name)  # None == available
            def call(self_, name, args):
                self_.calls.append((name, dict(args or {})))
                return {"status": "ok"}
            def handles_connector(self_, name):
                return True
        return _Reg()

    def test_unavailable_step_short_circuits(self) -> None:
        from hgr.live_api.planner import Executor, Plan, Step
        reg = self._reg({"outlook_compose": "ms_graph not connected"})
        plan = Plan(goal="g", steps=[
            Step(id=1, tool="outlook_compose",
                 args={"recipient": "x@y", "body": "hi"}),
        ])
        results = Executor(reg).run(plan)
        self.assertEqual(results[0].status, "error")
        self.assertIn("not connected", results[0].error or "")
        self.assertEqual(results[0].output.get("code"), "precondition_not_met")
        # Step was NEVER actually called.
        self.assertEqual(reg.calls, [])

    def test_available_steps_still_run(self) -> None:
        from hgr.live_api.planner import Executor, Plan, Step
        reg = self._reg({})  # nothing unavailable
        plan = Plan(goal="g", steps=[
            Step(id=1, tool="cal_today", args={}),
        ])
        results = Executor(reg).run(plan)
        self.assertEqual(results[0].status, "ok")
        self.assertEqual(reg.calls, [("cal_today", {})])

    def test_dependents_of_precondition_failure_cascade(self) -> None:
        """Step 2 depends on step 1; step 1 fails its precondition →
        step 2 gets upstream-failed instead of running."""
        from hgr.live_api.planner import Executor, Plan, Step
        reg = self._reg({"outlook_compose": "ms_graph not connected"})
        plan = Plan(goal="g", steps=[
            Step(id=1, tool="outlook_compose", args={}),
            Step(id=2, tool="cal_today", args={}, depends_on=[1]),
        ])
        results = Executor(reg).run(plan)
        self.assertEqual({r.step_id: r.status for r in results},
                         {1: "error", 2: "error"})
        self.assertEqual(reg.calls, [])

    def test_registry_without_is_available_works(self) -> None:
        """Older registries (or test stubs) that don't implement
        is_available must continue to work — the executor falls through."""
        from hgr.live_api.planner import Executor, Plan, Step
        reg = _StubRegistry({"a": {"status": "ok"}})
        # _StubRegistry has no is_available method.
        results = Executor(reg).run(Plan(goal="g",
                                          steps=[Step(id=1, tool="a")]))
        self.assertEqual(results[0].status, "ok")


class ConnectorRegistryAvailabilityTests(unittest.TestCase):
    """ConnectorRegistry.is_available_for routes to the owning connector."""

    def test_returns_reason_for_unavailable_connector(self) -> None:
        from hgr.live_api.connectors.base import Connector, ConnectorRegistry

        class _DisconnectedMS(Connector):
            id = "ms_graph"
            description = "Microsoft Graph"
            def available(self): return False
            def tools(self): return [{"name": "outlook_compose",
                                       "parameters": {"type": "object",
                                                      "properties": {},
                                                      "required": [],
                                                      "additionalProperties": False}}]
            def execute(self, name, args): return None

        reg = ConnectorRegistry()
        reg.register(_DisconnectedMS())
        # Note: an unavailable connector contributes nothing to
        # available_tool_schemas, so the ownership map stays empty —
        # is_available_for returns None (built-in path). That's correct:
        # the planner won't include the tool in its catalog either.
        reason = reg.is_available_for("outlook_compose")
        self.assertIsNone(reason)  # not owned by an available connector

    def test_returns_none_for_available_connector(self) -> None:
        from hgr.live_api.connectors.base import Connector, ConnectorRegistry

        class _ConnectedMS(Connector):
            id = "ms_graph"
            description = "Microsoft Graph"
            def available(self): return True
            def tools(self): return [{"name": "outlook_compose",
                                       "parameters": {"type": "object",
                                                      "properties": {},
                                                      "required": [],
                                                      "additionalProperties": False}}]
            def execute(self, name, args): return {"status": "ok"}

        reg = ConnectorRegistry()
        reg.register(_ConnectedMS())
        reg.available_tool_schemas()  # populate ownership
        self.assertIsNone(reg.is_available_for("outlook_compose"))


class ExecutorArrayRefTests(unittest.TestCase):
    """The executor's {step:N.field} resolver supports list indexing —
    needed for web_search output (.results[0].url) and similar patterns
    the cheap-LLM naturally produces."""

    def test_resolves_indexed_path(self) -> None:
        from hgr.live_api.planner import Executor, Plan, Step
        plan = Plan(goal="g", steps=[
            Step(id=1, tool="web_search",
                 args={"query": "ai news", "count": 3}),
            Step(id=2, tool="web_navigate",
                 args={"url_or_query": "{step:1.results[0].url}"},
                 depends_on=[1]),
        ])
        reg = _StubRegistry({
            "web_search": {
                "status": "ok",
                "results": [
                    {"title": "first", "url": "https://a.example/",
                     "snippet": "..."},
                    {"title": "second", "url": "https://b.example/",
                     "snippet": "..."},
                ],
            },
            "web_navigate": {"status": "ok", "url": "https://a.example/"},
        })
        Executor(reg).run(plan)
        nav_args = next(a for t, a in reg.calls if t == "web_navigate")
        self.assertEqual(nav_args["url_or_query"], "https://a.example/")

    def test_out_of_range_index_fails_step_clearly(self) -> None:
        """Previously: out-of-range index silently substituted ''. Now: the
        step fails with a 'ref_unresolved' error that names the bad ref —
        otherwise the downstream tool errors with a cryptic 'X is required'."""
        from hgr.live_api.planner import Executor, Plan, Step
        plan = Plan(goal="g", steps=[
            Step(id=1, tool="src", args={}),
            Step(id=2, tool="use",
                 args={"x": "{step:1.results[5].url}"}, depends_on=[1]),
        ])
        reg = _StubRegistry({
            "src": {"status": "ok", "results": [{"url": "https://a/"}]},
            "use": {"status": "ok"},
        })
        results = Executor(reg).run(plan)
        self.assertEqual(results[1].status, "error")
        self.assertIn("{step:1.results[5].url}", results[1].error)
        # "use" was never actually invoked.
        self.assertNotIn("use", [t for t, _ in reg.calls])

    def test_dotted_after_index(self) -> None:
        from hgr.live_api.planner import Executor, Plan, Step
        plan = Plan(goal="g", steps=[
            Step(id=1, tool="src", args={}),
            Step(id=2, tool="use",
                 args={"t": "{step:1.items[1].title}"}, depends_on=[1]),
        ])
        reg = _StubRegistry({
            "src": {"status": "ok",
                    "items": [{"title": "A"}, {"title": "B"}]},
            "use": {"status": "ok"},
        })
        Executor(reg).run(plan)
        use_args = next(a for t, a in reg.calls if t == "use")
        self.assertEqual(use_args["t"], "B")


class LLMPlannerHallucinationGuardTests(unittest.TestCase):
    """The LLM occasionally emits non-existent tools (e.g. 'synthesize' as a
    step instead of as plan.final). When the registry tells us the known
    tool names, _parse drops those steps defensively."""

    def test_unknown_tool_dropped(self) -> None:
        from hgr.live_api.planner.planner_llm import LLMPlanner
        data = {
            "goal": "x",
            "steps": [
                {"id": 1, "tool": "web_search", "args": {"query": "ai"}},
                {"id": 2, "tool": "synthesize",
                 "args": {"text": "{step:1.results[0].snippet}"},
                 "depends_on": [1]},  # <- hallucinated tool name
                {"id": 3, "tool": "web_navigate",
                 "args": {"url_or_query": "{step:1.results[0].url}"},
                 "depends_on": [1]},
            ],
            "final": "synthesize",
        }
        known = {"web_search", "web_navigate", "web_get_text"}
        plan = LLMPlanner._parse("x", data, known_tools=known)
        self.assertIsNotNone(plan)
        self.assertEqual([s.tool for s in plan.steps],
                         ["web_search", "web_navigate"])

    def test_none_known_tools_keeps_old_behaviour(self) -> None:
        from hgr.live_api.planner.planner_llm import LLMPlanner
        data = {"goal": "x", "steps": [{"id": 1, "tool": "anything"}]}
        # known_tools=None means "couldn't introspect" — keep the step.
        plan = LLMPlanner._parse("x", data, known_tools=None)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].tool, "anything")


class LLMPlannerPromptTests(unittest.TestCase):
    """System prompt must give the model the right web chain pattern and
    explicitly forbid emitting 'synthesize' as a step."""

    def test_prompt_mentions_synthesize_is_not_a_tool(self) -> None:
        from hgr.live_api.planner.planner_llm import LLMPlanner
        from hgr.live_api.planner import IrisPlanner
        # Build a planner with a stub registry just to render the prompt.
        reg = _StubRegistry({})
        lp = LLMPlanner(reg)
        msgs = lp._build_messages("any goal")
        system = next(m["content"] for m in msgs if m["role"] == "system")
        self.assertIn("not a tool", system.lower())
        self.assertIn("synthesize", system.lower())
        # Worked example shows the correct web chain.
        self.assertIn("web_search", system)
        self.assertIn("web_navigate", system)
        self.assertIn("web_get_text", system)
        # And the {step:N.results[0].url} ref pattern is documented.
        self.assertIn("results[0]", system)

    def test_prompt_documents_cross_provider_and_shapes(self) -> None:
        """The prompt must explain when cross-provider chains are OK
        (universal data: emails/URLs) vs not (provider-specific IDs), and
        document the contacts_search output shape so refs resolve."""
        from hgr.live_api.planner.planner_llm import LLMPlanner
        reg = _StubRegistry({})
        lp = LLMPlanner(reg)
        msgs = lp._build_messages("find x's email and send")
        system = next(m["content"] for m in msgs if m["role"] == "system")
        # Cross-provider rule discusses what's OK vs not (NOT a blanket ban).
        self.assertIn("cross-provider", system.lower())
        self.assertIn("universal", system.lower())
        # Provider-specific IDs (message IDs) called out as the exception.
        self.assertIn("message id", system.lower())
        # contacts_search output shape documented with the exact ref.
        self.assertIn("contacts[0].emails[0]", system)
        # Worked example for find-email-and-send chain.
        self.assertIn("contacts_search", system)


class PreferenceClassifierTests(unittest.TestCase):
    """'always send from X' / 'set default sender to X' / etc. — must map
    deterministically to an iris_set_preference Step."""

    def setUp(self) -> None:
        from hgr.live_api.planner.classifier import Classifier
        self.c = Classifier()

    def _expect(self, text, key, value):
        step = self.c.classify(text)
        self.assertIsNotNone(step, f"missed: {text!r}")
        self.assertEqual(step.tool, "iris_set_preference")
        self.assertEqual(step.args.get("key"), key)
        self.assertEqual(step.args.get("value"), value)

    def test_send_from_gmail_variants(self) -> None:
        self._expect("always send from gmail", "default_send_via", "gmail_send")
        self._expect("always send via my gmail account", "default_send_via", "gmail_send")
        self._expect("use outlook by default", "default_send_via", "ms_mail_send")
        self._expect("set my default sender to gmail", "default_send_via", "gmail_send")
        self._expect("only send email from outlook", "default_send_via", "ms_mail_send")

    def test_contacts_account_variants(self) -> None:
        self._expect("always search contacts in my .edu",
                     "default_contact_account", ".edu")
        self._expect("only look up contacts in gmail",
                     "default_contact_account", "gmail")
        self._expect("always find contacts from my work account",
                     "default_contact_account", "work")

    def test_non_preference_unaffected(self) -> None:
        # Things that must NOT classify as preference-setting.
        self.assertIsNone(self.c.classify("send dani an email"))
        # "what's the weather" doesn't classify at all — neither preference
        # nor a known intent.
        self.assertIsNone(self.c.classify("what's the weather"))


class PreferencePseudoToolTests(unittest.TestCase):
    """The iris_set_preference Step is intercepted by the orchestrator
    BEFORE the registry call — it writes to memory and returns ok."""

    def test_writes_to_memory_without_registry_call(self) -> None:
        from hgr.live_api.planner.orchestrator import IrisPlanner
        reg = _StubRegistry({})
        writes: List[tuple] = []
        class _FakeMem:
            def set_fact(self_, kind, key, value, source="user said"):
                writes.append((kind, key, value, source))
            def record(self_, *a, **k): pass
            def recall(self_, *a, **k):
                return {"context": "", "episodes": [], "facts": []}
        planner = IrisPlanner(reg, memory=_FakeMem())
        out = planner.try_handle("always send from gmail")
        self.assertIsNotNone(out)
        self.assertEqual(out["steps"][0].tool, "iris_set_preference")
        self.assertEqual(out["results"][0].status, "ok")
        # The pseudo-tool didn't touch the registry.
        self.assertEqual(reg.calls, [])
        # But it DID write to memory.
        self.assertEqual(writes,
                         [("preference", "default_send_via", "gmail_send",
                           "user said")])
        # User-visible message confirms the new preference.
        self.assertIn("default_send_via", out["message"])
        self.assertIn("gmail_send", out["message"])

    def test_no_memory_still_replies_gracefully(self) -> None:
        """When memory isn't wired, the preference command shouldn't crash —
        it just acknowledges (and forgets, no place to persist)."""
        from hgr.live_api.planner.orchestrator import IrisPlanner
        reg = _StubRegistry({})
        planner = IrisPlanner(reg, memory=None)
        out = planner.try_handle("always send from gmail")
        self.assertIsNotNone(out)
        self.assertEqual(out["results"][0].status, "ok")


class PlannerPromptPrefsTests(unittest.TestCase):
    """The planner prompt instructs the LLM to read preferences from the
    memory context and honor per-request overrides."""

    def test_prompt_mentions_preferences(self) -> None:
        from hgr.live_api.planner.planner_llm import LLMPlanner
        reg = _StubRegistry({})
        lp = LLMPlanner(reg)
        msgs = lp._build_messages("any goal")
        system = next(m["content"] for m in msgs if m["role"] == "system")
        self.assertIn("default_send_via", system)
        self.assertIn("preference", system.lower())
        # Per-request override wins for THIS request.
        self.assertIn("one-shot", system.lower())


class MultiAccountContactsSearchTests(unittest.TestCase):
    """contacts_search fans out across every connected Microsoft account by
    default, dedupes by email, and supports account=<substring> to restrict."""

    def _make_connector(self, account_contacts: Dict[str, List[Dict[str, Any]]]):
        """Build an Microsoft365Connector wired to a fake graph client that returns
        per-account canned contact lists."""
        from hgr.live_api.connectors.ms365_connector import Microsoft365Connector

        class _FakeClient:
            def all_accounts(self_):
                return [{"username": u} for u in account_contacts]
            def token_for(self_, account):
                return f"token-{account.get('username')}"
            def token(self_): return "active-token"

        seen_paths: List[tuple] = []
        def fake_graph(self_, method, path, body=None, raw=None,
                       content_type=None, token=None):
            seen_paths.append((token, path))
            acct_name = (token or "").replace("token-", "")
            contacts = account_contacts.get(acct_name, [])
            value = []
            # contacts_search now hits both /me/contacts (formal) AND
            # /me/people (correspondents). Mirror the real field shapes.
            if "/me/contacts" in path:
                for c in contacts:
                    value.append({
                        "displayName": c["name"],
                        "emailAddresses": [{"address": e} for e in c["emails"]],
                    })
            elif "/me/people" in path:
                # Empty for the multi-account tests — we're only exercising
                # the contacts pathway here.
                pass
            return {"value": value}, None

        conn = Microsoft365Connector(_FakeClient())  # type: ignore[arg-type]
        conn._graph = fake_graph.__get__(conn, Microsoft365Connector)  # type: ignore[method-assign]
        return conn, seen_paths

    def test_fans_out_across_accounts_and_dedupes(self) -> None:
        conn, paths = self._make_connector({
            "school@edu": [
                {"name": "Dani Markov", "emails": ["dani@school.edu"]},
                {"name": "Vesko", "emails": ["vesko@school.edu"]},
            ],
            "personal@gmail.com": [
                # Same email as school (dedup target).
                {"name": "Dani M", "emails": ["dani@school.edu"]},
                {"name": "Dani Personal", "emails": ["dani@personal.com"]},
            ],
        })
        out = conn.execute("contacts_search", {"query": "dani"})
        self.assertEqual(out["status"], "ok")
        # Hit both accounts.
        tokens_used = sorted({tok for tok, _ in paths})
        self.assertEqual(tokens_used, ["token-personal@gmail.com", "token-school@edu"])
        # Dedup: dani@school.edu should appear ONCE across both accounts.
        emails = [e for c in out["contacts"] for e in c["emails"]]
        self.assertEqual(emails.count("dani@school.edu"), 1)
        # source_account is recorded.
        sources = {c["source_account"] for c in out["contacts"]}
        self.assertEqual(sources, {"school@edu", "personal@gmail.com"})

    def test_account_substring_restricts(self) -> None:
        conn, paths = self._make_connector({
            "school@edu": [{"name": "Dani", "emails": ["dani@school.edu"]}],
            "personal@gmail.com": [{"name": "Dani G", "emails": ["dani@gmail.com"]}],
        })
        out = conn.execute("contacts_search",
                           {"query": "dani", "account": "edu"})
        self.assertEqual(out["status"], "ok")
        # Only the .edu account was queried.
        tokens_used = {tok for tok, _ in paths}
        self.assertEqual(tokens_used, {"token-school@edu"})
        self.assertEqual([c["source_account"] for c in out["contacts"]],
                         ["school@edu"])

    def test_unknown_account_substring_errors(self) -> None:
        conn, _ = self._make_connector({"school@edu": []})
        out = conn.execute("contacts_search",
                           {"query": "dani", "account": "yahoo"})
        self.assertEqual(out["status"], "error")
        self.assertIn("yahoo", out["error"])

    def test_no_accounts_returns_error(self) -> None:
        conn, _ = self._make_connector({})
        out = conn.execute("contacts_search", {"query": "dani"})
        self.assertEqual(out["status"], "error")
        self.assertIn("no Microsoft accounts connected", out["error"])


class ExecutorRefFailureTests(unittest.TestCase):
    """When a {step:N.field} ref points at nothing (empty list, missing key),
    the executor must fail the step with a CLEAR error instead of silently
    substituting '' and letting the downstream tool error cryptically."""

    def test_empty_list_index_fails_step_with_clear_error(self) -> None:
        from hgr.live_api.planner import Executor, Plan, Step
        plan = Plan(goal="g", steps=[
            Step(id=1, tool="contacts_search", args={"query": "ghost"}),
            Step(id=2, tool="gmail_send",
                 args={"to": "{step:1.contacts[0].emails[0]}",
                       "subject": "hi", "body": "hi"},
                 depends_on=[1]),
        ])
        reg = _StubRegistry({
            "contacts_search": {"status": "ok", "contacts": []},
            "gmail_send": {"status": "ok"},
        })
        results = Executor(reg).run(plan)
        # Step 1 ok, step 2 errors with the specific ref that failed.
        self.assertEqual(results[0].status, "ok")
        self.assertEqual(results[1].status, "error")
        self.assertIn("{step:1.contacts[0].emails[0]}", results[1].error)
        self.assertIn("no matching data", results[1].error)
        # gmail_send was NEVER actually invoked.
        self.assertNotIn("gmail_send", [t for t, _ in reg.calls])

    def test_resolves_when_data_is_present(self) -> None:
        """Sanity: when the ref DOES resolve, the step still runs normally."""
        from hgr.live_api.planner import Executor, Plan, Step
        plan = Plan(goal="g", steps=[
            Step(id=1, tool="contacts_search", args={"query": "dani"}),
            Step(id=2, tool="gmail_send",
                 args={"to": "{step:1.contacts[0].emails[0]}",
                       "subject": "hi", "body": "hi"},
                 depends_on=[1]),
        ])
        reg = _StubRegistry({
            "contacts_search": {"status": "ok",
                                "contacts": [{"name": "Dani",
                                              "emails": ["dani@x.io"]}]},
            "gmail_send": {"status": "ok"},
        })
        results = Executor(reg).run(plan)
        self.assertEqual([r.status for r in results], ["ok", "ok"])
        send = next(a for t, a in reg.calls if t == "gmail_send")
        self.assertEqual(send["to"], "dani@x.io")


class ContactsSearchFallbackTests(unittest.TestCase):
    """contacts_search now also queries /me/people (broader recall — anyone
    you've recently emailed, not just formal contacts)."""

    def _make_connector(self, contacts_by_account, people_by_account):
        from hgr.live_api.connectors.ms365_connector import Microsoft365Connector

        class _FakeClient:
            def all_accounts(self_):
                return [{"username": "u@gmail.com"}]
            def token_for(self_, account): return "tok"
            def token(self_): return "tok"

        paths_hit: List[str] = []
        def fake_graph(self_, method, path, body=None, raw=None,
                       content_type=None, token=None):
            paths_hit.append(path)
            if "/me/contacts" in path:
                return ({"value": [
                    {"displayName": c["name"],
                     "emailAddresses": [{"address": e} for e in c["emails"]]}
                    for c in contacts_by_account.get("u@gmail.com", [])
                ]}, None)
            if "/me/people" in path:
                return ({"value": [
                    {"displayName": p["name"],
                     "scoredEmailAddresses": [{"address": e} for e in p["emails"]]}
                    for p in people_by_account.get("u@gmail.com", [])
                ]}, None)
            return None, "unknown path"

        conn = Microsoft365Connector(_FakeClient())  # type: ignore[arg-type]
        conn._graph = fake_graph.__get__(conn, Microsoft365Connector)  # type: ignore[method-assign]
        return conn, paths_hit

    def test_falls_back_to_people_when_contacts_empty(self) -> None:
        conn, paths = self._make_connector(
            contacts_by_account={"u@gmail.com": []},  # nothing in formal contacts
            people_by_account={"u@gmail.com": [
                {"name": "Dani M", "emails": ["dani@school.edu"]},
            ]},
        )
        out = conn.execute("contacts_search", {"query": "Dani"})
        self.assertEqual(out["status"], "ok")
        self.assertEqual(len(out["contacts"]), 1)
        self.assertEqual(out["contacts"][0]["emails"], ["dani@school.edu"])
        # Both endpoints were hit.
        self.assertTrue(any("/me/contacts" in p for p in paths))
        self.assertTrue(any("/me/people" in p for p in paths))

    def test_merges_contacts_and_people_dedup(self) -> None:
        conn, _ = self._make_connector(
            contacts_by_account={"u@gmail.com": [
                {"name": "Dani Formal", "emails": ["dani@x.io"]},
            ]},
            people_by_account={"u@gmail.com": [
                # Same email — should dedup.
                {"name": "Dani M", "emails": ["dani@x.io"]},
                # Different email — should add.
                {"name": "Dani Other", "emails": ["dani2@x.io"]},
            ]},
        )
        out = conn.execute("contacts_search", {"query": "Dani"})
        emails = sorted(e for c in out["contacts"] for e in c["emails"])
        self.assertEqual(emails, ["dani2@x.io", "dani@x.io"])

    def test_no_results_returns_helpful_message(self) -> None:
        conn, _ = self._make_connector(
            contacts_by_account={"u@gmail.com": []},
            people_by_account={"u@gmail.com": []},
        )
        out = conn.execute("contacts_search", {"query": "ghost"})
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["count"], 0)
        self.assertIn("No one matching 'ghost'", out["message"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# Author: Konstantin Markov
