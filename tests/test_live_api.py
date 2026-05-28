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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# Author: Konstantin Markov
