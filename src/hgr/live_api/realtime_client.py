"""WebSocket client for the OpenAI Realtime API.

Threading model:
  * `start()` spawns one background reader thread that pumps events
    from the websocket and forwards them to per-event callbacks.
  * Sends are made directly from any thread (websocket-client's
    `WebSocketApp.sock.send` is thread-safe enough for our usage —
    we serialize writes with a lock).
  * The reader thread is a plain daemon thread; the manager owns its
    lifecycle.

We deliberately use `websocket-client` (the sync library) instead of
`websockets` (asyncio) so we don't need to run an event loop inside
Qt. This module degrades gracefully if `websocket-client` is not
installed — the manager surfaces a clear error in the UI.

Reference for the protocol:
  https://platform.openai.com/docs/api-reference/realtime
The exact event names ('session.update', 'response.create',
'input_audio_buffer.append', 'response.function_call_arguments.done',
...) come from the public Realtime spec.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from . import cortex_emit
from .config import LiveApiConfig
from .live_api_logger import LiveApiLogger


# Callback types
EventCallback = Callable[[Dict[str, Any]], None]
ConnectedCallback = Callable[[], None]
ClosedCallback = Callable[[Optional[str]], None]
ErrorCallback = Callable[[str], None]


class RealtimeClient:
    """Sync WebSocket client wrapping the OpenAI Realtime API."""

    def __init__(
        self,
        *,
        config: LiveApiConfig,
        logger: LiveApiLogger,
        tools: List[Dict[str, Any]],
        system_instructions: str,
        on_event: EventCallback,
        on_connected: Optional[ConnectedCallback] = None,
        on_closed: Optional[ClosedCallback] = None,
        on_error: Optional[ErrorCallback] = None,
        text_only: bool = False,
        voice_output: bool = False,
    ) -> None:
        self._config = config
        self._logger = logger
        self._tools = tools
        self._system_instructions = system_instructions
        # Typed-command mode: no mic, no input audio config.
        self._text_only = bool(text_only)
        # Voice output: when True AND text_only, request text+audio out
        # (no mic). When True AND NOT text_only, normal voice mode
        # already includes audio output, so this is a no-op there.
        self._voice_output = bool(voice_output)
        self._on_event = on_event
        self._on_connected = on_connected
        self._on_closed = on_closed
        self._on_error = on_error

        self._ws = None
        self._send_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._connected = False

    # ---- lifecycle ----

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> bool:
        if self._reader_thread is not None and self._reader_thread.is_alive():
            return True

        if not self._config.api_key:
            self._fire_error("missing OPENAI_API_KEY environment variable")
            return False

        try:
            import websocket  # type: ignore
        except Exception as exc:
            self._logger.exception("websocket_import_failed", exc)
            self._fire_error(
                "websocket-client package not installed (pip install websocket-client)"
            )
            return False

        self._stop_requested = False
        self._reader_thread = threading.Thread(
            target=self._run, name="LiveApiRealtime", daemon=True
        )
        self._reader_thread.start()
        return True

    def stop(self) -> None:
        self._stop_requested = True
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        self._connected = False

    def join(self, timeout: float = 2.0) -> None:
        """Wait for the reader thread to exit. Safe to call after stop()."""
        thread = self._reader_thread
        if thread is None or not thread.is_alive():
            return
        try:
            thread.join(timeout=timeout)
        except Exception:
            pass

    def _run(self) -> None:
        import websocket  # local — already validated in start()

        attempts = 0
        backoff = max(0.5, float(self._config.reconnect_backoff_sec))
        while not self._stop_requested and attempts <= self._config.reconnect_max_attempts:
            attempts += 1
            url = f"{self._config.realtime_url}?model={self._config.model}"
            # GA Realtime API: the OpenAI-Beta header is gone (sending it
            # gets you the retired beta shape -> beta_api_shape_disabled).
            headers = [
                f"Authorization: Bearer {self._config.api_key}",
            ]
            self._logger.event(
                "ws_connecting",
                attempt=attempts,
                url=self._config.realtime_url,
                model=self._config.model,
            )
            ws = None
            try:
                ws = websocket.create_connection(url, header=headers, timeout=10)
                self._ws = ws
                self._connected = True
                self._logger.event("ws_connected", attempt=attempts)
                self._send_session_update()
                if self._on_connected:
                    try:
                        self._on_connected()
                    except Exception as exc:
                        self._logger.exception("on_connected_callback_failed", exc)
                self._read_loop(ws)
            except Exception as exc:
                self._logger.exception("ws_run_error", exc, attempt=attempts)
                self._fire_error(f"{type(exc).__name__}: {exc}")
            finally:
                self._connected = False
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                self._ws = None

            if self._stop_requested:
                break
            if attempts <= self._config.reconnect_max_attempts:
                self._logger.event("ws_reconnect_wait", seconds=backoff, next_attempt=attempts + 1)
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

        if self._on_closed:
            try:
                self._on_closed(None if not self._stop_requested else "stopped")
            except Exception as exc:
                self._logger.exception("on_closed_callback_failed", exc)

    def _read_loop(self, ws) -> None:
        import websocket as _ws_mod  # for the timeout exception type

        while not self._stop_requested:
            try:
                raw = ws.recv()
            except _ws_mod.WebSocketTimeoutException:
                # The socket-level read timeout (set on connect) firing is
                # NOT a disconnect — a realtime session sends nothing for
                # long idle stretches. Keep waiting; just loop back so we
                # re-check the stop flag. Previously this fell into the
                # generic handler below and forced a reconnect every ~10s,
                # which exhausted the retry budget and dropped the session.
                continue
            except Exception as exc:
                self._logger.exception("ws_recv_failed", exc)
                return
            if not raw:
                self._logger.event("ws_recv_empty")
                return
            try:
                event = json.loads(raw)
            except Exception:
                self._logger.warning("ws_recv_unparseable", length=len(raw))
                continue
            kind = str(event.get("type") or "<no-type>")
            self._logger.event("ws_recv", event_type=kind)
            # Cortex viz: pulse voice-mic -> core when a user transcript
            # arrives, and core -> voice-tts when an assistant audio reply
            # completes. Best-effort — must NEVER break realtime dispatch.
            try:
                if kind == "conversation.item.input_audio_transcription.completed":
                    cortex_emit.edge_pulse("voice-mic", "core",
                                           color="cyan", duration_ms=100)
                elif kind in {"response.output_audio.done",
                              "response.audio.done",
                              "response.output_text.done",
                              "response.output_audio_transcript.done",
                              "response.audio_transcript.done",
                              "response.text.done"}:
                    # Pick target by output modality: voice-tts when
                    # speaking, rt-output for text-only realtime replies.
                    # Mirrors LiveApiManager._emit_reply_output_pulse so
                    # both code paths light up the same node.
                    # FULL output path — matches
                    # LiveApiManager._emit_reply_output_pulse. Always
                    # fire core → cap-realtime → rt-output; when voice
                    # is on, also core → cap-voice → voice-tts.
                    try:
                        cortex_emit.edge_pulse("core", "cap-realtime",
                                               color="magenta", duration_ms=280)
                    except Exception:
                        pass
                    try:
                        cortex_emit.edge_pulse("cap-realtime", "rt-output",
                                               color="magenta", duration_ms=280)
                    except Exception:
                        pass
                    if self._voice_output:
                        try:
                            cortex_emit.edge_pulse("core", "cap-voice",
                                                   color="magenta", duration_ms=280)
                        except Exception:
                            pass
                        try:
                            cortex_emit.edge_pulse("cap-voice", "voice-tts",
                                                   color="magenta", duration_ms=280)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                self._on_event(event)
            except Exception as exc:
                self._logger.exception("event_handler_failed", exc, event_type=kind)

    # ---- send helpers ----

    def _send(self, payload: Dict[str, Any]) -> bool:
        if self._ws is None or not self._connected:
            self._logger.warning("ws_send_skipped_not_connected", event_type=payload.get("type"))
            return False
        try:
            data = json.dumps(payload)
        except Exception as exc:
            self._logger.exception("ws_send_serialize_failed", exc)
            return False
        with self._send_lock:
            try:
                self._ws.send(data)
            except Exception as exc:
                self._logger.exception("ws_send_failed", exc)
                return False
        self._logger.event("ws_send", event_type=payload.get("type"), bytes=len(data))
        return True

    def _send_session_update(self) -> None:
        """Initial session config in the GA Realtime shape.

        GA differences from the retired beta:
          * `session.type` ("realtime") is required and the model lives
            in the session object.
          * `modalities` -> `output_modalities`.
          * audio config moved under `session.audio.input` /
            `session.audio.output` (format is now an object, not the
            "pcm16" string).

        Text-only (typed-command) sessions skip the audio block entirely
        and request text output — no mic, no voice synthesis.
        """
        session: Dict[str, Any] = {
            "type": "realtime",
            "model": self._config.model,
            "instructions": self._system_instructions,
            "tools": self._tools,
            "tool_choice": "auto",
        }
        # Unified audio path: by default Realtime emits TEXT ONLY and the
        # manager funnels every assistant reply (LLM, Layer-0 router,
        # Layer-1 planner, task queue) through the local TTS pipeline
        # (_speak_text -> _http_openai_tts) using the user's configured
        # voice. This eliminates the prosodic mismatch between Realtime's
        # streaming-audio voice and TTS REST's voice.
        #
        # Opt-out: set TOUCHLESS_REALTIME_AUDIO=1 to fall back to the
        # legacy behaviour where the model itself streams PCM (lower
        # latency, but inconsistent voice character across reply sources).
        realtime_audio = os.environ.get("TOUCHLESS_REALTIME_AUDIO", "0") == "1"

        if self._text_only and not self._voice_output:
            # Typed-command, no-speech mode: text out, no mic.
            session["output_modalities"] = ["text"]
        elif self._text_only and self._voice_output:
            # Typed-command + spoken-reply. Default = text only (TTS in
            # the manager handles speech); opt-in = legacy text+audio.
            if realtime_audio:
                rate = int(self._config.audio_sample_rate)
                session["output_modalities"] = ["audio"]
                session["audio"] = {
                    "output": {
                        "format": {"type": "audio/pcm", "rate": rate},
                        "voice": getattr(self._config, "voice", "marin"),
                    },
                }
            else:
                session["output_modalities"] = ["text"]
        else:
            # Normal voice path (mic input). Default = text out + TTS in
            # the manager; opt-in = legacy text+audio streaming. The
            # input audio config is kept either way so the user can speak.
            rate = int(self._config.audio_sample_rate)
            if realtime_audio:
                session["output_modalities"] = ["audio"]
                session["audio"] = {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": rate},
                        "transcription": {"model": "whisper-1"},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 2500,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": rate},
                        "voice": getattr(self._config, "voice", "marin"),
                    },
                }
            else:
                session["output_modalities"] = ["text"]
                session["audio"] = {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": rate},
                        "transcription": {"model": "whisper-1"},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 2500,
                        },
                    },
                }
        self._send({"type": "session.update", "session": session})

    def update_tools(self, tools: List[Dict[str, Any]]) -> bool:
        """Replace the session's tool list mid-conversation and re-push it.

        The capability-search router calls this to load a connector's tools
        on demand (keeping the initial tool list lean). A `session.update`
        with only `tools` merges into the live session — instructions/audio
        are untouched. Safe before connect: we just stash the list and the
        initial session.update sends it.
        """
        self._tools = list(tools or [])
        if not self._connected:
            return False
        return self._send({
            "type": "session.update",
            "session": {"type": "realtime", "tools": self._tools},
        })

    def send_audio_chunk(self, pcm16_bytes: bytes) -> bool:
        if not pcm16_bytes:
            return True
        b64 = base64.b64encode(pcm16_bytes).decode("ascii")
        return self._send({"type": "input_audio_buffer.append", "audio": b64})

    def send_text_message(self, text: str) -> bool:
        return self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    def send_session_note(self, text: str) -> bool:
        """Inject a system-role note into the conversation WITHOUT triggering
        a response. Used by the iris planner to tell realtime what was just
        handled outside the model, so follow-ups like 'send him a thank you
        too' have the prior context to resolve."""
        text = (text or "").strip()
        if not text:
            return True
        return self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    def send_screen_image(self, jpeg_b64: str, *, caption: str = "") -> bool:
        """Send a screenshot as an image content part on a user message."""
        text_part = {"type": "input_text", "text": caption or "Current screen context."}
        # The Realtime API accepts `input_image` with `image` (data URL or
        # `image_url`). We use a data: URL to keep it self-contained.
        image_part = {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{jpeg_b64}",
        }
        return self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [text_part, image_part],
                },
            }
        )

    def send_tool_result(self, call_id: str, output: Dict[str, Any]) -> bool:
        return self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(output),
                },
            }
        )

    def request_response(self) -> bool:
        return self._send({"type": "response.create"})

    # ---- internal ----

    def _fire_error(self, message: str) -> None:
        self._logger.error("ws_client_error", message=message)
        if self._on_error:
            try:
                self._on_error(message)
            except Exception as exc:
                self._logger.exception("on_error_callback_failed", exc)

# Author: Konstantin Markov
