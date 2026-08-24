"""Browser-engine receiver for the pairing-code phone camera.

Validated to beat the pure-Python aiortc path: a hidden Chromium page
(QtWebEngine) HARDWARE-decodes the phone's WebRTC stream, draws each
frame to a canvas, JPEG-encodes it, and POSTs to a tiny 127.0.0.1 server
in the app. localhost is a secure context, so WebRTC works with no
certificate, and the frames land in the SAME PhoneCameraCapture sink the
QR flow + engine already consume.

  phone (touchless-control.com/connect)
      └─WebRTC─▶ hidden QtWebEngine page ─canvas→JPEG→POST /frame─▶
                local aiohttp server ─push_jpeg─▶ PhoneCameraCapture ─▶ engine

Public surface mirrors PhoneCameraServer (`capture`, `audio_source`,
`info`, `is_running`, `connected_clients`, `seconds_since_last_frame`,
`connected_phone_label`, `set_status_callback`, `start`, `stop`) so
main_window's adopt path uses it unchanged.

NOTE: start()/stop() must be called on the GUI thread (they create/destroy
a QWebEngineView). The local HTTP server runs on its own daemon thread.

Author: Konstantin Markov
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from aiohttp import web

from .audio_source import PhoneAudioSource
from .capture import PhoneCameraCapture

SIGNALING_WS = "wss://touchless-signaling.konstantinvmarkov.workers.dev/ws"
CONNECT_PAGE_URL = "https://touchless-control.com/connect"
_LOCAL_PORT = 8771  # local receiver server (distinct from the QR server's 8765)

StatusCallback = Callable[[str, dict], None]


def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"[phone-webengine {time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


# Receiver page loaded into the hidden QtWebEngine view. WebRTC host that
# hardware-decodes the phone stream, then ships frames to the app as JPEG
# over same-origin localhost. Honors the phone's Mirror toggle by flipping
# the canvas (so the flip is baked into what the engine receives).
_HOST_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  html,body{margin:0;background:#000;overflow:hidden}
  video{width:100%;height:100vh;object-fit:contain;display:block}
</style></head>
<body>
  <video id="v" autoplay playsinline muted></video>
  <canvas id="c" style="display:none"></canvas>
  <script>
    const SIG = "%SIG%";
    const ICE = [{ urls: "stun:stun.l.google.com:19302" },
                 { urls: "stun:stun1.l.google.com:19302" }];
    const code = new URLSearchParams(location.search).get("code");
    const v = document.getElementById("v");
    const c = document.getElementById("c");
    let ws, pc, inflight = false, mirror = false, ka = null;

    function connect() {
      ws = new WebSocket(`${SIG}?code=${code}&role=host`);
      // Keepalive: idle WebSockets get closed (~100s) which would otherwise
      // flap the signalling channel and churn the connection.
      ws.onopen = () => {
        clearInterval(ka);
        ka = setInterval(() => {
          try { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "ping" })); } catch (_) {}
        }, 25000);
      };
      ws.onclose = () => { clearInterval(ka); setTimeout(connect, 1500); };
      ws.onmessage = async (e) => {
        const m = JSON.parse(e.data);
        if (m.type === "ready") { if (!pc) makePeer(); }
        else if (m.type === "offer") {
          if (!pc) makePeer();
          await pc.setRemoteDescription(m.sdp);
          const a = await pc.createAnswer();
          await pc.setLocalDescription(a);
          ws.send(JSON.stringify({ type: "answer", sdp: pc.localDescription }));
        } else if (m.type === "ice" && pc && m.candidate) {
          try { await pc.addIceCandidate(m.candidate); } catch (_) {}
        } else if (m.type === "mirror") {
          mirror = !!m.on;
        }
        // NOTE: deliberately ignore "peer-left" — the phone's signalling
        // socket flapping must NOT tear down the live P2P media (that was
        // turning the view black on every blip). connectionstatechange
        // handles a genuine disconnect.
      };
    }
    function makePeer() {
      pc = new RTCPeerConnection({ iceServers: ICE });
      pc.onicecandidate = (e) => { if (e.candidate) ws.send(JSON.stringify({ type: "ice", candidate: e.candidate })); };
      pc.ontrack = (e) => { v.srcObject = e.streams[0]; pump(); startAudio(e.streams[0]); };
      // Inbound DataChannel from the phone for text commands.
      // The connect page (touchless-control.com/connect) opens a
      // channel named "commands" and sends one text frame per Send tap.
      // We POST each message body to /command on this local server; the
      // host's text-command callback dispatches it through the voice
      // processor on the GUI thread.
      //
      // Result toast return path: after dispatch, the GUI thread calls
      // _LocalServer.publish_command_result({ok, heard_text, ...}),
      // which fans out to the SSE /results stream we subscribe to
      // below. Each event is forwarded back through the SAME DataChannel
      // so the phone's "commands" channel receives a JSON
      // {kind:"command_result", ...} message and renders a toast.
      pc.ondatachannel = (e) => {
        try {
          const ch = e.channel;
          commandsChannel = ch;
          ch.onmessage = (msg) => {
            const text = (typeof msg.data === "string" ? msg.data : "").trim();
            if (!text) return;
            fetch("/command", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ text }),
            }).catch(() => {});
          };
          ch.onclose = () => { if (commandsChannel === ch) commandsChannel = null; };
        } catch (_) {}
      };
    }
    // Track the OPEN inbound DataChannel so result events can be
    // mirrored back to the phone. SSE listener below sends each
    // "command_result" event through this channel.
    let commandsChannel = null;
    function subscribeResults() {
      let es = null;
      try { es = new EventSource("/results"); } catch (_) { return; }
      es.onmessage = (evt) => {
        let payload = null;
        try { payload = JSON.parse(evt.data || ""); } catch (_) { return; }
        if (!payload || typeof payload !== "object") return;
        const ch = commandsChannel;
        if (ch && ch.readyState === "open") {
          try { ch.send(JSON.stringify(payload)); } catch (_) {}
        }
      };
      // EventSource auto-reconnects on transient failures. Long-lived
      // connection drops trigger a fresh subscribe via this handler.
      es.onerror = () => {
        try { es.close(); } catch (_) {}
        setTimeout(subscribeResults, 1500);
      };
    }
    subscribeResults();
    // Capture the phone's mic track as 48k mono Int16 PCM and POST it to
    // /audio (same wire format + sink the QR flow uses). Routed into voice
    // commands only when the user ticks "Use phone microphone".
    let audioStarted = false;
    function startAudio(stream) {
      if (audioStarted) return;
      try {
        const at = stream.getAudioTracks();
        if (!at.length) return;
        audioStarted = true;
        const ac = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
        // This page is loaded programmatically with no user gesture, so the
        // AudioContext starts suspended under Chromium's autoplay policy and
        // onaudioprocess would never fire. The host disables
        // PlaybackRequiresUserGesture on the QWebEngineView; resume() here
        // kicks the context into "running" so the mic PCM actually flows.
        ac.resume().catch(() => {});
        const src = ac.createMediaStreamSource(new MediaStream([at[0]]));
        const proc = ac.createScriptProcessor(4096, 1, 1);
        const mute = ac.createGain(); mute.gain.value = 0;  // process without audible echo
        proc.onaudioprocess = (ev) => {
          const f32 = ev.inputBuffer.getChannelData(0);
          const i16 = new Int16Array(f32.length);
          for (let i = 0; i < f32.length; i++) {
            let s = Math.max(-1, Math.min(1, f32[i]));
            i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          fetch("/audio", { method: "POST", body: i16.buffer }).catch(() => {});
        };
        src.connect(proc); proc.connect(mute); mute.connect(ac.destination);
      } catch (_) {}
    }
    function pump() {
      const ctx = c.getContext("2d");
      // Downscale before sending. Hand tracking downsamples to a small
      // input internally, so this doesn't hurt accuracy — but it cuts the
      // per-frame JPEG encode (here) + decode (the app) that competes with
      // the gesture engine for CPU when a hand is in frame. 640px (~360p)
      // closes most of the fps gap vs a local webcam. Raise MAX_W (e.g.
      // 854 or 1280) for a crisper live preview at the cost of some fps.
      const MAX_W = 640;
      function loop() {
        if (v.videoWidth && !inflight) {
          let dw = v.videoWidth, dh = v.videoHeight;
          if (dw > MAX_W) { dh = Math.round(dh * MAX_W / dw); dw = MAX_W; }
          c.width = dw; c.height = dh;
          if (mirror) { ctx.save(); ctx.translate(dw, 0); ctx.scale(-1, 1); ctx.drawImage(v, 0, 0, dw, dh); ctx.restore(); }
          else { ctx.drawImage(v, 0, 0, dw, dh); }
          inflight = true;
          c.toBlob((b) => {
            if (!b) { inflight = false; return; }
            fetch("/frame", { method: "POST", body: b }).catch(() => {}).finally(() => { inflight = false; });
          }, "image/jpeg", 0.72);
        }
        requestAnimationFrame(loop);
      }
      requestAnimationFrame(loop);
    }
    connect();
  </script>
</body></html>
""".replace("%SIG%", SIGNALING_WS)


@dataclass(frozen=True)
class WebEnginePhoneServerInfo:
    code: str
    connect_url: str


class _LocalServer:
    """127.0.0.1 HTTP server: serves the receiver page + accepts JPEG frames."""

    def __init__(self, capture: PhoneCameraCapture, port: int,
                 on_frame: Optional[Callable[[], None]] = None,
                 audio_source=None) -> None:
        self._capture = capture
        self._port = port
        self._on_frame = on_frame
        self._audio_source = audio_source
        self._audio_logged = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._runner: Optional[web.AppRunner] = None
        # Phone → PC text command handler. Set by the MainWindow at
        # engine start (same callback voice commands dispatch through).
        # The hidden QtWebEngine page (see _HOST_HTML) listens for a
        # WebRTC DataChannel from the phone and POSTs its messages to
        # /command on this server, which fires the callback.
        self._on_text_command: Optional[Callable[[str], None]] = None
        # SSE result fan-out. Each /results subscriber holds an
        # asyncio.Queue; publish_command_result enqueues a JSON-
        # encoded payload onto every queue via call_soon_threadsafe so
        # the QtWebEngine page (the only subscriber) can forward it
        # back to the phone over the open WebRTC DataChannel.
        self._result_queues: set[asyncio.Queue] = set()
        self._result_queues_lock = threading.Lock()

    def start(self) -> None:
        ready = threading.Event()
        err: list[BaseException] = []

        def run():
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            app = web.Application(client_max_size=16 * 1024 * 1024)
            app.router.add_get("/", self._index)
            app.router.add_post("/frame", self._frame)
            app.router.add_post("/audio", self._audio)
            app.router.add_post("/command", self._command)
            # SSE: command_result events published from the GUI thread
            # (after voice_processor.execute returns) are fanned out
            # here so the QtWebEngine host page can forward each one
            # back to the phone over the WebRTC "commands" DataChannel.
            app.router.add_get("/results", self._results)
            self._runner = web.AppRunner(app)
            try:
                loop.run_until_complete(self._runner.setup())
                site = web.TCPSite(self._runner, "127.0.0.1", self._port)
                loop.run_until_complete(site.start())
            except Exception as exc:
                err.append(exc)
                ready.set()
                return
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True, name="WebEnginePhoneServer")
        self._thread.start()
        ready.wait(timeout=4.0)
        if err:
            raise err[0]

    def stop(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            async def _shutdown():
                try:
                    if self._runner is not None:
                        await self._runner.cleanup()
                except Exception:
                    pass
                loop.stop()
            try:
                asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None

    async def _index(self, request):
        return web.Response(text=_HOST_HTML, content_type="text/html",
                            headers={"Cache-Control": "no-store"})

    async def _frame(self, request):
        body = await request.read()
        if body:
            self._capture.push_jpeg(body)
            if self._on_frame is not None:
                try:
                    self._on_frame()
                except Exception:
                    pass
        return web.Response(status=204)

    async def _audio(self, request):
        body = await request.read()
        if body and self._audio_source is not None:
            try:
                self._audio_source.push_pcm_int16(body)
                if not self._audio_logged:
                    self._audio_logged = True
                    _log(f"POST /audio first chunk size={len(body)} bytes (phone mic flowing)")
            except Exception:
                pass
        return web.Response(status=204)

    async def _command(self, request):
        """Text command coming from the hidden QtWebEngine page after it
        receives a DataChannel message from the phone (Connect/pairing-
        code flow). Body is JSON {"text": "..."} or raw text. Hands the
        string off to the host's text-command callback (same callback
        the QR flow uses) — dispatch runs on the GUI thread through the
        voice processor.

        TODO(iris-prompt): accept a `mode` field ({"text", "mode":"iris"})
        once Iris paid prompts are implemented; route those to the Live
        API agent instead of the voice processor.
        """
        try:
            raw = await request.read()
        except Exception:
            return web.Response(status=400, text="read failed")
        import json as _json
        text = ""
        try:
            payload = _json.loads(raw.decode("utf-8")) if raw else {}
            if isinstance(payload, dict):
                text = str(payload.get("text", "") or "").strip()
            elif isinstance(payload, str):
                text = payload.strip()
        except (ValueError, UnicodeDecodeError):
            try:
                text = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                text = ""
        if not text:
            return web.Response(status=400, text="empty command")
        if self._on_text_command is None:
            return web.Response(status=503, text="text commands not available")
        try:
            self._on_text_command(text)
        except Exception as exc:
            _log(f"POST /command dispatch error: {type(exc).__name__}: {exc}")
            return web.Response(status=500, text="dispatch failed")
        return web.json_response({"queued": True, "text": text}, status=202)

    def set_text_command_callback(self, on_text_command: Optional[Callable[[str], None]]) -> None:
        """Install the phone-text-command bridge for the Connect flow.
        Called from MainWindow when the engine starts."""
        self._on_text_command = on_text_command

    async def _results(self, request: web.Request) -> web.StreamResponse:
        """SSE endpoint consumed by the QtWebEngine host page. Each
        `command_result` payload pushed via `publish_command_result`
        becomes one SSE frame here; the page forwards it back to the
        phone over the WebRTC DataChannel.

        Only one subscriber is expected (the hidden QtWebEngine page),
        but the implementation handles N for resilience: the page may
        reconnect on a transient error, briefly producing two queues.
        """
        response = web.StreamResponse(
            status=200, reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-store",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)
        queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        with self._result_queues_lock:
            self._result_queues.add(queue)
        try:
            await response.write(b"event: hello\ndata: {}\n\n")
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    try:
                        await response.write(b": heartbeat\n\n")
                    except (ConnectionResetError, asyncio.CancelledError):
                        break
                    continue
                if payload is None:
                    break
                try:
                    await response.write(payload)
                except (ConnectionResetError, asyncio.CancelledError):
                    break
        finally:
            with self._result_queues_lock:
                self._result_queues.discard(queue)
        return response

    def publish_command_result(self, payload: dict) -> None:
        """Fan out a command result to every /results SSE subscriber so
        the host page can mirror it back to the phone via DataChannel.
        Safe to call from any thread — we marshal onto the asyncio loop
        with call_soon_threadsafe.

        `payload` shape mirrors the QR flow's SSE: {ok, heard_text,
        control_text, info_text, kind:"command_result"}. `kind` is
        injected here so the JS receiver can dispatch by message type.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            body = dict(payload or {})
            body.setdefault("kind", "command_result")
            payload_json = json.dumps(body, ensure_ascii=False)
        except (TypeError, ValueError):
            return
        message = f"data: {payload_json}\n\n".encode("utf-8")

        def _broadcast() -> None:
            with self._result_queues_lock:
                queues = list(self._result_queues)
            for q in queues:
                try:
                    q.put_nowait(message)
                except asyncio.QueueFull:
                    # Subscriber is lagging — drop oldest by getting
                    # then putting. Better than blocking the loop.
                    try:
                        q.get_nowait()
                        q.put_nowait(message)
                    except Exception:
                        pass
                except Exception:
                    pass

        try:
            loop.call_soon_threadsafe(_broadcast)
        except RuntimeError:
            # Loop is shutting down; drop the message.
            pass


class WebEnginePhoneServer:
    """Receives a phone camera via a hidden Chromium (QtWebEngine) page."""

    def __init__(self, on_status: Optional[StatusCallback] = None) -> None:
        self._on_status = on_status
        self._capture = PhoneCameraCapture()
        self._audio_source = PhoneAudioSource()
        self._server: Optional[_LocalServer] = None
        self._view = None  # QWebEngineView (GUI thread)
        self._info: Optional[WebEnginePhoneServerInfo] = None
        self._active_clients = 0
        self._last_frame_at = 0.0
        self._announced = False
        self._connected_phone_label: Optional[str] = None

    # ---- PhoneCameraServer-compatible surface ----
    @property
    def capture(self) -> PhoneCameraCapture:
        return self._capture

    @property
    def audio_source(self) -> PhoneAudioSource:
        return self._audio_source

    @property
    def info(self) -> Optional[WebEnginePhoneServerInfo]:
        return self._info

    @property
    def is_running(self) -> bool:
        return self._view is not None

    @property
    def connected_clients(self) -> int:
        return self._active_clients

    @property
    def connected_phone_label(self) -> Optional[str]:
        return self._connected_phone_label

    @property
    def seconds_since_last_frame(self) -> float:
        if self._last_frame_at <= 0.0:
            return float("inf")
        return time.monotonic() - self._last_frame_at

    def set_status_callback(self, on_status: Optional[StatusCallback]) -> None:
        self._on_status = on_status

    def set_text_command_callback(self, on_text_command: Optional[Callable[[str], None]]) -> None:
        """Install the phone → PC text-command bridge. Forwarded to the
        underlying _LocalServer; the hidden QtWebEngine page POSTs
        DataChannel messages to its /command endpoint, which fires this
        callback. MainWindow wires it at engine start (same callback the
        QR PhoneCameraServer uses)."""
        if self._server is not None:
            self._server.set_text_command_callback(on_text_command)

    def publish_command_result(self, payload: dict) -> None:
        """Fan the result of a phone-sent text command back to the phone
        via the SSE → DataChannel relay. Safe to call from any thread —
        the underlying _LocalServer marshals onto its asyncio loop."""
        if self._server is not None:
            self._server.publish_command_result(payload)

    def _emit(self, event: str, data: dict) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(event, data)
        except Exception:
            pass

    # Called from the server thread on every received frame.
    def _on_frame(self) -> None:
        self._last_frame_at = time.monotonic()
        if not self._announced:
            self._announced = True
            self._active_clients = 1
            self._connected_phone_label = "Phone (WebRTC)"
            self._emit("client_connected", {"total": 1, "label": self._connected_phone_label})
            self._emit("streaming", {})

    # ---- Lifecycle (GUI thread) ----
    def start(self) -> WebEnginePhoneServerInfo:
        if self.is_running:
            assert self._info is not None
            return self._info
        from PySide6.QtCore import Qt, QUrl
        from PySide6.QtWebEngineWidgets import QWebEngineView

        code = f"{random.randint(0, 999999):06d}"
        self._server = _LocalServer(self._capture, _LOCAL_PORT, on_frame=self._on_frame,
                                    audio_source=self._audio_source)
        self._server.start()
        self._info = WebEnginePhoneServerInfo(code=code, connect_url=CONNECT_PAGE_URL)

        # The page must stay rendered or Chromium throttles it to ~1 fps,
        # so the view is shown — but off-screen and frameless/tool so no
        # stray window appears in the UI or taskbar.
        view = QWebEngineView()
        view.setWindowFlag(Qt.Tool, True)
        view.setWindowFlag(Qt.FramelessWindowHint, True)
        view.setAttribute(Qt.WA_ShowWithoutActivating, True)
        view.setFixedSize(320, 200)
        view.move(-10000, -10000)
        # The phone's mic audio is captured in-page via a WebRTC track ->
        # AudioContext -> ScriptProcessor -> POST /audio. Chromium's autoplay
        # policy starts that AudioContext SUSPENDED unless a user gesture has
        # occurred — but this page is hidden and loaded programmatically, so
        # no gesture ever happens and the audio worklet never runs (video is
        # unaffected; it uses requestAnimationFrame, not audio). Disabling the
        # user-gesture requirement lets the context run so phone-mic PCM
        # actually reaches the voice pipeline.
        try:
            from PySide6.QtWebEngineCore import QWebEngineSettings
            attr = getattr(
                getattr(QWebEngineSettings, "WebAttribute", QWebEngineSettings),
                "PlaybackRequiresUserGesture",
            )
            view.settings().setAttribute(attr, False)
        except Exception as exc:  # pragma: no cover - defensive
            _log(f"could not disable PlaybackRequiresUserGesture: {exc}")
        view.load(QUrl(f"http://127.0.0.1:{_LOCAL_PORT}/?code={code}"))
        view.show()
        self._view = view
        _log(f"started, code={code}")
        self._emit("listening", {"code": code, "connect_url": CONNECT_PAGE_URL})
        return self._info

    def stop(self) -> None:
        if self._view is not None:
            try:
                self._view.close()
                self._view.deleteLater()
            except Exception:
                pass
            self._view = None
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                pass
            self._server = None
        try:
            self._capture.release()
        except Exception:
            pass
        try:
            self._audio_source.close()
        except Exception:
            pass
        self._emit("stopped", {})
