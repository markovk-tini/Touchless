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
    }
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
