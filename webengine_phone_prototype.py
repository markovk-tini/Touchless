"""Standalone latency/quality prototype for the browser-engine phone camera.

Proves the "replicate connect-test.html inside the app" approach end to end
WITHOUT touching the main app:

  phone (touchless-control.com/connect)  --WebRTC-->  this app
        |                                                 |
        |                            Chromium (QtWebEngine) HARDWARE-decodes
        |                            the stream in a hidden web page, draws
        |                            each frame to a <canvas>, JPEG-encodes
        |                            it, and POSTs to http://127.0.0.1 (a
        |                            secure context, so WebRTC works and
        |                            there's no certificate).
        |                                                 v
        |                            local aiohttp server -> push_jpeg ->
        |                            PhoneCameraCapture (the SAME sink the
        |                            QR flow + the engine already use).
        v                                                 v
   camera stream                          this window shows what the APP
                                          received, with an FPS counter.

How to run:
    python webengine_phone_prototype.py

Then on your phone open the connect page (currently the preview:
redesign.touchless-website.pages.dev/connect), type the 6-digit code this
window prints, and watch the FPS / clarity / lag. If it's good, we wire the
same pieces into the app; if not, we've spent one file finding out.

Author: Konstantin Markov
"""
from __future__ import annotations

import asyncio
import random
import sys
import threading
import time

import cv2
import numpy as np
from aiohttp import web

sys.path.insert(0, "src")
from hgr.debug.phone_camera.capture import PhoneCameraCapture  # noqa: E402

from PySide6.QtCore import Qt, QTimer, QUrl  # noqa: E402
from PySide6.QtGui import QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: E402

SIGNALING_WS = "wss://touchless-signaling.konstantinvmarkov.workers.dev/ws"
PORT = 8770
CODE = f"{random.randint(0, 999999):06d}"

# Receiver page: WebRTC host that decodes the phone stream (hardware, via
# Chromium) then ships frames to the app as JPEG over same-origin localhost.
HOST_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  body { margin:0; background:#04111a; color:#5af0c1; font:13px monospace; }
  video { width:100%; height:78vh; object-fit:contain; background:#000; display:block; }
  #s { padding:6px 10px; }
</style></head>
<body>
  <video id="v" autoplay playsinline muted></video>
  <div id="s">starting…</div>
  <canvas id="c" style="display:none"></canvas>
  <script>
    const SIG = "%SIG%";
    const ICE = [{ urls: "stun:stun.l.google.com:19302" },
                 { urls: "stun:stun1.l.google.com:19302" }];
    const code = new URLSearchParams(location.search).get("code");
    const v = document.getElementById("v");
    const s = document.getElementById("s");
    const c = document.getElementById("c");
    let ws, pc, inflight = false, mirror = false;
    const log = (t) => { s.textContent = t; };

    ws = new WebSocket(`${SIG}?code=${code}&role=host`);
    ws.onopen = () => log("waiting for phone… code " + code);
    ws.onmessage = async (e) => {
      const m = JSON.parse(e.data);
      // Diagnostic: surface anything that isn't the high-rate ICE chatter.
      if (m.type !== "ice") log("rx: " + m.type + (m.type === "mirror" ? (" on=" + m.on) : ""));
      if (m.type === "ready") { if (!pc) makePeer(); }
      else if (m.type === "offer") {
        if (!pc) makePeer();
        await pc.setRemoteDescription(m.sdp);
        const a = await pc.createAnswer();
        await pc.setLocalDescription(a);
        ws.send(JSON.stringify({ type: "answer", sdp: pc.localDescription }));
        log("pairing…");
      } else if (m.type === "ice" && pc && m.candidate) {
        try { await pc.addIceCandidate(m.candidate); } catch (_) {}
      } else if (m.type === "mirror") {
        mirror = !!m.on;                                  // flip what we send
        v.style.transform = mirror ? "scaleX(-1)" : "none"; // and the preview
        log("MIRROR " + (mirror ? "ON" : "OFF"));
      }
    };
    function makePeer() {
      pc = new RTCPeerConnection({ iceServers: ICE });
      pc.onicecandidate = (e) => { if (e.candidate) ws.send(JSON.stringify({ type: "ice", candidate: e.candidate })); };
      pc.onconnectionstatechange = () => log("connection: " + pc.connectionState);
      pc.ontrack = (e) => { v.srcObject = e.streams[0]; log("connected — streaming"); pump(); };
    }
    // Draw each rendered frame to a canvas, JPEG-encode, POST to the app.
    // Skip while a POST is in flight so latency can't accumulate.
    function pump() {
      const ctx = c.getContext("2d");
      function loop() {
        if (v.videoWidth && !inflight) {
          c.width = v.videoWidth; c.height = v.videoHeight;
          if (mirror) {
            ctx.save(); ctx.translate(c.width, 0); ctx.scale(-1, 1);
            ctx.drawImage(v, 0, 0); ctx.restore();
          } else {
            ctx.drawImage(v, 0, 0);
          }
          inflight = true;
          c.toBlob((b) => {
            if (!b) { inflight = false; return; }
            fetch("/frame", { method: "POST", body: b })
              .catch(() => {})
              .finally(() => { inflight = false; });
          }, "image/jpeg", 0.85);
        }
        requestAnimationFrame(loop);
      }
      requestAnimationFrame(loop);
    }
  </script>
</body></html>
""".replace("%SIG%", SIGNALING_WS)


class LocalServer:
    """Tiny 127.0.0.1 HTTP server: serves the receiver page + takes frames."""

    def __init__(self, capture: PhoneCameraCapture, port: int) -> None:
        self.capture = capture
        self.port = port
        self._loop = None
        self._thread = None

    def start(self) -> None:
        ready = threading.Event()

        def run():
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            app = web.Application(client_max_size=16 * 1024 * 1024)
            app.router.add_get("/", self._index)
            app.router.add_post("/frame", self._frame)
            runner = web.AppRunner(app)
            loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, "127.0.0.1", self.port)
            loop.run_until_complete(site.start())
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True, name="LocalServer")
        self._thread.start()
        ready.wait(timeout=4.0)

    async def _index(self, request):
        return web.Response(text=HOST_HTML, content_type="text/html",
                            headers={"Cache-Control": "no-store"})

    async def _frame(self, request):
        body = await request.read()
        if body:
            self.capture.push_jpeg(body)
        return web.Response(status=204)


def main() -> None:
    app = QApplication(sys.argv)
    capture = PhoneCameraCapture()
    server = LocalServer(capture, PORT)
    server.start()

    win = QWidget()
    win.setWindowTitle("WebEngine phone prototype — 0 fps")
    win.setStyleSheet("background:#0c2331; color:#f2fbff;")
    lay = QVBoxLayout(win)

    info = QLabel(f"On your phone open the connect page and enter:   {CODE}")
    info.setStyleSheet("font-size:18px; font-weight:800; color:#58e3ff; padding:6px;")
    lay.addWidget(info)

    # The web view MUST stay visible — Chromium throttles hidden pages
    # (video would drop to ~1 fps). Kept small here.
    view = QWebEngineView()
    view.setFixedSize(360, 230)
    view.load(QUrl(f"http://127.0.0.1:{PORT}/?code={CODE}"))
    lay.addWidget(view)

    disp = QLabel("waiting for frames…")
    disp.setFixedSize(720, 405)
    disp.setAlignment(Qt.AlignCenter)
    disp.setStyleSheet("background:#000; border:1px solid #1de9b6;")
    lay.addWidget(disp)

    state = {"n": 0, "t": time.time()}

    def tick():
        ok, frame = capture.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            disp.setPixmap(QPixmap.fromImage(img).scaled(
                720, 405, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            state["n"] += 1
        now = time.time()
        if now - state["t"] >= 1.0:
            win.setWindowTitle(f"WebEngine phone prototype — {state['n']} fps (frames reaching the app)")
            state["n"] = 0
            state["t"] = now

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(12)  # poll faster than the stream so we never add wait latency

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
