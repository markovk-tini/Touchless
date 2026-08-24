"""HTML served to the phone browser.

Kept as a Python string (rather than a separate .html asset) so PyInstaller
bundling needs no extra datas entry. The page asks for the rear camera,
captures JPEG frames from a `<canvas>`, and streams them over the
WebSocket at `/ws`.
"""

CLIENT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, user-scalable=no">
  <title>Touchless Phone Camera</title>
  <style>
    :root { color-scheme: dark; }
    html, body { margin: 0; padding: 0; background: #0B3D91; color: #E5F6FF;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      -webkit-user-select: none; user-select: none; overscroll-behavior: none; }
    body { display: flex; flex-direction: column; min-height: 100vh;
      padding: 18px 16px; box-sizing: border-box; }
    h1 { margin: 0 0 4px; font-size: 18px; font-weight: 600; }
    .subtitle { margin: 0 0 12px; font-size: 13px; opacity: 0.8; }
    .status { font-size: 14px; padding: 10px 12px; border-radius: 10px;
      background: rgba(255,255,255,0.08); margin-bottom: 10px; min-height: 22px; }
    .status.ok { background: rgba(29,233,182,0.18); }
    .status.err { background: rgba(255,99,99,0.22); }
    .preview-wrap { position: relative; flex: 1; min-height: 260px;
      background: #00112a; border-radius: 14px; overflow: hidden;
      display: flex; align-items: center; justify-content: center; }
    /* Fullscreen: some iOS Safaris don't support element.requestFullscreen,
       so we implement our own CSS fallback that stretches the preview
       to cover the viewport. .fs-active is toggled from JS. */
    .preview-wrap.fs-active { position: fixed; inset: 0; z-index: 999;
      border-radius: 0; }
    /* The <video> element is the raw camera source. Once streaming
       starts it's hidden behind #previewCanvas (below); before Start it
       shows the live camera so the user can frame themselves. */
    video { width: 100%; height: 100%; object-fit: cover; }
    /* Device-independent preview. Front-camera mirroring differs per
       browser (some auto-mirror the <video>, some don't), so a CSS flip
       on the <video> can't reliably match the app. Instead we render the
       preview from the SAME canvas whose frames are POSTed to the PC —
       drawImage(video) always yields the raw, un-mirrored buffer
       regardless of how the browser displays the <video> — and flip it
       horizontally to match the engine's selfie flip (cv2.flip). That
       makes the phone preview show exactly what the app displays, on
       every device. Hidden until streaming starts (.live). */
    canvas#previewCanvas { position: absolute; inset: 0; width: 100%;
      height: 100%; object-fit: cover; transform: scaleX(-1);
      display: none; }
    canvas#previewCanvas.live { display: block; }
    .stats { position: absolute; left: 10px; bottom: 10px; font-size: 11px;
      padding: 4px 8px; background: rgba(0,0,0,0.55); border-radius: 6px;
      font-variant-numeric: tabular-nums; }
    .fs-exit { position: absolute; right: 10px; top: 10px;
      font-size: 13px; padding: 8px 12px; background: rgba(0,0,0,0.7);
      color: #E5F6FF; border-radius: 20px; display: none; z-index: 1000;
      border: 1px solid rgba(255,255,255,0.3); }
    .preview-wrap.fs-active .fs-exit { display: inline-block; }
    .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
      margin-top: 12px; }
    .ctl-row { display: flex; align-items: center;
      background: rgba(255,255,255,0.06); border-radius: 10px; padding: 8px 12px; }
    .ctl-label { font-size: 11px; opacity: 0.7; margin-right: 8px; }
    .ctl-select { flex: 1; background: transparent; color: #E5F6FF;
      border: none; font-size: 14px; font-weight: 600; outline: none;
      -webkit-appearance: none; appearance: none; text-align: right;
      padding-right: 8px; }
    .ctl-select option { background: #0B3D91; color: #E5F6FF; }
    .buttons { display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
    button { flex: 1; min-width: 100px; padding: 14px 10px;
      font-size: 15px; font-weight: 600;
      border: none; border-radius: 10px; background: #1DE9B6; color: #003d2a;
      -webkit-appearance: none; }
    button:disabled { opacity: 0.5; }
    button.secondary { background: rgba(255,255,255,0.15); color: #E5F6FF; }
    /* Text-command prompt row. Input fills available width; Send is
       sized to its label. iOS Safari likes explicit min-heights here
       so the input doesn't render at 28px (default) on a tall phone. */
    .command-row { display: flex; gap: 8px; margin-top: 12px; }
    .command-row input { flex: 1; min-width: 0; padding: 14px 12px;
      font-size: 15px; border: 1px solid rgba(255,255,255,0.22);
      border-radius: 10px; background: rgba(255,255,255,0.06);
      color: #E5F6FF; -webkit-appearance: none; }
    .command-row input::placeholder { color: rgba(229,246,255,0.4); }
    .command-row input:focus { outline: none;
      border-color: rgba(29,233,182,0.65); }
    .command-row button { flex: 0 0 auto; min-width: 80px; }
    .hint { font-size: 12px; opacity: 0.75; margin-top: 12px; line-height: 1.4; }
    .hint.compact { display: none; }
    body.mini h1,
    body.mini .subtitle,
    body.mini .hint { display: none; }
    body.mini .status { padding: 6px 10px; font-size: 12px; }
    body.mini .preview-wrap { min-height: 180px; }
    body.mini .controls { grid-template-columns: 1fr; }
    body.mini .ctl-row { padding: 6px 10px; }
    body.mini .audio-stats { font-size: 10px; }

    .audio-stats { font-size: 11px; opacity: 0.55; margin-top: 8px;
      font-variant-numeric: tabular-nums; text-align: center;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      min-height: 14px; line-height: 14px; }

    /* Toast notifications pushed from the PC via SSE.
       Sit fixed at the top of the viewport, slide-in/fade-out,
       capped to the latest 3 so a flurry of gestures doesn't
       fill the screen. The z-index sits above the CSS-fullscreen
       preview overlay (.fs-active uses 999), and the .toast-stack
       gets a state hook (`.fs-mode`) we toggle from JS so the
       offset matches the safe-area in fullscreen mode (iOS
       notch / Dynamic Island clearance). The stack is also
       reparented under the preview-wrap when CSS-fullscreen is
       active so it survives the browser's native-fullscreen API
       (which clips out-of-tree fixed-position elements). */
    .toast-stack { position: fixed; left: 50%; top: max(8px, env(safe-area-inset-top, 8px));
      transform: translateX(-50%); display: flex;
      flex-direction: column; gap: 6px; align-items: center;
      pointer-events: none; z-index: 2147483646; max-width: 92vw; }
    .toast-stack.fs-mode { top: max(18px, env(safe-area-inset-top, 18px)); }
    .toast { pointer-events: auto;
      background: rgba(11, 61, 145, 0.92);
      color: #E5F6FF;
      border: 1px solid rgba(29, 233, 182, 0.35);
      box-shadow: 0 4px 14px rgba(0,0,0,0.35);
      backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
      border-radius: 999px; padding: 10px 18px;
      font-size: 14px; font-weight: 600; line-height: 1.2;
      max-width: 100%;
      animation: toastIn 220ms ease-out;
      transition: opacity 280ms ease-out, transform 280ms ease-out; }
    .toast.gesture { border-color: rgba(29, 233, 182, 0.55); }
    .toast.voice   { border-color: rgba(255, 200, 80, 0.65); }
    .toast .toast-kind { font-size: 11px; font-weight: 600;
      opacity: 0.7; text-transform: uppercase;
      letter-spacing: 0.06em; margin-right: 8px; }
    .toast.fade-out { opacity: 0; transform: translateY(-6px); }
    @keyframes toastIn {
      from { opacity: 0; transform: translateY(-12px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div id="toastStack" class="toast-stack" aria-live="polite"></div>
  <h1>Touchless Phone Camera</h1>
  <div class="subtitle">Keep this tab open while you use Touchless.</div>
  <div id="status" class="status">Tap Start to share your camera.</div>

  <div class="preview-wrap" id="previewWrap">
    <video id="preview" autoplay playsinline muted></video>
    <canvas id="previewCanvas"></canvas>
    <div class="stats" id="stats"></div>
    <button class="fs-exit" id="fsExit" type="button">Exit fullscreen</button>
  </div>

  <div class="controls">
    <label class="ctl-row">
      <span class="ctl-label">Resolution</span>
      <select class="ctl-select" id="resSelect">
        <option value="1080">1080p</option>
        <option value="720" selected>720p</option>
        <option value="480">480p</option>
        <option value="360">360p</option>
      </select>
    </label>
    <label class="ctl-row">
      <span class="ctl-label">Frame rate</span>
      <select class="ctl-select" id="fpsSelect">
        <option value="15">15 fps</option>
        <option value="24">24 fps</option>
        <option value="30" selected>30 fps</option>
        <option value="60">60 fps</option>
      </select>
    </label>
    <label class="ctl-row">
      <span class="ctl-label">Quality</span>
      <select class="ctl-select" id="qualSelect">
        <option value="0.6">low</option>
        <option value="0.75" selected>medium</option>
        <option value="0.88">high</option>
      </select>
    </label>
    <label class="ctl-row">
      <span class="ctl-label">Camera</span>
      <select class="ctl-select" id="facingSelect">
        <option value="environment" selected>rear</option>
        <option value="user">front</option>
      </select>
    </label>
    <label class="ctl-row">
      <span class="ctl-label">Mic</span>
      <select class="ctl-select" id="micSelect">
        <option value="off" selected>off</option>
        <option value="on">send to PC</option>
      </select>
    </label>
    <label class="ctl-row">
      <span class="ctl-label">Mic distance</span>
      <select class="ctl-select" id="distSelect">
        <option value="near" selected>near (close-talk)</option>
        <option value="far">far (~1 m)</option>
      </select>
    </label>
  </div>
  <div id="audioStats" class="audio-stats"></div>

  <div class="buttons">
    <button id="start">Start</button>
    <button id="fs" class="secondary" type="button">Fullscreen</button>
    <button id="mini" class="secondary" type="button">Mini</button>
  </div>

  <!-- Text command prompt. Same intent surface as voice commands —
       "pause spotify", "open chrome", "play despacito on spotify",
       "open touchless settings", etc. The PC dispatches via the
       existing voice-command processor on the GUI thread and reports
       the outcome back through the SSE stream (kind="command_result")
       which the listener below renders as a toast. -->
  <div class="command-row">
    <input id="commandInput" type="text" inputmode="text"
           autocomplete="off" autocapitalize="sentences"
           placeholder="Type a command, e.g. pause spotify" />
    <button id="commandSend" type="button">Send</button>
  </div>

  <div class="hint">
    <div><strong>First time setup:</strong> your browser will ask for camera permission — tap Allow.</div>
    <div style="margin-top:8px"><strong>If Start fails on iPhone, install and trust the Touchless Root CA.</strong> You only do this once, ever — after it's trusted, Touchless works on this phone forever, including across LAN IP changes.</div>
    <ol style="margin:6px 0 6px 20px;padding:0;font-size:12px;line-height:1.5">
      <li><strong>Delete any older Touchless profile first.</strong> Settings → General → VPN &amp; Device Management → tap anything named "Touchless Phone Camera" or "Touchless Server" → Remove Profile. Earlier builds shipped certs iOS couldn't accept; leaving them installed blocks the new root from working.</li>
      <li>Tap <a href="/touchless-cert.cer" download="touchless-root-ca.cer" style="color:#1DE9B6;text-decoration:underline">Download Touchless Root CA</a>.</li>
      <li>iOS shows "Profile Downloaded." Open Settings — at the top you'll see <em>"Profile Downloaded"</em>. Tap it → Install → enter passcode → Install → Done.</li>
      <li><strong>Trust the root fully:</strong> Settings → General → About → Certificate Trust Settings. "Touchless Root CA" appears under "Enable Full Trust for Root Certificates." Toggle it on → Continue.</li>
      <li><strong>Close this Safari tab</strong> (swipe it away from the tab switcher). Safari caches cert state per-tab — a fresh tab is required for iOS to re-evaluate trust.</li>
      <li>Re-scan the QR on your PC from a fresh Safari tab and tap Start.</li>
    </ol>
    <div style="margin-top:8px;font-size:11px;opacity:0.7">If the profile doesn't appear as "Touchless Root CA" in Certificate Trust Settings, you're using an older build — pull latest Touchless on your PC and retry.</div>
  </div>

<script>
(() => {
  const statusEl = document.getElementById("status");
  const previewEl = document.getElementById("preview");
  const previewWrap = document.getElementById("previewWrap");
  const startBtn = document.getElementById("start");
  const fsBtn = document.getElementById("fs");
  const fsExitBtn = document.getElementById("fsExit");
  const miniBtn = document.getElementById("mini");
  const statsEl = document.getElementById("stats");
  const resSelect = document.getElementById("resSelect");
  const fpsSelect = document.getElementById("fpsSelect");
  const qualSelect = document.getElementById("qualSelect");
  const facingSelect = document.getElementById("facingSelect");
  const micSelect = document.getElementById("micSelect");
  const distSelect = document.getElementById("distSelect");
  const audioStatsEl = document.getElementById("audioStats");

  let stream = null;
  let canvas = null;
  let ctx = null;
  let sending = false;
  let wakeLock = null;
  let frameCount = 0;
  let lastStatsAt = 0;
  let lastSentBytes = 0;
  let lastSentAt = 0;
  let inflightPost = false;  // prevent overlapping POSTs — drop frames rather than pile up
  let consecutivePostErrors = 0;

  // Audio pipeline state.
  let audioContext = null;
  let audioSourceNode = null;
  let audioWorkletNode = null;
  let audioQueue = [];       // pending Int16 bytes not yet POSTed
  let audioQueuedBytes = 0;
  let audioInflight = false;
  let audioSampleRate = 48000;
  const AUDIO_CHUNK_BYTES = 9600;  // ~100ms of 48kHz mono Int16 (48000 * 2 * 0.1)

  function setStatus(text, kind) {
    statusEl.textContent = text;
    statusEl.className = "status" + (kind ? " " + kind : "");
  }

  function currentRes() {
    const v = parseInt(resSelect.value, 10) || 720;
    // Map short-edge target to width/height. Phone cameras deliver
    // landscape natively; we request width:height with 16:9 aspect.
    const heightMap = { 1080: 1080, 720: 720, 480: 480, 360: 360 };
    const h = heightMap[v] || 720;
    const w = Math.round(h * 16 / 9);
    return { width: w, height: h };
  }
  function currentFps() { return parseInt(fpsSelect.value, 10) || 30; }
  function currentQuality() { return parseFloat(qualSelect.value) || 0.78; }

  async function probeHttp() {
    try {
      const resp = await fetch("/healthz", { method: "GET", cache: "no-store" });
      return resp.ok;
    } catch (_) { return false; }
  }

  async function requestWakeLock() {
    try {
      if ("wakeLock" in navigator) {
        wakeLock = await navigator.wakeLock.request("screen");
      }
    } catch (_) {}
  }

  async function openCamera() {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    const res = currentRes();
    const fps = currentFps();
    const wantMic = micSelect.value === "on";
    // autoGainControl is OFF on purpose. iOS Safari's AGC drags speech
    // down to RMS ~0.0013 (~50x quieter than a normal desktop mic),
    // and the AGC curve lags by hundreds of ms when volume changes,
    // so quiet syllables at the start of a phrase get clipped to
    // near-silence. With AGC off we receive raw mic signal and apply
    // our own compressor + limiter inside the AudioWorklet, which
    // gives the steady, loud, Siri-like presence the PC voice
    // pipeline needs.
    const constraints = {
      audio: wantMic
        ? { echoCancellation: true, noiseSuppression: true, autoGainControl: false }
        : false,
      video: {
        facingMode: { ideal: facingSelect.value },
        width: { ideal: res.width },
        height: { ideal: res.height },
        frameRate: { ideal: fps, max: fps }
      }
    };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    previewEl.srcObject = stream;
    const track = stream.getVideoTracks()[0];
    const settings = track.getSettings ? track.getSettings() : {};
    if (wantMic) {
      try { await startAudioPipeline(stream); }
      catch (err) { console.warn("audio pipeline failed:", err); }
    } else {
      stopAudioPipeline();
    }
    return { width: settings.width || res.width, height: settings.height || res.height, fps: settings.frameRate || fps };
  }

  // --- Audio capture + upload ------------------------------------------------
  //
  // AudioWorklet "pcm-capture" runs a multi-stage processing chain on
  // the raw mic stream (AGC off — see openCamera). Per sample:
  //   1. DC-blocking high-pass at ~100 Hz — kills HVAC rumble, fan
  //      noise, monitor PSU hum, table thumps. These dominate the
  //      noise floor at distance and steal headroom from the compressor.
  //   2. Speech-band emphasis (gentle high-shelf, ~+4 dB above 2 kHz).
  //      Boosts consonants (s/t/k/ch) which lose disproportionate
  //      energy at distance and matter most for whisper's WER.
  //   3. Adaptive noise gate. Threshold tracks an estimated room-
  //      noise floor (rolling minimum of per-window mean RMS over
  //      ~10 s) so the gate self-tunes to the user's environment.
  //      Slew-limited rises stop a fluky speech-only window from
  //      slamming the gate shut; falls are instant so moving to a
  //      quieter spot recovers immediately. HOLD-style envelope as
  //      before: opens within a few samples of any signal above
  //      threshold, stays open for 120 ms after the last
  //      above-threshold sample so mid-phrase pauses don't get
  //      chopped, then mutes to true zero so whisper sees real
  //      silence at the tail and stops cleanly.
  //   4. PRE_GAIN — configurable per "mic distance" mode. Near (16x)
  //      matches the original close-talk tuning that compensates
  //      for iOS Safari's quiet baseline (AGC off pushes RMS to
  //      ~0.0013 raw); Far (32x) accounts for speech being ~5 dB
  //      quieter at monitor distance vs close-talk.
  //   5. Envelope follower — peak detector with 1 ms attack /
  //      80 ms release.
  //   6. Compressor with mode-dependent threshold (Near -3 dBFS,
  //      Far -7 dBFS so far speech also gets compressed reliably).
  //   7. Hard limiter at -0.3 dBFS — final safeguard before Int16.
  //
  // The worklet posts two kinds of messages on its port:
  //   - ArrayBuffer (Int16 PCM) — audio samples to upload
  //   - { type:'stats', noiseFloor, gateThreshold, gainReduction,
  //       envelope, gateOpen } — periodic telemetry for the live
  //     UI readout
  // The worklet accepts config messages of the form
  //   { preGain:number, compThreshold:number }
  // so changing the mic-distance mode doesn't require restarting
  // the audio stream.
  const PCM_WORKLET_SRC =
    "const HARD_LIMIT = 0.97;" +
    "const ATTACK_COEF = 0.3;" +
    "const RELEASE_COEF = 0.0008;" +
    "const GAIN_SMOOTH = 0.08;" +
    // 1-pole DC-blocking HPF: y = a * (y_prev + x - x_prev),
    // a = 1 - 2*pi*fc/Fs. fc = 100 Hz at Fs = 48 kHz → a = 0.9869.
    "const HPF_ALPHA = 0.9869;" +
    // 1-pole LPF cutoff for the high-shelf split.
    // alpha = 2*pi*fc/Fs ≈ 0.245 at fc = 1.9 kHz, Fs = 48 kHz.
    "const SHELF_LP_ALPHA = 0.245;" +
    // High-band gain. Output = lp + (1 + SHELF_GAIN) * (x - lp);
    // 0.6 → +4.1 dB shelf above ~2 kHz.
    "const SHELF_GAIN = 0.6;" +
    "const NOISE_GATE_HOLD_SAMPLES = 5760;" +  // 120 ms at 48 kHz
    // Adaptive noise floor (Martin-style minimum statistics):
    // accumulate per-window mean RMS, store in a ring of recent
    // windows, take the minimum. Speech can never push the minimum
    // smaller — only true silence does — so this is robust to long
    // monologues with no obvious gaps.
    "const NOISE_HISTORY_LEN = 40;" +     // 40 windows × ~267 ms ≈ 10.7 s
    "const NOISE_BLOCKS_PER_WIN = 100;" + // ~267 ms at 128-sample blocks
    "const NOISE_FLOOR_MIN = 0.0005;" +   // never trust an estimate below this (true silence + ADC noise)
    "const NOISE_FLOOR_SCALE = 3.0;" +    // gate opens at noise_floor × this (~+9.5 dB above floor)
    "const NOISE_RISE_PER_WIN = 1.06;" +  // +0.5 dB per 267 ms ≈ +1.9 dB/s — slew-limit upward jumps
    "const STATS_BLOCKS = 50;" +          // post stats every ~133 ms
    "class PcmCapture extends AudioWorkletProcessor {" +
    "  constructor() {" +
    "    super();" +
    "    this._preGain = 16.0;" +
    "    this._compThreshold = 0.7;" +
    "    this._env = 0; this._gr = 1.0; this._gateHold = 0;" +
    "    this._hpfX = 0; this._hpfY = 0; this._shelfLp = 0;" +
    "    this._noiseFloor = 0.002;" +
    "    this._gateThreshold = this._noiseFloor * NOISE_FLOOR_SCALE;" +
    "    this._noiseHist = new Float32Array(NOISE_HISTORY_LEN);" +
    "    this._noiseHist.fill(1.0);" +
    "    this._noiseHistIdx = 0;" +
    "    this._noiseHistFilled = 0;" +
    "    this._winSqSum = 0; this._winSamples = 0; this._winBlocks = 0;" +
    "    this._statsBlocks = 0;" +
    "    this.port.onmessage = (e) => {" +
    "      const d = e && e.data;" +
    "      if (!d || typeof d !== 'object') return;" +
    "      if (typeof d.preGain === 'number' && d.preGain > 0) this._preGain = d.preGain;" +
    "      if (typeof d.compThreshold === 'number' && d.compThreshold > 0) this._compThreshold = d.compThreshold;" +
    "    };" +
    "  }" +
    "  process(inputs) {" +
    "    const ch = inputs[0] && inputs[0][0];" +
    "    if (!ch || !ch.length) return true;" +
    "    const out = new Int16Array(ch.length);" +
    "    let env = this._env, gr = this._gr, gateHold = this._gateHold;" +
    "    let hpfX = this._hpfX, hpfY = this._hpfY, shelfLp = this._shelfLp;" +
    "    const preGain = this._preGain, compThreshold = this._compThreshold;" +
    "    const gateThreshold = this._gateThreshold;" +
    "    let winSqSum = this._winSqSum;" +
    "    for (let i = 0; i < ch.length; i++) {" +
    "      const xin = ch[i];" +
    // 1. DC-blocking HPF
    "      const newY = HPF_ALPHA * (hpfY + xin - hpfX);" +
    "      hpfX = xin; hpfY = newY;" +
    // 2. High-shelf via low-pass split
    "      shelfLp += SHELF_LP_ALPHA * (newY - shelfLp);" +
    "      const high = newY - shelfLp;" +
    "      const shaped = newY + high * SHELF_GAIN;" +
    // Accumulate energy on the SHAPED, pre-gain signal so the noise
    // floor we compare against has the same spectral shape as what
    // hits the gate.
    "      winSqSum += shaped * shaped;" +
    // 3. Adaptive noise gate
    "      const absS = shaped < 0 ? -shaped : shaped;" +
    "      if (absS >= gateThreshold) gateHold = NOISE_GATE_HOLD_SAMPLES;" +
    "      if (gateHold <= 0) {" +
    "        out[i] = 0;" +
    "        env *= 0.95;" +  // bleed envelope down so we don't pop on next gate-open
    "        continue;" +
    "      }" +
    "      gateHold--;" +
    // 4. Pre-gain
    "      const boosted = shaped * preGain;" +
    // 5. Envelope follower
    "      const absB = boosted < 0 ? -boosted : boosted;" +
    "      if (absB > env) env += (absB - env) * ATTACK_COEF;" +
    "      else env += (absB - env) * RELEASE_COEF;" +
    // 6. Compressor
    "      const target = env > compThreshold ? compThreshold / env : 1.0;" +
    "      gr += (target - gr) * GAIN_SMOOTH;" +
    "      let s = boosted * gr;" +
    // 7. Hard limiter
    "      if (s > HARD_LIMIT) s = HARD_LIMIT;" +
    "      else if (s < -HARD_LIMIT) s = -HARD_LIMIT;" +
    "      out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;" +
    "    }" +
    "    this._env = env; this._gr = gr; this._gateHold = gateHold;" +
    "    this._hpfX = hpfX; this._hpfY = hpfY; this._shelfLp = shelfLp;" +
    "    this._winSqSum = winSqSum;" +
    "    this._winSamples += ch.length;" +
    "    this._winBlocks++;" +
    "    if (this._winBlocks >= NOISE_BLOCKS_PER_WIN) {" +
    "      const winRms = Math.sqrt(this._winSqSum / Math.max(1, this._winSamples));" +
    "      this._noiseHist[this._noiseHistIdx % NOISE_HISTORY_LEN] = winRms;" +
    "      this._noiseHistIdx++;" +
    "      if (this._noiseHistFilled < NOISE_HISTORY_LEN) this._noiseHistFilled++;" +
    "      let minVal = this._noiseHist[0];" +
    "      const N = this._noiseHistFilled;" +
    "      for (let k = 1; k < N; k++) { if (this._noiseHist[k] < minVal) minVal = this._noiseHist[k]; }" +
    "      let target = minVal < NOISE_FLOOR_MIN ? NOISE_FLOOR_MIN : minVal;" +
    "      const maxRise = this._noiseFloor * NOISE_RISE_PER_WIN;" +
    "      this._noiseFloor = target > this._noiseFloor ? (target < maxRise ? target : maxRise) : target;" +
    "      this._gateThreshold = this._noiseFloor * NOISE_FLOOR_SCALE;" +
    "      this._winSqSum = 0; this._winSamples = 0; this._winBlocks = 0;" +
    "    }" +
    "    this.port.postMessage(out.buffer, [out.buffer]);" +
    "    this._statsBlocks++;" +
    "    if (this._statsBlocks >= STATS_BLOCKS) {" +
    "      this._statsBlocks = 0;" +
    "      this.port.postMessage({" +
    "        type: 'stats'," +
    "        noiseFloor: this._noiseFloor," +
    "        gateThreshold: this._gateThreshold," +
    "        gainReduction: this._gr," +
    "        envelope: this._env," +
    "        gateOpen: this._gateHold > 0," +
    "        preGain: this._preGain" +
    "      });" +
    "    }" +
    "    return true;" +
    "  }" +
    "}" +
    "registerProcessor('pcm-capture', PcmCapture);";

  // Plays a near-silent looping audio element so iOS Safari treats
  // this tab as actively producing audio. Without it, Safari
  // suspends the AudioContext when the screen dims, a notification
  // arrives, or the tab loses focus — which is exactly the
  // "cuts out quickly" symptom on iPhone. The audio is real (not
  // muted) so iOS keeps the audio session alive; volume is dropped
  // to 0.001 so the user doesn't hear it. Must be started inside a
  // user gesture (the Start button click); we kick it off lazily
  // from startAudioPipeline which only runs on Start.
  let silentKeepalive = null;
  function ensureSilentKeepalive() {
    if (silentKeepalive) return;
    try {
      // 1s of stereo silence in a base64 WAV. Loops indefinitely.
      const wavB64 =
        "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";
      const audio = new Audio("data:audio/wav;base64," + wavB64);
      audio.loop = true;
      audio.volume = 0.001;
      audio.muted = false;
      audio.preload = "auto";
      audio.setAttribute("playsinline", "");
      const playPromise = audio.play();
      if (playPromise && playPromise.catch) {
        playPromise.catch(() => { /* will retry on next user gesture */ });
      }
      silentKeepalive = audio;
    } catch (_) { /* best-effort */ }
  }

  // Resume the AudioContext when iOS suspends it. iOS Safari
  // suspends the context whenever the audio session is interrupted
  // (screen lock, incoming call, Siri, another tab playing audio).
  // Without explicit resume, the worklet stops emitting samples and
  // the desktop sees a silent stream — the "cuts out" failure.
  // We listen on three signals: AudioContext.statechange (most
  // direct), visibilitychange (catch tab refocus), and pageshow
  // (catch back-forward cache restore).
  function attachAudioResilienceHandlers() {
    if (!audioContext) return;
    const tryResume = () => {
      if (!audioContext) return;
      if (audioContext.state === "suspended") {
        audioContext.resume().catch(() => {});
      }
      if (silentKeepalive && silentKeepalive.paused) {
        silentKeepalive.play().catch(() => {});
      }
    };
    audioContext.addEventListener("statechange", tryResume);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) tryResume();
    });
    window.addEventListener("pageshow", tryResume);
    window.addEventListener("focus", tryResume);
  }

  async function startAudioPipeline(mediaStream) {
    const audioTrack = mediaStream.getAudioTracks()[0];
    if (!audioTrack) return;
    // iOS Safari 14.1+ supports AudioWorklet; earlier doesn't. If it
    // isn't available, silently skip — voice still works with the
    // laptop mic on the PC side.
    if (!window.AudioContext && !window.webkitAudioContext) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    audioContext = new AC({ sampleRate: 48000 });
    audioSampleRate = audioContext.sampleRate || 48000;
    if (!audioContext.audioWorklet) {
      try { await audioContext.close(); } catch (_) {}
      audioContext = null;
      return;
    }
    const blob = new Blob([PCM_WORKLET_SRC], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    try {
      await audioContext.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    audioSourceNode = audioContext.createMediaStreamSource(mediaStream);
    audioWorkletNode = new AudioWorkletNode(audioContext, "pcm-capture");
    audioWorkletNode.port.onmessage = (e) => {
      const d = e.data;
      // The worklet posts two message kinds: ArrayBuffer (Int16 PCM)
      // for upload, or a plain object for stats / telemetry.
      if (d instanceof ArrayBuffer) {
        if (!sending) return;
        audioQueue.push(d);
        audioQueuedBytes += d.byteLength;
        // Fire-and-forget flush at ~100ms worth of samples.
        if (audioQueuedBytes >= AUDIO_CHUNK_BYTES) {
          flushAudioQueue();
        }
        return;
      }
      if (d && typeof d === "object" && d.type === "stats") {
        renderAudioStats(d);
      }
    };
    audioSourceNode.connect(audioWorkletNode);
    // Push the initial config (pre_gain + comp_threshold) corresponding
    // to the currently-selected "Mic distance" mode. Mode changes after
    // this just re-post a config message; no need to rebuild the worklet.
    sendAudioConfigForMode(distSelect ? distSelect.value : "near");
    // Intentionally NOT connected to destination — we don't want to
    // play the mic back through the phone's speakers.
    ensureSilentKeepalive();
    attachAudioResilienceHandlers();
  }

  async function flushAudioQueue() {
    if (audioInflight || audioQueue.length === 0) return;
    // Coalesce pending chunks into one contiguous buffer.
    const merged = new Uint8Array(audioQueuedBytes);
    let offset = 0;
    for (const ab of audioQueue) {
      merged.set(new Uint8Array(ab), offset);
      offset += ab.byteLength;
    }
    audioQueue = [];
    audioQueuedBytes = 0;
    audioInflight = true;
    try {
      await fetch("/audio", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: merged,
        cache: "no-store",
        keepalive: false,
      });
    } catch (_) { /* drop on error; next chunk will retry */ }
    finally { audioInflight = false; }
  }

  // Mode → worklet config table. Near matches the original close-talk
  // tuning. Far bumps pre-gain by 6 dB (16x → 32x) and lowers the
  // compressor threshold from -3 dBFS to -7 dBFS so far-field speech,
  // which has a smaller dynamic range, also gets reliably compressed
  // up to a steady output level.
  function audioConfigForMode(mode) {
    if (mode === "far") return { preGain: 32.0, compThreshold: 0.45 };
    return { preGain: 16.0, compThreshold: 0.7 };
  }

  function sendAudioConfigForMode(mode) {
    if (!audioWorkletNode) return;
    try { audioWorkletNode.port.postMessage(audioConfigForMode(mode)); }
    catch (_) {}
  }

  // Live audio readout. Shows the estimated room-noise floor, the
  // gate threshold (= floor × scale), and an open/quiet indicator,
  // so the user can see at a glance whether the mic is in a
  // workable environment and pick a phone position accordingly.
  function renderAudioStats(s) {
    if (!audioStatsEl) return;
    const dB = (v) => v > 1e-7 ? (20 * Math.log10(v)).toFixed(0) + " dBFS" : "-inf";
    const status = s.gateOpen ? "speech" : "silence";
    const gainTxt = s.preGain ? Math.round(s.preGain) + "x" : "";
    audioStatsEl.style.display = "block";
    audioStatsEl.textContent =
      "noise " + dB(s.noiseFloor) +
      " · gate " + dB(s.gateThreshold) +
      " · " + status +
      (gainTxt ? " · " + gainTxt : "");
  }

  function clearAudioStats() {
    if (!audioStatsEl) return;
    audioStatsEl.style.display = "none";
    audioStatsEl.textContent = "";
  }

  function stopAudioPipeline() {
    if (audioWorkletNode) {
      try { audioWorkletNode.disconnect(); } catch (_) {}
      try { audioWorkletNode.port.close(); } catch (_) {}
      audioWorkletNode = null;
    }
    if (audioSourceNode) {
      try { audioSourceNode.disconnect(); } catch (_) {}
      audioSourceNode = null;
    }
    if (audioContext) {
      try { audioContext.close(); } catch (_) {}
      audioContext = null;
    }
    audioQueue = [];
    audioQueuedBytes = 0;
    audioInflight = false;
    clearAudioStats();
  }

  async function startLoop() {
    startBtn.disabled = true;
    startBtn.textContent = "Connecting...";
    setStatus("Opening camera...", "");
    let s;
    try {
      s = await openCamera();
      setStatus("Camera ready at " + s.width + "x" + s.height + " @ " + Math.round(s.fps || currentFps()) + "fps. Connecting...", "");
    } catch (err) {
      setStatus("Camera denied: " + (err && err.name || err), "err");
      startBtn.disabled = false;
      startBtn.textContent = "Start";
      return;
    }
    // Verify the PC is reachable before we start pumping frames. If this
    // probe fails the user has a network/firewall problem, not a cert
    // one — surface that clearly instead of silently dropping frames.
    const reachable = await probeHttp();
    if (!reachable) {
      setStatus("Could not reach the PC. Make sure Touchless is still open on your PC and your phone is on the same WiFi network.", "err");
      startBtn.disabled = false;
      startBtn.textContent = "Start";
      return;
    }
    consecutivePostErrors = 0;
    setStatus("Streaming. Keep this tab open.", "ok");
    startBtn.textContent = "Stop";
    startBtn.disabled = false;
    requestWakeLock();
    // Reuse the on-page #previewCanvas as the send canvas so the visible
    // preview IS the exact RAW frame being POSTed. The CSS scaleX(-1) on
    // #previewCanvas flips it to match the engine's selfie flip, so phone
    // preview == app display regardless of front-camera browser quirks.
    canvas = document.getElementById("previewCanvas");
    ctx = canvas.getContext("2d", { alpha: false });
    canvas.classList.add("live");
    sending = true;
    frameCount = 0;
    lastStatsAt = performance.now();
    lastSentBytes = 0;
    lastSentAt = 0;
    sendFrames();
  }

  async function sendFrames() {
    const loop = async (ts) => {
      if (!sending) return;
      const targetInterval = 1000 / currentFps();
      if (ts - lastSentAt >= targetInterval) {
        lastSentAt = ts;
        await sendOneFrame();
      }
      if (sending) requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  async function sendOneFrame() {
    if (!stream) return;
    // Skip the encode entirely if we're still waiting on the previous
    // POST — lets the network be the bottleneck instead of memory.
    if (inflightPost) return;
    const track = stream.getVideoTracks()[0];
    if (!track) return;
    const s = track.getSettings ? track.getSettings() : {};
    const w = s.width || 640;
    const h = s.height || 480;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    ctx.drawImage(previewEl, 0, 0, w, h);
    const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", currentQuality()));
    if (!blob) return;
    inflightPost = true;
    let ok = false;
    try {
      // fetch() over HTTPS honors the user's trusted root CA (as opposed
      // to WSS, which iOS WebKit rejects even with a trusted local root).
      // This is the transport that Just Works on iOS self-signed setups.
      const resp = await fetch("/frame", {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: blob,
        cache: "no-store",
        keepalive: false,
      });
      ok = resp.ok || resp.status === 204;
    } catch (_) {
      ok = false;
    } finally {
      inflightPost = false;
    }
    if (ok) {
      consecutivePostErrors = 0;
      frameCount += 1;
      lastSentBytes += blob.size;
    } else {
      consecutivePostErrors += 1;
      if (consecutivePostErrors >= 10) {
        setStatus("Lost connection to the PC. Tap Start to reconnect.", "err");
        stopLoop();
        return;
      }
    }
    const now = performance.now();
    if (now - lastStatsAt >= 1000) {
      const fps = frameCount * 1000 / (now - lastStatsAt);
      const kbps = (lastSentBytes * 8 / 1024) / ((now - lastStatsAt) / 1000);
      statsEl.textContent = fps.toFixed(1) + " fps | " + kbps.toFixed(0) + " kbps | " + w + "x" + h;
      frameCount = 0;
      lastSentBytes = 0;
      lastStatsAt = now;
    }
  }

  function stopLoop() {
    sending = false;
    // Hide the flipped preview canvas so the live <video> shows again
    // for re-framing before the next Start.
    try {
      const pc = document.getElementById("previewCanvas");
      if (pc) pc.classList.remove("live");
    } catch (_) {}
    stopAudioPipeline();
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (wakeLock) { try { wakeLock.release(); } catch (_) {} wakeLock = null; }
    statsEl.textContent = "";
    setStatus("Stopped.", "");
    startBtn.textContent = "Start";
    startBtn.disabled = false;
  }

  function enterFullscreen() {
    // Try the native Fullscreen API (Chrome / Android Firefox). On iOS
    // Safari, element.requestFullscreen is a no-op for non-video
    // elements and the video's webkitEnterFullscreen forces landscape
    // and strips UI — so for iOS we fall back to our CSS-positioned
    // .fs-active layout that at least fills the viewport.
    previewWrap.classList.add("fs-active");
    moveToastStackToFullscreen();
    const req = previewWrap.requestFullscreen || previewWrap.webkitRequestFullscreen;
    if (typeof req === "function") {
      try { req.call(previewWrap); } catch (_) {}
    }
  }

  function exitFullscreen() {
    previewWrap.classList.remove("fs-active");
    moveToastStackToBody();
    const exit = document.exitFullscreen || document.webkitExitFullscreen;
    if (typeof exit === "function") {
      try { exit.call(document); } catch (_) {}
    }
  }

  // Reparent the toast container into / out of the fullscreen
  // element. The browser's native Fullscreen API hides every DOM
  // node that isn't a descendant of the fullscreened element, so
  // a body-level fixed-position stack disappears when the user
  // hits the system fullscreen mode. Moving the toast-stack inside
  // previewWrap keeps it visible in both the CSS-fallback and
  // native-fullscreen paths.
  function moveToastStackToFullscreen() {
    const stack = document.getElementById("toastStack");
    if (!stack || !previewWrap) return;
    if (stack.parentElement !== previewWrap) {
      previewWrap.appendChild(stack);
    }
    stack.classList.add("fs-mode");
  }
  function moveToastStackToBody() {
    const stack = document.getElementById("toastStack");
    if (!stack) return;
    if (stack.parentElement !== document.body) {
      document.body.insertBefore(stack, document.body.firstChild);
    }
    stack.classList.remove("fs-mode");
  }

  async function restartStreamOnSettingsChange() {
    if (!sending) return;
    setStatus("Updating camera settings...", "");
    try {
      const s = await openCamera();
      setStatus("Streaming (" + s.width + "x" + s.height + " @ " + Math.round(s.fps || currentFps()) + "fps). Keep this tab open.", "ok");
    } catch (err) {
      setStatus("Could not change settings: " + (err && err.name || err), "err");
    }
  }

  startBtn.addEventListener("click", () => { if (sending) stopLoop(); else startLoop(); });

  // Text command prompt — POST {text} to /command. Same intent surface
  // as voice commands. The PC dispatches via the voice command
  // processor on the GUI thread; the outcome arrives back via the SSE
  // stream below (kind="command_result") and gets rendered as a toast.
  const commandInputEl = document.getElementById("commandInput");
  const commandSendBtn = document.getElementById("commandSend");
  async function sendCommandPrompt() {
    if (!commandInputEl) return;
    const text = (commandInputEl.value || "").trim();
    if (!text) return;
    commandSendBtn.disabled = true;
    try {
      const resp = await fetch("/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        cache: "no-store",
        keepalive: false,
      });
      if (resp.status === 503) {
        showToast("", "Start Touchless on the PC first.", "Command");
      } else if (resp.ok || resp.status === 202) {
        showToast("", text, "Sent");
        commandInputEl.value = "";
      } else {
        showToast("", "Send failed (" + resp.status + ")", "Command");
      }
    } catch (err) {
      showToast("", "Network error", "Command");
    } finally {
      commandSendBtn.disabled = false;
      commandInputEl.focus();
    }
  }
  if (commandSendBtn) commandSendBtn.addEventListener("click", sendCommandPrompt);
  if (commandInputEl) commandInputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); sendCommandPrompt(); }
  });
  fsBtn.addEventListener("click", () => {
    if (previewWrap.classList.contains("fs-active")) exitFullscreen();
    else enterFullscreen();
  });
  fsExitBtn.addEventListener("click", exitFullscreen);
  miniBtn.addEventListener("click", () => {
    document.body.classList.toggle("mini");
    miniBtn.textContent = document.body.classList.contains("mini") ? "Expand" : "Mini";
  });
  // Changing res/fps/facing while streaming re-opens the camera with
  // the new constraints. Quality change is instantaneous (applied to
  // the next canvas.toBlob call) so no restart needed.
  resSelect.addEventListener("change", restartStreamOnSettingsChange);
  fpsSelect.addEventListener("change", restartStreamOnSettingsChange);
  facingSelect.addEventListener("change", restartStreamOnSettingsChange);
  micSelect.addEventListener("change", restartStreamOnSettingsChange);
  // Mic-distance mode just re-pushes config to the running worklet —
  // no need to re-open the camera/audio stream, so the user gets an
  // immediate change with no hiccup in the active conversation.
  if (distSelect) {
    distSelect.addEventListener("change", () => sendAudioConfigForMode(distSelect.value));
  }

  // ESC / back-gesture to exit fullscreen
  document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement) {
      previewWrap.classList.remove("fs-active");
      moveToastStackToBody();
    }
  });

  // ----- Toast notifications pushed FROM the PC via SSE -------------
  //
  // Touchless-on-PC publishes events when a gesture triggers an action
  // ("Right swipe → Next track") or a voice command is recognized
  // ("play Spotify"). We subscribe once on page load and render each
  // event as a slide-in toast at the top of the screen, capped to the
  // most recent 3 so a flurry of detections doesn't fill the viewport.
  // ------------------------------------------------------------------

  const toastStack = document.getElementById("toastStack");
  const TOAST_MAX = 3;
  const TOAST_LIFETIME_MS = 1700;

  function showToast(kind, text, sublabel) {
    if (!toastStack || !text) return;
    const el = document.createElement("div");
    el.className = "toast " + (kind || "");
    if (sublabel) {
      const k = document.createElement("span");
      k.className = "toast-kind";
      k.textContent = sublabel;
      el.appendChild(k);
    }
    el.appendChild(document.createTextNode(text));
    toastStack.appendChild(el);
    while (toastStack.children.length > TOAST_MAX) {
      toastStack.removeChild(toastStack.firstElementChild);
    }
    setTimeout(() => {
      el.classList.add("fade-out");
      setTimeout(() => {
        if (el.parentElement) el.parentElement.removeChild(el);
      }, 300);
    }, TOAST_LIFETIME_MS);
  }

  let eventSource = null;
  let sseRetryDelay = 1000;
  function connectEventStream() {
    try {
      // Same-origin endpoint on this very server; no CORS, no auth
      // shenanigans, the same TLS chain the page itself loaded over.
      eventSource = new EventSource("/events");
    } catch (_) {
      eventSource = null;
      return;
    }
    eventSource.addEventListener("hello", () => {
      // Successful handshake — reset backoff so a long-lived connection
      // that drops once falls back to fast reconnect.
      sseRetryDelay = 1000;
    });
    eventSource.onmessage = (event) => {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch (_) {
        return;
      }
      if (!payload || typeof payload !== "object") return;
      const kind = String(payload.kind || "");
      if (kind === "gesture") {
        const label = String(payload.action_text || payload.label || "Gesture");
        showToast("gesture", label, "Gesture");
      } else if (kind === "voice") {
        const text = String(payload.text || "");
        if (text) showToast("voice", text, "Heard");
      } else if (kind === "status") {
        const message = String(payload.message || "");
        if (message) showToast("", message, "");
      } else if (kind === "command_result") {
        // Result of a phone-side /command POST. `ok` true means the
        // voice processor recognised + executed the intent; false means
        // it was either unrecognised or the action itself failed.
        const ok = Boolean(payload.ok);
        const heard = String(payload.heard_text || "");
        const detail = String(payload.info_text || payload.control_text || "");
        const head = ok ? "Done" : "Couldn't run";
        const body = detail || heard || "command";
        showToast(ok ? "voice" : "", body, head);
      }
    };
    eventSource.onerror = () => {
      // EventSource auto-reconnects on its own, but on iOS Safari that
      // sometimes goes dormant. Force a fresh connect after a backoff.
      try { eventSource.close(); } catch (_) {}
      eventSource = null;
      const delay = Math.min(sseRetryDelay, 10000);
      sseRetryDelay = Math.min(sseRetryDelay * 1.6, 10000);
      setTimeout(connectEventStream, delay);
    };
  }
  connectEventStream();

  window.addEventListener("beforeunload", () => {
    stopLoop();
    if (eventSource) {
      try { eventSource.close(); } catch (_) {}
    }
  });
})();
</script>
</body>
</html>
"""

# Author: Konstantin Markov
