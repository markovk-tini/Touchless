# Iris Cortex — 3D Cinematic Visualization Spec

A clickable, draggable, cinematic 3D view of Iris's live thinking.
Opens when the user clicks "IRIS" in the titlebar. Shows a glowing
central core with branches extending outward for every active
project, thought, memory, and tool Iris is engaged with. Responds
in real time to Iris's actual activity.

Iris code lives at `src/hgr/live_api/`. This visualization is a
new subsystem under `src/hgr/live_api/cortex/`.

This is both a debug surface and a marketing asset (see
`c:\touchless-marketing\`).

---

## Vision

```
                    ╭─── ⬡ recent voice query
                   ╱
    ⬡ KiCAD ────────╮
       ╲           ╲╲
        ╲           ╲ ╲              ⬡ calendar tool
         ╲           ╲ ╲            ╱
   ⬡ ─────╲           ╲ ╲          ╱
   memory ─╲╲          ╲ ╲       ╱
            ╲╲          ╲ ╲    ╱
             ╲╲╲    ╔═════╗ ╱╱
              ╲╲╲══╣ IRIS ╠══════ ⬡ Touchless dev
               ╲╲╲ ╚═════╝ ╲╲╲
              ╱╱╱    ║      ╲╲╲
             ╱╱╱     ║       ╲╲╲
            ╱╱      ║         ╲╲
           ╱        ║           ⬡ Jarvis project
          ╱        ╔═══╗
    ⬡ ───╱         ║TTS║
   voice           ╚═══╝
   stream                            ⬡ marketing strategy
```

- **Core:** glowing icosphere at origin, breathes at idle, brightens
  when active, ripples when listening
- **First ring (capabilities):** Memory · Voice · Tools · Realtime
  Link — persistent, fixed in orbit
- **Second ring (active context):** dynamic — every project,
  conversation, or thought Iris is currently engaged with; spawned
  by `node.spawn` events, fade out via `node.fade`
- **Leaves:** specific artifacts, files, tool results attached to
  context nodes (e.g. KiCAD node → `schematic.kicad_sch` leaf)
- **Edges:** thin lines connecting everything; animated light pulses
  travel along them when data flows (memory → core, core → tool)
- **Camera:** orbit controls — drag right-mouse to rotate, scroll to
  zoom, drag middle-mouse to pan
- **Interaction:** click-drag any node to pull it through space;
  physics springs the rest of the graph around it
- **Background:** dark void with subtle starfield + ambient fog
- **Post-processing:** UnrealBloom + slight film grain + faint
  chromatic aberration on edges

Everything has inertia and easing. Nodes don't snap — they spring.
Edges don't appear — they unfurl. Pulses travel at human-readable
speeds (~400–800ms). This is what makes it feel alive instead of
clinical.

---

## Tech stack (the actual choice)

| Layer | Tool | Why |
|---|---|---|
| Window | PySide6 `QMainWindow` (modeless) | Already the app stack |
| Host | `QWebEngineView` | Embeds a browser-grade renderer in the Qt app |
| Renderer | **Three.js** (r160+) | Battle-tested 3D, huge ecosystem |
| Graph engine | **3d-force-graph** | Node-edge graph with physics + drag built in |
| Post-processing | **UnrealBloomPass** + **FilmPass** | Cinematic glow + grain |
| Camera | **OrbitControls** | Drag-orbit-zoom out of the box |
| Bridge | **QWebChannel** | Python → JS event stream |
| Audio reactivity | Web Audio API (RX from Iris VAD) | Core pulses with voice in |

**Avoided alternatives:**
- ❌ Qt3D / QtQuick3D — works but ~5× the code for the same look;
  smaller example pool
- ❌ pyqtgraph.opengl — fast but visually utilitarian; would fight
  defaults the whole way
- ❌ VisPy — scientific aesthetic, not cinematic
- ❌ Panda3D embedded — heavyweight, awkward to embed
- ❌ raw PyOpenGL — possible, but you'd reimplement Three.js

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Iris (Python — src/hgr/live_api/)                           │
│  ─────────────────                                           │
│  - realtime_client.py    (GPT Realtime loop)                 │
│  - planner/              (classifier, orchestrator, plan)    │
│  - memory/manager.py     (memory retrieval)                  │
│  - tool_registry.py      (tool router)                       │
│  - connectors/           (kicad, drive, ollama, spotify, ms) │
│                                                              │
│    │ emits events                                            │
│    ▼                                                         │
│  cortex/event_bus.py     (Python)                            │
│    - Buffer + throttle to ~30 events/sec max                 │
│    - Coalesce rapid pulses                                   │
└──────────────────────────────────────────────────────────────┘
                  │ QWebChannel
                  ▼
┌──────────────────────────────────────────────────────────────┐
│  cortex/window.py — CortexWindow (QMainWindow, modeless)     │
│    QWebEngineView                                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  cortex/web/cortex.html  (JS)                         │  │
│  │  ─────────────                                        │  │
│  │  - Three.js scene                                     │  │
│  │  - 3d-force-graph instance                            │  │
│  │  - OrbitControls                                      │  │
│  │  - EffectComposer (UnrealBloom + FilmPass)            │  │
│  │  - Bridge listener (cortex.onEvent)                   │  │
│  │  - Core mesh (icosphere + custom shader)              │  │
│  │  - Edge pulse system (instanced points)               │  │
│  │  - Particle field                                     │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

The visualization is a **pure consumer** of the event stream. It
doesn't know anything about LLMs, memory implementations, or tool
internals. This separation matters because:

1. You can replay event logs to debug past sessions
2. You can fake event streams to record demo videos
3. You can swap the entire visualization without touching Iris
4. The same event stream powers a future minimal "headless"
   visualization for marketing reels

---

## Data model — events Iris must emit

One JSON message per event. Buffer and throttle to ~30/sec max in
`cortex/event_bus.py` before sending across the bridge.

```python
# Core state — fires on every state transition
{"type": "core.state", "state": "thinking", "intensity": 0.7}
# states: idle | listening | thinking | retrieving | speaking | error

# Voice waveform sample — fires at ~30hz during listening/speaking
{"type": "core.audio", "rms": 0.42}

# Spawn a new active-context node
{"type": "node.spawn",
 "id": "kicad-2026-05-29",
 "label": "KiCAD project",
 "category": "project",        # project|memory|tool|conversation
 "weight": 0.8,                 # 0..1, controls size + initial brightness
 "parent_id": null}            # null = attaches to core; else nests

# Activity on an existing node (lights it up)
{"type": "node.activity",
 "id": "memory-pool",
 "intensity": 0.9,
 "duration_ms": 600}

# An edge pulses (data flowing along the connection)
{"type": "edge.pulse",
 "from": "memory-pool",
 "to": "core",
 "color": "cyan",              # cyan|amber|magenta|red
 "duration_ms": 500}

# A leaf appears (artifact, tool result, retrieved memory)
{"type": "leaf.add",
 "parent_id": "kicad-2026-05-29",
 "label": "schematic.kicad_sch",
 "ttl_ms": 8000}               # auto-fades after ttl

# Node fades out (context ended)
{"type": "node.fade",
 "id": "kicad-2026-05-29",
 "duration_ms": 1500}

# JS → Python (user interaction)
{"type": "ui.node_focused", "id": "..."}
{"type": "ui.leaf_opened", "leaf_id": "..."}
```

**Coalescing rules in `event_bus.py`:**
- Multiple `core.audio` events within 33ms → keep last
- Multiple `edge.pulse` with same from/to within 100ms → merge
- `node.activity` on same id within 100ms → keep higher intensity

---

## Where events come from (Iris subsystems)

Map Iris's existing subsystems to events:

| Iris source | Event(s) emitted |
|---|---|
| `realtime_client.py` VAD | `core.audio`, `core.state` (listening/idle) |
| `realtime_client.py` token stream | `core.state` (speaking) |
| `planner/orchestrator.py` thinking | `core.state` (thinking) |
| `memory/manager.py` retrieve | `node.activity` on memory node, `edge.pulse` memory→core, `leaf.add` per retrieved memory |
| `tool_registry.py` dispatch | `edge.pulse` core→tool, `node.activity` on tool node |
| `tool_executor.py` result | `leaf.add` on tool node with result label |
| `connectors/*` open artifact | `node.spawn` for new project context, `leaf.add` for artifact |
| `live_api_manager.py` context switch | `node.fade` for the old context |

Don't wire every subsystem at once — see Build Plan v5.

---

## Visual design details

### Core
- Icosphere, radius 8, subdivision 4
- Custom GLSL shader: noise-displaced surface, time-varying
- Audio-reactive: `core.audio` events scale displacement amplitude
- Color: cyan when listening, amber when thinking, magenta when
  retrieving memory, white-ish when speaking
- Two-layer: solid inner sphere + slightly larger translucent shell
  for halo effect
- Idle breathing: sin(t * 0.5) * 0.05 scale modulation

### Capability nodes (first ring)
- Octahedrons, radius 2
- Fixed orbit at radius 20 from core, evenly spaced
- Labels rendered as billboarded sprites (always face camera)
- Faint connecting edge to core (always visible, dim)
- Brightens when its capability is invoked

### Context nodes (second ring)
- Spheres, radius scaled 1.5–4 by `weight`
- Force-directed: spring toward radius 40, repel each other
- Draggable: mouse pickup applies temporary force, release allows
  spring back
- Color tinted by category:
  - project = teal
  - memory = violet
  - tool = amber
  - conversation = soft blue
- Pulse on `node.activity` (radial scale 1.0 → 1.3 → 1.0 over
  `duration_ms`, plus emissive intensity boost)
- Spawn animation: scale 0 → target over 800ms, OutBack easing

### Leaves
- Small icosahedra, radius 0.7
- Attached to parent by short edge (length ~5)
- Force-clustered around parent
- Auto-fade after `ttl_ms` (default 8s)

### Edges
- Thin tube geometry (radius 0.05) — looks better than lines under
  bloom
- Base color faint cyan, low opacity (0.15)
- Pulse: bright traveling segment along the edge, ~10% of edge
  length, animated from source to target over `duration_ms`
- Implemented as instanced points moving along a curve for
  performance

### Background
- Solid #050a14 base
- Subtle radial gradient brighter near center
- Starfield: 3000 points, slight parallax with camera
- Ambient fog: exponential, density 0.005, color #050a14
- Optional: very faint volumetric god-rays from core

### Post-processing
- `UnrealBloomPass`: strength 1.5, threshold 0.2, radius 0.4
- `FilmPass`: noise intensity 0.15, scanline intensity 0.0 (we
  want grain, not scanlines)
- `RGBShiftPass` (very subtle, amount 0.001) for organic feel
- Disabled by a feature flag for low-end GPUs

---

## Interactions

| Input | Action |
|---|---|
| Left-click drag on node | Pull node through space (physics responds) |
| Release | Node springs back into orbit |
| Left-click on node (no drag) | Focus camera, show side panel with node details + recent payloads |
| Double-click leaf | Send `ui.leaf_opened` event to Iris → opens artifact |
| Right-click drag on background | Orbit camera around core |
| Scroll wheel | Zoom (camera dolly, not FOV) |
| Middle-click drag | Pan |
| Hover on node | Tooltip with label + recent activity timestamp |
| `R` key | Reset camera to default position |
| `F` key | Toggle fullscreen |
| `D` key | Toggle debug overlay (event rate, FPS, node count) |
| `B` key | Toggle bloom (for low-end GPUs) |

---

## Window behavior

- **Modeless** — `QMainWindow` shown via `.show()`, not `.exec()`.
  Iris keeps running in the background. Closing the window does
  not stop Iris. (Matches CLAUDE.md's "no blocking modal UI"
  rule.)
- **Frameless** with custom titlebar (matches the cinematic vibe).
  Custom titlebar has a small "✕" close, a "—" minimize, and the
  text "IRIS CORTEX" on the left.
- Click-and-drag the titlebar to move the window.
- Default size: 1280×800; restores last position/size to QSettings.
- Always on top: off by default; toggleable from the titlebar.

---

## Triggering from the Assistant window

The trigger lives on the **Assistant window** (`live_assistant_window.py`),
not the main Touchless gesture window. That window already has a
visible header label at the top reading "Touchless Assistant"
(see `_build_ui`, around line 118). The plan is:

1. **Rename** the header `QLabel` text from "Touchless Assistant" →
   "Iris"
2. **Restyle** it slightly so it reads as interactive (cursor
   changes to pointer on hover, subtle underline or color shift,
   maybe a small 🧠 or arrow glyph after the word)
3. **Make it clickable** — subclass `QLabel` to emit a `clicked`
   signal on `mousePressEvent`, or use a `QPushButton` styled to
   look like a label
4. **On click**, instantiate `CortexWindow` (modeless) and call
   `.show()` — Iris and the assistant keep running

Recommended approach: a tiny `ClickableLabel(QLabel)` subclass that
emits a `clicked = Signal()` on mouse press. ~10 lines. Keeps the
existing styling pipeline intact, no `QPushButton` chrome.

The window's OS title (set by `self.setWindowTitle(...)` at line
~93) can stay as "Touchless Assistant" or change to "Iris" —
that affects the taskbar entry, not the in-window label.

**State pill placement:** the existing colored state pill (Off /
Connecting / Listening / Thinking / Executing / Error) sits to the
right of the title. Keep it there — it complements "Iris" cleanly:

```
┌──────────────────────────────────────────────┐
│ Iris 🧠         [ Thinking… ]                │  ← clickable
├──────────────────────────────────────────────┤
│ (chat transcript / tool pills / etc.)        │
└──────────────────────────────────────────────┘
```

**Regression risk:** this is a low-risk UI change — only the
header label is touched, no titlebar flag changes, no main-window
modifications. Still verify per CLAUDE.md workflow:
- Assistant window still opens cleanly
- Header layout doesn't shift visually beyond the relabel
- Existing state-pill color/text behavior unchanged
- The new click handler doesn't interfere with title text
  selection (it shouldn't — `QLabel` text selection is off by
  default)

---

## Build plan

| Stage | What | Time |
|---|---|---|
| **v0 — scaffold** | `QWebEngineView` window, blank Three.js scene with camera + OrbitControls + a placeholder sphere. Verify QWebChannel round-trip with a debug event. | 1 evening |
| **v1 — idle world** | Core (no shader yet), 4 capability nodes in orbit, dim edges, starfield, ambient fog. Idle breathing on core. Everything still, no events. | 1 evening |
| **v2 — event-driven graph** | Wire `3d-force-graph`, implement `node.spawn`, `node.fade`, `edge.pulse`, `leaf.add`, `node.activity`. Replace force-directed default styling with custom node renderers. | 1 weekend |
| **v3 — cinematic polish** | Add `UnrealBloomPass` + `FilmPass`, write the custom core shader (noise displacement + audio reactivity), tube-geometry edges with traveling pulse, particle field, color palette per category. | 1 weekend |
| **v4 — interaction** | Drag nodes (physics integration), click to focus + side panel, double-click leaf to dispatch `ui.leaf_opened`, hover tooltip, keybindings, frameless window. | 1 evening |
| **v5 — Iris integration** | Implement `event_bus.py` in Iris, emit events from the actual subsystems (memory retrieval, tool calls, VAD), throttle + coalesce. | 1 weekend |

Total: ~2 weekends + 3 evenings to "the version worth filming."

---

## Performance notes

- 60fps target on integrated graphics with ≤ 200 active nodes
- LOD: nodes >150 units from camera render as billboarded sprites
  instead of meshes
- Cap simultaneous edge pulses at 30; queue extras
- `event_bus.py` throttles to 30 events/sec
- `requestAnimationFrame` for render loop; do NOT couple to event
  rate
- Disable bloom + film pass via `B` key for low-end GPUs (still
  looks decent, just less wow)

---

## File layout (when built)

```
c:\HGR App v1.0.0\
├── src\
│   └── hgr\
│       └── live_api\          # = Iris
│           ├── cortex\        # NEW subsystem (this spec)
│           │   ├── __init__.py
│           │   ├── event_bus.py      # CortexEventBus — throttle + coalesce
│           │   ├── window.py         # CortexWindow (QMainWindow modeless)
│           │   ├── bridge.py         # QWebChannel bridge object
│           │   └── web\
│           │       ├── cortex.html
│           │       ├── cortex.js     # Three.js scene + 3d-force-graph
│           │       ├── shaders\
│           │       │   ├── core.vert
│           │       │   ├── core.frag
│           │       │   └── edge_pulse.frag
│           │       ├── vendor\
│           │       │   ├── three.min.js
│           │       │   ├── 3d-force-graph.js
│           │       │   ├── OrbitControls.js
│           │       │   ├── EffectComposer.js
│           │       │   ├── UnrealBloomPass.js
│           │       │   └── FilmPass.js
│           │       └── styles.css
│           ├── realtime_client.py    # existing
│           ├── planner\              # existing
│           ├── memory\               # existing
│           └── ...                   # existing
└── docs\
    └── IRIS_VISUALIZATION.md         # this file
```

All vendored JS files are checked into the repo (no CDN) so it
works offline and without a network round-trip on startup. This
also matters for packaging: PyInstaller bundle needs to include
`cortex/web/` as a data folder (update `hgr_app.spec`).

---

## Packaging notes

When this ships in a built installer:

- Add `src/hgr/live_api/cortex/web/` to `hgr_app.spec` as a
  `datas` entry so vendored JS / HTML / shaders ship inside the
  bundle
- `QWebEngineView` requires the Qt WebEngine binaries — already
  bundled by PyInstaller via PySide6 hooks, but verify the
  installed `.exe` opens the cortex window without missing-DLL
  errors before tagging a release
- Path resolution: use `runtime_paths.app_base_path()` to locate
  `cortex.html` so it works from source AND from the installed
  bundle (see `project_installed_app_subprocess.md` memory for
  the install-vs-source pattern)

---

## Marketing crossover

The cinematic visualization is itself prime content for the
Touchless+Iris reels planned in `c:\touchless-marketing\`. A clip
showing you say something to Iris, the cortex panel pulsing as
it thinks, then Iris actually doing the thing on screen — is
exactly the "wow factor" hook that stops a scroll. Record at
60fps with OBS for crisp slowmo in post.

When recording for marketing:
- Toggle the debug overlay off (`D`)
- Set bloom slightly stronger (`bloomStrength = 1.8`)
- Use a slightly orbiting camera (auto-rotate) for kinetic feel
- Frame at 16:9 in the window, crop to 9:16 in post

---

## Status

Spec written 2026-05-29. Not yet implemented. Lives in
`c:\HGR App v1.0.0\docs\` — build work happens in the HGR App
session, under `src/hgr/live_api/cortex/`.
