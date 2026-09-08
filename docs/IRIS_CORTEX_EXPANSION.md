# Iris Cortex — Expansion Plan: Living World Model

The cortex is now **Iris's world model rendered as a network**. Anything
Iris learns, memorizes, or touches appears here and persists across
sessions. New rule:

> Every Iris interaction emits a `touch` event that creates or
> refreshes a node in the cortex world. The world is read from /
> written to `cortex_world.json` on every meaningful change.

This is a supplement to `IRIS_VISUALIZATION.md`. That doc covered the
renderer + event protocol. This one covers the **data**, the
**persistence**, and the **interaction model** for drilling into each
node.

---

## What appears in the cortex

### 1. Spine (always present, fixed)
- **IRIS core** — the center
- 4 capabilities in fixed orbit: **Memory · Voice · Tools · Realtime**

### 2. Body (semi-permanent, accumulates from interaction)
- **Projects** — folders Iris has touched (file read/written, app
  opened with that root, mentioned in conversation). Persist forever
  unless the user explicitly removes.
- **Apps Iris has opened** — chrome, vscode, kicad, blender, etc.
  Surface as leaves under **Tools › Apps**.
- **Files Iris has touched** — recently read/edited/created files.
  Persist with visual decay over weeks.

### 3. Memory-managed (under Memory when expanded)
- **Episodic memory** — last N conversation episodes as leaves
- **Semantic facts** — facts extracted by `memory/manager.py`,
  clustered by domain (people, preferences, places, projects)
- **Active context** — what was just loaded into the system prompt

### 4. Transient (the live pulse, not persisted)
- Current voice in (pulses Voice node when speaking)
- Current tool firing (leaf appears, fades after ~5s)
- Current state (colors the core)

---

## Capability expansions (click → fly in)

### MEMORY
When user clicks Memory, the camera flies in. Sub-nodes:

- **Facts cluster** — sub-spheres grouped by domain:
  - `you` (preferences, role, contacts)
  - `places` (frequently mentioned locations)
  - `projects` (projects Iris knows about)
  - `recurring topics` (themes that come up often)
  - Each fact is a clickable leaf showing key/value/source/date
- **Episodes ring** — last 12 conversation episodes orbiting, each
  labeled with goal-text snippet. Click → details bubble with the
  full episode.
- **Active context** — what's currently injected into the live
  system prompt. Pulses with the realtime turn.
- **Memory health** — small indicator: total facts count, episodic
  rows count, last embed time.

### VOICE
Sub-nodes:

- **Backend** — labeled chip showing the active whisper build
  (CUDA / Vulkan / CPU). Pulses orange if it fell back unexpectedly.
- **Recent transcripts** — last 8 phrases as leaves, fade with age
- **TTS voice** — current speaking voice (when audio-out is enabled)
- **Wake / hotword** — if any wake-word listener is armed
- **Dictation targets** — apps Iris has typed into recently
  (chrome, vscode, etc.) as small chips
- **Audio devices** — current mic + speaker chips with health dot

### TOOLS — human action groups (no AI tool names)
The user explicitly asked not to expose internal tool names. Map the
existing `tool_executor.py` ~50 tools to human-action groups:

| Group | Underlying tools (hidden from UI) |
|---|---|
| **Apps & windows** | `open_app`, `close_window`, `control_window`, `move_window_to_monitor` |
| **Files & folders** | `create_file`, `read_file`, `write_file`, `append_file`, `move_file`, `rename_file`, `delete_file`, `create_folder`, `list_files`, `list_recent_paths` |
| **Type & click** | `type_text`, `press_hotkey`, `click_screen`, `click_type`, `drag`, `draw_path`, `draw_shape` |
| **Read the screen** | `read_screen`, `get_screen_context`, `click_text_on_screen`, `wait_for_screen_text` |
| **Zoom & screenshot** | `zoom_screen`, `click_zoom` |
| **Web** | `open_url` (+ future web actions) |
| **Code & scripts** | `send_to_coding_agent`, `follow_up_coding_agent`, `run_matlab_script` |
| **Editor** | `open_in_editor` |
| **Touchless built-ins** | `run_existing_touchless_action`, `run_quick_command` |

Each group is a sphere. Click → expands into its leaves with
**human-facing** labels: "Open Chrome," "Type into focused window,"
"Read what's on screen," etc.

Apps that Iris has actually launched move from leaf to mini-sphere of
their own under **Apps & windows** as their touch count rises.

### REALTIME
Sub-nodes:

- **Connection** — session state pill (connected / idle / disconnected)
- **Model** — abstract label only ("Cloud model" / "Local model"), no
  specific name shown
- **Recent turns** — last 6 message exchanges as leaves
- **Active turn** — current message in flight (transient, fades when
  the assistant finishes speaking)
- **Cost** — tokens used this session, remaining budget if tracked

---

## Project sub-trees

### Schema — `cortex_world.json`

Lives at `%LOCALAPPDATA%\Touchless\cortex_world.json` (project-specific
state) with the same path-resolution pattern as `memory.db`.

```json
{
  "version": 1,
  "projects": {
    "touchless-dev": {
      "label": "Touchless dev",
      "root_path": "c:/HGR App v1.0.0",
      "color": "teal",
      "created_at": "2026-05-29T14:22:00Z",
      "last_touched_at": "2026-05-29T16:01:14Z",
      "touch_count": 47,
      "categories": [
        {"label": "Bugs / planned work", "source": "file:OPEN_ISSUES.md", "kind": "issues_doc"},
        {"label": "Subsystem docs",      "source": "glob:docs/*.md",      "kind": "docs"},
        {"label": "Recent commits",      "source": "git:HEAD~10..HEAD",   "kind": "git_log"},
        {"label": "Active branch",       "source": "git:current",         "kind": "git_branch"}
      ]
    },
    "marketing": {
      "label": "Marketing strategy",
      "root_path": "c:/touchless-marketing",
      "color": "soft-blue",
      "created_at": "2026-05-28T11:05:00Z",
      "last_touched_at": "2026-05-29T15:42:00Z",
      "touch_count": 18,
      "categories": [
        {"label": "Strategy", "source": "glob:docs/*.md", "kind": "docs"},
        {"label": "Planner",  "source": "file:docs/CONTENT_STRATEGY.md"},
        {"label": "Ideas",    "source": "section:docs/CONTENT_STRATEGY.md#hooks-bank"}
      ]
    },
    "kicad-cli": {
      "label": "KiCAD CLI",
      "root_path": null,
      "color": "amber",
      "created_at": "2026-05-27T09:18:00Z",
      "last_touched_at": "2026-05-27T09:18:00Z",
      "touch_count": 2,
      "categories": []
    }
  },
  "files": {
    "c:/HGR App v1.0.0/docs/IRIS_VISUALIZATION.md": {
      "touches": 4,
      "last_touched_at": "2026-05-29T15:55:00Z",
      "project_id": "touchless-dev"
    }
  },
  "apps": {
    "chrome": {"launches": 23, "last_at": "..."},
    "vscode": {"launches": 11, "last_at": "..."}
  },
  "cross_links": [
    {
      "from": "marketing:ideas:ai-pipeline",
      "to":   "touchless-dev:features:iris-content-gen",
      "kind": "manual",
      "weight": 0.8
    }
  ]
}
```

### Auto-detect new projects

When `cortex_emit.touch(kind="file", path=<absolute>)` fires:

1. Walk up the path looking for a folder containing `CLAUDE.md`,
   `README.md`, `pyproject.toml`, `package.json`, `.git/`, or
   `OPEN_ISSUES.md` — that's the inferred project root.
2. If the root is not in `projects`, auto-add it with derived label
   (folder name, title-cased) and default color.
3. Bump touch count + last_touched_at on the project.
4. Log the file under `files`.

### Project sphere size

Computed each tick:
```
weight = clamp(
  log1p(touch_count) / 5.0
  + log1p(file_count) / 8.0
  + recent_activity_bonus(last_touched_at)
  , 0.15, 1.0
)
```
- KiCAD CLI (2 touches, no root): weight ~0.20 → small
- Marketing (18 touches, 5 docs): weight ~0.55 → medium
- Touchless dev (47 touches, hundreds of files): weight ~0.95 → large

### Project content (when expanded)

Categories from the project's schema → sub-spheres around the project.

For **Touchless dev**:
- **Bugs / planned work** — parses sections of `OPEN_ISSUES.md`; each
  section heading becomes a sub-leaf, clickable to open the file at
  that section
- **Subsystem docs** — every `docs/*.md` as a clickable leaf
- **Recent commits** — last 10 git commits as leaves, label = commit
  subject, click → shows the diff (or opens `git show <sha>` in
  default editor)
- **Active branch** — current branch + modified files (live)
- **Features (planned)** — sections of OPEN_ISSUES under "Features"
- **Code** — folders under `src/hgr/` as sub-spheres (don't enumerate
  files — too noisy)

For **Marketing strategy**:
- **Strategy docs** — `docs/*.md`
- **Planner** — clickable leaf opening `docs/CONTENT_STRATEGY.md`
- **Hooks bank** — sub-leaves parsed from the "Hook bank" section,
  one leaf per hook example
- **Setup notes** — Instagram setup, Linktree walkthrough, etc.

For **KiCAD CLI**:
- (Initially empty — just a small sphere)
- When user starts a KiCAD project, root becomes the schematic file's
  parent dir; categories populate

### Clickable file leaves

- **Single click** — focus / show side-panel info card
- **Double-click** — Python opens the file via `os.startfile(path)`
  (Windows-native default-handler)
- **Right-click** — context menu: copy path / show in explorer

Wiring: JS sends `pythonBridge.leafOpened(leaf_id)` → Python looks up
`leaf_id → file_path` in the world model → calls `os.startfile`.

---

## Cross-links

When two nodes in different sub-trees are related, connect them with
a thin dotted line. Three discovery paths:

### A. Manual (start here)
`cortex_world.json["cross_links"]` — user maintains. Each entry:
```json
{"from": "<node_id>", "to": "<node_id>", "kind": "manual", "weight": 0.0..1.0}
```

### B. Markdown-link inference
When scanning a project doc, if it contains a relative path to another
project's file (or an absolute path matching another project root),
auto-create a cross-link.

### C. Semantic-similarity inference (later)
Use the existing `memory/embedder.py` to embed every node's label +
its file contents (truncated). Pairs over a similarity threshold get
auto-cross-linked. Computed in a background thread; results cached.

### Visual treatment
- Line: thin (0.4× edge thickness)
- Style: dotted (Three.js `LineDashedMaterial`)
- Color: muted gray-purple `#6e7099` at 25% opacity
- Brighten + opacify when either endpoint is focused or hovered

---

## Persistence + lifecycle

### Where state lives
- `%LOCALAPPDATA%\Touchless\cortex_world.json` — the world model
- `%LOCALAPPDATA%\Touchless\memory.db` — existing memory store (unchanged)
- Both are loaded on `LiveApiManager.start()`, saved on `_set_state(OFF)`
  + on every meaningful touch (debounced to once per 5s)

### Decay rules
- Fresh touch → +5% scale (capped at 1.5×)
- No touch for 24h → start dimming (-2% opacity / day)
- No touch for 30 days → auto-archive: scale → 0.4, opacity → 0.35,
  push to outer ring, fade out of normal hover
- Archive is reversible: any new touch revives full visibility

### User can pin/unpin
Right-click → "Pin" prevents archival. Pinned nodes keep full
visibility forever.

### What's NOT persisted
- Active state (idle/thinking/etc.)
- Voice level / audio reactivity
- In-flight tool calls (transient pulses)

---

## Interaction model

### Click a capability or project sphere
1. Camera animates toward node center (~1.2s, ease-out cubic)
2. Sibling nodes fade to 30% opacity + push outward 1.5×
3. The selected node expands — sub-nodes appear with stagger animation
4. Breadcrumb at top-left: `IRIS › Memory`

### Click a sub-node
Same drill-down. Breadcrumb extends: `IRIS › Memory › Facts › Preferences`.
Max depth = 4 levels (anything deeper opens the file directly).

### Esc / right-click background
Camera reverses to the previous level. Breadcrumb pops.

### Hover any node
- Tooltip: label, kind (project/file/fact/etc.), last-touched, touch count
- Connected nodes brighten briefly (peek the network)

### Double-click leaf with file path
Opens the file via Python `os.startfile(path)`.

### Right-click any node
Context menu: Pin / Unpin / Hide / Open path / Show in explorer.

---

## Implementation phases

### Phase 1 — Foundation (1 evening)
- `world_state.py` — load/save `cortex_world.json`, debounced writes
- `cortex_emit.touch(kind, label, path=None, project_hint=None)`
  general-purpose entry, called from any subsystem
- Auto-detect-project logic (walk up looking for marker files)
- Replace `_seed_demo` with `_seed_from_world(world)`
- Hook the existing emit calls (state, tool_call, memory_retrieve) to
  also push touches

### Phase 2 — Focus mode + sub-graphs (1 weekend)
- JS: camera fly-in animation on `node.focused` event
- JS: render sub-graph when in focused state, push siblings to outer ring
- JS: breadcrumb UI element
- JS: Esc / back navigation
- Python: when JS sends `node.focused`, query world model for that node's
  children, emit them as a sub-graph

### Phase 3 — Project sub-trees + file links (1 weekend)
- `project_scanner.py` — given a project root + categories, return
  the sub-tree structure (with caching)
- `OPEN_ISSUES.md` section parser (for Touchless dev)
- Markdown section/heading extractor (for Marketing hooks bank, etc.)
- File-leaf rendering with file-icon look
- `bridge.openFile(path)` — Python opens file via os.startfile

### Phase 4 — Cross-links (1 evening)
- Render dotted edges from `cortex_world["cross_links"]`
- Brighten on focus / hover
- Markdown-link inference at scan time

### Phase 5 — Real subsystem wiring (continuous)
- Memory facts → Memory sub-nodes (already wired for retrieval; add
  catalog view)
- Voice config → Voice sub-nodes (need to expose backend + transcripts)
- Tool catalog → Tools sub-nodes (with user-facing labels — table above)
- Auto-detect projects when files touched
- Decay timer + archival

### Phase 6 — Optional polish (later)
- Semantic-similarity cross-link inference
- Right-click context menu
- Pin / unpin UI
- Side-panel info card on click
- Search bar to jump to a node

---

## Open questions for the user

(See the response that accompanies this doc.)
