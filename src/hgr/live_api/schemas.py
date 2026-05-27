"""JSON-schema definitions for tools exposed to the Realtime model.

Each schema follows the OpenAI tool-format used by the Realtime
`session.update.tools` field:
    {"type": "function", "name": ..., "description": ..., "parameters": {...}}

`ToolRegistry.openai_tools()` returns the full list. `validate_args`
enforces required fields and basic types before any executor code
runs — the model is *not* trusted to follow the schema perfectly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_screen_context",
        "description": (
            "Return the current active window title, process name, and "
            "screenshot metadata. Optionally include a fresh screenshot. "
            "Use this whenever you need to see what is on the user's screen "
            "right now."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "include_image": {
                    "type": "boolean",
                    "description": "If true, include a fresh JPEG screenshot.",
                    "default": True,
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_screen",
        "description": (
            "Read the TEXT currently visible on screen (active window) via local "
            "OCR + the accessibility tree. Accurate and cheap — returns text, "
            "not an image, so no vision tokens. Use this to READ or SUMMARIZE "
            "on-screen content: emails in the open Outlook/Mail window, a "
            "document, a chat, a web page, etc. Prefer this over "
            "get_screen_context when you need to read words rather than see "
            "layout. (The user's email lives in their open Outlook window — "
            "read it here; there is no mail-reading API.) Set scroll_passes>0 "
            "(e.g. 6) to auto-scroll the window down and accumulate everything "
            "below the fold — use this to read a WHOLE inbox or long document, "
            "not just the visible top. It stops early once nothing new appears.) "
            "Returns clickable_elements: each visible control/text with its "
            "SCREEN-PIXEL center x,y. To click one, call click_screen with "
            "coordinate_space='screen' and that x,y — exact, no guessing. After "
            "one read_screen you have everything; click and type from it "
            "WITHOUT screenshotting or reading again (re-read only if the UI "
            "actually changed)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scroll_passes": {
                    "type": "integer",
                    "description": "Times to scroll down and re-read (0 = visible "
                                   "screen only; 6 captures a typical full inbox).",
                    "default": 0,
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "click_type",
        "description": (
            "Click a location, type text, and optionally press Enter — ALL IN "
            "ONE call. Use after read_screen to fill a message box or field and "
            "send it in a single step: click_type(x=<px>, y=<py>, "
            "coordinate_space='screen', text='...', submit=true). This replaces "
            "doing separate click + type + send turns (faster, fewer round "
            "trips, avoids rate limits). Confirm with the user before "
            "submit=true on an outward message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "text": {"type": "string", "description": "Text to type after clicking."},
                "coordinate_space": {"type": "string", "enum": ["screen", "normalized"],
                                     "default": "screen"},
                "submit": {"type": "boolean",
                           "description": "Press Enter after typing (e.g. to send).",
                           "default": False},
            },
            "required": ["x", "y", "text"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "click_screen",
        "description": (
            "Move the mouse and click. ALWAYS use coordinate_space='normalized' "
            "with x and y as FRACTIONS of the screen from 0.0 to 1.0 "
            "(x=0.0 is the left edge, x=1.0 the right edge; y=0.0 the top, "
            "y=1.0 the bottom). The screenshots you receive are downscaled, so "
            "raw pixel coordinates are meaningless and land in the wrong place "
            "— estimate the target's position as a fraction of the image you "
            "see. Do NOT use coordinate_space='screen' unless you were given "
            "exact desktop pixel coordinates. If unsure where something is, "
            "call get_screen_context first to refresh your view."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "coordinate_space": {
                    "type": "string",
                    "enum": ["screen", "normalized"],
                    "default": "normalized",
                },
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "default": "left",
                },
                "double_click": {"type": "boolean", "default": False},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "type_text",
        "description": (
            "Type text into a window. Prefer method='clipboard_paste' for long "
            "text/code; 'keyboard_type' only for short text. ALWAYS pass "
            "window_title naming the app you just opened (e.g. 'Notepad') so "
            "the text lands there — the Iris chat is always-on-top, so without "
            "a target the keystrokes can go to the wrong window and type "
            "nothing. NOT for coding agents — use send_to_coding_agent for "
            "Claude/Codex."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "window_title": {
                    "type": "string",
                    "description": (
                        "The target window's title or app name (matched against "
                        "title OR process, e.g. 'Notepad', 'Untitled - Notepad'). "
                        "Strongly recommended; omit only to type into whatever's "
                        "already focused."
                    ),
                },
                "method": {
                    "type": "string",
                    "enum": ["clipboard_paste", "keyboard_type"],
                    "default": "clipboard_paste",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "send_to_coding_agent",
        "description": (
            "Give a coding agent (Claude Code / Codex in VS Code) a task. Use "
            "THIS — never type_text/set_field/click_ui, and never type the "
            "word 'claude'. `prompt` is the actual task. "
            "via='extension' (default): ALWAYS opens a FRESH agent TAB via the "
            "Command Palette (Ctrl+Shift+P -> 'Claude Code: Open in New Tab' -> "
            "Enter), which focuses its input, then types the prompt there — it "
            "does this every time so the prompt never lands in the bottom side "
            "panel. via='terminal': runs the CLI ('claude') in the integrated "
            "terminal and sends the prompt there. "
            "project_folder: when the user wants the work in a NEW project/"
            "folder (e.g. 'make a folder called iris demo'), pass that name "
            "here — it creates the folder under ~/Documents, opens it in VS "
            "Code as the workspace, THEN opens the Claude tab, so the agent's "
            "files land inside that folder. "
            "Do NOT set assume_focused for a normal task — that SKIPS opening "
            "the tab and risks the wrong panel; it's a recovery-only flag. "
            "background=false (default): leave VS Code in front so the user "
            "watches it work. background=true: ONLY when the user says to run it "
            "in the background / without watching — does the setup, then returns "
            "focus to their previous window so they keep working while it runs. "
            "By DEFAULT this also starts a background watcher that clicks every "
            "Yes/Allow/Keep prompt the agent shows — so you do NOT need a "
            "separate auto_approve call. Set auto_approve=false only if the "
            "user explicitly wants to approve prompts themselves. "
            "open_only=true: just OPEN a fresh agent tab and stop, WITHOUT a "
            "prompt — use when the user only asks to 'open a Claude tab' and has "
            "NOT given Claude a task (don't invent one)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "agent": {"type": "string", "enum": ["claude", "codex"], "default": "claude"},
                "via": {"type": "string", "enum": ["extension", "terminal"], "default": "extension"},
                "open_command": {"type": "string"},
                "open_wait_sec": {"type": "number", "default": 2.5},
                "startup_wait_sec": {"type": "number", "default": 4},
                "assume_focused": {"type": "boolean", "default": False},
                "auto_approve": {"type": "boolean", "default": True},
                "background": {"type": "boolean", "default": False},
                "project_folder": {"type": "string"},
                "open_only": {"type": "boolean", "default": False},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "follow_up_coding_agent",
        "description": (
            "Send a FOLLOW-UP message into the SAME Claude Code tab/conversation "
            "that send_to_coding_agent already started — it does NOT open a new "
            "tab (that would lose context). Use it to paste the program's actual "
            "output/error back to Claude and ask it to confirm the result is "
            "correct, do more work, or fix-and-rerun. `message` is the full text "
            "to send (include the quoted output). Keeps the approve-watcher "
            "running so Claude's fixes get auto-approved. Prefer instructing "
            "Claude to self-verify in the original prompt; use this when Claude "
            "couldn't see the output itself or stopped early."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "auto_approve": {"type": "boolean", "default": True},
                "window_title": {"type": "string"},
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "click_text_on_screen",
        "description": (
            "Find VISIBLE TEXT on screen with OCR and click the exact pixel "
            "center of it. Use this for buttons that UIA CANNOT see — custom / "
            "Chromium / webview UIs like Valorant's PLAY button or a launcher "
            "button — and as a precise alternative to click_screen (which only "
            "estimates a fraction on a downscaled image). `text` is the visible "
            "label to click (e.g. 'PLAY', 'Yes', 'Sign in'). Prefers an exact "
            "word match; `occurrence` (1-based, reading order) picks among "
            "multiple matches. Returns no_match if the text isn't on screen — "
            "then wait/re-check or fall back to click_screen. If the target may "
            "still be LOADING (just launched an app), pass timeout_sec (e.g. 30) "
            "and it polls until the text appears before clicking — so you never "
            "look too early. Set double=true for a double-click. Set retries "
            "(e.g. 3) to VERIFY-AND-RETRY: after clicking it re-checks, and if "
            "the text is still there (the click missed a still-shifting layout) "
            "it re-clicks at fresh coordinates — use for slow launcher buttons."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "occurrence": {"type": "integer", "default": 1},
                "timeout_sec": {"type": "number", "default": 0},
                "double": {"type": "boolean", "default": False},
                "retries": {"type": "integer", "default": 0},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "drag",
        "description": (
            "Click-and-drag from one point to another (move a file/icon, drag a "
            "slider, reorder, select a region). from_x/from_y and to_x/to_y are "
            "NORMALIZED 0-1 fractions of the full screen (like click_screen). It "
            "presses, glides through intermediate points, and releases so the "
            "app registers a real drag. Estimate endpoints from get_screen_"
            "context; for a precise endpoint on an icon, zoom_screen first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_x": {"type": "number"},
                "from_y": {"type": "number"},
                "to_x": {"type": "number"},
                "to_y": {"type": "number"},
            },
            "required": ["from_x", "from_y", "to_x", "to_y"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "draw_path",
        "description": (
            "Draw a CONNECTED stroke through a list of points as ONE continuous "
            "pen stroke (press → glide through every point → release). USE THIS "
            "for shapes in a drawing app instead of multiple drags: a SQUARE is "
            "5 points (4 corners, last = first), a TRIANGLE is 4 (3 corners + "
            "back to start), a LINE is 2, and any polygon/freeform is its "
            "vertices. Points are NORMALIZED 0-1 [x,y] fractions of the full "
            "screen. Pick corners that sit INSIDE the canvas; for a square keep "
            "equal width and height. One call draws the whole shape — don't loop "
            "drag per edge, and don't hunt for the app's shape-tool buttons."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "minItems": 2,
                    "description": "Ordered [x,y] points (0-1). Close a shape by repeating the first point at the end.",
                },
            },
            "required": ["points"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "draw_shape",
        "description": (
            "Draw a primitive SHAPE in a drawing app (Paint, etc.) — the tool "
            "computes and draws the stroke itself, so you do NOT need to "
            "screenshot, zoom, or find the app's shape-tool buttons (that wastes "
            "tokens and rate-limits you). PREFER THIS for 'draw a circle/square/"
            "triangle/line/oval'. Give a bounding box as NORMALIZED 0-1 corners: "
            "x1,y1 = top-left, x2,y2 = bottom-right; pick a box well inside the "
            "canvas (e.g. 0.4,0.4 to 0.6,0.6) and offset it for 'to the right "
            "of' / 'below' the previous shape. square and circle are made truly "
            "equal-sided automatically (you don't need to match width/height); "
            "rectangle and oval use the box as-is."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "shape": {
                    "type": "string",
                    "enum": ["square", "rectangle", "circle", "oval", "ellipse", "triangle", "diamond", "line"],
                },
                "x1": {"type": "number", "description": "Left edge, 0-1."},
                "y1": {"type": "number", "description": "Top edge, 0-1."},
                "x2": {"type": "number", "description": "Right edge, 0-1."},
                "y2": {"type": "number", "description": "Bottom edge, 0-1."},
            },
            "required": ["shape", "x1", "y1", "x2", "y2"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "wait_for_screen_text",
        "description": (
            "Readiness check: wait until some TEXT is visible on screen (OCR), "
            "up to timeout_sec. Use it to CONFIRM a step finished before doing "
            "the next one — e.g. after launching an app, wait_for_screen_text "
            "for a label you expect on its loaded page (Valorant: 'PLAY') so you "
            "don't act before it's ready. Returns appeared=true/false."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "timeout_sec": {"type": "number", "default": 20},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_quick_command",
        "description": (
            "Run a simple BUILT-IN Touchless command through its deterministic "
            "processor — mainly Spotify / media control: 'play <song> on "
            "spotify', 'next song', 'previous song', 'pause', 'resume', "
            "'shuffle'. Use this for the media part of a multi-step request "
            "(e.g. while also doing a web search) instead of scripting "
            "Spotify's UI. `command` is the natural phrase."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "control_window",
        "description": (
            "Close / minimize / maximize / restore / focus a SPECIFIC window "
            "(matched by title or app/process name). Acts on THAT window only — "
            "close sends a graceful close like clicking its X. NEVER use "
            "press_hotkey Alt+F4 to close a window (that hits the focused "
            "window, usually the assistant itself). If several windows match, "
            "returns status='ambiguous' with their names — ask which, then call "
            "again. Use for 'minimize chrome', 'close the explorer window', "
            "'maximize discord', 'bring vs code to the front'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "window_title": {"type": "string"},
                "action": {"type": "string", "enum": ["close", "minimize", "maximize", "restore", "focus"]},
            },
            "required": ["window_title", "action"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_matlab_script",
        "description": (
            "Run a MATLAB .m script — launches the MATLAB desktop in the "
            "script's folder and executes it, so plots/figures appear and "
            "MATLAB stays open. Use to run a .m script (e.g. one Claude just "
            "wrote into a MATLAB project folder); no MATLAB UI clicking needed. "
            "Pass the script's path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "script_path": {"type": "string"},
            },
            "required": ["script_path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "move_window_to_monitor",
        "description": (
            "Move a window to a monitor. window_title matches by title OR app/"
            "process name ('Chrome', 'VS Code', 'file explorer', 'Applications'). "
            "monitor='primary'/'main' or 'secondary'/'second' (or 1-based index). "
            "placement: 'maximize' (default, fill the monitor), 'center' (keep "
            "the window's size, centered), or a SNAP — 'left'/'right' (half the "
            "screen), 'top'/'bottom', or a quadrant 'top-left'/'top-right'/"
            "'bottom-left'/'bottom-right'. USE THIS for 'move/put/drag a window "
            "to the right/left side of monitor X' — placement='right' on that "
            "monitor — do NOT pixel-drag the title bar (unreliable). If it "
            "returns status='ambiguous' (several windows match), it lists their "
            "names in `matches` — ask which, then call again with that exact "
            "name. Use for 'put this on my second monitor', 'snap Spotify to the "
            "right side of monitor one'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "window_title": {"type": "string"},
                "monitor": {"type": "string", "default": "secondary"},
                "placement": {
                    "type": "string",
                    "enum": ["maximize", "center", "left", "right", "top", "bottom",
                             "top-left", "top-right", "bottom-left", "bottom-right"],
                    "default": "maximize",
                },
            },
            "required": ["window_title"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "zoom_screen",
        "description": (
            "Get a HIGH-RES close-up of a screen area so you can click a small "
            "or unlabeled target precisely (an ICON, an X/close, a gear, a "
            "thumbnail, a region — anything click_text_on_screen can't match by "
            "text). Pass x,y as your best 0-1 guess of where the target is on "
            "the full screen; `size` is the crop's fraction of the screen "
            "(default 0.25 — smaller = more zoom). It sends you a zoomed image; "
            "then call click_zoom with coords relative to THAT crop. Use this "
            "instead of click_screen when precision matters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "size": {"type": "number", "default": 0.25},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "click_zoom",
        "description": (
            "Click inside the most recent zoom_screen crop. x,y are fractions "
            "0-1 of THAT crop (center = 0.5,0.5), not the full screen. Maps back "
            "to exact screen pixels and clicks. Call zoom_screen first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "press_hotkey",
        "description": (
            "Press a keyboard shortcut. `keys` is an ordered list of "
            "modifier+key names: ['ctrl','s'], ['alt','f4'], "
            "['ctrl','shift','t']. Pass window_title to send it to a specific "
            "app (e.g. Ctrl+S into 'Notepad') — recommended, since the Iris "
            "chat is always-on-top and the shortcut otherwise hits whatever's "
            "focused."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "window_title": {
                    "type": "string",
                    "description": "Optional: focus this window (title or app name) before pressing.",
                },
            },
            "required": ["keys"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "open_in_editor",
        "description": (
            "Open a folder (and optionally a specific file) in a code "
            "editor. PREFER THIS over open_app when the user wants to "
            "see / edit code — it ensures the editor opens WITH the "
            "folder loaded in the sidebar AND optionally with a "
            "specific file already open in the editor pane. "
            "Always pass the absolute folder_path you got from "
            "create_folder; passing relative paths or omitting it gives "
            "the user an empty editor window which is never what they "
            "want."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "editor": {
                    "type": "string",
                    "enum": ["code", "notepad"],
                    "default": "code",
                    "description": "Which editor to launch. 'code' = VS Code.",
                },
                "folder_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the folder to open in the "
                        "editor's sidebar."
                    ),
                },
                "file_to_open": {
                    "type": "string",
                    "description": (
                        "Optional. Absolute path to a specific file to "
                        "open in the editor pane (must be inside "
                        "folder_path). Pass this so the user immediately "
                        "sees the file you just created."
                    ),
                },
            },
            "required": ["folder_path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "open_app",
        "description": (
            "Open a desktop application by name (e.g. 'chrome', 'spotify', "
            "'notepad', 'code'). Uses the OS's registered handler. "
            "Optionally pass `arguments` to forward command-line args — "
            "e.g. open_app(app_name='code', arguments=['C:\\\\path\\\\to\\\\folder']) "
            "opens VS Code with that folder already loaded in the sidebar. "
            "To open an app ON a monitor in one step (e.g. 'open Discord on my "
            "second monitor, maximized'), pass monitor='secondary' (or "
            "'primary'/index) and placement='maximize'/'center' — it launches, "
            "WAITS for the app's window, and places it there. Much more reliable "
            "than launching then moving separately."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string"},
                "arguments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional command-line arguments to pass to the app.",
                },
                "monitor": {"type": "string"},
                "placement": {"type": "string", "enum": ["maximize", "center"], "default": "maximize"},
            },
            "required": ["app_name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "open_url",
        "description": (
            "Open a URL or web search query in the user's default or "
            "specified browser."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url_or_query": {"type": "string"},
                "browser": {"type": "string"},
            },
            "required": ["url_or_query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "create_folder",
        "description": (
            "Create a new folder. If `base_dir` is omitted, uses the safe "
            "workspace dir. Returns the absolute path created."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "base_dir": {"type": "string"},
                "folder_name": {"type": "string"},
            },
            "required": ["folder_name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "create_file",
        "description": (
            "Create a new file with optional content. Refuses to overwrite "
            "unless `overwrite=true`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "base_dir": {"type": "string"},
                "relative_path": {"type": "string"},
                "content": {"type": "string"},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["relative_path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": (
            "Write/replace the entire contents of a file. If the file "
            "exists and `overwrite=false`, returns an error requiring "
            "confirmation. A `.bak` backup is created before overwriting."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "base_dir": {"type": "string"},
                "relative_path": {"type": "string"},
                "content": {"type": "string"},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["relative_path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "append_file",
        "description": "Append text to an existing file (created if missing).",
        "parameters": {
            "type": "object",
            "properties": {
                "base_dir": {"type": "string"},
                "relative_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["relative_path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read a file's TEXT content so you can SUMMARIZE or answer "
            "questions about it WITHOUT opening it. Pass an absolute `path`, or "
            "just a `name` to find it via file search (e.g. 'my resume', "
            "'config.json', 'the build log'). Handles text/code/csv/json/md, "
            "PDF, and Word (.docx). Returns the content (truncated to "
            "max_chars) — then give the user a concise summary or answer, don't "
            "recite it verbatim. Use whenever the user says things like "
            "'summarize X', 'what's in X', 'what does X say', 'read me X'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "name": {"type": "string"},
                "max_chars": {"type": "integer", "default": 8000},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_files",
        "description": (
            "Find files with their REAL paths, so you act on the right file "
            "instead of guessing a path. Use this whenever the user refers to "
            "a file by description rather than exact name — 'my most recent "
            "screenshot', 'the latest PDF', 'the three latest touchless "
            "drawings', 'the newest file on my desktop'. Results are sorted "
            "newest-first, so 'the N latest X' = the FIRST N results. Then pass "
            "the chosen file's 'path' straight into move_file / read_file / "
            "open_path. NEVER invent a file path for a described file — "
            "list_files first. If you don't know which folder it's in, OMIT "
            "folder (or pass 'all') to search everywhere — don't guess one "
            "folder and give up."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": (
                        "Optional. A friendly name (downloads, desktop, "
                        "documents, pictures, screenshots, videos, music, home) "
                        "or absolute path. OMIT it (or pass 'all'/'anywhere') "
                        "to search the user's common folders recursively — best "
                        "when you don't know where the file is."
                    ),
                },
                "file_type": {
                    "type": "string",
                    "description": (
                        "Optional filter: image, screenshot, document, pdf, "
                        "video, audio, archive, code — or a specific extension "
                        "like '.png'. Omit for all files."
                    ),
                },
                "name_contains": {
                    "type": "string",
                    "description": (
                        "Optional: only files whose name contains this text. "
                        "Spaces, underscores and hyphens are interchangeable, "
                        "so 'touchless drawing' matches 'Touchless_Drawing_9.png'."
                    ),
                },
                "sort": {
                    "type": "string",
                    "enum": ["recent", "oldest", "name"],
                    "default": "recent",
                    "description": "recent (newest first, default), oldest, or name.",
                },
                "recursive": {
                    "type": "boolean",
                    "default": False,
                    "description": "Also search subfolders (forced on when searching everywhere).",
                },
                "limit": {
                    "type": "number",
                    "default": 20,
                    "description": "Max files to return (1-100).",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "move_file",
        "description": (
            "Move or rename a file or folder. Use this when the user "
            "asks to relocate something you already created (e.g. \"move "
            "it to Documents\"), instead of trying to fake it with "
            "delete + create. Both paths must be absolute. Refuses to "
            "move into system directories (Windows, Program Files, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "Absolute path of the file or folder to move.",
                },
                "destination_path": {
                    "type": "string",
                    "description": (
                        "Absolute path of the new location. If the "
                        "destination's parent doesn't exist, the call "
                        "fails — use create_folder first if needed."
                    ),
                },
                "overwrite": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "If true, replace any existing file at the "
                        "destination. If false (default) and the "
                        "destination exists, returns needs_confirmation."
                    ),
                },
            },
            "required": ["source_path", "destination_path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "rename_file",
        "description": (
            "Rename a file or folder in place. Pass the absolute current "
            "path and the NEW NAME (just the name, not a path)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path of the file or folder to rename.",
                },
                "new_name": {
                    "type": "string",
                    "description": "New name (basename only, e.g. 'final.py'). No slashes.",
                },
            },
            "required": ["path", "new_name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "delete_file",
        "description": (
            "Delete a file or folder. ALWAYS ask the user for explicit "
            "confirmation in plain text BEFORE calling this tool — never "
            "delete without asking. Refuses to touch system directories. "
            "Folders are deleted recursively."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path of the file or folder to delete.",
                },
                "confirmed": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Must be true to actually delete. The model is "
                        "responsible for asking the user for explicit "
                        "permission BEFORE setting this true."
                    ),
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_recent_paths",
        "description": (
            "Return the absolute paths the agent has created or written "
            "in this session, most-recent first. Use this when you've "
            "lost track of where you put something the user is now "
            "asking about."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_existing_touchless_action",
        "description": (
            "Bridge to the existing Touchless action router. Use this for "
            "spotify/chrome/youtube/system actions already implemented in "
            "the app."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action_name": {"type": "string"},
                "parameters": {"type": "object"},
            },
            "required": ["action_name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_python_script",
        "description": (
            "Run a Python script and capture its output. Use this to "
            "EXECUTE a script you just created — never try to focus VS "
            "Code's terminal and type 'python main.py' manually. This "
            "tool spawns python directly and returns stdout/stderr. The "
            "script is launched DETACHED so GUI scripts (tkinter, "
            "pygame, etc.) can show their windows without blocking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "script_path": {
                    "type": "string",
                    "description": "Absolute path to the .py file to run.",
                },
                "wait_for_exit": {
                    "type": "boolean",
                    "description": (
                        "If true, block until the script finishes and "
                        "return its stdout/stderr. Use for short "
                        "terminal scripts. If false (default), launch "
                        "and return immediately — required for GUI "
                        "scripts like tkinter/pygame that show a window."
                    ),
                    "default": False,
                },
                "timeout_sec": {
                    "type": "number",
                    "description": "Max wait when wait_for_exit=true (default 15).",
                    "default": 15,
                },
            },
            "required": ["script_path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "skip_youtube_ad",
        "description": (
            "Try to click YouTube's user-visible Skip Ad button using the "
            "existing template-matching pipeline. Does NOT bypass non-"
            "skippable ads or block ads in any way."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "ask_user_confirmation",
        "description": (
            "Show a confirmation dialog before a risky action. Returns "
            "{'approved': bool}. risk_level affects the dialog styling."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "web_scroll",
        "description": (
            "Scroll the current web page. to='bottom' (jump to end), 'top', "
            "'down' (~one screen), or 'up'. Use this when asked to scroll a "
            "page; for reading the whole page prefer web_get_text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "enum": ["bottom", "top", "down", "up"],
                    "default": "bottom",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "web_fill",
        "description": (
            "Type text into a form field on the current page, found by its "
            "label / placeholder / name. Fires input+change so the page "
            "reacts (search boxes, login fields, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"field": {"type": "string"}, "text": {"type": "string"}},
            "required": ["field", "text"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "web_wait_for",
        "description": (
            "Wait until the page shows a CSS selector or visible text "
            "(`query`), up to timeout_sec. Use after an action that loads "
            "content asynchronously, before reading/clicking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "timeout_sec": {"type": "number", "default": 15},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "web_eval",
        "description": (
            "Run a JavaScript expression in the current page and return its "
            "value. Advanced/escape-hatch for page control the other web_* "
            "tools don't cover. Returns the result as text."
        ),
        "parameters": {
            "type": "object",
            "properties": {"javascript": {"type": "string"}},
            "required": ["javascript"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_ui",
        "description": (
            "Read the interactive UI elements (buttons, fields, menu items, "
            "tabs, checkboxes) of a desktop app window via Windows "
            "accessibility — names + types, with exact positions known "
            "internally. Use this for ANY native/desktop app (dialogs, "
            "settings, small apps) instead of screenshot-guessing. Defaults "
            "to the foreground window; pass window_title to target another."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "window_title": {"type": "string"},
                "limit": {"type": "integer", "default": 40},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "click_ui",
        "description": (
            "Click a desktop UI element BY NAME via accessibility (e.g. "
            "click_ui('Yes'), click_ui('Save')). Precise — no pixel guessing. "
            "Use read_ui first if unsure of the exact name. Prefer this over "
            "click_screen for native app buttons/menus."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "window_title": {"type": "string"},
            },
            "required": ["target"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "set_field",
        "description": (
            "Type text into a desktop app's text field by name via "
            "accessibility (e.g. set_field('Search', 'invoices'))."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "text": {"type": "string"},
                "window_title": {"type": "string"},
            },
            "required": ["target", "text"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "wait_and_click",
        "description": (
            "Wait (locally, no tokens) for a desktop UI element named "
            "`target` to APPEAR, then click it. Use for 'when a prompt/dialog "
            "appears, click Yes': wait_and_click('Yes'). Times out after "
            "timeout_sec."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "timeout_sec": {"type": "number", "default": 30},
                "window_title": {"type": "string"},
            },
            "required": ["target"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "wait_and_press",
        "description": (
            "Wait (locally, no tokens) until `text` appears in a window's "
            "content, then press `keys`. For TERMINAL/CLI prompts that have "
            "no clickable button and are answered with a keystroke — e.g. a "
            "Claude Code or installer prompt 'Do you want to proceed? 1. Yes "
            "2. No': wait_and_press(text='proceed', keys=['1']). Use Enter via "
            "keys=['enter']. (For native dialog BUTTONS use wait_and_click "
            "instead.)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "keys": {"type": "array", "items": {"type": "string"}},
                "timeout_sec": {"type": "number", "default": 30},
                "window_title": {"type": "string"},
            },
            "required": ["text", "keys"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "auto_approve",
        "description": (
            "Start a BACKGROUND watcher that keeps clicking approval buttons "
            "(Yes/Allow/Keep/Accept/...) every time one appears, AND detects "
            "when the app pauses (no approve button + screen stops changing). "
            "Returns immediately; you'll get a follow-up '[auto-approve "
            "update]' message when it pauses — read its on-screen text and "
            "tell the user whether the app finished or is showing a prompt "
            "that needs them. Use for 'keep approving until Claude is done "
            "editing' — NOT a one-shot click, and don't click a 'Done' button "
            "to end it. `idle_sec` = how long of no activity counts as paused. "
            "This is the RIGHT tool when the user already started Claude/Codex "
            "themselves and asks you to 'watch my VS Code and approve' — just "
            "call it, no other setup needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "targets": {"type": "array", "items": {"type": "string"}},
                "duration_sec": {"type": "number", "default": 600},
                "idle_sec": {"type": "number", "default": 20},
                "check_interval_sec": {"type": "number", "default": 5},
                "window_title": {
                    "type": "string",
                    "description": (
                        "Defaults to VS Code (where coding agents run) — omit it "
                        "for the usual case. Only set it to watch a different app."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "stop_auto_approve",
        "description": "Stop the background auto-approve watcher started by auto_approve.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "open_path",
        "description": (
            "Find existing file(s)/folder(s) BY NAME using Touchless's own "
            "file search, then open them — do NOT ask the user for the path "
            "first. Pass the name the user said (e.g. 'HGR App v1.0.0', "
            "'resume', 'Downloads'). To open SEVERAL at once (e.g. 'open "
            "Documents, Downloads and Pictures'), pass them all in `queries` "
            "in ONE call so none are skipped. On a single confident match it "
            "opens it; if several files match one name it returns "
            "status='ambiguous' with a `matches` list — show those and re-call "
            "with the chosen exact path. `kind` narrows the search. If a "
            "normal call returns not_found, retry with deep=true for a "
            "thorough (slower) search across all drives."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "queries": {"type": "array", "items": {"type": "string"}},
                "kind": {
                    "type": "string",
                    "enum": ["file", "folder", "any"],
                    "default": "any",
                },
                "deep": {"type": "boolean", "default": False},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "web_navigate",
        "description": (
            "Open a URL (or run a Google search for a plain query) in the "
            "controlled Chrome and wait for it to load. Returns the final "
            "{url, title}. PREFER this + web_get_links/web_get_text for web "
            "tasks over pixel-clicking — it reads the real page, so it's "
            "exact and cheap. To open a specific search result: web_navigate "
            "the query, web_get_links, then web_navigate the chosen link's url."
        ),
        "parameters": {
            "type": "object",
            "properties": {"url_or_query": {"type": "string"}},
            "required": ["url_or_query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "web_get_links",
        "description": (
            "Return the current page's visible links as a numbered list of "
            "{index, text, url}. Use this to pick a specific result/link by "
            "its text, then web_navigate its url. Optional `contains` filters "
            "links whose text or url include that substring."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "contains": {"type": "string"},
                "limit": {"type": "integer", "default": 30},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "web_get_text",
        "description": (
            "Return the current page's visible text (truncated). Use to read "
            "or summarize a page instead of screenshotting it."
        ),
        "parameters": {
            "type": "object",
            "properties": {"max_chars": {"type": "integer", "default": 4000}},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "web_click",
        "description": (
            "Click the first link/button/control on the current page whose "
            "visible text contains `text` (case-insensitive). Use for buttons "
            "that don't have a URL (e.g. 'Accept', 'Sign in'). For navigating "
            "to a link, prefer web_navigate with the link's url."
        ),
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
]


def all_tool_schemas() -> List[Dict[str, Any]]:
    """Return a fresh copy of the full tool schema list."""
    # Shallow copy is fine — callers should not mutate the inner dicts.
    return [dict(s) for s in _TOOL_SCHEMAS]


_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_args(tool_name: str, args: Any) -> Tuple[bool, str, Dict[str, Any]]:
    """Light JSON-schema validation. Returns (ok, error_message, normalised_args)."""
    schema = next(
        (s for s in _TOOL_SCHEMAS if s["name"] == tool_name),
        None,
    )
    if schema is None:
        return False, f"unknown tool: {tool_name}", {}
    if not isinstance(args, dict):
        return False, "tool arguments must be a JSON object", {}

    params = schema["parameters"]
    properties = params.get("properties", {})
    required = params.get("required", [])

    for key in required:
        if key not in args:
            return False, f"missing required argument: {key}", {}

    normalised: Dict[str, Any] = {}
    for key, value in args.items():
        if key not in properties:
            # tolerate extras to keep the model loop unblocked, but log them
            normalised[key] = value
            continue
        prop = properties[key]
        expected = _TYPE_MAP.get(str(prop.get("type", "")))
        if expected is not None and not isinstance(value, expected):
            # accept ints where numbers are expected
            if expected == (int, float) and isinstance(value, bool):
                return False, f"{key} expected number, got bool", {}
            if not (expected == (int, float) and isinstance(value, (int, float))):
                if expected != bool and isinstance(value, bool):
                    return False, f"{key} expected {prop.get('type')}, got bool", {}
                if not isinstance(value, expected):
                    return False, f"{key} expected {prop.get('type')}", {}
        if "enum" in prop and value not in prop["enum"]:
            return False, f"{key} must be one of {prop['enum']}", {}
        normalised[key] = value

    # apply declared defaults for missing optional fields
    for key, prop in properties.items():
        if key not in normalised and "default" in prop:
            normalised[key] = prop["default"]

    return True, "", normalised

# Author: Konstantin Markov
