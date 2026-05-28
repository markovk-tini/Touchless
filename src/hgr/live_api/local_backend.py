"""Local (offline) backend for the Live API agent.

This is the Phase 1 local-mode counterpart to `realtime_client.py`.
It provides the SAME public surface as `RealtimeClient` so
`LiveApiManager` can swap them via a config flag without any other
changes:

    start() / stop() / join()
    send_audio_chunk(pcm16_bytes)
    send_text_message(text)
    send_screen_image(jpeg_b64, caption=...)   -- no-op in Phase 1 (no vision)
    send_tool_result(call_id, output)
    request_response()
    .connected (bool)
    callbacks: on_event, on_connected, on_closed, on_error

Internally:
  * Spawns `llama-server.exe` (CUDA / Vulkan / CPU auto-detected) with
    the configured Qwen 2.5 7B Instruct GGUF.
  * Buffers mic audio, runs an energy-based VAD, and on end-of-turn
    writes the accumulated PCM16 to a temp WAV and shells out to
    `whisper-cli.exe` for transcription. The VAD threshold and silence
    window are tunable via env (LIVE_API_LOCAL_VAD_*).
  * Sends the transcript + tool schemas to llama-server's
    OpenAI-compatible `/v1/chat/completions` endpoint with the `tools`
    array. Parses `tool_calls` and synthesises Realtime-style events
    (`response.function_call_arguments.done`, `response.text.delta`,
    `response.text.done`, `response.done`) so `LiveApiManager`'s event
    router doesn't care which backend produced them.
  * Maintains a per-session `messages` history. After tool execution
    the manager calls `request_response()` which kicks the next LLM
    round with the tool output appended to the history.

Phase 1 explicitly does NOT include screen-vision. `send_screen_image`
is a no-op so the manager's existing screen worker can run unchanged
without affecting tokens or latency. Phase 2 will route screenshots
through `llama-qwen2vl-cli.exe` (already built in the project).
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import LiveApiConfig
from .live_api_logger import LiveApiLogger
from ..utils.runtime_paths import app_base_path
from ..utils.subprocess_utils import hidden_subprocess_kwargs


# Tools that don't make sense in local (no-vision) mode. Filtering them
# out of the schema sent to llama-server reduces the model's "choice
# fatigue" and stops it from picking screen-vision tools it can't use.
# `ask_user_confirmation` is removed because the model can ask in plain
# text via the conversation — having a tool for it just confuses Qwen.
_LOCAL_TOOL_BLACKLIST = {
    "get_screen_context",
    "click_screen",
    "ask_user_confirmation",
}


def _filter_tools_for_local(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [t for t in tools if t.get("name") not in _LOCAL_TOOL_BLACKLIST]


# Callback types match RealtimeClient so LiveApiManager treats both
# backends identically.
EventCallback = Callable[[Dict[str, Any]], None]
ConnectedCallback = Callable[[], None]
ClosedCallback = Callable[[Optional[str]], None]
ErrorCallback = Callable[[str], None]


# ---------- discovery helpers (mirroring llama_server.py / voice_command_listener.py)

def _candidate_llama_roots() -> List[Path]:
    roots: List[Path] = []
    base = app_base_path()
    roots.append(base / "llama.cpp")
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "llama.cpp"
        if candidate not in roots:
            roots.append(candidate)
    env = os.getenv("HGR_LLAMA_CPP_ROOT", "").strip()
    if env:
        roots.insert(0, Path(env))
    home_candidate = Path.home() / "Documents" / "llama.cpp"
    if home_candidate not in roots:
        roots.append(home_candidate)
    return roots


def _candidate_model_roots() -> List[Path]:
    roots: List[Path] = []
    env = os.getenv("HGR_LLAMA_MODEL_ROOT", "").strip()
    if env:
        roots.append(Path(env))
    roots.append(Path.home() / "Documents" / "TouchlessVoiceModels")
    roots.append(Path.home() / "Documents" / "HGRVoiceModels")
    for llama_root in _candidate_llama_roots():
        roots.append(llama_root / "models")
    return roots


def _detect_nvidia_gpu() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        proc = subprocess.run(
            [exe, "-L"],
            capture_output=True,
            text=True,
            timeout=4.0,
            **hidden_subprocess_kwargs(),
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return proc.returncode == 0 and "GPU" in (proc.stdout or "")


def _resolve_llama_server_executable() -> Optional[Tuple[str, Path]]:
    override = os.getenv("HGR_LLAMA_BACKEND", "").strip().lower()
    if override in {"cuda", "vulkan", "cpu"}:
        backend_order = [override]
    else:
        backend_order = []
        if _detect_nvidia_gpu():
            backend_order.append("cuda")
        backend_order.extend(["vulkan", "cpu"])

    build_dirs = {"cuda": "build_cuda", "vulkan": "build_vulkan", "cpu": "build_cpu"}
    for backend in backend_order:
        for root in _candidate_llama_roots():
            for rel in ("bin/Release/llama-server.exe", "bin/llama-server.exe"):
                candidate = root / build_dirs[backend] / rel
                if candidate.exists():
                    return backend, candidate
    return None


def _resolve_model_file(filename: str) -> Optional[Path]:
    """Return the absolute path of `filename` if it exists in any candidate root."""
    for root in _candidate_model_roots():
        candidate = root / filename
        if candidate.exists():
            return candidate
    return None


def _resolve_whisper_command() -> Optional[Tuple[str, ...]]:
    env_value = os.getenv("HGR_WHISPER_CPP", "").strip()
    if env_value:
        if Path(env_value).exists():
            return (env_value,)
        resolved = shutil.which(env_value)
        if resolved:
            return (resolved,)
    for command_name in ("whisper-cli.exe", "whisper-cli"):
        resolved = shutil.which(command_name)
        if resolved:
            return (resolved,)
    base = app_base_path()
    roots = [base, base / "whisper.cpp"]
    here = Path(__file__).resolve()
    for parent in here.parents:
        roots.append(parent)
        roots.append(parent / "whisper.cpp")
    home = Path.home() / "Documents" / "whisper.cpp"
    if home not in roots:
        roots.append(home)
    for root in roots:
        for build_dir in ("build", "build_cuda", "build_vulkan", "build_stream"):
            bin_dir = root / build_dir / "bin"
            for candidate in (bin_dir / "Release" / "whisper-cli.exe", bin_dir / "whisper-cli.exe"):
                if candidate.exists():
                    return (str(candidate),)
    return None


def _resolve_whisper_model() -> Optional[Path]:
    env_value = os.getenv("HGR_WHISPER_CPP_MODEL", "").strip()
    if env_value and Path(env_value).exists():
        return Path(env_value)
    candidate_roots = [
        Path.home() / "Documents" / "TouchlessVoiceModels",
        Path.home() / "Documents" / "HGRVoiceModels",
    ]
    base = app_base_path()
    candidate_roots.append(base / "whisper.cpp" / "models")
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate_roots.append(parent / "whisper.cpp" / "models")
    for root in candidate_roots:
        if not root.exists():
            continue
        for name in (
            "ggml-medium.en.bin",
            "ggml-small.en.bin",
            "ggml-base.en.bin",
            "ggml-medium.bin",
            "ggml-small.bin",
            "ggml-base.bin",
        ):
            candidate = root / name
            if candidate.exists():
                return candidate
        extras = sorted(p for p in root.glob("ggml-*.bin") if not p.name.startswith("for-tests-"))
        if extras:
            return extras[0]
    return None


def _find_free_port(preferred: int = 8758) -> int:
    """8758 instead of llama_server.py's 8757 so both can co-exist."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", preferred))
            return preferred
    except OSError:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ---------- backend itself ----------

_LOCAL_SYSTEM_PROMPT = """You are Touchless Local Agent, an offline assistant controlling \
the user's Windows PC via tools. ALWAYS use a tool — never claim \
something was done unless a tool result confirms it. Be concise.

═══════════════════════════════════════════════════════════════════
RULE #1 — DECISION CHECKLIST. RUN THIS BEFORE EMITTING ANY TOOL CALL.
═══════════════════════════════════════════════════════════════════

For the user's request, walk through this checklist mentally:

  [ ] Did the user say WHERE to put any new files/folders?
      ▶ If NO and the action creates/writes a file or folder → ASK.
        Example: "create a folder X" → did they say where?
                 If not, STOP and ask.
  [ ] Did the user describe WHAT a script should do, specifically?
      ▶ If they said "show on screen" or "display" → did they say
        full-screen overlay vs popup vs system notification?
        If not, STOP and ask which form they want.
      ▶ If they said "write code" with no specifics → STOP and ask.
  [ ] Will any tool OVERWRITE or DELETE existing data?
      ▶ STOP and confirm first.
  [ ] Are there exactly ZERO unspecified choices that affect the
      outcome?
      ▶ Only then proceed to emit tool calls.

If ANY checkbox is unchecked, EMIT NO TOOLS THIS TURN. Reply with a \
plain-text question listing 2-4 concrete options. Wait for the user's \
reply. Do NOT call tools and ask in the same turn — pick one.

CLARIFICATION FORMAT (good vs bad):
  BAD:  "Where would you like the folder?"
  GOOD: "Where should I create 'Live API Test'?
         1. Documents (~/Documents/Live API Test)
         2. Desktop (~/Desktop/Live API Test)
         3. Touchless sandbox (~/Documents/Touchless/live_api_workspace)
         Reply with a number or type a custom path."

  BAD:  "What should the script do?"
  GOOD: "How should 'HELLO USER' appear?
         1. Fullscreen black overlay with big white text (press Esc to close)
         2. Small popup window in the center of the screen
         3. Windows toast notification
         4. Just print to terminal
         Reply with a number."

Only SKIP asking when there is exactly one reasonable interpretation \
(e.g. "open chrome" — no choice to make).

═══════════════════════════════════════════════════════════════════
RULE #2 — APP / FOLDER COMBO. WHEN OPENING AN EDITOR FOR A FOLDER:
═══════════════════════════════════════════════════════════════════

NEVER use open_app(app_name="code") to open VS Code on a folder. \
ALWAYS use open_in_editor(folder_path=<absolute path>, \
file_to_open=<absolute file path>) — this guarantees the editor opens \
WITH the folder loaded in the sidebar AND the specific file already \
visible. Calling open_app("code") gives an empty window which is \
never what the user wants.

Use open_app ONLY for apps that don't take a folder/file (Spotify, \
Calculator, Chrome to launch the browser without a URL, etc.).

═══════════════════════════════════════════════════════════════════
LOCAL MODE LIMITATION: no screen vision. If asked "what's on my \
screen" or to click a visual element you can't name, say so plainly \
and suggest a tool alternative.
═══════════════════════════════════════════════════════════════════

ACTION RESOLUTION ORDER. For every action, prefer tools in this order:
1. DIRECT API/FILESYSTEM. Most reliable, never fails on UI. Examples:
   - Launching an app: open_app(app_name="code")
   - Opening a URL/website: open_url(url_or_query="https://...")
   - Creating folders / files / writing code: create_folder, create_file, write_file, append_file
   - Bridges into existing Touchless features: run_existing_touchless_action
2. KEYBOARD SHORTCUT. When the target app is already running and the \
   action has a known hotkey. Examples:
   - New VS Code window when VS Code is already open: press_hotkey(keys=["ctrl","shift","n"])
   - Save current file: press_hotkey(keys=["ctrl","s"])
   - Switch app: press_hotkey(keys=["alt","tab"])
3. CLICK SCREEN (LAST RESORT). Only when 1 and 2 don't apply. Local \
   mode has no vision so you usually can't use this — prefer 1 or 2.

APP NAMES on Windows for open_app: "code" (VS Code), "chrome", \
"spotify", "notepad", "msedge", "calc", "explorer", "wt" (Terminal). \
If open_app returns an error, retry with a shorter name.

ASK WHEN AMBIGUOUS. Before doing anything that depends on a choice the \
user did NOT explicitly make, ASK them in plain text and STOP — do not \
emit any tools that turn. Wait for their reply, then proceed. Things \
that are AMBIGUOUS by default and require asking:
   - WHERE to create a folder or file when no path was given. Common \
     options to offer: "Documents folder", "Desktop", "the current VS \
     Code workspace", "Touchless's safe sandbox at \
     ~/Documents/Touchless/live_api_workspace". Pick 2-3 sensible \
     choices to suggest based on the request, but accept any path the \
     user types back.
   - WHAT to put in a file when the user says "write a script" without \
     specifying language, framework, or behavior. Confirm the gist \
     before generating code.
   - WHICH browser if multiple are reasonable and the request matters \
     (e.g. "log in to my work account" — different browsers have \
     different cookies).
   - OVERWRITE permission when a file already exists. Always show the \
     existing path and ask before overwriting non-trivial files.
   - DESTRUCTIVE actions: delete, close-without-saving, force-quit, \
     wipe a folder. ALWAYS confirm first.

DON'T over-ask. If the user gave enough info for ONE reasonable \
interpretation (e.g. "open chrome" — no ambiguity), just do it. The \
goal is to ask only when a wrong guess would put files in the wrong \
place or destroy data.

CLARIFICATION FORMAT. Keep questions short and offer concrete options. \
Bad: "Where would you like the folder?" \
Good: "Where should I create 'Live API Test'? \
  1. Documents folder (~/Documents/Live API Test) \
  2. Desktop (~/Desktop/Live API Test) \
  3. Touchless sandbox (~/Documents/Touchless/live_api_workspace/Live API Test) \
Reply with a number or type a custom path."

MULTI-STEP COMMANDS. Once you have all the info you need (checklist \
in Rule #1 fully checked off!), plan the full sequence and emit ALL \
the tool calls together in your response (the runtime executes them \
in order and feeds all results back before you respond again). \
Standard shape for "create folder X with main.py inside, open it in \
VS Code, then run it":
   1. create_folder(base_dir="<user's chosen dir>", folder_name="X")
      → system returns absolute path, e.g. "C:\\\\Users\\\\u\\\\Documents\\\\X"
   2. create_file(base_dir="<from step 1>", relative_path="main.py", content="...")
      → system returns absolute file path
   3. open_in_editor(folder_path="<from step 1>", file_to_open="<from step 2>")
      ← THIS opens VS Code WITH the folder loaded AND the file open.
        Do not use open_app for this. Ever.
   4. run_python_script(script_path="<from step 2>", wait_for_exit=False)
      ← only if the user asked to run it. Use wait_for_exit=False for
        GUI scripts (tkinter/pygame), True for terminal scripts.

After all tool results come back, give the user a one-line confirmation \
that mentions WHERE things were created so they can find them.

DEFAULT BASE_DIR. If — and only if — the user explicitly says "use the \
default" or "anywhere is fine", omit base_dir from create_folder / \
create_file and the system will use ~/Documents/Touchless/live_api_workspace/.

PATH FORMAT — VERY IMPORTANT.
  - The user's Documents folder on Windows is NOT "C:\\Documents". \
    It's "C:\\Users\\<username>\\Documents", which in tilde form is \
    "~/Documents". ALWAYS use the tilde form: ~/Documents/X.
  - Same for Desktop: ~/Desktop/X (NOT C:\\Desktop).
  - The system expands ~ to the user's actual home directory.
  - If the user types a literal "C:\\Documents\\X", interpret it as \
    "~/Documents/X" — they meant their personal Documents, not the \
    nonexistent root.
  - The base_dir tool argument REJECTS paths whose parent doesn't \
    exist. If you get error code "invalid_base_dir", retry with the \
    tilde form (~/Documents/...).

OVERWRITE HANDLING. If create_file or write_file returns \
status="needs_confirmation" with code="exists":
  - Within the SAME user request (e.g. user just asked you to create \
    something), the user has implicitly approved overwriting their \
    own previous attempt — RETRY with overwrite=true on the next \
    tool call. Do not ask the user again.
  - If you're not sure it's the same task, ASK the user once, then \
    retry with overwrite=true if they confirm.

MOVING / RENAMING / DELETING. Use the dedicated tools — never fake it \
by combining write_file + create_file:
  - To MOVE a folder or file the user wants relocated: \
    move_file(source_path, destination_path).
  - To RENAME in place: rename_file(path, new_name).
  - To DELETE: ALWAYS ask the user for explicit confirmation in plain \
    text first, then call delete_file(path, confirmed=true).
  - These tools refuse system paths (Windows, Program Files, ...) so \
    the user can't accidentally tell you to break their machine.

WHEN YOU LOSE TRACK OF WHERE YOU PUT SOMETHING. If the user asks "where \
is the script?" or you're not sure which folder you most recently \
created a file in, call list_recent_paths() — it returns the absolute \
paths the agent has touched this session, most-recent first. Use this \
INSTEAD of guessing or apologizing about being lost.

═══════════════════════════════════════════════════════════════════
RESPONSE SHAPE — VERY IMPORTANT.
═══════════════════════════════════════════════════════════════════

You communicate in TWO channels:
  1. TOOL CALLS (the function-calling mechanism). Use these to take \
     actions on the user's PC. The runtime intercepts them, runs the \
     real Python function, and feeds the result back to you.
  2. PLAIN TEXT. Use this to talk TO the user — questions, \
     confirmations, explanations.

NEVER write the literal syntax of a tool call (like \
"create_folder(folder_name=...)") into your text reply. That is just \
text — nothing executes. To actually create a folder, EMIT A TOOL \
CALL using the function-calling mechanism. The text channel is only \
for prose.

WORKED SCENARIOS — what should happen, in plain English.

SCENARIO 1 — User's request has unspecified choices.
  USER: "Open a new VS Code window, create a folder called Live API \
        Test, then create main.py that prints Hello User to show up in \
        the middle of my screen, not just terminal. Then run it please."
  Your job: emit ZERO tool calls. Reply with text only, asking the \
  two clarifying questions:
    1. Where to create 'Live API Test' (Documents / Desktop / sandbox)
    2. How should "Hello User" appear (fullscreen overlay / popup / \
       toast notification)
  Wait for the user's reply before doing anything.

SCENARIO 2 — User answered the clarifying questions.
  USER: "Documents folder, fullscreen overlay"
  Your job: emit FOUR tool calls in this turn (using the function-\
  calling mechanism, NOT text):
    • create_folder for "Live API Test" under Documents
    • create_file for main.py with tkinter fullscreen overlay code
    • open_in_editor on the folder + file
    • run_python_script with wait_for_exit=False (GUI script)
  After all four results come back, send ONE short text message \
  confirming what happened (e.g. "Done — folder created, VS Code open \
  on it, overlay running. Press Esc to close.")

SCENARIO 3 — Unambiguous request.
  USER: "Open Chrome."
  Your job: emit ONE tool call (open_app with app_name="chrome"), then \
  one short confirmation in text ("Chrome is open.").

SCENARIO 4 — Request needs vision.
  USER: "Click the blue Save button."
  Your job: emit ZERO tool calls. Reply in text that local mode has no \
  vision, suggest pressing Ctrl+S as an alternative.

INTENT TRANSLATION. Listen for what the user actually wants visually, \
not just what they literally say. Two common cases:

  "PRINT X TO TERMINAL" → write a .py with print('X'). Run with \
  run_python_script(wait_for_exit=true) so you can return the output.

  "SHOW X ON SCREEN" / "DISPLAY X" / "POP UP X" / "X in the middle of \
  my screen" → the user wants a VISUAL overlay, NOT terminal output. \
  Generate a tkinter GUI script and run it with \
  run_python_script(wait_for_exit=false). Example template:
      import tkinter as tk
      root = tk.Tk()
      root.attributes('-fullscreen', True)
      root.attributes('-topmost', True)
      root.configure(bg='black')
      label = tk.Label(root, text='HELLO USER',
                       font=('Arial', 120), fg='white', bg='black')
      label.place(relx=0.5, rely=0.5, anchor='center')
      root.bind('<Escape>', lambda e: root.destroy())
      root.mainloop()
  Always include the Escape-to-close binding so the user can dismiss it.

EXECUTING SCRIPTS. After creating a .py file the user wants to run, \
ALWAYS use run_python_script(script_path=<absolute path>) instead of \
trying to type 'python main.py' into a terminal — terminal manipulation \
via type_text is unreliable and slow. \
  - For terminal scripts: wait_for_exit=true (default false), so you \
    get stdout/stderr back and can report it. \
  - For GUI scripts (tkinter/pygame/pyside): wait_for_exit=false so the \
    GUI window stays open without blocking.

DON'T RETRY SUCCESSFUL TOOLS. If a tool returned status="ok", DO NOT \
call it again in the same turn. Specifically: only open VS Code ONCE \
per request — never call open_app(app_name="code", ...) multiple times.

DON'T CALL type_text TO INTERACT WITH OTHER APPS. type_text pastes into \
whatever window is in the foreground. It is unreliable for multi-step \
UI manipulation (typing in a terminal, clicking through dialogs). Use \
direct tools instead (run_python_script, create_file, press_hotkey)."""


class LocalBackend:
    """Local equivalent of RealtimeClient — same interface, different brain."""

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
        require_audio: bool = True,
    ) -> None:
        self._config = config
        self._logger = logger
        # Trim down to tools that work locally before any LLM call sees
        # them. Vision-only tools just waste context tokens and tempt
        # the model into wrong decisions.
        self._tools = _filter_tools_for_local(tools)
        # When False (text-only mode) we skip whisper-cli discovery so a
        # missing whisper.cpp model doesn't block startup.
        self._require_audio = bool(require_audio)
        # Phase 1 prepends the local-specific system prompt to the
        # generic instructions (which mention vision). Local mode has no
        # vision, so the local prompt overrides where they conflict.
        self._system_instructions = _LOCAL_SYSTEM_PROMPT + "\n\n" + system_instructions
        self._on_event = on_event
        self._on_connected = on_connected
        self._on_closed = on_closed
        self._on_error = on_error

        self._llama_proc: Optional[subprocess.Popen] = None
        self._llama_port: Optional[int] = None
        self._llama_backend: Optional[str] = None
        self._llama_model_path: Optional[Path] = None
        self._whisper_command: Optional[Tuple[str, ...]] = None
        self._whisper_model_path: Optional[Path] = None

        self._connected = False
        self._stop_requested = False
        self._reader_thread: Optional[threading.Thread] = None
        self._llama_stderr_thread: Optional[threading.Thread] = None

        # Audio buffering / VAD state
        self._audio_buffer = bytearray()
        self._audio_lock = threading.Lock()
        self._speech_started_at: Optional[float] = None
        self._last_voice_at: Optional[float] = None
        self._turn_ready = threading.Event()
        self._turn_pending: Optional[bytes] = None
        # Conversation state for the LLM. messages is a list of
        # {"role": ..., "content": ...} dicts compatible with the
        # OpenAI chat completion schema that llama-server speaks.
        self._messages: List[Dict[str, Any]] = []
        # Pending tool outputs to send on the next inference round.
        self._pending_tool_outputs: List[Dict[str, Any]] = []
        # Tool calls we just dispatched, keyed by call_id, so we know
        # what call_id maps back to which tool name when the result
        # comes in via send_tool_result.
        self._inflight_tool_calls: Dict[str, str] = {}

    # ---- lifecycle ----

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> bool:
        # Resolve binaries + models. Each missing dep gives a precise
        # error so the UI can surface exactly what to install.
        resolution = _resolve_llama_server_executable()
        if resolution is None:
            self._fire_error(
                "llama-server.exe not found — build llama.cpp first "
                "(see docs/INSTALL_LLAMA.md or run "
                "`cmake --preset cuda && cmake --build build_cuda`)"
            )
            return False
        self._llama_backend, llama_exe = resolution

        model_path = _resolve_model_file(self._config.local_llm_model_filename)
        if model_path is None:
            target_dir = Path.home() / "Documents" / "TouchlessVoiceModels"
            target_path = target_dir / self._config.local_llm_model_filename
            # Hugging Face URL is well-known for Qwen 2.5 7B GGUF; if the
            # user has overridden the model filename to something else,
            # we still tell them WHERE to put it, just not the exact
            # download URL.
            hf_url = (
                "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/"
                "resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
            )
            self._fire_error(
                f"Local LLM model not found.\n\n"
                f"Expected: {target_path}\n\n"
                f"Download (PowerShell):\n"
                f"    mkdir \"{target_dir}\" -Force | Out-Null\n"
                f"    curl.exe -L -o \"{target_path}\" {hf_url}\n\n"
                f"Then click Start Local Agent again."
            )
            return False
        self._llama_model_path = model_path

        # Whisper is only required when audio (voice) input is enabled.
        # Text-only sessions skip the check so missing whisper.cpp model
        # doesn't block typed-command startup.
        if self._require_audio:
            whisper_cmd = _resolve_whisper_command()
            if whisper_cmd is None:
                self._fire_error("whisper-cli.exe not found (needed for local STT).")
                return False
            whisper_model = _resolve_whisper_model()
            if whisper_model is None:
                self._fire_error(
                    "whisper.cpp model (ggml-*.bin) not found under "
                    "~/Documents/TouchlessVoiceModels/."
                )
                return False
            self._whisper_command = whisper_cmd
            self._whisper_model_path = whisper_model
        else:
            self._whisper_command = None
            self._whisper_model_path = None

        # Spawn llama-server.
        port = _find_free_port()
        self._llama_port = port
        args = [
            str(llama_exe),
            "-m", str(model_path),
            "-c", str(self._config.local_llm_context_size),
            "-t", str(max(2, min(8, os.cpu_count() or 4))),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--no-webui",
        ]
        if self._llama_backend in {"cuda", "vulkan"}:
            args.extend(["-ngl", "999"])
        try:
            self._llama_proc = subprocess.Popen(
                args,
                cwd=str(llama_exe.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **hidden_subprocess_kwargs(),
            )
        except OSError as exc:
            self._fire_error(f"failed to launch llama-server: {exc}")
            return False

        self._logger.event(
            "local_llama_spawned",
            backend=self._llama_backend,
            model=self._llama_model_path.name,
            port=port,
            pid=self._llama_proc.pid if self._llama_proc else None,
        )

        # Drain stderr in a daemon so it doesn't fill the pipe + block.
        self._llama_stderr_thread = threading.Thread(
            target=self._drain_llama_stderr, daemon=True, name="LiveApiLlamaStderr"
        )
        self._llama_stderr_thread.start()

        # Reader/orchestrator thread waits for llama-server to come up,
        # then drives the per-turn loop.
        self._stop_requested = False
        self._reader_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="LiveApiLocal"
        )
        self._reader_thread.start()
        return True

    def stop(self) -> None:
        self._stop_requested = True
        self._turn_ready.set()  # unblock any waiter
        self._connected = False
        proc = self._llama_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def join(self, timeout: float = 3.0) -> None:
        for thread in (self._reader_thread, self._llama_stderr_thread):
            if thread is not None and thread.is_alive():
                try:
                    thread.join(timeout=timeout)
                except Exception:
                    pass
        proc = self._llama_proc
        if proc is not None:
            try:
                proc.wait(timeout=timeout)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._llama_proc = None

    # ---- public send-side (mirrors RealtimeClient) ----

    def send_audio_chunk(self, pcm16_bytes: bytes) -> bool:
        if not pcm16_bytes or self._stop_requested:
            return False
        # Append + run the energy VAD inline. Cheap (np.frombuffer + RMS).
        arr = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
        if arr.size == 0:
            return True
        rms = float(np.sqrt(np.mean(arr * arr)) / 32768.0)
        now = time.time()
        is_voice = rms >= self._config.local_vad_rms_threshold
        with self._audio_lock:
            self._audio_buffer.extend(pcm16_bytes)
            if is_voice:
                if self._speech_started_at is None:
                    self._speech_started_at = now
                self._last_voice_at = now
            elif self._speech_started_at is not None and self._last_voice_at is not None:
                silent_for = now - self._last_voice_at
                speech_len = self._last_voice_at - self._speech_started_at
                if (
                    silent_for >= self._config.local_vad_silence_seconds
                    and speech_len >= self._config.local_vad_min_speech_seconds
                ):
                    self._flush_turn_locked()
            # Hard cap: even if VAD never sees clean silence, flush after N seconds.
            if (
                self._speech_started_at is not None
                and (now - self._speech_started_at) >= self._config.local_vad_max_turn_seconds
            ):
                self._flush_turn_locked()
        return True

    def send_text_message(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        self._messages.append({"role": "user", "content": text})
        self._logger.text("local_user_text", text)
        self._turn_ready.set()
        return True

    def send_session_note(self, text: str) -> bool:
        """Background note injected after a planner-handled turn so the model
        has context for follow-ups. Local backend stores it but never triggers
        a fresh turn (mirrors realtime semantics)."""
        text = (text or "").strip()
        if not text:
            return True
        self._messages.append({"role": "system", "content": text})
        if self._logger:
            self._logger.text("local_session_note", text)
        return True

    def send_screen_image(self, jpeg_b64: str, *, caption: str = "") -> bool:
        # Phase 1: no vision. Quietly accept so the manager's screen
        # worker can run without errors. Logged so the dev knows why it
        # has no effect.
        if self._logger:
            self._logger.event("local_send_screen_image_noop", bytes_b64=len(jpeg_b64 or ""))
        return True

    def update_tools(self, tools: List[Dict[str, Any]]) -> bool:
        """Replace the available tool list (capability-search router loads
        connector tools on demand). Local backend reads self._tools fresh on
        every turn, so this just swaps the list — no protocol message needed.
        """
        self._tools = _filter_tools_for_local(tools or [])
        return True

    def send_tool_result(self, call_id: str, output: Dict[str, Any]) -> bool:
        tool_name = self._inflight_tool_calls.pop(call_id, "unknown")
        try:
            content = json.dumps(output, default=str)
        except Exception:
            content = str(output)
        self._messages.append(
            {"role": "tool", "tool_call_id": call_id, "name": tool_name, "content": content}
        )
        self._pending_tool_outputs.append({"call_id": call_id, "name": tool_name})
        self._logger.event(
            "local_tool_result_queued",
            call_id=call_id,
            tool=tool_name,
            status=output.get("status") if isinstance(output, dict) else None,
        )
        return True

    def request_response(self) -> bool:
        # The manager calls this after dispatching tool results. We
        # signal the loop to run another inference round.
        self._turn_ready.set()
        return True

    # ---- internal: orchestrator ----

    def _run_loop(self) -> None:
        # Wait for llama-server to accept connections.
        if not self._wait_for_llama_ready(timeout=45.0):
            self._fire_error("llama-server failed to become ready within 45s")
            return
        self._connected = True
        self._messages = [{"role": "system", "content": self._system_instructions}]
        self._logger.event("local_backend_ready", port=self._llama_port)
        if self._on_connected:
            try:
                self._on_connected()
            except Exception as exc:
                self._logger.exception("on_connected_failed", exc)

        # Main loop: wait for a "turn ready" signal, then process it.
        # A turn-ready can be: user finished speaking (audio flushed and
        # transcribed), explicit text message, or tool results queued
        # for the next inference round.
        while not self._stop_requested:
            self._turn_ready.wait(timeout=0.5)
            if self._stop_requested:
                break
            if not self._turn_ready.is_set():
                # Loop iteration with no work — keep going to check
                # _stop_requested + give VAD a tick.
                continue
            self._turn_ready.clear()

            # If audio has been flushed, transcribe before LLM call.
            audio = self._take_pending_audio()
            if audio is not None:
                transcript = self._transcribe(audio)
                if transcript:
                    self._messages.append({"role": "user", "content": transcript})
                    self._emit_event(
                        {
                            "type": "conversation.item.input_audio_transcription.completed",
                            "transcript": transcript,
                        }
                    )

            # Run inference rounds until the model produces a final
            # message (no tool calls), or we hit the safety cap.
            self._run_inference_rounds()

        if self._on_closed:
            try:
                self._on_closed("stopped" if self._stop_requested else None)
            except Exception as exc:
                self._logger.exception("on_closed_failed", exc)

    def _run_inference_rounds(self) -> None:
        for hop in range(self._config.local_max_tool_hops):
            if self._stop_requested:
                return
            self._emit_event({"type": "response.created"})
            try:
                response = self._call_llama()
            except Exception as exc:
                self._logger.exception("local_llm_call_failed", exc)
                self._emit_event({"type": "error", "error": {"message": str(exc), "code": "llm_error"}})
                return
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = (message.get("content") or "").strip()
            tool_calls = message.get("tool_calls") or []

            # Append the assistant message so the next hop has context.
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if content:
                assistant_msg["content"] = content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            self._messages.append(assistant_msg)

            if content:
                # Emit as a single delta + done so the manager's
                # transcript UI updates.
                self._emit_event({"type": "response.text.delta", "delta": content})
                self._emit_event({"type": "response.text.done", "text": content})

            if tool_calls:
                # Dispatch each tool call. The manager will execute and
                # call back send_tool_result + request_response, which
                # will set _turn_ready — but we DO NOT wait here in the
                # current hop; instead we return so the outer loop runs
                # another inference round once results come in.
                for call in tool_calls:
                    call_id = str(call.get("id") or "")
                    fn = call.get("function") or {}
                    name = str(fn.get("name") or "")
                    raw_args = fn.get("arguments") or "{}"
                    self._inflight_tool_calls[call_id] = name
                    self._emit_event(
                        {
                            "type": "response.function_call_arguments.done",
                            "call_id": call_id,
                            "name": name,
                            "arguments": raw_args,
                        }
                    )
                self._emit_event(
                    {
                        "type": "response.done",
                        "response": {
                            "status": "completed",
                            "output": [{"type": "function_call"} for _ in tool_calls],
                        },
                    }
                )
                # Wait for the manager to send results back via
                # send_tool_result + request_response. The manager's
                # dispatch is SYNCHRONOUS (called from our own emit_event
                # via _on_event), so by the time we get here the manager
                # has already executed all tools, appended their outputs
                # to self._messages, and called request_response — which
                # set _turn_ready N times. Pre-clearing here would wipe
                # that signal and cause a 30s spurious hang. Just wait.
                if not self._turn_ready.wait(timeout=60.0):
                    self._logger.warning(
                        "local_tool_result_wait_timeout",
                        hop=hop,
                        tool_count=len(tool_calls),
                        pending=len(self._pending_tool_outputs),
                    )
                    return
                if self._stop_requested:
                    return
                # Drain the signal so the next iteration's wait blocks
                # until NEW request_response calls fire.
                self._turn_ready.clear()
                self._pending_tool_outputs.clear()
                self._logger.event(
                    "local_tool_round_complete",
                    hop=hop,
                    tool_count=len(tool_calls),
                )
                continue

            # No tool calls — final answer this turn.
            self._emit_event(
                {
                    "type": "response.done",
                    "response": {"status": "completed", "output": [{"type": "message"}]},
                }
            )
            return
        self._logger.warning("local_max_tool_hops_reached", limit=self._config.local_max_tool_hops)
        self._emit_event(
            {
                "type": "response.done",
                "response": {"status": "incomplete", "reason": "max_tool_hops"},
            }
        )

    def _take_pending_audio(self) -> Optional[bytes]:
        with self._audio_lock:
            if self._turn_pending is None:
                return None
            data = self._turn_pending
            self._turn_pending = None
            return data

    def _flush_turn_locked(self) -> None:
        """Move the audio buffer to `_turn_pending` and signal the loop. Caller holds the lock."""
        if not self._audio_buffer:
            self._speech_started_at = None
            self._last_voice_at = None
            return
        self._turn_pending = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        self._speech_started_at = None
        self._last_voice_at = None
        self._turn_ready.set()
        try:
            self._logger.event("local_vad_turn_flushed", bytes=len(self._turn_pending))
        except Exception:
            pass

    # ---- whisper transcription ----

    def _transcribe(self, pcm16: bytes) -> str:
        if not self._whisper_command or not self._whisper_model_path:
            return ""
        # Write to a temp WAV so whisper-cli can read it. PCM16 mono
        # @ self._config.audio_sample_rate (24kHz default — whisper
        # accepts that without resampling).
        tmp_path: Optional[Path] = None
        started = time.time()
        try:
            tmp_dir = Path(tempfile.gettempdir()) / "Touchless_LiveApi"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / f"turn_{int(started * 1000)}.wav"
            with wave.open(str(tmp_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._config.audio_sample_rate)
                wf.writeframes(pcm16)

            cmd = [
                *self._whisper_command,
                "-m", str(self._whisper_model_path),
                "-f", str(tmp_path),
                "-l", "en",
                "-nt",  # no timestamps in output
                "-otxt",  # write a .txt sidecar
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30.0,
                    **hidden_subprocess_kwargs(),
                )
            except subprocess.TimeoutExpired:
                self._logger.error("whisper_timeout", duration_sec=30.0)
                return ""
            if proc.returncode != 0:
                self._logger.error(
                    "whisper_failed",
                    returncode=proc.returncode,
                    stderr=(proc.stderr or "")[-500:],
                )
                return ""
            text_path = tmp_path.with_suffix(tmp_path.suffix + ".txt")
            text = ""
            if text_path.exists():
                try:
                    text = text_path.read_text(encoding="utf-8", errors="replace").strip()
                except Exception:
                    text = ""
            if not text:
                # Some whisper-cli builds print directly to stdout
                # instead of writing the .txt file. Fall back to it.
                text = (proc.stdout or "").strip()
            self._logger.latency("whisper_transcribe", started, chars=len(text))
            self._logger.text("local_transcript", text)
            return text
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                    txt = tmp_path.with_suffix(tmp_path.suffix + ".txt")
                    if txt.exists():
                        txt.unlink()
                except Exception:
                    pass

    # ---- llama-server HTTP ----

    def _wait_for_llama_ready(self, *, timeout: float) -> bool:
        if self._llama_port is None:
            return False
        url = f"http://127.0.0.1:{self._llama_port}/v1/models"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._stop_requested:
                return False
            if self._llama_proc and self._llama_proc.poll() is not None:
                self._logger.error(
                    "local_llama_died_early",
                    returncode=self._llama_proc.returncode,
                )
                return False
            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(0.5)
        return False

    def _call_llama(self) -> Dict[str, Any]:
        if self._llama_port is None:
            raise RuntimeError("llama-server not started")
        # Transform tool schemas from the OpenAI Realtime API format
        # (flat: {type, name, description, parameters}) into the OpenAI
        # Chat Completions format that llama-server expects (nested:
        # {type, function: {name, description, parameters}}). Sending the
        # flat form makes llama-server return HTTP 500 because it can't
        # find a `function` field on each tool entry.
        tools_for_chat: List[Dict[str, Any]] = []
        for tool in self._tools:
            tool_type = tool.get("type", "function")
            inner = {k: v for k, v in tool.items() if k != "type"}
            tools_for_chat.append({"type": tool_type, "function": inner})
        payload = {
            "model": "local",
            "messages": self._messages,
            "tools": tools_for_chat,
            "tool_choice": "auto",
            # Low temperature for an agent that follows multi-rule
            # prompts. 0.4 was creative enough to ignore the rules
            # ~30% of the time. 0.2 is more deterministic — the model
            # is more likely to hit the same rule-following branch.
            "temperature": 0.2,
            # Cap output so a runaway model can't spew pages of text.
            "max_tokens": 1024,
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"http://127.0.0.1:{self._llama_port}/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as http_exc:
            # On 4xx/5xx, llama-server usually returns a JSON body
            # describing what it didn't like. Surface it in the log so
            # the next failure mode is diagnosable instead of opaque.
            err_body = ""
            try:
                err_body = http_exc.read().decode("utf-8", errors="replace")[:1000]
            except Exception:
                pass
            self._logger.event(
                "local_llm_http_error",
                code=http_exc.code,
                reason=str(http_exc.reason),
                body=err_body,
            )
            raise RuntimeError(
                f"HTTP {http_exc.code}: {http_exc.reason}"
                + (f" — {err_body[:200]}" if err_body else "")
            )
        elapsed_ms = round((time.time() - started) * 1000.0, 1)
        try:
            parsed = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as exc:
            raise RuntimeError(f"llama-server returned non-JSON: {exc}")
        usage = parsed.get("usage") or {}
        self._logger.event(
            "local_llm_call",
            elapsed_ms=elapsed_ms,
            messages=len(self._messages),
            usage=usage,
        )
        return parsed

    # ---- misc ----

    def _drain_llama_stderr(self) -> None:
        proc = self._llama_proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            line = (line or "").rstrip()
            if not line:
                continue
            # Whitelist: only log lines that look interesting.
            lowered = line.lower()
            if any(k in lowered for k in ("error", "fail", "loaded", "served", "abort")):
                try:
                    self._logger.event("local_llama_stderr", line=line[:300])
                except Exception:
                    pass

    def _emit_event(self, event: Dict[str, Any]) -> None:
        try:
            self._on_event(event)
        except Exception as exc:
            self._logger.exception("local_event_callback_failed", exc, event_type=event.get("type"))

    def _fire_error(self, message: str) -> None:
        self._logger.error("local_backend_error", message=message)
        if self._on_error:
            try:
                self._on_error(message)
            except Exception as exc:
                self._logger.exception("on_error_callback_failed", exc)

# Author: Konstantin Markov
