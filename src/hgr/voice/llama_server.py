from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

from ..utils.runtime_paths import app_base_path


_CORRECTION_SYSTEM_PROMPT = (
    "You clean up speech-to-text dictation transcripts. Apply these fixes:\n"
    "1. Capitalize the first word of every sentence and all proper nouns "
    "(names, days, months, places, brands, languages).\n"
    "2. Add missing punctuation: periods at sentence ends, commas at natural "
    "pauses, question marks for questions.\n"
    "3. Fix misheard homophones (their/there/they're, to/too/two, your/you're, "
    "its/it's, hear/here, then/than).\n"
    "4. Remove obvious dictation stutters where the same word or phrase "
    "repeats back-to-back (e.g., 'the the cat' becomes 'the cat'; 'we shipped "
    "we shipped a feature' becomes 'we shipped a feature').\n"
    "5. Convert spoken structured formats to their standard written form, but "
    "ONLY when surrounding context makes the intent unambiguous:\n"
    "   - Email: 'X at Y dot Z' -> 'X@Y.Z' (e.g., 'p.sharma at example.com' -> "
    "'p.sharma@example.com'; 'john dot smith at company dot com' -> "
    "'john.smith@company.com'). Trigger phrases like 'email', 'address', "
    "'send it to <name> at <domain>' make this unambiguous. DO NOT rewrite "
    "'at' when temporal ('meet at noon', 'starts at 3pm') or locational "
    "('arrive at the office', 'look at this').\n"
    "   - URL: 'www dot X dot Y' -> 'www.X.Y'; 'X dot com slash path' -> "
    "'X.com/path'.\n"
    "   - Phone number: digit sequences that follow a clear area-code pattern "
    "-> '555-123-4567' (or '+1-555-123-4567' when prefixed with 'plus one'). "
    "Leave ambiguous digit strings alone.\n"
    "6. NEVER paraphrase, reword, summarize, or add new content. The format "
    "conversions in rule 5 are the ONLY content-shape changes allowed; keep "
    "every intentional word otherwise. In particular: NEVER convert digits to "
    "words ('3' stays as '3', not 'three') or words to digits ('twenty' stays "
    "as 'twenty', not '20'). A chunk may start with a bare digit because of "
    "mid-utterance chunking; preserve it as-is.\n"
    "7. If the input is already clean, return it exactly unchanged.\n"
    "8. Output ONLY the cleaned text. No preamble, no commentary, no quotes, "
    "no markdown."
)


def _candidate_llama_roots() -> list[Path]:
    roots: list[Path] = []
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


def _candidate_model_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.getenv("HGR_LLAMA_MODEL_ROOT", "").strip()
    if env:
        roots.append(Path(env))
    roots.append(Path.home() / "Documents" / "TouchlessVoiceModels")
    roots.append(Path.home() / "Documents" / "HGRVoiceModels")
    for llama_root in _candidate_llama_roots():
        roots.append(llama_root / "models")
    return roots


def _first_existing_model(names: Iterable[str]) -> Optional[Path]:
    for root in _candidate_model_roots():
        if not root.exists():
            continue
        for name in names:
            path = root / name
            if path.exists():
                return path
        extras = sorted(p for p in root.glob("*.gguf"))
        if extras:
            return extras[0]
    return None


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
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return proc.returncode == 0 and "GPU" in (proc.stdout or "")


def _detect_vulkan() -> bool:
    exe = shutil.which("vulkaninfo")
    if not exe:
        return False
    try:
        proc = subprocess.run(
            [exe, "--summary"],
            capture_output=True,
            text=True,
            timeout=4.0,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    if proc.returncode != 0:
        return False
    return "deviceName" in (proc.stdout or "") or "GPU" in (proc.stdout or "")


def _resolve_backend_executable() -> Optional[tuple[str, Path]]:
    override = os.getenv("HGR_LLAMA_BACKEND", "").strip().lower()
    backend_order: list[str]
    is_mac = sys.platform == "darwin"
    if is_mac:
        # macOS: llama.cpp Metal build (Apple GPU), extensionless Mach-O in
        # build_metal/. No CUDA/Vulkan on mac; CPU is the fallback.
        backend_order = [override] if override in {"metal", "cpu"} else ["metal", "cpu"]
    elif override in {"cuda", "vulkan", "cpu"}:
        backend_order = [override]
    else:
        backend_order = []
        if _detect_nvidia_gpu():
            backend_order.append("cuda")
        if _detect_vulkan():
            backend_order.append("vulkan")
        backend_order.append("cpu")

    build_dirs = {
        "cuda": "build_cuda",
        "vulkan": "build_vulkan",
        "cpu": "build_cpu",
        "metal": "build_metal",
    }
    exe_names = (
        ("bin/llama-server", "llama-server")
        if is_mac
        else ("bin/Release/llama-server.exe", "bin/llama-server.exe")
    )

    for backend in backend_order:
        build_dir = build_dirs[backend]
        for root in _candidate_llama_roots():
            for rel in exe_names:
                candidate = root / build_dir / rel
                if candidate.exists():
                    return backend, candidate
    return None


def _find_free_port(preferred: int = 8757) -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", preferred))
            return preferred
    except OSError:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LlamaServer:
    def __init__(
        self,
        *,
        context_size: int = 4096,
        gpu_layers: int = 999,
        threads: Optional[int] = None,
        request_timeout: float = 12.0,
    ) -> None:
        self._backend: Optional[str] = None
        self._executable: Optional[Path] = None
        self._model_path: Optional[Path] = _first_existing_model(
            (
                "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
                "qwen2.5-3b-instruct-q4_k_m.gguf",
                "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            )
        )
        self._context_size = max(1024, int(context_size))
        self._gpu_layers = int(gpu_layers)
        self._threads = threads or max(2, min(8, os.cpu_count() or 4))
        self._request_timeout = float(request_timeout)

        self._process: Optional[subprocess.Popen[str]] = None
        self._port: Optional[int] = None
        self._available = False
        self._message = "llama-server not configured"
        self._lock = threading.Lock()
        self._stderr_tail: list[str] = []
        self._starting = False

        resolution = _resolve_backend_executable()
        if resolution is None:
            self._message = "llama-server binary not found (llama.cpp/build_cuda|build_vulkan|build_cpu missing)"
            return
        self._backend, self._executable = resolution
        if self._model_path is None:
            self._message = "llama gguf model not found (place Qwen2.5-3B-Instruct-Q4_K_M.gguf under Documents/TouchlessVoiceModels or llama.cpp/models)"
            return
        self._available = True
        self._message = f"llama-server ready ({self._backend}, {self._model_path.name})"

    @property
    def available(self) -> bool:
        return self._available

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    @property
    def message(self) -> str:
        return self._message

    @property
    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(self) -> bool:
        if not self._available or self._executable is None or self._model_path is None:
            return False
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return True
            self._starting = True
            self._port = _find_free_port()
            args = [
                str(self._executable),
                "-m",
                str(self._model_path),
                "-c",
                str(self._context_size),
                "-t",
                str(self._threads),
                "--host",
                "127.0.0.1",
                "--port",
                str(self._port),
                "--no-webui",
            ]
            if self._backend in {"cuda", "vulkan"}:
                args.extend(["-ngl", str(self._gpu_layers)])
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            try:
                self._process = subprocess.Popen(
                    args,
                    cwd=str(self._executable.parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creation_flags,
                )
            except OSError as exc:
                self._message = f"failed to launch llama-server: {exc}"
                self._process = None
                self._starting = False
                return False
            stderr_pipe = self._process.stderr
            threading.Thread(target=self._drain_stderr, args=(stderr_pipe,), name="hgr-llama-stderr", daemon=True).start()

        ready = self._wait_for_ready(deadline=time.monotonic() + 60.0)
        with self._lock:
            self._starting = False
        if ready:
            self._message = f"llama-server listening on :{self._port} ({self._backend})"
        else:
            self._message = "llama-server failed to become ready: " + (" | ".join(self._stderr_tail[-3:]) or "no response")
            self.stop()
        return ready

    def _drain_stderr(self, pipe) -> None:
        if pipe is None:
            return
        for line in pipe:
            stripped = line.rstrip()
            if stripped:
                self._stderr_tail.append(stripped)
                if len(self._stderr_tail) > 40:
                    self._stderr_tail = self._stderr_tail[-40:]

    def _wait_for_ready(self, deadline: float) -> bool:
        if self._port is None:
            return False
        url = f"http://127.0.0.1:{self._port}/health"
        while time.monotonic() < deadline:
            if self._process is None or self._process.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    if 200 <= resp.status < 500:
                        return True
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(0.4)
        return False

    def stop(self) -> None:
        with self._lock:
            proc = self._process
            self._process = None
            self._port = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass

    def correct(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return text
        if not self.running or self._port is None:
            return None
        payload = {
            "model": "hgr-corrector",
            "messages": [
                {"role": "system", "content": _CORRECTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Correct this dictation transcript:\n{text}"},
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": min(1024, max(128, int(len(text) * 1.6))),
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"http://127.0.0.1:{self._port}/v1/chat/completions"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._request_timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError):
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        choices = data.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "").strip()
        if not content:
            return None
        return _strip_wrapper_quotes(content)


def _strip_wrapper_quotes(text: str) -> str:
    stripped = text.strip()
    for _ in range(2):
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'", "`"}:
            stripped = stripped[1:-1].strip()
        else:
            break
    return stripped

# Author: Konstantin Markov
