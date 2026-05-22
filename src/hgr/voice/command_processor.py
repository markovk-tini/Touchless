from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Optional

from ..config.app_config import CONFIG_DIR
from ..debug.chrome_controller import KNOWN_WEB_TARGETS, ChromeController
from ..debug.desktop_controller import DesktopAppEntry, DesktopController
from ..debug.discord_controller import DiscordController
from ..debug.spotify_controller import SpotifyController
from ..debug.text_input_controller import TextInputController
from ..debug.youtube_controller import YouTubeController


COMMON_FILLERS = (
    "please",
    "for me",
    "right now",
    "thanks",
    "thank you",
    "can you",
    "could you",
    "would you",
    "will you",
    "hey hgr",
    "hey app",
    "just",
    "uh",
    "um",
    "and",
)

COMMAND_CORRECTIONS = (
    ("google chrome", "chrome"),
    ("chrome browser", "chrome"),
    ("windows explorer", "file explorer"),
    ("file explore", "file explorer"),
    ("files explorer", "file explorer"),
    ("blu tooth", "bluetooth"),
    ("wi fi", "wifi"),
    ("e mail", "email"),
    ("chat gpt", "chatgpt"),
    ("visual stdios", "visual studio"),
    ("visual studios", "visual studio"),
    ("searchup", "search up"),
    ("a c dc", "ac/dc"),
    ("ac dc", "ac/dc"),
    ("v s code", "vs code"),
    ("vs code", "visual studio code"),
    ("discord app", "discord"),
    ("steam app", "steam"),
    ("you tube", "youtube"),
    ("key card", "kicad"),
    ("key cards", "kicad"),
    ("key cad", "kicad"),
    ("ki cad", "kicad"),
    ("k i cad", "kicad"),
    ("google browser", "chrome"),
    ("chrome browser", "chrome"),
    ("near by", "nearby"),
    ("clothes", "close"),
    ("cloths", "close"),
    ("flows", "close"),
    ("close d ", "close "),
    ("and close", "close"),
)

APP_ALIASES: dict[str, tuple[str, ...]] = {
    "spotify": ("spotify",),
    "chrome": ("google chrome", "chrome browser", "chrome", "browser"),
    "settings": ("settings", "device settings", "windows settings", "system settings"),
    "file_explorer": ("file explorer", "explorer"),
    "outlook": ("outlook", "mail app"),
    "discord": ("discord",),
}


# Voice command "open touchless [tab]" → maps tab keywords to a
# stable section key. The executor finds the live MainWindow
# (in-process) and translates the key to the section index, so
# we don't need to import section constants here (no circular
# import). Order matters: longer / more-specific phrases first
# so "custom gestures" wins over plain "gestures".
TOUCHLESS_TAB_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("custom gestures", "custom_gestures"),
    ("custom gesture", "custom_gestures"),
    ("control guide", "gestures"),
    ("preset gestures", "gestures"),
    ("gesture guide", "gestures"),
    ("gesture binds", "gesture_binds"),
    ("gesture bindings", "gesture_binds"),
    ("save locations", "save_locations"),
    ("save location", "save_locations"),
    ("instructions", "instructions"),
    ("walkthrough", "tutorial"),
    ("microphone", "microphone"),
    ("gestures", "gestures"),
    ("rebind", "gesture_binds"),
    ("bindings", "gesture_binds"),
    ("binds", "gesture_binds"),
    ("camera", "camera"),
    ("webcam", "camera"),
    ("colors", "colors"),
    ("color", "colors"),
    ("theme", "colors"),
    ("appearance", "colors"),
    ("tutorial", "tutorial"),
    ("updates", "updates"),
    ("update", "updates"),
    ("intro", "instructions"),
    ("home", "_home"),
    ("main page", "_home"),
    ("homepage", "_home"),
    ("mic", "microphone"),
    ("saves", "save_locations"),
)

SPOTIFY_PLAY_PHRASES = ("play", "put on", "listen to", "queue", "queue up", "start playing")
CHROME_SEARCH_PHRASES = ("search up", "search for", "search", "look up", "look for", "find", "go to", "navigate to", "take me to")
GENERIC_OPEN_PHRASES = ("open", "launch", "start", "bring up", "show")
APP_FOCUS_PHRASES = GENERIC_OPEN_PHRASES + ("switch to", "focus", "focus on", "pull up", "open up", "show me")
APP_LAUNCH_PHRASES = APP_FOCUS_PHRASES + (
    "run",
    "start up",
    "boot up",
    "fire up",
    "load",
    "use",
    "go into",
    "enter",
    "i need",
    "need",
)
DEDICATED_APP_LAUNCH_PHRASES = tuple(phrase for phrase in APP_LAUNCH_PHRASES if phrase != "focus")
CLOSE_WINDOW_PHRASES = ("close", "close the", "exit", "quit", "dismiss")

# Touchless clip-export phrases. The voice command "clip that" auto-
# exports the most-recent N seconds of footage from the rolling clip
# cache and saves it to the default clips folder without asking the
# user where to save (different from the gesture path, which prompts
# for a save location). 1m is the default duration; "clip 30 seconds"
# / "clip last 30" routes to the 30 s variant.
CLIP_TRIGGER_PHRASES = (
    "clip that",
    "clip this",
    "clip it",
    "clip me",
    "save clip",
    "save the clip",
    "save that clip",
    "save this clip",
    "make clip",
    "make a clip",
    "create clip",
    "create a clip",
    "grab clip",
    "grab a clip",
    "grab that clip",
    "record clip",
    "clip the last minute",
    "clip last minute",
    "clip the past minute",
    "clip past minute",
    "clip last 60 seconds",
    "clip last sixty seconds",
    "clip the last 30 seconds",
    "clip last 30 seconds",
    "clip last thirty seconds",
)
CLIP_30S_HINT_PHRASES = (
    "30 second",
    "30 seconds",
    "thirty second",
    "thirty seconds",
    "last 30",
    "last thirty",
    "past 30",
    "past thirty",
    "half a minute",
)
APP_OBJECT_HINTS = (
    "app called",
    "application called",
    "program called",
    "app named",
    "application named",
    "program named",
    "app",
    "application",
    "program",
    "window",
    "called",
    "named",
    "titled",
)
SPOTIFY_CONTEXT_PHRASES = ("on spotify", "in spotify", "from spotify", "using spotify")

# Phrases that gate the Discord parser. If any of these appear in the
# normalised text, the parser will accept its action keywords (mute,
# deafen, leave, join, send, open) as Discord-targeted. The aliases
# below (in APP_ALIASES["discord"]) and the parser's matched_alias
# check do the same job — context phrases let users say things like
# "mute me on discord" instead of "mute discord" / "discord mute".
DISCORD_CONTEXT_PHRASES = (
    "on discord",
    "in discord",
    "from discord",
    "using discord",
    "to discord",
)
# Disconnect phrases — any of these (with or without discord context)
# treated as "leave the current voice call".
DISCORD_LEAVE_PHRASES = (
    "leave call",
    "leave the call",
    "hang up",
    "hangup",
    "disconnect call",
    "disconnect from call",
    "leave voice",
    "leave voice channel",
    "leave channel",
    "drop call",
    "end call",
)
# Phrases that prefix a voice-channel join. "Join" alone is too
# generic (it could mean a server join, a chat join, etc.) so we
# require "join" + optional qualifier + channel name.
DISCORD_JOIN_PHRASES = (
    "join voice",
    "join voice channel",
    "join channel",
    "join the channel",
    "join",
)
# Phrases that prefix a text-channel focus. Two tiers:
#
#   STRONG — explicit "channel" keyword. Fire regardless of whether
#   the user mentioned Discord. "open channel general" is
#   unambiguous; no other app uses the word "channel" the same way.
#
#   WEAK — bare "switch to" / "go to". Require Discord context
#   (matched alias, "on discord", etc.) so the parser doesn't grab
#   "switch to dark mode" / "go to settings".
DISCORD_OPEN_TEXT_PHRASES_STRONG = (
    "go to channel",
    "switch to channel",
    "switch channel to",
    "open channel",
    "go to the channel",
    "open the channel",
)
DISCORD_OPEN_TEXT_PHRASES_WEAK = (
    "switch to",
    "go to",
)
# Phrases that prefix a server (guild) switch.
DISCORD_OPEN_SERVER_PHRASES = (
    "switch to server",
    "open server",
    "go to server",
    "switch server to",
    "open the server",
)
# Send-message prefixes. Anything starting with these is treated as
# "send a message to <target>". The destination and message body are
# separated by " in " / " to " in the spoken text — last occurrence
# wins to avoid splitting on an "in"/"to" inside the message itself.
DISCORD_SEND_PHRASES = (
    "send message",
    "send a message",
    "send dm",
    "send",
)
DISCORD_SEND_TARGET_SEPARATORS = (" in ", " to ")
YOUTUBE_CONTEXT_PHRASES = (
    "on youtube",
    "in youtube",
    "from youtube",
    "using youtube",
    "on you tube",
    "in you tube",
    "from you tube",
    "using you tube",
)
CHROME_CONTEXT_PHRASES = ("on chrome", "in chrome", "using chrome", "in the browser", "on the browser")
WEB_FALLBACK_PREFIXES = (
    ("search", "search up"),
    ("search", "search for"),
    ("search", "search"),
    ("search", "look up"),
    ("search", "look for"),
    ("search", "find"),
    ("open", "go to"),
    ("open", "navigate to"),
    ("open", "take me to"),
    ("open", "open"),
)
WEB_TARGET_HINTS = {"youtube", "github", "gmail", "chatgpt", "reddit", "wikipedia", "maps", "docs", "drive", "calendar"}
MUSIC_NOUNS = ("music", "song", "songs", "track", "tracks", "playlist", "playlists", "album", "albums", "artist")
EDGE_NOISE_WORDS = (
    "a",
    "an",
    "app",
    "application",
    "called",
    "for",
    "from",
    "in",
    "inside",
    "my",
    "me",
    "named",
    "on",
    "program",
    "the",
    "to",
    "titled",
    "up",
    "window",
    "with",
)

SETTINGS_TOPICS = {
    "apps": ("apps", "applications", "installed apps", "programs"),
    "bluetooth": ("bluetooth",),
    "camera": ("camera", "webcam"),
    "display": ("display", "screen", "monitor"),
    "email": ("email", "mail", "accounts"),
    "network": ("network",),
    "privacy": ("privacy", "permissions"),
    "sound": ("sound", "audio"),
    "storage": ("storage", "disk", "drive"),
    "update": ("update", "updates", "windows update"),
    "volume": ("volume",),
    "wifi": ("wifi", "wi-fi", "wireless"),
}

KNOWN_FOLDERS = {
    "desktop": ("desktop",),
    "documents": ("documents", "document"),
    "downloads": ("downloads", "download"),
    "music": ("music",),
    "pictures": ("pictures", "photos"),
    "videos": ("videos", "video"),
}

FILE_REQUEST_HINTS = {
    "assignment",
    "budget",
    "contract",
    "csv",
    "doc",
    "docx",
    "document",
    "draft",
    "essay",
    "excel",
    "homework",
    "invoice",
    "json",
    "note",
    "notes",
    "paper",
    "pdf",
    "powerpoint",
    "presentation",
    "project",
    "proposal",
    "report",
    "resume",
    "sheet",
    "slides",
    "spreadsheet",
    "summary",
    "text",
    "thesis",
    "txt",
    "word",
    "xlsx",
}

SELECTION_LETTERS = "ABCDEFGH"
SELECTION_LETTER_WORDS = {
    "a": 1,
    "ay": 1,
    "alpha": 1,
    "b": 2,
    "bee": 2,
    "bravo": 2,
    "c": 3,
    "cee": 3,
    "charlie": 3,
    "d": 4,
    "dee": 4,
    "delta": 4,
    "e": 5,
    "echo": 5,
    "f": 6,
    "foxtrot": 6,
    "g": 7,
    "gee": 7,
    "golf": 7,
    "h": 8,
    "aitch": 8,
    "hotel": 8,
}


@dataclass
class VoiceCommandContext:
    preferred_app: str | None = None


@dataclass
class ParsedVoiceCommand:
    raw_text: str
    normalized_text: str
    app_name: str
    action: str
    confidence: float
    query: str | None = None
    matched_alias: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceExecutionResult:
    success: bool
    target: str
    heard_text: str
    control_text: str
    info_text: str
    intent: ParsedVoiceCommand | None = None
    display_text: str | None = None


class VoiceProfileStore:
    _volatile_cache: dict[str, dict[str, Any]] = {}

    def __init__(self, *, path: Path | None = None) -> None:
        self._path = path or (CONFIG_DIR / "voice_profile.json")
        self._lock = threading.Lock()

    def best_match(self, utterance: str) -> dict[str, Any] | None:
        normalized = self._normalize(utterance)
        if not normalized:
            return None
        best_entry: dict[str, Any] | None = None
        best_score = 0.0
        for entry in self._read_history():
            previous = self._normalize(str(entry.get("utterance", "")))
            if not previous:
                continue
            ratio = SequenceMatcher(None, normalized, previous).ratio()
            overlap = self._token_overlap(normalized, previous)
            score = ratio + overlap * 0.20 + min(int(entry.get("count", 1)), 8) * 0.02
            if score > best_score:
                best_score = score
                best_entry = dict(entry)
                best_entry["score"] = score
        if best_entry is None or best_score < 0.92:
            return None
        return best_entry

    def record_success(self, intent: ParsedVoiceCommand) -> None:
        self._upsert_history(intent=intent, corrected=False)

    def record_correction(self, *, utterance: str, app_name: str, action: str, query: str | None = None) -> None:
        intent = ParsedVoiceCommand(
            raw_text=utterance,
            normalized_text=self._normalize(utterance),
            app_name=app_name,
            action=action,
            confidence=1.0,
            query=query,
        )
        self._upsert_history(intent=intent, corrected=True)

    def history_entries(self) -> list[dict[str, Any]]:
        return list(self._read_history())

    def _upsert_history(self, *, intent: ParsedVoiceCommand, corrected: bool) -> None:
        normalized = self._normalize(intent.raw_text)
        if not normalized:
            return
        with self._lock:
            data = self._load_data()
            history = data.setdefault("history", [])
            now = int(time.time())
            for entry in history:
                if (
                    self._normalize(str(entry.get("utterance", ""))) == normalized
                    and entry.get("app_name") == intent.app_name
                    and entry.get("action") == intent.action
                    and (entry.get("query") or None) == intent.query
                ):
                    entry["count"] = int(entry.get("count", 0)) + 1
                    entry["updated_at"] = now
                    entry["corrected"] = bool(entry.get("corrected")) or corrected
                    break
            else:
                history.append(
                    {
                        "utterance": normalized,
                        "app_name": intent.app_name,
                        "action": intent.action,
                        "query": intent.query,
                        "count": 1,
                        "corrected": corrected,
                        "updated_at": now,
                    }
                )
            history.sort(key=lambda item: (int(item.get("count", 0)), int(item.get("updated_at", 0))), reverse=True)
            data["history"] = history[:64]
            self._save_data(data)

    def _read_history(self) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load_data()
        history = data.get("history")
        return history if isinstance(history, list) else []

    def _load_data(self) -> dict[str, Any]:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            cached = self._volatile_cache.get(str(self._path))
            return dict(cached) if isinstance(cached, dict) else {"history": []}

    def _save_data(self, data: dict[str, Any]) -> None:
        self._volatile_cache[str(self._path)] = json.loads(json.dumps(data))
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _normalize(self, text: str) -> str:
        return " ".join(str(text or "").lower().split()).strip()

    def _token_overlap(self, left: str, right: str) -> float:
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens or not right_tokens:
            return 0.0
        shared = len(left_tokens & right_tokens)
        return shared / max(len(left_tokens), len(right_tokens))


class VoiceCommandProcessor:
    def __init__(
        self,
        *,
        chrome_controller: ChromeController | None = None,
        spotify_controller: SpotifyController | None = None,
        desktop_controller: DesktopController | None = None,
        youtube_controller: YouTubeController | None = None,
        discord_controller: DiscordController | None = None,
        text_input_controller: TextInputController | None = None,
        profile_store: VoiceProfileStore | None = None,
        log_step: "Optional[Callable[[str], None]]" = None,
    ) -> None:
        # Step-emit callback for the detailed log surface. The engine
        # wires this to its `engine_log` Qt signal so every step of
        # every multi-step command (resolve destination → fuzzy-match
        # → focus window → paste → press enter) becomes a line the
        # user sees in real time. When a command fails midway, the
        # log tells them exactly which step broke and why instead of
        # the bare "(failed)" summary the executor returns. Optional
        # — falls back to a no-op so unit tests and standalone uses
        # don't have to provide it.
        self._log_step: Callable[[str], None] = log_step or (lambda _msg: None)
        self.chrome_controller = chrome_controller or ChromeController()
        self.spotify_controller = spotify_controller or SpotifyController()
        self.desktop_controller = desktop_controller or DesktopController()
        # Voice's YouTube auto-play handler. Distinct instance from
        # the engine's youtube_controller (which drives playback
        # gestures) — both operate on the same Chrome window via
        # separate API surfaces, so sharing isn't required.
        self.youtube_controller = youtube_controller or YouTubeController()
        # Discord RPC controller — same instance as the engine's so
        # commands run against the connection the user already
        # authorised. Falls back to a fresh instance for unit tests
        # / from-cold launches.
        self.discord_controller = discord_controller or DiscordController()
        # Text input controller is reused for the Discord "send X"
        # voice command — after SELECT_TEXT_CHANNEL focuses the
        # channel, the message is typed via clipboard-paste + Enter.
        # Same instance the engine uses so we don't fight over the
        # clipboard.
        self.text_input_controller = text_input_controller or TextInputController()
        self.profile_store = profile_store or VoiceProfileStore()
        self._pending_selection: dict[str, Any] | None = None
        # Awaiting-destination state for the Discord "send X" voice
        # command when no destination was provided AND the controller
        # fallback chain (voice channel → last focused) found nothing.
        # The next user utterance is consumed as the channel name; the
        # parser is bypassed and the message is sent verbatim. Stored
        # as None when no send is pending. Cleared on success/cancel/
        # 30 s timeout.
        self._discord_pending_send: dict[str, Any] | None = None

    def parse(self, spoken_text: str, *, context: VoiceCommandContext | None = None) -> ParsedVoiceCommand | None:
        normalized = self._normalize_text(spoken_text)
        if not normalized:
            return None

        candidates = [
            self._parse_clip(normalized, raw_text=spoken_text, context=context),
            self._parse_touchless_app(normalized, raw_text=spoken_text, context=context),
            self._parse_spotify(normalized, raw_text=spoken_text, context=context),
            # YouTube parser runs BEFORE chrome so "play X on youtube"
            # is handled as a YouTube play (search + open) rather than
            # being intercepted by the chrome web-target heuristic.
            self._parse_youtube(normalized, raw_text=spoken_text, context=context),
            # Discord parser runs before chrome / catalog so phrases
            # like "open general" (channel) aren't swallowed by the
            # generic-open heuristic. The discord context check is
            # strict: parser only fires if the text has a discord
            # marker OR uses one of the high-specificity verb prefixes
            # (mute/deafen/hang up/leave call/join voice).
            self._parse_discord(normalized, raw_text=spoken_text, context=context),
            self._parse_chrome(normalized, raw_text=spoken_text, context=context),
            self._parse_settings(normalized, raw_text=spoken_text, context=context),
            self._parse_close_window(normalized, raw_text=spoken_text, context=context),
            self._parse_file_explorer(normalized, raw_text=spoken_text, context=context),
            self._parse_outlook(normalized, raw_text=spoken_text, context=context),
        ]
        intents = [intent for intent in candidates if intent is not None]
        early = self._best_intent(intents)
        if early is not None and self._can_skip_catalog_lookup(early):
            return early if early.confidence >= 0.56 else None

        catalog_matches = self.desktop_controller.rank_applications_in_text(normalized)
        candidates.extend(
            [
                self._parse_catalog_open(normalized, raw_text=spoken_text, catalog_matches=catalog_matches),
                self._parse_generic_open(normalized, raw_text=spoken_text),
            ]
        )
        intents = [intent for intent in candidates if intent is not None]
        learned = self.profile_store.best_match(normalized)
        if learned is not None:
            for intent in intents:
                if intent.app_name == learned.get("app_name"):
                    intent.confidence += 0.08
                if intent.action == learned.get("action"):
                    intent.confidence += 0.04
        if not intents:
            return None
        for intent in intents:
            self._apply_catalog_bias(intent, catalog_matches, context=context)
            intent.confidence += self._intent_template_bonus(normalized, intent)
        best = self._best_intent(intents)
        if best is None:
            return None
        return best if best.confidence >= 0.56 else None

    def execute(self, spoken_text: str, *, context: VoiceCommandContext | None = None) -> VoiceExecutionResult:
        # Discord send-message in awaiting-destination state takes
        # precedence over normal parsing. The next utterance after a
        # destination-less "send X" prompt is consumed as the channel
        # name verbatim (no parser run — the channel name might
        # otherwise look like a different intent, e.g. "general" →
        # generic-open).
        pending_send_result = self._execute_discord_pending_send(spoken_text)
        if pending_send_result is not None:
            return pending_send_result

        pending_result = self._execute_pending_selection(spoken_text)
        if pending_result is not None:
            return pending_result

        intent = self.parse(spoken_text, context=context)
        if intent is None:
            return VoiceExecutionResult(
                success=False,
                target="voice",
                heard_text=spoken_text,
                control_text="voice command not understood",
                info_text="-",
                intent=None,
                display_text=self._normalize_text(spoken_text) or spoken_text,
            )

        result = self._execute_intent(intent)
        if result.success:
            self.profile_store.record_success(intent)
        if result.display_text is None:
            result.display_text = self._display_text_for_intent(intent)
        return result

    @staticmethod
    def _selection_key_for_index(index: int) -> str:
        if index < 0:
            return "?"
        if index < len(SELECTION_LETTERS):
            return SELECTION_LETTERS[index]
        return str(index + 1)

    def _extract_selection_number(self, spoken_text: str) -> int | None:
        normalized = self._normalize_text(spoken_text)
        if not normalized:
            return None
        _CANCEL_EXACT = {
            "cancel", "nevermind", "never mind", "stop", "go back",
            "close list", "wrong files", "wrong file", "wrong folders", "wrong folder",
            "wrong apps", "wrong app", "not these", "none of these", "cancel search",
            "no", "nope", "abort", "exit", "quit", "close", "dismiss", "hide",
            "not that", "forget it", "scratch that", "skip", "leave it", "exit list",
        }
        if normalized in _CANCEL_EXACT:
            return -1
        _CANCEL_PATTERNS = (
            r"\bcancel\b", r"\bnevermind\b", r"\bnever mind\b", r"\bgo back\b",
            r"\bforget it\b", r"\bscratch that\b", r"\bnone of (?:these|those|them)\b",
            r"\bclose (?:the )?list\b", r"\bnot (?:this|that|these|those)\b",
        )
        for pat in _CANCEL_PATTERNS:
            if re.search(pat, normalized):
                return -1
        word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
        prefix = (
            r"(?:(?:can\s+(?:you|i)\s+)?(?:open|show(?:\s+me)?|launch|pull\s+up|choose|select|see|"
            r"let(?:(?:'|\s)?s)\s+(?:see|open|look\s+at|check)|i(?:'|\s)?ll\s+take)"
            r")?(?:\s+(?:file|folder|item|number|option))?"
        )
        match = re.search(rf"\b{prefix}\s*(\d{{1,2}})\b", normalized)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
        word_match = re.search(rf"\b{prefix}\s*(one|two|three|four|five|six|seven|eight)\b", normalized)
        if word_match:
            return word_map.get(word_match.group(1))
        letter_match = re.search(
            rf"^(?:{prefix}\s*)?(?:option|number|letter|choice|pick)?\s*"
            r"(a|ay|alpha|b|bee|bravo|c|cee|charlie|d|dee|delta|e|echo|f|foxtrot|g|gee|golf|h|aitch|hotel)\s*$",
            normalized,
        )
        if letter_match:
            return SELECTION_LETTER_WORDS.get(letter_match.group(1))
        return None

    def _serialize_desktop_entry(self, entry: DesktopAppEntry) -> dict[str, Any]:
        return {
            "label": entry.display_name,
            "app_name": entry.display_name,
            "display_name": entry.display_name,
            "normalized_name": entry.normalized_name,
            "target": entry.target,
            "source": entry.source,
            "aliases": list(entry.aliases),
            "category": entry.category,
        }

    def _deserialize_desktop_entry(self, payload: dict[str, Any]) -> DesktopAppEntry | None:
        target = str(payload.get("target", "") or "").strip()
        display_name = str(
            payload.get("display_name")
            or payload.get("app_name")
            or payload.get("label")
            or ""
        ).strip()
        if not display_name or not target:
            return None
        normalized_name = str(payload.get("normalized_name", "") or "").strip()
        if not normalized_name:
            normalized_name = self.desktop_controller._normalize_application_name(display_name)
        aliases_raw = payload.get("aliases") or ()
        if isinstance(aliases_raw, (list, tuple)):
            aliases = tuple(str(alias or "").strip() for alias in aliases_raw if str(alias or "").strip())
        else:
            aliases = ()
        return DesktopAppEntry(
            display_name=display_name,
            normalized_name=normalized_name,
            target=target,
            source=str(payload.get("source", "voice_selection") or "voice_selection"),
            aliases=aliases,
            category=str(payload.get("category", "generic") or "generic"),
        )

    def _create_pending_selection(self, *, kind: str, query: str, heard_text: str, options: list[dict[str, Any]]) -> str:
        cleaned: list[dict[str, Any]] = []
        seen: set[str] = set()
        for option in options:
            label = str(option.get("label", "") or "").strip()
            path_text = str(option.get("path", option.get("app_name", "")) or "").strip()
            key = f"{label.lower()}|{path_text.lower()}"
            if not label or key in seen:
                continue
            seen.add(key)
            cleaned.append(dict(option))
        self._pending_selection = {
            "kind": kind,
            "query": query,
            "heard_text": heard_text,
            "options": cleaned[:8],
        }
        title = "Which file/folder?" if kind == "file" else "Which app?"
        lines = [title]
        for index, option in enumerate(self._pending_selection["options"], start=1):
            selection_key = self._selection_key_for_index(index - 1)
            if kind == "file":
                lines.append(f"{selection_key}. {option['label']} — {option.get('path', '')}")
            else:
                lines.append(f"{selection_key}. {option['label']}")
        lines.append("Say the corresponding letter.")
        return "\n".join(lines)

    def _execute_pending_selection(self, spoken_text: str) -> VoiceExecutionResult | None:
        pending = self._pending_selection
        if pending is None:
            return None
        choice = self._extract_selection_number(spoken_text)
        if choice is None:
            prompt = self._create_pending_selection(
                kind=str(pending.get("kind", "file")),
                query=str(pending.get("query", "")),
                heard_text=str(pending.get("heard_text", "")),
                options=list(pending.get("options", [])),
            )
            return VoiceExecutionResult(
                success=False,
                target="voice_selection",
                heard_text=spoken_text,
                control_text="awaiting selection",
                info_text=prompt,
                display_text=prompt,
            )
        if choice == -1:
            self._pending_selection = None
            return VoiceExecutionResult(
                success=False,
                target="voice",
                heard_text=spoken_text,
                control_text="selection cancelled",
                info_text="-",
                display_text="selection cancelled",
            )
        options = list(pending.get("options", []))
        if choice < 1 or choice > len(options):
            prompt = self._create_pending_selection(
                kind=str(pending.get("kind", "file")),
                query=str(pending.get("query", "")),
                heard_text=str(pending.get("heard_text", "")),
                options=options,
            )
            return VoiceExecutionResult(
                success=False,
                target="voice_selection",
                heard_text=spoken_text,
                control_text="that number was not listed",
                info_text=prompt,
                display_text=prompt,
            )
        selected = options[choice - 1]
        self._pending_selection = None
        if pending.get("kind") == "app":
            selected_entry = self._deserialize_desktop_entry(selected)
            if selected_entry is not None:
                success = self.desktop_controller.open_desktop_entry(selected_entry)
            else:
                success = self.desktop_controller.open_named_application(str(selected.get("app_name", selected.get("label", ""))))
            return VoiceExecutionResult(
                success=success,
                target="system",
                heard_text=spoken_text,
                control_text=self.desktop_controller.message,
                info_text=self.desktop_controller.message,
                display_text=str(selected.get("label", "")),
            )
        success = self.desktop_controller.open_resolved_path(Path(str(selected.get("path", ""))))
        return VoiceExecutionResult(
            success=success,
            target="file_explorer",
            heard_text=spoken_text,
            control_text=self.desktop_controller.message,
            info_text=self.desktop_controller.message,
            display_text=str(selected.get("label", "")),
        )

    def export_training_bundle(self, *, output_dir: Path | None = None) -> dict[str, Path]:
        from .training_data import VoiceCommandDatasetBuilder

        builder = VoiceCommandDatasetBuilder(
            desktop_controller=self.desktop_controller,
            profile_store=self.profile_store,
        )
        return builder.export_bundle(output_dir=output_dir)

    def _execute_intent(self, intent: ParsedVoiceCommand) -> VoiceExecutionResult:
        if intent.app_name == "touchless":
            result = self._execute_clip(intent)
        elif intent.app_name == "touchless_app":
            result = self._execute_touchless_app(intent)
        elif intent.app_name == "spotify":
            result = self._execute_spotify(intent)
        elif intent.app_name == "chrome":
            result = self._execute_chrome(intent)
        elif intent.app_name == "youtube":
            result = self._execute_youtube(intent)
        elif intent.app_name == "discord":
            result = self._execute_discord(intent)
        elif intent.app_name == "settings":
            result = self._execute_settings(intent)
        elif intent.app_name == "file_explorer":
            result = self._execute_file_explorer(intent)
        elif intent.app_name == "outlook":
            result = self._execute_outlook(intent)
        else:
            result = self._execute_generic(intent)
        result.intent = intent
        return result

    def _execute_touchless_app(self, intent: ParsedVoiceCommand) -> VoiceExecutionResult:
        """Bring the Touchless main window to the foreground and,
        if a tab keyword was matched, navigate to that section.
        Runs in-process: we find the MainWindow via QApplication's
        top-level widgets and dispatch the show + navigate calls
        on the GUI thread via QTimer.singleShot(0, ...).

        Special action 'open_save_folder' (set by _parse_touchless_app
        when the user says e.g. "open touchless clip / drawing /
        screenshot / recording"): instead of focusing Touchless,
        open the user's configured save folder for that output
        kind in Windows Explorer. Without this the parser would
        have routed those phrases to the generic launch path and
        the user would see Touchless come to the foreground —
        which is what they reported as wrong."""
        if intent.action == "open_save_folder":
            output_kind = ""
            try:
                slots = dict(getattr(intent, "slots", {}) or {})
                output_kind = str(slots.get("output_kind", "") or intent.query or "").strip().lower()
            except Exception:
                output_kind = str(intent.query or "").strip().lower()
            try:
                # Locate the MainWindow so we can read its live
                # config (the user's chosen save folders). Falls
                # back to the AppConfig defaults if MainWindow is
                # somehow missing.
                from PySide6.QtWidgets import QApplication
                from ...config.app_config import (
                    SAVE_LOCATION_CONFIG_FIELDS,
                    SAVE_LOCATION_LABELS,
                    default_save_directory,
                )
                target_dir: Path | None = None
                main_window = None
                app = QApplication.instance()
                if app is not None:
                    for widget in app.topLevelWidgets():
                        if widget.__class__.__name__ == "MainWindow":
                            main_window = widget
                            break
                config_field = SAVE_LOCATION_CONFIG_FIELDS.get(output_kind, "")
                if main_window is not None and config_field:
                    raw_path = str(getattr(main_window.config, config_field, "") or "").strip()
                    if raw_path:
                        candidate = Path(raw_path).expanduser()
                        if candidate.exists() and candidate.is_dir():
                            target_dir = candidate
                if target_dir is None:
                    target_dir = default_save_directory(output_kind)
                if target_dir is None or not target_dir.exists():
                    label = SAVE_LOCATION_LABELS.get(output_kind, output_kind or "file")
                    info = f"touchless {label.lower()} folder not found"
                    return VoiceExecutionResult(
                        success=False,
                        target="touchless_app",
                        heard_text=intent.raw_text,
                        control_text=info,
                        info_text=info,
                    )
                import os as _os
                _os.startfile(str(target_dir))
                label = SAVE_LOCATION_LABELS.get(output_kind, output_kind)
                info = f"opening Touchless {label.lower()} folder"
                return VoiceExecutionResult(
                    success=True,
                    target="touchless_app",
                    heard_text=intent.raw_text,
                    control_text=info,
                    info_text=info,
                )
            except Exception as exc:
                info = f"open touchless {output_kind} folder failed: {exc}"
                return VoiceExecutionResult(
                    success=False,
                    target="touchless_app",
                    heard_text=intent.raw_text,
                    control_text=info,
                    info_text=info,
                )
        target_key = intent.query
        info: str
        try:
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import QTimer
            app = QApplication.instance()
            if app is None:
                return VoiceExecutionResult(
                    success=False,
                    target="touchless_app",
                    heard_text=intent.raw_text,
                    control_text="touchless not running",
                    info_text="touchless not running",
                )
            main_window = None
            for widget in app.topLevelWidgets():
                if widget.__class__.__name__ == "MainWindow":
                    main_window = widget
                    break
            if main_window is None:
                return VoiceExecutionResult(
                    success=False,
                    target="touchless_app",
                    heard_text=intent.raw_text,
                    control_text="touchless main window not found",
                    info_text="touchless main window not found",
                )

            def _focus_and_navigate():
                try:
                    if main_window.isMinimized():
                        main_window.showNormal()
                    else:
                        main_window.show()
                    main_window.raise_()
                    main_window.activateWindow()
                    if target_key and target_key != "_home":
                        section_idx = self._touchless_section_index(
                            main_window, target_key
                        )
                        if section_idx is not None:
                            try:
                                main_window.show_settings_page(section_idx)
                            except Exception:
                                pass
                    elif target_key == "_home":
                        try:
                            main_window.show_home_page()
                        except Exception:
                            pass
                    # Win32 SetForegroundWindow as a belt-and-
                    # suspenders fallback — Qt's activateWindow
                    # is sometimes a no-op on Windows when
                    # another app stole focus recently.
                    try:
                        import ctypes
                        hwnd = int(main_window.winId())
                        if hwnd:
                            ctypes.windll.user32.SetForegroundWindow(hwnd)
                    except Exception:
                        pass
                except Exception:
                    pass

            QTimer.singleShot(0, _focus_and_navigate)
            label = target_key or "home"
            info = f"opening Touchless ({label})" if target_key else "focusing Touchless"
            return VoiceExecutionResult(
                success=True,
                target="touchless_app",
                heard_text=intent.raw_text,
                control_text=info,
                info_text=info,
            )
        except Exception as exc:
            info = f"open touchless failed: {exc}"
            return VoiceExecutionResult(
                success=False,
                target="touchless_app",
                heard_text=intent.raw_text,
                control_text=info,
                info_text=info,
            )

    @staticmethod
    def _touchless_section_index(main_window, target_key: str):
        """Map a TOUCHLESS_TAB_KEYWORDS value (e.g. 'gestures',
        'camera') to the SECTION_* integer constant defined in
        main_window. We import the constants here (deferred) so
        the voice processor doesn't have a top-level dependency
        on main_window."""
        try:
            from ..app.ui import main_window as _mw
        except Exception:
            return None
        mapping = {
            "instructions": getattr(_mw, "SECTION_INSTRUCTIONS", None),
            "gestures": getattr(_mw, "SECTION_GESTURES", None),
            "custom_gestures": getattr(_mw, "SECTION_CUSTOM_GESTURE", None),
            "gesture_binds": getattr(_mw, "SECTION_GESTURE_BINDS", None),
            "camera": getattr(_mw, "SECTION_CAMERA", None),
            "microphone": getattr(_mw, "SECTION_MICROPHONE", None),
            "save_locations": getattr(_mw, "SECTION_SAVE_LOCATIONS", None),
            "colors": getattr(_mw, "SECTION_COLORS", None),
            "tutorial": getattr(_mw, "SECTION_TUTORIAL", None),
            "updates": getattr(_mw, "SECTION_UPDATES", None),
        }
        return mapping.get(target_key)

    def _execute_clip(self, intent: ParsedVoiceCommand) -> VoiceExecutionResult:
        """Voice-triggered clip export. Side effect (firing the
        actual clip + skipping the save-location prompt) is handled
        by the engine layer, which inspects this intent and queues
        a utility request — the processor itself has no UI handle
        to call _export_recent_clip directly. We just return a
        success result with a clear control_text so the live status
        line reads naturally."""
        action = intent.action or "clip_1m"
        seconds_label = "1-minute" if action == "clip_1m" else "30-second"
        info = f"saving last {seconds_label} clip"
        return VoiceExecutionResult(
            success=True,
            target="touchless_clip",
            heard_text=intent.raw_text,
            control_text=info,
            info_text=info,
        )

    def _best_intent(self, intents: list[ParsedVoiceCommand]) -> ParsedVoiceCommand | None:
        if not intents:
            return None
        return max(intents, key=lambda item: item.confidence)

    def _can_skip_catalog_lookup(self, intent: ParsedVoiceCommand) -> bool:
        if intent.app_name == "touchless":
            return True
        if intent.app_name == "touchless_app":
            return True
        if intent.app_name == "file_explorer" and bool(intent.query):
            return True
        if intent.app_name == "settings":
            return True
        if intent.app_name == "outlook" and intent.action in {"open_folder", "compose"}:
            return True
        if intent.app_name in {"chrome", "spotify"} and intent.matched_alias is not None:
            return True
        if intent.app_name == "chrome" and intent.confidence >= 0.88:
            return True
        if intent.app_name == "spotify" and intent.confidence >= 0.88:
            return True
        if intent.app_name == "youtube" and intent.confidence >= 0.84:
            return True
        return intent.confidence >= 0.98 and intent.app_name != "system"

    def _display_text_for_intent(self, intent: ParsedVoiceCommand) -> str:
        query = str(intent.query or "").strip()
        if intent.app_name == "spotify":
            if intent.action == "play" and query:
                return f"play {query} on spotify"
            if intent.action == "open":
                return "open spotify"
            if intent.action == "next":
                return "next song on spotify"
            if intent.action == "previous":
                return "previous song on spotify"
            if intent.action == "shuffle":
                return "shuffle on spotify"
            if intent.action == "repeat":
                return "repeat on spotify"
            if intent.action in {"pause", "resume"}:
                return f"{intent.action} spotify"
        elif intent.app_name == "youtube":
            if intent.action == "play" and query:
                return f"play {query} on youtube"
            return "youtube"
        elif intent.app_name == "chrome":
            if intent.action in {"open", "search"} and query:
                return f"search {query} on chrome"
            if intent.action == "open":
                return "open chrome"
            if intent.action == "back":
                return "go back in chrome"
            if intent.action == "forward":
                return "go forward in chrome"
            if intent.action == "refresh":
                return "refresh chrome"
            if intent.action == "new_tab":
                return "new tab in chrome"
            if intent.action == "incognito":
                return "new incognito tab in chrome"
        elif intent.app_name == "system":
            resolved = str(intent.slots.get("resolved_app_name") or query or "").strip()
            if intent.action == "open_app" and resolved:
                return f"open {resolved}"
        elif intent.app_name == "file_explorer" and query:
            return f"open {query} in file explorer"
        elif intent.app_name == "settings" and query:
            return f"open {query} settings"
        elif intent.app_name == "outlook" and query:
            return f"open {query} in outlook"
        elif intent.app_name == "touchless":
            if intent.action == "clip_30s":
                return "save 30-second clip"
            return "save 1-minute clip"
        return intent.normalized_text or intent.raw_text

    def _execute_spotify(self, intent: ParsedVoiceCommand) -> VoiceExecutionResult:
        success = False
        info_text = "-"
        self._log_step(
            f"[spotify] action={intent.action} query={(intent.query or '')[:60]!r}"
        )
        if intent.action == "open":
            if self._is_spotify_resume_phrase(intent.raw_text):
                self._log_step("[spotify] resume phrase detected; focusing then playing")
                self._log_step("[spotify] focus_or_open_window...")
                focused = self.spotify_controller.focus_or_open_window()
                self._log_step(
                    f"[spotify] focus → {'ok' if focused else 'failed'} "
                    f"({self.spotify_controller.message})"
                )
                self._log_step("[spotify] sending play...")
                played = self.spotify_controller.play()
                self._log_step(
                    f"[spotify] play → {'ok' if played else 'failed'} "
                    f"({self.spotify_controller.message})"
                )
                success = focused and played
                details = self.spotify_controller.get_current_track_details() if success else None
                if details is not None:
                    info_text = details.summary()
                    self._log_step(f"[spotify] now playing: {info_text}")
            else:
                self._log_step("[spotify] focus_or_open_window...")
                success = self.spotify_controller.focus_or_open_window()
                self._log_step(
                    f"[spotify] open → {'ok' if success else 'failed'} "
                    f"({self.spotify_controller.message})"
                )
        elif intent.action == "play":
            preferred_types = tuple(intent.slots.get("preferred_types", ()))
            self._log_step(
                f"[spotify] play_search_request query={intent.query!r} "
                f"types={preferred_types or 'default'}"
            )
            success = self.spotify_controller.play_search_request(intent.query or "", preferred_types=preferred_types)
            self._log_step(
                f"[spotify] play_search_request → {'ok' if success else 'failed'} "
                f"({self.spotify_controller.message})"
            )
            details = self.spotify_controller.get_current_track_details() if success else None
            if details is not None:
                info_text = details.summary()
                self._log_step(f"[spotify] now playing: {info_text}")
            elif intent.query:
                info_text = f"Spotify request: {intent.query}"
        elif intent.action == "pause":
            self._log_step("[spotify] sending pause...")
            success = self.spotify_controller.pause()
            self._log_step(
                f"[spotify] pause → {'ok' if success else 'failed'} "
                f"({self.spotify_controller.message})"
            )
        elif intent.action == "resume":
            self._log_step("[spotify] resume: focus then play")
            focused = self.spotify_controller.focus_or_open_window()
            self._log_step(
                f"[spotify] focus → {'ok' if focused else 'failed'}"
            )
            played = self.spotify_controller.play()
            self._log_step(
                f"[spotify] play → {'ok' if played else 'failed'} "
                f"({self.spotify_controller.message})"
            )
            success = focused and played
            details = self.spotify_controller.get_current_track_details() if success else None
            if details is not None:
                info_text = details.summary()
                self._log_step(f"[spotify] now playing: {info_text}")
        elif intent.action == "next":
            self._log_step("[spotify] next_track...")
            success = self.spotify_controller.next_track()
            self._log_step(
                f"[spotify] next_track → {'ok' if success else 'failed'} "
                f"({self.spotify_controller.message})"
            )
        elif intent.action == "previous":
            self._log_step("[spotify] previous_track...")
            success = self.spotify_controller.previous_track()
            self._log_step(
                f"[spotify] previous_track → {'ok' if success else 'failed'} "
                f"({self.spotify_controller.message})"
            )
        elif intent.action == "shuffle":
            self._log_step("[spotify] toggle_shuffle...")
            success = self.spotify_controller.toggle_shuffle()
            self._log_step(
                f"[spotify] toggle_shuffle → {'ok' if success else 'failed'} "
                f"({self.spotify_controller.message})"
            )
        elif intent.action == "repeat":
            self._log_step("[spotify] toggle_repeat_track...")
            success = self.spotify_controller.toggle_repeat_track()
            self._log_step(
                f"[spotify] toggle_repeat_track → {'ok' if success else 'failed'} "
                f"({self.spotify_controller.message})"
            )
        if info_text == "-" and intent.query:
            info_text = f"Spotify request: {intent.query}"
        return VoiceExecutionResult(
            success=success,
            target="spotify",
            heard_text=intent.raw_text,
            control_text=self.spotify_controller.message,
            info_text=info_text,
        )

    # ---- Discord -----------------------------------------------------------

    def _execute_discord(self, intent: ParsedVoiceCommand) -> VoiceExecutionResult:
        """Drive the Discord RPC controller from a parsed intent.

        Action contract (`intent.action` → controller call):

          * `mute` / `unmute`       → controller.set_self_mute(bool)
          * `deafen` / `undeafen`   → controller.set_self_deafen(bool)
          * `leave_call`            → controller.leave_voice_channel()
          * `join_voice`            → find voice channel by name + select
          * `open_text`             → find text channel by name + select
          * `switch_server`         → find guild + focus its first text channel
          * `send`                  → focus channel (if specified) + paste
                                       the message via text_input_controller
                                       and press Enter

        Returns the standard VoiceExecutionResult shape with `target="discord"`.

        Currently short-circuits to a "Coming soon" response in every
        build — Discord is shelved behind a future feature update,
        matching the Settings → General → Discord card placeholder and
        the engine's idle Discord-mode router. The full parser + action
        contract is kept intact below for when we re-enable it.
        """
        return VoiceExecutionResult(
            success=False,
            target="discord",
            heard_text=intent.raw_text,
            control_text="Discord control coming soon",
            info_text="Discord support is coming in a future Touchless update.",
        )
        action = intent.action
        slots = intent.slots or {}
        target_name = str(slots.get("target") or "").strip()
        message_text = str(slots.get("message") or "").strip()
        success = False
        info_text = "-"

        # Every Discord action emits a step log so the user can see
        # exactly which RPC call ran and whether the response was
        # success / not-found / RPC error. The controller methods
        # already update their `.message` field with the underlying
        # reason; we mirror it here so the user sees it via the log
        # without having to read controller-internal state.
        self._log_step(f"[discord] action={action} target={target_name!r}")

        if action == "mute":
            self._log_step("[discord] setting self_mute=True...")
            success = self.discord_controller.set_self_mute(True)
            self._log_step(
                f"[discord] mute → {'ok' if success else 'failed'} "
                f"({self.discord_controller.message})"
            )
        elif action == "unmute":
            self._log_step("[discord] setting self_mute=False...")
            success = self.discord_controller.set_self_mute(False)
            self._log_step(
                f"[discord] unmute → {'ok' if success else 'failed'} "
                f"({self.discord_controller.message})"
            )
        elif action == "deafen":
            self._log_step("[discord] setting self_deafen=True...")
            success = self.discord_controller.set_self_deafen(True)
            self._log_step(
                f"[discord] deafen → {'ok' if success else 'failed'} "
                f"({self.discord_controller.message})"
            )
        elif action == "undeafen":
            self._log_step("[discord] setting self_deafen=False...")
            success = self.discord_controller.set_self_deafen(False)
            self._log_step(
                f"[discord] undeafen → {'ok' if success else 'failed'} "
                f"({self.discord_controller.message})"
            )
        elif action == "leave_call":
            self._log_step("[discord] leaving current voice channel...")
            success = self.discord_controller.leave_voice_channel()
            self._log_step(
                f"[discord] leave_call → {'ok' if success else 'failed'} "
                f"({self.discord_controller.message})"
            )
        elif action == "join_voice":
            self._log_step(
                f"[discord] searching voice channels for {target_name!r}..."
            )
            channel = self.discord_controller.find_voice_channel_by_name(target_name)
            if channel is None:
                self._log_step(
                    f"[discord] no voice channel matched {target_name!r} "
                    f"(searched across all joined guilds)"
                )
                info_text = f"voice channel '{target_name}' not found"
            else:
                self._log_step(
                    f"[discord] matched voice #{channel.get('name', '?')} "
                    f"in {channel.get('_guild_name', '?')} "
                    f"(id={channel.get('id', '?')})"
                )
                self._log_step("[discord] joining voice channel...")
                success = self.discord_controller.select_voice_channel(channel.get("id", ""))
                if success:
                    info_text = (
                        f"joined #{channel.get('name', '?')} "
                        f"in {channel.get('_guild_name', '?')}"
                    )
                    self._log_step(f"[discord] joined: {info_text}")
                else:
                    self._log_step(
                        f"[discord] join failed: {self.discord_controller.message}"
                    )
        elif action == "open_text":
            self._log_step(
                f"[discord] searching text channels for {target_name!r}..."
            )
            channel = self.discord_controller.find_text_channel_by_name(target_name)
            if channel is None:
                self._log_step(
                    f"[discord] no text channel matched {target_name!r}"
                )
                info_text = f"text channel '{target_name}' not found"
            else:
                self._log_step(
                    f"[discord] matched #{channel.get('name', '?')} "
                    f"in {channel.get('_guild_name', '?')} "
                    f"(id={channel.get('id', '?')})"
                )
                self._log_step("[discord] focusing text channel in UI...")
                success = self.discord_controller.select_text_channel(
                    channel.get("id", ""), channel_info=channel
                )
                if success:
                    info_text = (
                        f"opened #{channel.get('name', '?')} "
                        f"in {channel.get('_guild_name', '?')}"
                    )
                    self._log_step(f"[discord] opened: {info_text}")
                else:
                    self._log_step(
                        f"[discord] open failed: {self.discord_controller.message}"
                    )
        elif action == "switch_server":
            self._log_step(
                f"[discord] searching guilds for {target_name!r}..."
            )
            guild = self.discord_controller.find_guild_by_name(target_name)
            if guild is None:
                self._log_step(f"[discord] no guild matched {target_name!r}")
                info_text = f"server '{target_name}' not found"
            else:
                self._log_step(
                    f"[discord] matched guild {guild.get('name', '?')} "
                    f"(id={guild.get('id', '?')}); "
                    "selecting its first text channel..."
                )
                success = self.discord_controller.focus_guild_first_text_channel(
                    guild.get("id", "")
                )
                if success:
                    info_text = f"switched to {guild.get('name', '?')}"
                    self._log_step(f"[discord] {info_text}")
                else:
                    self._log_step(
                        f"[discord] switch_server failed: "
                        f"{self.discord_controller.message}"
                    )
        elif action == "send":
            success, info_text = self._execute_discord_send(target_name, message_text)
        return VoiceExecutionResult(
            success=success,
            target="discord",
            heard_text=intent.raw_text,
            control_text=self.discord_controller.message,
            info_text=info_text if info_text != "-" else self.discord_controller.message,
        )

    def _execute_discord_pending_send(
        self, spoken_text: str
    ) -> VoiceExecutionResult | None:
        """If `_discord_pending_send` is set, treat `spoken_text` as
        the destination channel name and complete the send.

        Cancel words ("cancel", "nevermind", "stop") clear the pending
        state without sending. Unrecognised channel names re-prompt
        (state stays set). 30 s expiry — if the user goes silent for
        too long, the pending message is dropped.
        """
        pending = self._discord_pending_send
        if pending is None:
            return None
        # 30 s timeout — drop stale pending if the user moved on.
        if time.time() - float(pending.get("started_at", 0)) > 30.0:
            self._discord_pending_send = None
            return None
        normalized = self._normalize_text(spoken_text)
        if not normalized:
            return None
        # Cancel words clear the pending state without firing.
        if normalized in {
            "cancel", "nevermind", "never mind", "stop", "forget it",
            "abort", "scratch that", "no", "nope",
        }:
            self._discord_pending_send = None
            return VoiceExecutionResult(
                success=False,
                target="discord",
                heard_text=spoken_text,
                control_text="send cancelled",
                info_text="-",
                display_text="send cancelled",
            )
        # Treat the entire utterance as the channel name and resolve.
        message_text = str(pending.get("message", "")).strip()
        target_name = normalized
        # Strip common filler prefixes the user might say when
        # responding to the prompt — "in general", "the general
        # channel", etc.
        for prefix in ("in the ", "in ", "the ", "to "):
            if target_name.startswith(prefix):
                target_name = target_name[len(prefix):].strip()
                break
        for suffix in (" channel", " channels"):
            if target_name.endswith(suffix):
                target_name = target_name[: -len(suffix)].strip()
                break
        # Clear pending state BEFORE the send call — if it fails, we
        # don't want to leave it dangling for the next utterance to
        # consume.
        self._discord_pending_send = None
        success, info_text = self._execute_discord_send(target_name, message_text)
        return VoiceExecutionResult(
            success=success,
            target="discord",
            heard_text=spoken_text,
            control_text=self.discord_controller.message,
            info_text=info_text if info_text != "-" else self.discord_controller.message,
            display_text=info_text,
        )

    def _execute_discord_send(
        self, target_name: str, message_text: str
    ) -> tuple[bool, str]:
        """Resolve the destination, focus the Discord window + the
        target channel, paste the message, and press Enter.

        Destination resolution (in order):
          1. Explicit `target_name` from the voice command (fuzzy
             match against every text channel).
          2. `default_send_target()` — the controller-side fallback
             chain: currently-joined voice channel's in-call chat,
             then the last text channel we focused via Touchless.
          3. Set `_discord_pending_send` and return a prompt asking
             the user which channel. The next utterance is consumed
             as the destination.

        Critical step the v1 build missed: focus the Discord WINDOW
        before pasting. `SELECT_TEXT_CHANNEL` navigates Discord's
        internal UI but doesn't raise the window — paste keystrokes
        would otherwise land in whichever app currently has focus
        (Touchless's own voice-listener overlay, browser, IDE, etc.).
        """
        self._log_step(
            f"[discord] send: message={message_text[:60]!r} target={target_name!r}"
        )
        if not message_text:
            self._log_step("[discord] send aborted: no message text")
            return False, "no message text — try 'send <message> in <channel>'"
        info_prefix = ""
        channel: dict[str, Any] | None = None

        # Stage 1: explicit destination from the voice command.
        if target_name:
            self._log_step(
                f"[discord] resolving destination by name: {target_name!r}"
            )
            channel = self.discord_controller.find_text_channel_by_name(target_name)
            if channel is None:
                self._log_step(
                    f"[discord] no text channel matched {target_name!r}; "
                    "send aborted"
                )
                return False, f"send target '{target_name}' not found"
            self._log_step(
                f"[discord] matched #{channel.get('name', '?')} "
                f"in {channel.get('_guild_name', '?')} "
                f"(id={channel.get('id', '?')})"
            )
        # Stage 2: fall back to the controller's smart default.
        else:
            self._log_step(
                "[discord] no destination given; checking fallback chain "
                "(voice-channel → last focused)..."
            )
            channel = self.discord_controller.default_send_target()
            if channel is None:
                # Stage 3: nothing to fall back to — stash the message
                # and prompt the user for a destination on the next
                # utterance.
                self._log_step(
                    "[discord] fallback chain empty; entering "
                    "awaiting-destination state"
                )
                self._discord_pending_send = {
                    "message": message_text,
                    "started_at": time.time(),
                }
                prompt = (
                    f"Which channel for: \"{message_text[:80]}\"?\n"
                    "Say a channel name (e.g. 'general'), or 'cancel' to abort."
                )
                return False, prompt
            self._log_step(
                f"[discord] fallback target: #{channel.get('name', '?')} "
                f"({channel.get('_via', 'unknown')})"
            )

        # We have a channel — focus the Discord window so the paste
        # lands in its compose box, then run SELECT_TEXT_CHANNEL.
        self._log_step("[discord] focusing Discord desktop window...")
        if not self.discord_controller.focus_discord_window():
            self._log_step(
                "[discord] could not focus Discord window — paste may go "
                "to the wrong app; continuing anyway"
            )
        time.sleep(0.20)
        self._log_step(
            f"[discord] SELECT_TEXT_CHANNEL id={channel.get('id', '?')}..."
        )
        ok = self.discord_controller.select_text_channel(
            channel.get("id", ""), channel_info=channel
        )
        if not ok:
            self._log_step(
                f"[discord] SELECT_TEXT_CHANNEL failed: "
                f"{self.discord_controller.message}"
            )
            return False, "couldn't focus destination channel"
        self._log_step(
            "[discord] channel focused; settling 150 ms before paste"
        )
        time.sleep(0.15)
        info_prefix = (
            f"#{channel.get('name', '?')} in {channel.get('_guild_name', '?')}: "
        )
        # Step 3: paste the body via clipboard, then synthesize a
        # separate VK_RETURN keypress.
        #
        # Two calls, not one with "\n" baked in. The reason: paste-
        # mode shoves the entire string onto the clipboard and hits
        # Ctrl+V, which delivers ALL characters verbatim — including
        # newlines, which Discord renders as in-message line breaks
        # instead of treating as "send". For an actual Send we need
        # the Enter key event, not a newline character. The typed
        # path (prefer_paste=False) calls _text_to_inputs which
        # remaps "\n" to VK_RETURN — exactly what we want for the
        # Enter step. Splitting body and Enter into two calls keeps
        # the body fast (one clipboard op) while making the send a
        # real keypress.
        self._log_step(
            f"[discord] pasting message body ({len(message_text)} chars) "
            "via clipboard..."
        )
        try:
            ok_paste = bool(
                self.text_input_controller.insert_text(message_text, prefer_paste=True)
            )
        except Exception as exc:
            self._log_step(f"[discord] paste raised: {exc}")
            return False, f"send failed (paste step): {exc}"
        if not ok_paste:
            self._log_step(
                f"[discord] paste returned False: "
                f"{self.text_input_controller.message}"
            )
            return False, "send failed (paste step)"
        self._log_step("[discord] paste ok; sending Enter keypress...")
        try:
            ok_enter = bool(
                self.text_input_controller.insert_text("\n", prefer_paste=False)
            )
        except Exception as exc:
            self._log_step(f"[discord] enter raised: {exc}")
            return False, f"send failed (enter step): {exc}"
        if not ok_enter:
            self._log_step(
                f"[discord] enter returned False: "
                f"{self.text_input_controller.message}"
            )
            return False, "send failed (enter step)"
        self._log_step(
            f"[discord] sent {message_text[:60]!r} to "
            f"#{channel.get('name', '?')}"
        )
        return True, f"{info_prefix}sent \"{message_text[:80]}\""

    def _parse_discord(
        self,
        text: str,
        *,
        raw_text: str,
        context: VoiceCommandContext | None,
    ) -> ParsedVoiceCommand | None:
        """Parse Discord voice commands.

        Action coverage:
          * mute / unmute / deafen / undeafen (require Discord context
            so "mute me" alone doesn't capture system-mute users)
          * leave_call / hang up / disconnect (no Discord context
            required — these phrases are Discord-specific enough)
          * join_voice — "join <voice channel>"
          * open_text — "open <channel>" / "switch to <channel>"
          * switch_server — "open <server>" / "switch to server <name>"
          * send — "send <message>" / "send <message> in <target>" /
                   "send <message> to <target>"

        Returns None when the text doesn't look like a Discord command.
        """
        matched_alias = self._matched_alias(text, "discord")
        discord_context = (
            matched_alias is not None
            or self._contains_any(text, DISCORD_CONTEXT_PHRASES)
            or (context is not None and context.preferred_app == "discord")
        )

        # leave / hang up — high-specificity, no context required
        if self._contains_any(text, DISCORD_LEAVE_PHRASES):
            return ParsedVoiceCommand(
                raw_text=raw_text,
                normalized_text=text,
                app_name="discord",
                action="leave_call",
                confidence=0.94,
                matched_alias=matched_alias,
                slots={},
            )

        # mute / unmute — require Discord context so they don't grab
        # generic "mute" intents.
        if discord_context:
            if self._contains_any(text, ("unmute", "un-mute")):
                return ParsedVoiceCommand(
                    raw_text=raw_text, normalized_text=text, app_name="discord",
                    action="unmute", confidence=0.94, matched_alias=matched_alias,
                )
            if "mute" in text:
                return ParsedVoiceCommand(
                    raw_text=raw_text, normalized_text=text, app_name="discord",
                    action="mute", confidence=0.94, matched_alias=matched_alias,
                )
            if self._contains_any(text, ("undeafen", "un-deafen", "un deafen")):
                return ParsedVoiceCommand(
                    raw_text=raw_text, normalized_text=text, app_name="discord",
                    action="undeafen", confidence=0.94, matched_alias=matched_alias,
                )
            if "deafen" in text:
                return ParsedVoiceCommand(
                    raw_text=raw_text, normalized_text=text, app_name="discord",
                    action="deafen", confidence=0.94, matched_alias=matched_alias,
                )

        # send — long-tail; match before "join"/"open" because "send a
        # message" must not be confused with anything else.
        for prefix in DISCORD_SEND_PHRASES:
            if text.startswith(prefix + " "):
                payload = text[len(prefix) + 1:].strip()
                if not payload:
                    return None
                # Split out the destination if present. Use the LAST
                # " in " / " to " — the message itself may contain
                # earlier instances of those words. Prefer "in" over
                # "to" when both appear; "in" is the canonical phrase.
                target = ""
                message = payload
                for sep in DISCORD_SEND_TARGET_SEPARATORS:
                    idx = payload.rfind(sep)
                    if idx > 0:
                        target = payload[idx + len(sep):].strip()
                        message = payload[:idx].strip()
                        break
                # Confidence: high when destination is given (parser
                # has more signal); moderate when implicit (just "send
                # X"). Drop below the 0.56 acceptance floor when there's
                # no Discord marker AND no destination — otherwise
                # "send a file" would always be a Discord intent.
                # Always claim "send <something>". The executor runs
                # a 3-stage fallback chain (voice channel chat → last-
                # focused text channel → pending prompt) so any send
                # shape produces useful behaviour. Confidence ladder:
                #   target + discord context  → 1.20 (unambiguous)
                #   target alone              → 1.05 (still beats
                #     most catalog matches)
                #   discord context, no target → 1.05 (use fallback)
                #   nothing                    → 0.60 (just above
                #     accept floor 0.56 — stronger parsers like
                #     "send file to printer" → catalog can win)
                if target and (matched_alias or discord_context):
                    confidence = 1.20
                elif target or matched_alias or discord_context:
                    confidence = 1.05
                else:
                    confidence = 0.60
                return ParsedVoiceCommand(
                    raw_text=raw_text,
                    normalized_text=text,
                    app_name="discord",
                    action="send",
                    confidence=confidence,
                    matched_alias=matched_alias,
                    slots={"target": target, "message": message},
                )

        # join — voice channel. Strip the " in <server>" qualifier
        # from the trailing part so we don't search for a channel
        # literally called "everyone in garbage". Heuristic: if the
        # tail contains " in " AND there's at least one word before
        # it, treat everything before " in " as the channel name and
        # discard the server hint (we fuzzy-match channels across all
        # guilds anyway, so server context is informational only).
        # Same for the singular "channel" filler (e.g. "join voice
        # channel garbage" → target "garbage", not "channel garbage").
        for prefix in DISCORD_JOIN_PHRASES:
            if text.startswith(prefix + " "):
                rest = text[len(prefix) + 1:].strip()
                if not rest:
                    return None
                target = rest
                for sep in (" in the ", " in "):
                    idx = target.rfind(sep)
                    if idx > 0:
                        target = target[:idx].strip()
                        break
                if target.startswith("channel "):
                    target = target[len("channel "):].strip()
                if not target:
                    return None
                return ParsedVoiceCommand(
                    raw_text=raw_text,
                    normalized_text=text,
                    app_name="discord",
                    action="join_voice",
                    confidence=0.88 if (matched_alias or discord_context) else 0.74,
                    matched_alias=matched_alias,
                    slots={"target": target, "message": ""},
                )

        # open server (check BEFORE channel since "open server X"
        # would also match "open X"). High confidence because "server"
        # is an explicit, unambiguous Discord keyword — boosted past
        # the catalog parser's typical 1.0+ confidence for app-name
        # matches like "AppAdminServer" that grab the word "server".
        for prefix in DISCORD_OPEN_SERVER_PHRASES:
            if text.startswith(prefix + " "):
                target = text[len(prefix) + 1:].strip()
                if not target:
                    return None
                return ParsedVoiceCommand(
                    raw_text=raw_text,
                    normalized_text=text,
                    app_name="discord",
                    action="switch_server",
                    confidence=1.05,
                    matched_alias=matched_alias,
                    slots={"target": target, "message": ""},
                )

        # open / switch text channel — STRONG patterns (explicit
        # "channel" keyword) fire unconditionally. WEAK patterns
        # ("switch to X", "go to X") require Discord context to avoid
        # grabbing "switch to dark mode" / "go to settings".
        for prefix in DISCORD_OPEN_TEXT_PHRASES_STRONG:
            if text.startswith(prefix + " "):
                target = text[len(prefix) + 1:].strip()
                if not target:
                    return None
                return ParsedVoiceCommand(
                    raw_text=raw_text,
                    normalized_text=text,
                    app_name="discord",
                    action="open_text",
                    confidence=0.98,
                    matched_alias=matched_alias,
                    slots={"target": target, "message": ""},
                )
        if discord_context:
            for prefix in DISCORD_OPEN_TEXT_PHRASES_WEAK:
                if text.startswith(prefix + " "):
                    target = text[len(prefix) + 1:].strip()
                    if not target:
                        return None
                    return ParsedVoiceCommand(
                        raw_text=raw_text,
                        normalized_text=text,
                        app_name="discord",
                        action="open_text",
                        confidence=0.86,
                        matched_alias=matched_alias,
                        slots={"target": target, "message": ""},
                    )

        return None

    def _execute_youtube(self, intent: ParsedVoiceCommand) -> VoiceExecutionResult:
        """Open YouTube search results for the query in Chrome AND
        auto-click the first video card on the resulting page.

        Two-stage execution:
          1. `chrome_controller.search_youtube(query)` returns
             immediately after dispatching the URL — synchronous
             from the voice pipeline's perspective so we can return
             a fast "OK heard" response to the user.
          2. `youtube_controller.play_first_search_result(query)`
             runs on a background thread because it polls for the
             freshly-opened tab and walks the page's accessibility
             tree (~1.5–4 s of waiting). Putting this on the voice
             pipeline thread would block the next utterance.

        Failure paths degrade gracefully: if step 2 can't find the
        tab or the first link, the user is still on the search-
        results page (filtered to videos) and one click away from
        playback. Same fallback as the pre-auto-play behaviour."""
        query = (intent.query or "").strip()
        success = False
        message = "youtube search query missing"
        self._log_step(f"[youtube] action=play query={query[:60]!r}")
        # Helper: telemetry fire-and-forget. Lazy import keeps the
        # voice processor importable in environments / tests where
        # the telemetry module isn't on the path.
        def _fire(event: str, props: dict) -> None:
            try:
                from ..telemetry import track as _track
                _track(event, props)
            except Exception:
                pass

        if not query:
            self._log_step("[youtube] aborting: no query")

        if query:
            self._log_step(
                f"[youtube] stage 1: opening search URL in Chrome..."
            )
            success = self.chrome_controller.search_youtube(query)
            message = self.chrome_controller.message
            self._log_step(
                f"[youtube] search open → {'ok' if success else 'failed'} "
                f"({message})"
            )
            if success:
                self._log_step(
                    f"[youtube] stage 2: dispatching auto-play worker for "
                    f"{query[:40]!r}..."
                )
            # Stage-1 telemetry: the search-page open is the user's
            # first observable signal that the voice command worked.
            # Fires whether the open succeeded or not so the dashboard
            # can chart conversion (intent -> search opened -> auto-
            # play started -> video actually playing).
            _fire(
                "youtube_search_opened",
                {
                    "success": bool(success),
                    "via": "voice",
                },
            )
            if success:
                # Stage 2 — UIAutomation auto-click on a worker.
                # Capture the controller reference so the closure
                # doesn't depend on `self` after the executor returns.
                yt = self.youtube_controller
                target_query = query

                def _autoplay_worker() -> None:
                    # Surface ANY exception from the worker — the
                    # previous silent-pass version made it impossible
                    # to tell whether the worker crashed (rare but
                    # possible) vs. just landed in a non-success
                    # path. The autoplay path's diagnostics live on
                    # stderr behind a [yt-autoplay] tag so they're
                    # easy to grep alongside the rest of the engine
                    # log.
                    import sys as _sys
                    import traceback as _tb
                    autoplay_ok = False
                    autoplay_error = ""
                    try:
                        autoplay_ok = bool(yt.play_first_search_result(target_query))
                    except Exception as exc:
                        autoplay_error = f"{type(exc).__name__}: {exc}"
                        try:
                            _sys.stderr.write(
                                f"[yt-autoplay] worker crashed: {autoplay_error}\n"
                            )
                            _tb.print_exc(file=_sys.stderr)
                            _sys.stderr.flush()
                        except Exception:
                            pass
                    # Stage-2 telemetry: the auto-play click is the
                    # 'YouTube is actually playing' signal. Fires for
                    # both success and failure so the dashboard can
                    # see the full funnel (search opened -> autoplay
                    # attempted -> autoplay succeeded vs. fell back
                    # to the search-results page).
                    _fire(
                        "youtube_autoplay",
                        {
                            "success": autoplay_ok,
                            "error": autoplay_error,
                            "via": "voice",
                        },
                    )

                threading.Thread(
                    target=_autoplay_worker,
                    name="youtube-autoplay",
                    daemon=True,
                ).start()
        return VoiceExecutionResult(
            success=success,
            target="youtube",
            heard_text=intent.raw_text,
            control_text=message,
            info_text=f"YouTube: {query}" if query else message,
        )

    def _execute_chrome(self, intent: ParsedVoiceCommand) -> VoiceExecutionResult:
        success = False
        self._log_step(
            f"[chrome] action={intent.action} query={(intent.query or '')[:60]!r}"
        )
        if intent.action == "open" and not intent.query:
            self._log_step("[chrome] focus_or_open_window...")
            success = self.chrome_controller.focus_or_open_window()
        elif intent.action in {"open", "search"}:
            self._log_step(f"[chrome] open_or_search {intent.query!r}...")
            success = self.chrome_controller.open_or_search(intent.query or "")
        elif intent.action == "back":
            self._log_step("[chrome] navigate_back...")
            success = self.chrome_controller.navigate_back()
        elif intent.action == "forward":
            self._log_step("[chrome] navigate_forward...")
            success = self.chrome_controller.navigate_forward()
        elif intent.action == "refresh":
            self._log_step("[chrome] refresh_page...")
            success = self.chrome_controller.refresh_page()
        elif intent.action == "new_tab":
            self._log_step("[chrome] new_tab...")
            success = self.chrome_controller.new_tab()
        elif intent.action == "incognito":
            self._log_step("[chrome] new_incognito_tab...")
            success = self.chrome_controller.new_incognito_tab()
        self._log_step(
            f"[chrome] {intent.action} → {'ok' if success else 'failed'} "
            f"({self.chrome_controller.message})"
        )
        return VoiceExecutionResult(
            success=success,
            target="chrome",
            heard_text=intent.raw_text,
            control_text=self.chrome_controller.message,
            info_text=f"Chrome target: {intent.query}" if intent.query else self.chrome_controller.message,
        )

    def _execute_settings(self, intent: ParsedVoiceCommand) -> VoiceExecutionResult:
        self._log_step(f"[settings] opening settings panel {intent.query!r}...")
        success = self.desktop_controller.open_settings(intent.query)
        self._log_step(
            f"[settings] open → {'ok' if success else 'failed'} "
            f"({self.desktop_controller.message})"
        )
        return VoiceExecutionResult(
            success=success,
            target="settings",
            heard_text=intent.raw_text,
            control_text=self.desktop_controller.message,
            info_text=self.desktop_controller.message,
        )

    def _execute_file_explorer(self, intent: ParsedVoiceCommand) -> VoiceExecutionResult:
        self._log_step(
            f"[file_explorer] action={intent.action} query={(intent.query or '')[:60]!r} "
            f"folder_hint={intent.slots.get('folder_hint')!r}"
        )
        if intent.action == "search":
            self._log_step(
                f"[file_explorer] searching File Explorer for {intent.query!r}..."
            )
            success = self.desktop_controller.search_file_explorer(intent.query or "")
            self._log_step(
                f"[file_explorer] search → {'ok' if success else 'failed'} "
                f"({self.desktop_controller.message})"
            )
            return VoiceExecutionResult(
                success=success,
                target="file_explorer",
                heard_text=intent.raw_text,
                control_text=self.desktop_controller.message,
                info_text=self.desktop_controller.message,
            )
        if intent.query:
            self._log_step(
                f"[file_explorer] resolving named file {intent.query!r} "
                f"(preferred_root={intent.slots.get('preferred_root')!r})..."
            )
            success = self.desktop_controller.open_named_file(
                intent.query,
                preferred_root=intent.slots.get("preferred_root"),
                folder_hint=intent.slots.get("folder_hint"),
            )
            message = self.desktop_controller.message
            self._log_step(
                f"[file_explorer] open_named_file → "
                f"{'ok' if success else 'no/multi match'} ({message})"
            )
            if success:
                return VoiceExecutionResult(
                    success=True,
                    target="file_explorer",
                    heard_text=intent.raw_text,
                    control_text=message,
                    info_text=message,
                )
            if str(message).lower().startswith("multiple matching files found"):
                self._log_step(
                    "[file_explorer] multiple matches — prompting user to pick"
                )
                resolved, ambiguous = self.desktop_controller.resolve_named_file(
                    intent.query,
                    preferred_root=intent.slots.get("preferred_root"),
                    folder_hint=intent.slots.get("folder_hint"),
                )
                prompt = self._create_pending_selection(
                    kind="file",
                    query=intent.query,
                    heard_text=intent.raw_text,
                    options=[{"path": str(path), "label": path.name} for path in ([resolved] if resolved is not None else []) + list(ambiguous)],
                )
                return VoiceExecutionResult(
                    success=False,
                    target="voice_selection",
                    heard_text=intent.raw_text,
                    control_text="awaiting file selection",
                    info_text=prompt,
                    display_text=prompt,
                )
            return VoiceExecutionResult(
                success=False,
                target="file_explorer",
                heard_text=intent.raw_text,
                control_text=message or f"could not find file: {intent.query}",
                info_text=message,
                display_text=intent.query,
            )
        else:
            self._log_step("[file_explorer] opening File Explorer window...")
            success = self.desktop_controller.open_file_explorer(intent.query)
            self._log_step(
                f"[file_explorer] open → {'ok' if success else 'failed'} "
                f"({self.desktop_controller.message})"
            )
        return VoiceExecutionResult(
            success=success,
            target="file_explorer",
            heard_text=intent.raw_text,
            control_text=self.desktop_controller.message,
            info_text=self.desktop_controller.message,
        )

    def _execute_outlook(self, intent: ParsedVoiceCommand) -> VoiceExecutionResult:
        self._log_step(f"[outlook] action={intent.action}")
        if intent.action == "compose":
            self._log_step(
                f"[outlook] composing email to={intent.slots.get('recipient')!r} "
                f"subject={intent.slots.get('subject')!r}"
            )
            success = self.desktop_controller.compose_email(
                recipient=intent.slots.get("recipient"),
                subject=intent.slots.get("subject"),
                body=intent.slots.get("body"),
            )
        elif intent.action == "open_folder":
            self._log_step(
                f"[outlook] opening folder {intent.query!r}..."
            )
            success = self.desktop_controller.open_outlook_folder(intent.query)
        else:
            self._log_step("[outlook] opening Outlook...")
            success = self.desktop_controller.open_outlook()
        self._log_step(
            f"[outlook] {intent.action} → {'ok' if success else 'failed'} "
            f"({self.desktop_controller.message})"
        )
        return VoiceExecutionResult(
            success=success,
            target="outlook",
            heard_text=intent.raw_text,
            control_text=self.desktop_controller.message,
            info_text=self.desktop_controller.message,
        )

    def _execute_generic(self, intent: ParsedVoiceCommand) -> VoiceExecutionResult:
        self._log_step(
            f"[system] action={intent.action} query={(intent.query or '')[:60]!r}"
        )
        if intent.action == "close_window":
            if intent.query:
                parts = [p.strip() for p in re.split(r"[,;]\s*|\s+and\s+", intent.query) if p.strip()]
                self._log_step(
                    f"[system] close_window: targeting {parts or [intent.query]}"
                )
                results: list[bool] = []
                messages: list[str] = []
                for part in (parts if parts else [intent.query]):
                    self._log_step(f"[system] closing window matching {part!r}...")
                    ok = self.desktop_controller.close_named_window(part)
                    self._log_step(
                        f"[system] close_named_window({part!r}) → "
                        f"{'ok' if ok else 'failed'} "
                        f"({self.desktop_controller.message})"
                    )
                    results.append(ok)
                    messages.append(self.desktop_controller.message)
                success = any(results)
                msg = "; ".join(m for m in messages if m)
            else:
                self._log_step("[system] closing active window")
                success = self.desktop_controller.close_active_window()
                msg = self.desktop_controller.message
                self._log_step(
                    f"[system] close_active_window → {'ok' if success else 'failed'} "
                    f"({msg})"
                )
            return VoiceExecutionResult(
                success=success,
                target="system",
                heard_text=intent.raw_text,
                control_text=msg,
                info_text=msg,
            )
        self._log_step(
            f"[system] searching app catalog for {intent.query!r}..."
        )
        best_entry, ambiguous_entries = self.desktop_controller.resolve_named_application_options(intent.query or "")
        if ambiguous_entries:
            ambig_names = [getattr(e, "name", str(e)) for e in ambiguous_entries[:5]]
            self._log_step(
                f"[system] multiple matches; prompting user. Candidates: "
                f"{ambig_names}"
            )
            prompt = self._create_pending_selection(
                kind="app",
                query=intent.query or "",
                heard_text=intent.raw_text,
                options=[
                    self._serialize_desktop_entry(entry)
                    for entry in ([best_entry] if best_entry is not None else []) + list(ambiguous_entries)
                ],
            )
            return VoiceExecutionResult(
                success=False,
                target="voice_selection",
                heard_text=intent.raw_text,
                control_text="awaiting app selection",
                info_text=prompt,
                display_text=prompt,
            )
        if best_entry is not None:
            entry_name = getattr(best_entry, "name", "?")
            self._log_step(f"[system] matched app {entry_name!r}; launching...")
            success = self.desktop_controller.open_desktop_entry(best_entry)
            self._log_step(
                f"[system] open_desktop_entry → {'ok' if success else 'failed'} "
                f"({self.desktop_controller.message})"
            )
        else:
            self._log_step(
                f"[system] no catalog match; trying open_named_application "
                f"({intent.query!r})..."
            )
            success = self.desktop_controller.open_named_application(intent.query or "")
            self._log_step(
                f"[system] open_named_application → "
                f"{'ok' if success else 'failed'} "
                f"({self.desktop_controller.message})"
            )
        return VoiceExecutionResult(
            success=success,
            target="system",
            heard_text=intent.raw_text,
            control_text=self.desktop_controller.message,
            info_text=self.desktop_controller.message,
        )

    def _parse_touchless_app(
        self,
        text: str,
        *,
        raw_text: str,
        context: VoiceCommandContext | None,
    ) -> ParsedVoiceCommand | None:
        """Recognise 'open touchless' / 'show touchless camera' /
        'switch to touchless settings' style phrases. Routes to
        app_name='touchless_app' so the executor can focus the
        Touchless main window and (optionally) navigate to a
        specific settings tab.

        Distinct from _parse_clip's 'touchless' intent — that's
        for clip-export commands like 'clip that'. The 'app' suffix
        on this app_name keeps the two routes separate inside
        _execute_intent."""
        trimmed = (text or "").strip()
        if not trimmed:
            return None
        # Need an explicit "touchless" mention so we don't grab
        # generic "open camera" / "open settings" commands.
        if "touchless" not in trimmed:
            return None
        if not self._contains_any(trimmed, APP_LAUNCH_PHRASES):
            return None
        # File-type disambiguation. "open touchless clip" /
        # "open touchless drawing" / "open touchless screenshot" /
        # "open touchless recording" are NOT app-launch commands —
        # the user wants the corresponding save folder opened in
        # Explorer (or in a future extension, the most recent
        # matching file). We route the intent through this same
        # parser with the file-type stashed in slots so
        # _execute_touchless_app can branch on it. The plain
        # "open touchless" / "open touchless settings" path is
        # unaffected.
        TOUCHLESS_FILE_KEYWORDS_TO_OUTPUT = {
            "clip": "clips",
            "clips": "clips",
            "drawing": "drawings",
            "drawings": "drawings",
            "screenshot": "screenshots",
            "screenshots": "screenshots",
            "recording": "screen_recordings",
            "recordings": "screen_recordings",
        }
        # Word-boundary check so "clipboard" / "drawing-tool" / etc.
        # don't accidentally match; we split on whitespace and check
        # each token verbatim against the keyword map.
        tokens = trimmed.split()
        touchless_file_kind: str | None = None
        for tok in tokens:
            if tok in TOUCHLESS_FILE_KEYWORDS_TO_OUTPUT:
                touchless_file_kind = TOUCHLESS_FILE_KEYWORDS_TO_OUTPUT[tok]
                break
        # Find the most-specific tab keyword that matches.
        matched_tab: str | None = None
        for phrase, key in TOUCHLESS_TAB_KEYWORDS:
            if phrase in trimmed:
                matched_tab = key
                break
        # File-kind beats tab-keyword (file-kind tokens like "clip"
        # don't overlap with TOUCHLESS_TAB_KEYWORDS entries, but
        # being explicit makes the precedence obvious).
        if touchless_file_kind is not None:
            return ParsedVoiceCommand(
                raw_text=raw_text,
                normalized_text=text,
                app_name="touchless_app",
                action="open_save_folder",
                confidence=1.10,
                query=touchless_file_kind,
                matched_alias="touchless",
                slots={"output_kind": touchless_file_kind},
            )
        confidence = 1.04 if matched_tab else 0.96
        return ParsedVoiceCommand(
            raw_text=raw_text,
            normalized_text=text,
            app_name="touchless_app",
            action="navigate" if matched_tab else "open",
            confidence=confidence,
            query=matched_tab,
            matched_alias="touchless",
        )

    def _parse_clip(
        self,
        text: str,
        *,
        raw_text: str,
        context: VoiceCommandContext | None,
    ) -> ParsedVoiceCommand | None:
        """Recognise 'clip that' / 'clip the last minute' / 'save
        clip' style phrases. Routes to app_name='touchless',
        action='clip_1m' (or 'clip_30s' when a 30-second hint is
        present). The engine catches this intent and fires the
        clip-export with auto-save enabled."""
        trimmed = (text or "").strip()
        if not trimmed:
            return None
        # Bare "clip" alone is ambiguous (could be misheard "click",
        # "clipboard", a Spotify queue follow-up, etc.). Require
        # either an explicit phrase or 'clip' adjacent to a
        # demonstrative / time qualifier.
        matched = False
        for phrase in CLIP_TRIGGER_PHRASES:
            if phrase in trimmed:
                matched = True
                break
        if not matched:
            # Loose match: "clip" with a recency qualifier.
            if re.search(r"\bclip\b.*\b(that|this|it|now|here|just|right now|last|past|previous|recent)\b", trimmed):
                matched = True
            elif re.search(r"\b(clip|record)\b.*\b(last|past)\b.*\b(minute|sixty|60|30|thirty)\b", trimmed):
                matched = True
        if not matched:
            return None
        # Pick duration variant.
        action = "clip_1m"
        if any(hint in trimmed for hint in CLIP_30S_HINT_PHRASES):
            action = "clip_30s"
        elif re.search(r"\b30\b|\bthirty\b", trimmed) and "minute" not in trimmed:
            action = "clip_30s"
        return ParsedVoiceCommand(
            raw_text=raw_text,
            normalized_text=text,
            app_name="touchless",
            action=action,
            confidence=0.93,
        )

    def _parse_close_window(
        self,
        text: str,
        *,
        raw_text: str,
        context: VoiceCommandContext | None,
    ) -> ParsedVoiceCommand | None:
        trimmed = self._strip_common_prefix(text)
        if not self._contains_launch_request(trimmed, CLOSE_WINDOW_PHRASES):
            return None
        query = self._cleanup_query(
            trimmed,
            app_name=None,
            extra_phrases=CLOSE_WINDOW_PHRASES + APP_OBJECT_HINTS + ("window", "app", "application", "program"),
            trim_edge_noise=True,
        )
        query = self._cleanup_app_launch_query(query)
        if query in {"close", "window", "app", "application", "program"}:
            query = None
        confidence = 0.92 if query else 0.88
        return ParsedVoiceCommand(
            raw_text=raw_text,
            normalized_text=text,
            app_name="system",
            action="close_window",
            confidence=confidence,
            query=query,
        )

    def _parse_spotify(
        self,
        text: str,
        *,
        raw_text: str,
        context: VoiceCommandContext | None,
    ) -> ParsedVoiceCommand | None:
        matched_alias = self._matched_alias(text, "spotify")
        spotify_context = (
            matched_alias is not None
            or self._contains_any(text, SPOTIFY_CONTEXT_PHRASES)
            or (context is not None and context.preferred_app == "spotify")
        )
        music_hint = self._contains_any(text, MUSIC_NOUNS)
        has_play_phrase = self._contains_any(text, SPOTIFY_PLAY_PHRASES)

        action = None
        if spotify_context and self._contains_any(text, ("pause", "stop music")):
            action = "pause"
        elif spotify_context and self._contains_any(text, ("resume", "continue", "unpause")):
            action = "resume"
        elif spotify_context and self._contains_any(text, ("next song", "next track", "skip song", "skip track", "skip")):
            action = "next"
        elif spotify_context and self._contains_any(text, ("previous song", "previous track", "last song", "last track", "go back")):
            action = "previous"
        elif spotify_context and "shuffle" in text:
            action = "shuffle"
        elif spotify_context and "repeat" in text:
            action = "repeat"
        elif has_play_phrase and (spotify_context or music_hint):
            action = "play"
        elif matched_alias and self._contains_launch_request(text, DEDICATED_APP_LAUNCH_PHRASES):
            action = "open"

        if action is None:
            return None

        confidence = 0.76
        if matched_alias:
            confidence = 0.94
        elif spotify_context:
            confidence = 0.84
        elif music_hint:
            confidence = 0.68

        preferred_types = self._spotify_preferred_types(text)
        query = None
        if action == "play":
            query = self._cleanup_query(
                text,
                app_name="spotify",
                extra_phrases=SPOTIFY_PLAY_PHRASES + SPOTIFY_CONTEXT_PHRASES,
                trim_edge_noise=False,
            )
            query = self._strip_music_tail(query)
            if not query:
                if matched_alias:
                    action = "resume"
                    query = None
                else:
                    return None

        return ParsedVoiceCommand(
            raw_text=raw_text,
            normalized_text=text,
            app_name="spotify",
            action=action,
            confidence=confidence,
            query=query,
            matched_alias=matched_alias,
            slots={"preferred_types": preferred_types},
        )

    def _parse_youtube(
        self,
        text: str,
        *,
        raw_text: str,
        context: VoiceCommandContext | None,
    ) -> ParsedVoiceCommand | None:
        """Mirror of `_parse_spotify` for the YouTube/Chrome surface.

        Recognises "play <query> on youtube" and the related play-
        verb variants. The execute step opens YouTube search results
        for the query in Chrome (filtered to videos only) so the
        first card on the page is the playable result the user
        wants. Without this parser, "play X on youtube" used to fall
        through every other parser and land on "command not
        understood".
        """
        youtube_context = (
            self._contains_any(text, YOUTUBE_CONTEXT_PHRASES)
            or (context is not None and context.preferred_app == "youtube")
        )
        if not youtube_context:
            return None

        has_play_phrase = self._contains_any(text, SPOTIFY_PLAY_PHRASES)
        if not has_play_phrase:
            return None

        query = self._cleanup_query(
            text,
            app_name="youtube",
            extra_phrases=SPOTIFY_PLAY_PHRASES + YOUTUBE_CONTEXT_PHRASES + ("on youtube", "you tube"),
            trim_edge_noise=True,
        )
        # Strip any lingering "video"/"music"/"song" tail that the
        # generic music-tail trimmer would catch — keeps the search
        # query tight ("broken boulevard by green day video" →
        # "broken boulevard by green day").
        query = self._strip_music_tail(query)
        if not query:
            return None

        return ParsedVoiceCommand(
            raw_text=raw_text,
            normalized_text=text,
            app_name="youtube",
            action="play",
            confidence=0.86,
            query=query,
            matched_alias=None,
        )

    def _parse_chrome(
        self,
        text: str,
        *,
        raw_text: str,
        context: VoiceCommandContext | None,
    ) -> ParsedVoiceCommand | None:
        matched_alias = self._matched_alias(text, "chrome")
        web_target_match = self._matched_known_web_target(text)
        chrome_context = (
            matched_alias is not None
            or self._contains_any(text, CHROME_CONTEXT_PHRASES)
            or (context is not None and context.preferred_app == "chrome")
            or web_target_match is not None
        )

        action: str | None = None
        if chrome_context and self._contains_any(text, ("go back", "back page", "previous page")):
            action = "back"
        elif chrome_context and self._contains_any(text, ("go forward", "forward page", "next page")):
            action = "forward"
        elif chrome_context and self._contains_any(text, ("refresh", "reload")):
            action = "refresh"
        elif chrome_context and self._contains_any(text, ("incognito", "private tab")):
            action = "incognito"
        elif chrome_context and self._contains_any(text, ("new tab", "open tab")):
            action = "new_tab"
        elif chrome_context and self._contains_any(text, GENERIC_OPEN_PHRASES):
            action = "open"
        elif self._contains_any(text, CHROME_SEARCH_PHRASES) or (context is not None and context.preferred_app == "chrome"):
            action = "search"

        if action is None:
            return None

        confidence = 0.74
        if matched_alias:
            confidence = 0.92
        elif web_target_match is not None and action in {"open", "search"}:
            confidence = 0.90
        elif chrome_context:
            confidence = 0.84

        query = None
        if action in {"open", "search"}:
            browser_context_phrases = CHROME_SEARCH_PHRASES + CHROME_CONTEXT_PHRASES + (
                "on google",
                "in google",
                "using google",
                "with google",
                "google chrome",
                "chrome browser",
                "browser",
            )
            extra_phrases = browser_context_phrases
            if action == "open":
                extra_phrases = extra_phrases + GENERIC_OPEN_PHRASES
            query = self._cleanup_query(
                text,
                app_name="chrome",
                extra_phrases=extra_phrases,
                trim_edge_noise=True,
            )
            query = self._normalize_browser_query(query)
            if action == "open" and not query:
                query = None
            elif not query and matched_alias:
                action = "open"
            elif not query:
                return None
            if query:
                query = self._normalize_browser_query(self.chrome_controller.normalize_spoken_target(query))
            if matched_alias and action == "open" and not query:
                confidence = max(confidence, 1.08)

        return ParsedVoiceCommand(
            raw_text=raw_text,
            normalized_text=text,
            app_name="chrome",
            action=action,
            confidence=confidence,
            query=query,
            matched_alias=matched_alias,
        )

    def _parse_settings(
        self,
        text: str,
        *,
        raw_text: str,
        context: VoiceCommandContext | None,
    ) -> ParsedVoiceCommand | None:
        matched_alias = self._matched_alias(text, "settings")
        topic = self._matched_settings_topic(text)
        if matched_alias is None and topic is None:
            return None
        if not self._contains_any(text, APP_LAUNCH_PHRASES + ("settings",)):
            return None
        explicit_settings = "settings" in text or matched_alias is not None
        confidence = 1.02 if explicit_settings else 0.90
        if topic is not None and explicit_settings:
            confidence = max(confidence, 1.06)
        elif topic is not None:
            confidence = max(confidence, 0.92)
        if context is not None and context.preferred_app == "settings":
            confidence += 0.04
        return ParsedVoiceCommand(
            raw_text=raw_text,
            normalized_text=text,
            app_name="settings",
            action="open",
            confidence=confidence,
            query=topic,
            matched_alias=matched_alias,
        )

    def _parse_file_explorer(
        self,
        text: str,
        *,
        raw_text: str,
        context: VoiceCommandContext | None,
    ) -> ParsedVoiceCommand | None:
        matched_alias = self._matched_alias(text, "file_explorer")
        if self._contains_launch_request(text, CLOSE_WINDOW_PHRASES):
            return None
        folder = self._matched_folder(text)
        has_folder_language = self._contains_any(text, ("folder", "directory", "files"))
        has_launch_request = self._contains_launch_request(text, APP_LAUNCH_PHRASES)
        bare_file_request = False
        if matched_alias is None and folder is None and not has_folder_language:
            if has_launch_request:
                rough_query = self._cleanup_query(
                    text,
                    app_name=None,
                    extra_phrases=APP_LAUNCH_PHRASES,
                    trim_edge_noise=True,
                )
                rough_query = self._normalize_file_request_query(rough_query)
                bare_file_request = self._looks_like_file_request(text, rough_query, context=context)
            if not bare_file_request:
                return None
        if matched_alias is None and folder is not None and not has_folder_language and not has_launch_request and not bare_file_request:
            return None

        search_phrases = ("search files", "find file", "find files", "search for")
        alias_search_hint = matched_alias is not None and self._contains_any(text, ("search", "find", "look for", "look up"))
        if self._contains_any(text, search_phrases) or alias_search_hint:
            query = self._cleanup_query(
                text,
                app_name="file_explorer",
                extra_phrases=search_phrases + ("find", "look for", "look up", "in file explorer", "in explorer"),
                trim_edge_noise=True,
            )
            query = self._strip_file_search_tail(query)
            if not query:
                return None
            action = "search"
        else:
            action = "open"
            cleaned_query = self._cleanup_query(
                text,
                app_name="file_explorer",
                extra_phrases=APP_LAUNCH_PHRASES + ("folder", "directory", "files"),
                trim_edge_noise=True,
            )
            cleaned_query = self._normalize_file_request_query(cleaned_query)
            query = cleaned_query or folder
            if not matched_alias and not folder and not query:
                return None
            if matched_alias and not query:
                query = None

        confidence = 0.90 if matched_alias else 0.78
        if matched_alias is not None:
            confidence = 1.02
            if action == "open" and not query:
                confidence = 1.08
        if folder is not None:
            confidence += 0.08
        if bare_file_request:
            confidence += 0.12
        if context is not None and context.preferred_app == "file_explorer":
            confidence += 0.04
        slots: dict[str, Any] = {}
        if folder is not None and query is not None:
            normalized_query = self.desktop_controller._normalize_application_name(query)
            if normalized_query != folder:
                slots["preferred_root"] = folder
        return ParsedVoiceCommand(
            raw_text=raw_text,
            normalized_text=text,
            app_name="file_explorer",
            action=action,
            confidence=confidence,
            query=query,
            matched_alias=matched_alias,
            slots=slots,
        )

    def _parse_outlook(
        self,
        text: str,
        *,
        raw_text: str,
        context: VoiceCommandContext | None,
    ) -> ParsedVoiceCommand | None:
        matched_alias = self._matched_alias(text, "outlook")
        if "settings" in text and matched_alias in {"email", "mail"}:
            return None
        folder_name = self._matched_outlook_folder(text)
        mentions_outlook = matched_alias is not None or re.search(r"\boutlook\b", text) is not None
        email_hint = self._contains_any(text, ("email", "mail", "inbox")) or folder_name is not None
        if matched_alias is None and not email_hint:
            return None

        if folder_name is not None and self._contains_any(text, GENERIC_OPEN_PHRASES):
            confidence = 0.92 if mentions_outlook else 0.78
            if context is not None and context.preferred_app == "outlook":
                confidence += 0.04
            return ParsedVoiceCommand(
                raw_text=raw_text,
                normalized_text=text,
                app_name="outlook",
                action="open_folder",
                confidence=confidence,
                query=folder_name,
                matched_alias=matched_alias,
            )

        if self._contains_any(text, ("compose", "draft", "write", "send email", "email")):
            action = "compose"
            recipient = self._strip_outlook_tail(self._extract_slot(text, start_phrase="to ", stop_phrases=(" subject ", " body ", " message ")))
            subject = self._strip_outlook_tail(self._extract_slot(text, start_phrase="subject ", stop_phrases=(" body ", " message ")))
            body = self._strip_outlook_tail(self._extract_slot(text, start_phrase="body ", stop_phrases=()))
            confidence = 0.88 if matched_alias else 0.76
            if context is not None and context.preferred_app == "outlook":
                confidence += 0.04
            return ParsedVoiceCommand(
                raw_text=raw_text,
                normalized_text=text,
                app_name="outlook",
                action=action,
                confidence=confidence,
                query=recipient,
                matched_alias=matched_alias,
                slots={"recipient": recipient, "subject": subject, "body": body},
            )

        if self._contains_launch_request(text, DEDICATED_APP_LAUNCH_PHRASES + ("open outlook", "open inbox", "open calendar")):
            confidence = 0.90 if matched_alias else 0.76
            if matched_alias is not None:
                confidence = max(confidence, 1.02)
            if context is not None and context.preferred_app == "outlook":
                confidence += 0.04
            return ParsedVoiceCommand(
                raw_text=raw_text,
                normalized_text=text,
                app_name="outlook",
                action="open",
                confidence=confidence,
                matched_alias=matched_alias,
            )
        return None

    def _parse_catalog_open(
        self,
        text: str,
        *,
        raw_text: str,
        catalog_matches: list[tuple[DesktopAppEntry, float, str | None]],
    ) -> ParsedVoiceCommand | None:
        if not self._contains_any(text, APP_LAUNCH_PHRASES):
            return None
        if not catalog_matches:
            return None
        entry, match_score, matched_alias = catalog_matches[0]
        if match_score < 0.82:
            return None
        entry_name = self.desktop_controller._normalize_application_name(entry.display_name)
        alias_name = self.desktop_controller._normalize_application_name(matched_alias or "")
        if self._should_skip_catalog_open(text, entry_name=entry_name, matched_alias=alias_name):
            return None
        query = self._cleanup_query(
            text,
            app_name=None,
            extra_phrases=APP_LAUNCH_PHRASES + APP_OBJECT_HINTS,
            trim_edge_noise=True,
        )
        if self._looks_like_file_request(text, self._normalize_file_request_query(query), context=None):
            return None
        query = self._cleanup_app_launch_query(query)
        normalized_query = self.desktop_controller._normalize_application_name(query or "")
        app_alias = self.desktop_controller._normalize_application_name(matched_alias or entry.normalized_name)
        if query and not self.desktop_controller.can_resolve_application(query):
            if app_alias and app_alias not in normalized_query:
                return None
            resolved_query = entry.display_name
        else:
            resolved_query = query or entry.display_name
        return ParsedVoiceCommand(
            raw_text=raw_text,
            normalized_text=text,
            app_name="system",
            action="open_app",
            confidence=min(0.96, 0.68 + match_score * 0.22),
            query=resolved_query,
            matched_alias=matched_alias,
            slots={
                "resolved_app_name": entry.display_name,
                "app_category": entry.category,
                "catalog_match_score": match_score,
            },
        )

    def _parse_generic_open(self, text: str, *, raw_text: str) -> ParsedVoiceCommand | None:
        if not self._contains_any(text, APP_LAUNCH_PHRASES):
            return None
        query = self._cleanup_query(
            text,
            app_name=None,
            extra_phrases=APP_LAUNCH_PHRASES + APP_OBJECT_HINTS,
            trim_edge_noise=True,
        )
        if self._looks_like_file_request(text, self._normalize_file_request_query(query), context=None):
            return None
        query = self._cleanup_app_launch_query(query)
        if not query:
            return None
        normalized_query = self.desktop_controller._normalize_application_name(query)
        if self._query_mentions_hgr_asset(query, normalized_query):
            return ParsedVoiceCommand(
                raw_text=raw_text,
                normalized_text=text,
                app_name="file_explorer",
                action="open",
                confidence=0.62,
                query=query,
            )
        matched_alias: str | None = None
        slots: dict[str, Any] = {}
        confidence = 0.78
        if not self.desktop_controller.can_resolve_application(query):
            recovered = self._best_fuzzy_launch_match(query, text=text)
            if recovered is not None:
                entry, recovered_score, matched_alias = recovered
                if self._entry_is_running_hgr_app(entry):
                    return ParsedVoiceCommand(
                        raw_text=raw_text,
                        normalized_text=text,
                        app_name="file_explorer",
                        action="open",
                        confidence=0.62,
                        query=query,
                    )
                query = entry.display_name
                confidence = max(confidence, min(1.04, 0.86 + recovered_score * 0.28))
                slots = {
                    "resolved_app_name": entry.display_name,
                    "app_category": entry.category,
                    "catalog_match_score": recovered_score,
                }
            elif normalized_query in {"calendar", "system calendar"}:
                query = "calendar"
            else:
                if query and len(query.split()) >= 2:
                    return ParsedVoiceCommand(
                        raw_text=raw_text,
                        normalized_text=text,
                        app_name="file_explorer",
                        action="open",
                        confidence=0.60,
                        query=query,
                    )
                return None
        else:
            resolved = self.desktop_controller._resolve_application(query)
            if resolved is not None and self._entry_is_running_hgr_app(resolved):
                return ParsedVoiceCommand(
                    raw_text=raw_text,
                    normalized_text=text,
                    app_name="file_explorer",
                    action="open",
                    confidence=0.62,
                    query=query,
                )
        return ParsedVoiceCommand(
            raw_text=raw_text,
            normalized_text=text,
            app_name="system",
            action="open_app",
            confidence=confidence,
            query=query,
            matched_alias=matched_alias,
            slots=slots,
        )

    @staticmethod
    def _query_mentions_hgr_asset(query: str, normalized_query: str) -> bool:
        text = f" {(normalized_query or query or '').strip().lower()} "
        if not text.strip():
            return False
        tokens = text.split()
        if "hgr" in tokens and any(tok in {"clip", "clips", "recording", "recordings", "screenshot", "screenshots", "drawing", "drawings", "capture", "captures"} for tok in tokens):
            return True
        return False

    @staticmethod
    def _entry_is_running_hgr_app(entry: Any) -> bool:
        display = str(getattr(entry, "display_name", "") or "").strip().lower()
        target = str(getattr(entry, "target", "") or "").strip().lower()
        if target.endswith("hgr app.exe") or target.endswith("hgr_app.exe"):
            return True
        if display == "hgr app" or display == "hgr":
            return True
        return False

    def _best_fuzzy_launch_match(
        self,
        query: str | None,
        *,
        text: str,
    ) -> tuple[DesktopAppEntry, float, str | None] | None:
        normalized_query = self.desktop_controller._normalize_application_query(query or "")
        if not normalized_query:
            return None
        matches = self.desktop_controller.rank_applications_in_text(normalized_query)
        if not matches:
            return None
        entry, score, matched_alias = matches[0]
        token_count = len(normalized_query.split())
        has_object_hint = self._contains_any(text, APP_OBJECT_HINTS)
        if not has_object_hint and self._contains_any(
            text,
            CHROME_CONTEXT_PHRASES + SPOTIFY_CONTEXT_PHRASES + ("outlook", "file explorer", "settings"),
        ):
            return None
        threshold = 0.58
        if has_object_hint and token_count <= 1:
            threshold = 0.38
        elif has_object_hint and token_count <= 2:
            threshold = 0.46
        elif token_count <= 1:
            threshold = 0.50
        if entry.source == "known":
            threshold = max(0.34, threshold - 0.04)
        if score < threshold:
            return None
        return entry, score, matched_alias

    def _apply_catalog_bias(
        self,
        intent: ParsedVoiceCommand,
        catalog_matches: list[tuple[DesktopAppEntry, float, str | None]],
        *,
        context: VoiceCommandContext | None,
    ) -> None:
        best_score = 0.0
        catalog_keys = self._intent_catalog_keys(intent)
        for entry, score, matched_alias in catalog_matches:
            entry_keys = {entry.normalized_name, *(alias for alias in entry.aliases if alias)}
            if entry_keys & catalog_keys:
                best_score = max(best_score, score)
            elif matched_alias is not None and matched_alias in catalog_keys:
                best_score = max(best_score, score)
        if best_score > 0.0:
            intent.confidence += min(0.15, best_score * 0.12)
        elif intent.app_name == "system" and catalog_matches and intent.query:
            top_entry, top_score, _matched_alias = catalog_matches[0]
            if self.desktop_controller._normalize_application_name(intent.query) == top_entry.normalized_name:
                intent.confidence += min(0.12, top_score * 0.10)
        if context is not None and context.preferred_app and context.preferred_app == intent.app_name:
            intent.confidence += 0.04

    def _intent_catalog_keys(self, intent: ParsedVoiceCommand) -> set[str]:
        keys = {intent.app_name}
        if intent.app_name == "chrome":
            keys.update({"chrome", "google chrome"})
        elif intent.app_name == "spotify":
            keys.add("spotify")
        elif intent.app_name == "outlook":
            keys.add("outlook")
        elif intent.app_name == "system":
            resolved_name = str(intent.slots.get("resolved_app_name", "") or "").strip()
            if resolved_name:
                keys.add(self.desktop_controller._normalize_application_name(resolved_name))
            if intent.query:
                keys.add(self.desktop_controller._normalize_application_name(intent.query))
        return {key for key in keys if key}

    def _intent_template_bonus(self, text: str, intent: ParsedVoiceCommand) -> float:
        best_score = 0.0
        for phrase in self._intent_reference_phrases(intent):
            score = SequenceMatcher(None, text, phrase).ratio()
            score += self._token_overlap(text, phrase) * 0.22
            if score > best_score:
                best_score = score
        return max(0.0, min(0.12, (best_score - 0.55) * 0.30))

    def _intent_reference_phrases(self, intent: ParsedVoiceCommand) -> tuple[str, ...]:
        query = (intent.query or "").strip()
        if intent.app_name == "spotify":
            if intent.action == "play" and query:
                return (f"play {query} on spotify", f"spotify play {query}")
            if intent.action == "open":
                return ("open spotify", "launch spotify")
            return (f"spotify {intent.action}",)
        if intent.app_name == "chrome":
            if intent.action in {"open", "search"} and query:
                return (f"open {query} on chrome", f"search {query} in chrome")
            return (f"chrome {intent.action}",)
        if intent.app_name == "settings":
            topic = query or "settings"
            return (f"open {topic} settings", f"show {topic} settings")
        if intent.app_name == "file_explorer":
            if intent.action == "search" and query:
                return (f"search files for {query}", f"find files named {query}")
            if query:
                return (f"open {query} folder", f"open {query} in file explorer")
            return ("open file explorer",)
        if intent.app_name == "outlook":
            if intent.action == "open_folder" and query:
                return (f"open {query.lower()} from outlook", f"show {query.lower()} in outlook")
            if intent.action == "compose":
                return ("compose email in outlook", "write email in outlook")
            return ("open outlook",)
        if intent.app_name == "system" and query:
            return (f"open {query}", f"launch {query}", f"switch to {query}", f"run {query}")
        return ()

    def _token_overlap(self, left: str, right: str) -> float:
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens or not right_tokens:
            return 0.0
        shared = len(left_tokens & right_tokens)
        return shared / max(len(left_tokens), len(right_tokens))

    def _normalize_text(self, text: str) -> str:
        normalized = str(text or "").replace("\n", " ").lower()
        normalized = re.sub(r"[?!,]+", " ", normalized)
        normalized = normalized.replace("'", "")
        normalized = re.sub(r"\bshowme\b", "show me", normalized)
        normalized = re.sub(r"\bpullup\b", "pull up", normalized)
        normalized = re.sub(r"\b(open|close|launch|start|boot|show)(?=[a-z0-9])", lambda m: m.group(1) + " ", normalized)
        normalized = f" {normalized} "
        for source, target in COMMAND_CORRECTIONS:
            normalized = normalized.replace(f" {source} ", f" {target} ")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip(" .")

    def _matched_alias(self, text: str, app_name: str) -> str | None:
        for alias in sorted(APP_ALIASES.get(app_name, ()), key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return alias
        if app_name == "outlook":
            for alias in ("outlook", "email", "mail", "inbox"):
                if re.search(rf"\b{re.escape(alias)}\b", text):
                    return alias
        return None

    def _matched_known_web_target(self, text: str) -> str | None:
        for label in sorted(KNOWN_WEB_TARGETS.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(label)}\b", text):
                return label
        return None

    def _matched_settings_topic(self, text: str) -> str | None:
        for topic, aliases in SETTINGS_TOPICS.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", text):
                    return topic
        return None

    def _matched_folder(self, text: str) -> str | None:
        for folder, aliases in KNOWN_FOLDERS.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", text):
                    return folder
        return None

    def _matched_outlook_folder(self, text: str) -> str | None:
        for canonical, aliases in self.desktop_controller.OUTLOOK_FOLDERS.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", text):
                    return self.desktop_controller.outlook_folder_display_name(canonical)
        return None

    def _spotify_preferred_types(self, text: str) -> tuple[str, ...]:
        if "playlist" in text:
            return ("playlist", "track", "album", "artist")
        if "album" in text:
            return ("album", "track", "playlist", "artist")
        if "artist" in text or "songs by " in text:
            return ("artist", "track", "playlist", "album")
        return ("track", "playlist", "album", "artist")

    def _strip_music_tail(self, text: str | None) -> str | None:
        value = str(text or "").strip()
        if not value:
            return None
        preserved = {"liked songs"}
        if value.lower() in preserved:
            return value
        for tail in (" music", " song", " songs", " track", " tracks", " playlist", " playlists", " album", " albums"):
            if value.endswith(tail):
                candidate = value[: -len(tail)].strip()
                if candidate.lower() in preserved:
                    return candidate + tail
                value = candidate
                break
        return value or None

    def _strip_file_search_tail(self, text: str | None) -> str | None:
        value = str(text or "").strip()
        if not value:
            return None
        for tail in (" file", " files", " folder", " folders"):
            if value.endswith(tail):
                value = value[: -len(tail)].strip()
        return value or None

    def _normalize_file_request_query(self, text: str | None) -> str | None:
        value = self._strip_file_search_tail(text)
        if not value:
            return None
        value = re.sub(r"\s+(?:in|from)\s+(?:file explorer|explorer)$", "", value).strip()
        value = re.sub(r"^\s*(?:file|document|folder)\s+", "", value).strip()
        return value or None

    def _looks_like_file_request(
        self,
        text: str,
        query: str | None,
        *,
        context: VoiceCommandContext | None,
    ) -> bool:
        value = str(query or "").strip()
        if not value:
            return False
        normalized_query = self.desktop_controller._normalize_application_name(value)
        if not normalized_query:
            return False
        if self._matched_outlook_folder(text) is not None:
            return False
        if self._contains_any(text, APP_OBJECT_HINTS):
            return False
        tokens = [token for token in normalized_query.split() if token]
        if not tokens:
            return False
        if context is not None and context.preferred_app == "file_explorer":
            return True
        explicit_file_language = self._contains_any(
            text,
            ("file", "document", "pdf", "word", "excel", "powerpoint", "spreadsheet", "slides", "txt", "json", "csv"),
        )
        if explicit_file_language:
            return True
        if any(token in FILE_REQUEST_HINTS for token in tokens):
            return True
        if any(re.fullmatch(r"[a-z]{1,4}\d{2,4}", token) for token in tokens) and len(tokens) >= 2:
            return True
        if "." in value and re.search(r"\.[a-z0-9]{1,5}\b", value):
            return True
        if self._contains_any(text, CHROME_CONTEXT_PHRASES + SPOTIFY_CONTEXT_PHRASES):
            return False
        for size in range(min(4, len(tokens)), 0, -1):
            candidate = " ".join(tokens[:size])
            if self.desktop_controller.can_resolve_application(candidate):
                return False
        if self.desktop_controller.can_resolve_application(normalized_query) and len(tokens) <= 4:
            return False
        if self._contains_launch_request(text, APP_LAUNCH_PHRASES) and 1 <= len(tokens) <= 5:
            return True
        if self._looks_like_web_target(value):
            return False
        return False

    def _strip_outlook_tail(self, text: str | None) -> str | None:
        value = str(text or "").strip()
        if not value:
            return None
        value = re.sub(r"\s+(?:in|from)\s+outlook$", "", value).strip()
        return value or None

    def _parse_bare_browser_request(self, text: str) -> tuple[str, str] | None:
        trimmed = self._strip_common_prefix(text)
        for action, prefix in WEB_FALLBACK_PREFIXES:
            if re.search(rf"^\b{re.escape(prefix)}\b", trimmed):
                query = trimmed[len(prefix):].strip()
                query = self._trim_edge_noise(query)
                query = self._normalize_browser_query(query)
                if query and self._looks_like_web_target(query):
                    return action, query
        return None

    def _looks_like_web_target(self, query: str | None) -> bool:
        value = str(query or "").strip().lower()
        if not value:
            return False
        if "." in value or "/" in value or value.startswith("www "):
            return True
        if value in WEB_TARGET_HINTS:
            return True
        tokens = [token for token in value.split() if token]
        if len(tokens) >= 2:
            return True
        return False

    def _contains_launch_request(self, text: str, phrases: tuple[str, ...]) -> bool:
        trimmed = self._strip_common_prefix(text)
        return any(re.search(rf"^\b{re.escape(phrase)}\b", trimmed) for phrase in phrases)

    def _strip_common_prefix(self, text: str) -> str:
        value = f" {str(text or '').strip()} "
        changed = True
        while changed:
            changed = False
            for phrase in sorted(COMMON_FILLERS, key=len, reverse=True):
                updated = re.sub(rf"^\s*{re.escape(phrase)}\b\s*", " ", value)
                if updated != value:
                    value = updated
                    changed = True
        return re.sub(r"\s+", " ", value).strip()

    def _should_skip_catalog_open(self, text: str, *, entry_name: str, matched_alias: str) -> bool:
        folder = self._matched_folder(text)
        if "settings" in text and entry_name not in {"settings", "windows settings"}:
            return True
        if entry_name in {"google chrome", "chrome"} or matched_alias in {"chrome", "browser"}:
            if self._contains_launch_request(text, DEDICATED_APP_LAUNCH_PHRASES):
                return True
            if self._contains_any(text, ("new tab", "open tab", "incognito", "private tab", "go back", "back page", "previous page", "go forward", "forward page", "refresh", "reload")):
                return True
        if entry_name in {"file explorer", "explorer"} or matched_alias in {"file explorer", "explorer"}:
            if self._contains_launch_request(text, DEDICATED_APP_LAUNCH_PHRASES):
                return True
            if folder is not None or self._contains_any(text, ("search files", "find file", "find files", "search for", "look for", "look up")):
                return True
        if entry_name == "spotify" or matched_alias == "spotify":
            if self._contains_launch_request(text, DEDICATED_APP_LAUNCH_PHRASES):
                return True
            if self._contains_any(text, SPOTIFY_PLAY_PHRASES + ("pause", "stop music", "resume", "continue", "unpause", "next song", "next track", "skip song", "skip track", "skip", "previous song", "previous track", "last song", "last track", "go back", "shuffle", "repeat")):
                return True
        if entry_name == "outlook" or matched_alias in {"outlook", "mail", "email", "inbox"}:
            if self._contains_launch_request(text, DEDICATED_APP_LAUNCH_PHRASES):
                return True
            if self._contains_any(text, ("compose", "draft", "write", "send email", "email", "inbox", "calendar", "sent items", "drafts", "deleted items", "outbox")):
                return True
        return False

    def _is_spotify_resume_phrase(self, raw_text: str) -> bool:
        normalized = self._normalize_text(raw_text)
        if not normalized:
            return False
        if "spotify" not in normalized:
            return False
        play_only = self._cleanup_query(
            normalized,
            app_name="spotify",
            extra_phrases=SPOTIFY_PLAY_PHRASES + SPOTIFY_CONTEXT_PHRASES,
            trim_edge_noise=False,
        )
        play_only = self._strip_music_tail(play_only)
        return not play_only

    def _normalize_browser_query(self, text: str | None) -> str | None:
        if text is None:
            return None
        working = str(text or "").strip().lower()
        if not working:
            return None
        working = re.sub(
            r"\b(search up|search for|search on|search|look up|look for|look on|find|google|open up|open|go to|navigate to|take me to)\b",
            " ",
            working,
        )
        working = re.sub(
            r"\b(on|in|using|with)\s+(google chrome|chrome browser|chrome|browser)\b",
            " ",
            working,
        )
        working = re.sub(
            r"\b(on|in|using|with)\s+google\b",
            " ",
            working,
        )
        working = re.sub(r"\b(google chrome|chrome browser|chrome|browser)\b", " ", working)
        working = re.sub(r"\bgoogle\b$", " ", working)
        working = re.sub(r"\bplease\b$", " ", working)
        working = re.sub(r"\s+", " ", working).strip(" .")
        working = self._trim_edge_noise(working)
        return working or None

    def _cleanup_app_launch_query(self, query: str | None) -> str | None:
        if query is None:
            return None
        working = str(query or "").strip()
        if not working:
            return None
        normalized = self.desktop_controller._normalize_application_query(working)
        if not normalized:
            return None

        tokens = normalized.split()
        tail_words = {"library", "client", "launcher", "app", "application", "program", "window"}
        while len(tokens) > 1 and tokens[-1] in tail_words:
            tokens.pop()
        cleaned = " ".join(tokens).strip()
        return cleaned or None

    def _cleanup_query(
        self,
        text: str,
        *,
        app_name: str | None,
        extra_phrases: tuple[str, ...],
        trim_edge_noise: bool = True,
    ) -> str | None:
        working = f" {text} "
        phrases = list(dict.fromkeys(COMMON_FILLERS + extra_phrases))
        for phrase in sorted(phrases, key=len, reverse=True):
            working = re.sub(rf"\b{re.escape(phrase)}\b", " ", working)
        if app_name is not None:
            for alias in sorted(APP_ALIASES.get(app_name, ()), key=len, reverse=True):
                working = re.sub(rf"\b{re.escape(alias)}\b", " ", working)
        working = re.sub(r"\s+", " ", working).strip(" .")
        if trim_edge_noise:
            working = self._trim_edge_noise(working)
        return working or None

    def _contains_any(self, text: str, phrases: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(phrase)}\b", text) for phrase in phrases)

    def _extract_slot(self, text: str, *, start_phrase: str, stop_phrases: tuple[str, ...]) -> str | None:
        start_index = text.find(start_phrase)
        if start_index < 0:
            return None
        value = text[start_index + len(start_phrase):]
        for stop_phrase in stop_phrases:
            stop_index = value.find(stop_phrase)
            if stop_index >= 0:
                value = value[:stop_index]
                break
        cleaned = " ".join(value.split()).strip(" .")
        return cleaned or None

    def _trim_edge_noise(self, text: str) -> str:
        tokens = [token for token in str(text or "").split() if token]
        while tokens and tokens[0] in EDGE_NOISE_WORDS:
            tokens.pop(0)
        while tokens and tokens[-1] in EDGE_NOISE_WORDS:
            tokens.pop()
        return " ".join(tokens).strip()

# Author: Konstantin Markov
