"""Static test of IrisPlanner routing — verifies the Tier-1 Ollama
patterns route as intended AND that existing intents (volume, email,
gdocs, weather, etc.) still win over the broader new patterns.

Also exercises:
  - command_router (Layer 0) defer rules for 'use the local model' / pronoun-open
  - executor key-alias resolution (url <-> link)
  - _looks_multistep heuristic in live_api_manager (catches 2-verb 'X and Y')

This is a one-off harness; safe to delete after.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable without installing the package.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.planner.classifier import Classifier  # noqa: E402

clf = Classifier()


# (input, expected_tool, why)
# expected_tool: a string ("ollama_generate"), None (no match -> realtime),
# or "ANY_BUT_OLLAMA" to assert it routes to SOMETHING but not Ollama.
CASES: list[tuple[str, str | None, str]] = [
    # ---- A: creative-write should route to ollama_generate -----------------
    ("write a haiku about debugging", "ollama_generate", "A.1 basic haiku"),
    ("write me a poem about the rain", "ollama_generate", "A.2 poem"),
    ("compose a tweet about Python", "ollama_generate", "A.3 tweet"),
    ("give me a joke", "ollama_generate", "A.4 short joke"),
    ("write a short caption", "ollama_generate", "A.5 caption"),
    ("draft a quick description", "ollama_generate", "A.6 description"),
    ("write a funny limerick about cats", "ollama_generate", "A.7 limerick"),
    ("make me a catchy slogan for a coffee shop", "ollama_generate", "A.8 slogan"),
    ("give me a clever quote", "ollama_generate", "A.9 quote"),
    ("compose a brief story about a robot", "ollama_generate", "A.10 story"),
    ("write a tagline", "ollama_generate", "A.11 tagline"),
    ("can you write a poem about coffee", "ollama_generate", "A.12 polite prefix"),
    ("please write me a haiku about snow", "ollama_generate", "A.13 please prefix"),
    ("write a sonnet", "ollama_generate", "A.14 sonnet"),
    ("give me a pun", "ollama_generate", "A.15 pun"),

    # ---- B: explicit override (use ollama / via ollama / etc.) -------------
    ("use ollama to write a haiku about food", "ollama_generate", "B.1 use ollama to"),
    ("use the local model to explain quantum entanglement", "ollama_generate", "B.2 use the local model"),
    ("with ollama, summarize the difference between TCP and UDP", "ollama_generate", "B.3 with ollama,"),
    ("using the local llm, generate 5 startup names", "ollama_generate", "B.4 using the local llm"),
    ("tell me a story via ollama", "ollama_generate", "B.5 suffix via ollama"),
    ("explain monads with the local model", "ollama_generate", "B.6 suffix with the local model"),
    ("use ollama: write me a sonnet", "ollama_generate", "B.7 colon form"),
    ("use local ai to draft an apology", "ollama_generate", "B.8 use local ai"),

    # ---- C: regex generation ---------------------------------------------
    ("give me a regex for email addresses", "ollama_generate", "C.1 give me a regex"),
    ("write a regex that matches phone numbers", "ollama_generate", "C.2 write a regex"),
    ("regex for IP addresses", "ollama_generate", "C.3 regex for"),
    ("regex matching dates", "ollama_generate", "C.4 regex matching"),
    ("make me a regex to capture URLs", "ollama_generate", "C.5 make me a regex"),
    ("generate a regex for postal codes", "ollama_generate", "C.6 generate a regex"),

    # ---- D: translate with inline source ----------------------------------
    ("translate hello world to Spanish", "ollama_generate", "D.1 X to Spanish"),
    ("translate to French: how are you doing today", "ollama_generate", "D.2 to French: X"),
    ("translate good morning into German", "ollama_generate", "D.3 X into German"),
    ("translate to Japanese: thank you very much", "ollama_generate", "D.4 to Japanese:"),

    # ---- E: rephrase / proofread with inline source -----------------------
    ('rephrase: "the meeting is at 3pm"', "ollama_generate", "E.1 rephrase: quoted"),
    ('proofread: "their going to the store"', "ollama_generate", "E.2 proofread:"),
    ('paraphrase "the cat sat on the mat"', "ollama_generate", "E.3 paraphrase quoted"),

    # ---- F: REGRESSION GUARDS — existing intents must still win ------------
    ("set volume to 50", "volume_set", "F.1 volume_set"),
    ("volume 30", "volume_set", "F.2 short volume"),
    ("mute", "volume_mute", "F.3 mute"),
    ("toggle mute", "volume_toggle_mute", "F.4 toggle mute"),
    ("create a google doc called Meeting Notes", "gdocs_create", "F.5 gdocs_create"),
    ("make a new spreadsheet titled Budget", "sheets_create", "F.6 sheets_create"),
    ("create a slideshow named Q4 Recap", "slides_create", "F.7 slides_create"),
    ("email dani@example.com saying hello there", "outlook_compose", "F.8 email compose"),
    ("send a message to dani@example.com saying hi", "outlook_compose", "F.9 send message"),
    ("what's Dani's email", "iris_lookup_contact", "F.10 lookup contact"),
    ("open it", "iris_open_last", "F.11 open last"),
    ("forget Dani", "iris_forget_contact", "F.12 forget"),
    ("weather in Paris", "weather_get", "F.13 weather_get"),
    ("what's the weather", "weather_get", "F.14 weather no location"),
    ("send a text to Vesko saying hello", "phone_link_send_text", "F.15 text send"),
    ("read my recent texts", "phone_link_read_recent", "F.16 read texts"),
    ("always send from gmail", "iris_set_preference", "F.17 set preference"),
    ("set up kicad", "iris_setup_tool", "F.18 setup tool"),
    ("add a task: buy milk", "todo_add", "F.19 todo_add"),
    ("remind me to call mom", "todo_add", "F.20 remind me"),
    ("upload C:/foo.pdf to drive", "drive_upload", "F.21 drive upload"),
    ("toggle discord mute", "discord_toggle_mute", "F.22 discord mute"),
    ("dani@example.com is for Vesko", "iris_remember_contact", "F.23 remember contact"),

    # ---- G: should NOT route to ollama — needs realtime/screen context -----
    # These return None (no Tier-1 match) so realtime handles them. The
    # assertion is "NOT ollama_generate".
    ("summarize this email", None, "G.1 summarize this (screen)"),
    ("translate that", None, "G.2 translate that (pronoun)"),
    ("translate this to French", None, "G.3 translate this (pronoun)"),
    ("translate the screen", None, "G.4 translate the screen"),
    ("rephrase that", None, "G.5 rephrase that (pronoun)"),
    ("what's in my notion", None, "G.6 what's in my X (realtime)"),
    ("what is python", None, "G.7 factual Q&A"),
    ("click the send button", None, "G.8 ui action"),
    ("open chrome", None, "G.9 app launch"),

    # ---- H: edge cases ----------------------------------------------------
    # "write me an email" — 'email' is NOT in my creative-write kind list,
    # so this should fall through (realtime/Tier-2 picks the right path).
    ("write me an email", None, "H.1 email not in kind list"),
    # "write code" — code is NOT in my kind list (realtime knows codebase).
    ("write a function to sort a list", None, "H.2 code falls through"),
    # "write a doc" — doc is NOT in my kind list (gdocs_create handles
    # 'create a doc'; 'write a doc' alone has no good Tier-1 home).
    ("write a doc", None, "H.3 doc falls through"),
    # Empty / very short
    ("", None, "H.4 empty"),
    ("hi", None, "H.5 hi"),
    ("haiku", None, "H.6 haiku alone, no verb"),
    # Use ollama with no task
    ("use ollama", None, "H.7 use ollama, no task"),
    # Translate to lang with no source
    ("translate to French", None, "H.8 translate to lang, no source"),
]


def name_of(result) -> str | None:
    if result is None:
        return None
    return getattr(result, "tool", None)


# ---- additional layer tests ------------------------------------------------

# Layer 0: command_router should DEFER ('not match') these — they belong to
# Tier 1 / iris_open_last / explicit Ollama route, not the catalog fuzzy
# matcher that previously opened OneNote / It Takes Two / processed only
# the volume-set fragment of a multi-action prompt.
ROUTER_DEFER_CASES: list[tuple[str, str]] = [
    ("use the local model to explain monads in one sentence", "Bug1: opened OneNote"),
    ("use ollama to write a haiku about food", "Bug1: should defer"),
    ("use the local llm, summarize this", "Bug1: defer to Tier-1"),
    ("with ollama, draft an apology", "Bug1: with-prefix defer"),
    ("open it", "Bug3: opened It Takes Two"),
    ("open that", "Bug3: pronoun defer"),
    ("open this", "Bug3: pronoun defer"),
    ("show me that", "Bug3: show variant"),
    ("open the doc", "Bug3: 'open the doc' defer"),
    ("open the page you just made", "Bug3: full iris_open_last phrase"),
    ("open the sheet", "Bug3: sheet variant"),
    # NEW: multi-action prompts where Layer 0 was fuzzy-matching ONE
    # embedded sub-action (e.g. 'set volume to 30') and executing it
    # standalone, hiding the other actions.
    ("add a task: review PR, set volume to 30, and tell me the weather",
     "Bug C19: multi-action with ', and'"),
    ("write me a haiku about coffee and add a task to drink some",
     "Bug C20: 2-verb 'and' multi-action"),
    ("open Notepad and type 'hi'", "2-verb multi-action via 'and'"),
    ("send an email to vesko and add a calendar event",
     "send + add multi-action"),
]

# Executor variable-alias resolution. Mocks a prior step's output and asks
# the resolver to substitute {step:1.url} when the actual output uses 'link'.
def _check_executor_aliases() -> tuple[int, int, list[str]]:
    from hgr.live_api.planner.executor import Executor
    from hgr.live_api.planner.plan import StepResult
    fails: list[str] = []
    cases: list[tuple[str, dict, str, str]] = [
        ("{step:1.url}", {"link": "https://x/y", "id": "abc"},
         "https://x/y", "url->link alias"),
        ("{step:1.link}", {"url": "https://x/y"},
         "https://x/y", "link->url alias"),
        ("{step:1.text}", {"response": "haiku here"},
         "haiku here", "text->response alias"),
        ("{step:1.title}", {"name": "Meeting Notes"},
         "Meeting Notes", "title->name alias"),
        ("{step:1.url}", {"url": "explicit", "link": "alias"},
         "explicit", "exact match wins over alias"),
        # Weather aliases — the D25 failure was {step:1.temp_f} when
        # weather_get returns {temperature, summary, ...} but not temp_f.
        ("{step:1.temp_f}",
         {"temperature": "70F", "summary": "Sofia: sunny, 70F"},
         "70F", "weather: temp_f -> temperature"),
        ("{step:1.weather}",
         {"summary": "Sofia: sunny, 70F", "description": "sunny"},
         "Sofia: sunny, 70F", "weather: weather -> summary"),
        ("{step:1.conditions}", {"description": "sunny"},
         "sunny", "weather: conditions -> description"),
        ("{step:1.body}",
         {"summary": "Sofia: sunny, 70F"},
         "Sofia: sunny, 70F", "email: body -> summary (cross-tool pipe)"),
    ]
    passed = 0
    for ref, output, expected, why in cases:
        prior = StepResult(step_id=1, tool="prior", status="ok", output=output)
        results = {1: prior}
        resolved = Executor._resolve(ref, results)
        if resolved == expected:
            passed += 1
        else:
            fails.append(f"  [{why}] expected {expected!r}, got {resolved!r}")
    return passed, len(cases), fails


# live_api_manager._looks_multistep: should detect 2-verb 'X and Y' prompts.
def _check_multistep_heuristic() -> tuple[int, int, list[str]]:
    # Inline the heuristic so the test doesn't pull in PySide6/etc.
    _ACTION_VERB_STARTS = (
        "open", "play", "search", "look", "move", "close", "minimize", "maximize",
        "restore", "focus", "start", "launch", "run", "create", "make", "build",
        "find", "show", "put", "summarize", "summarise", "write", "pull", "set",
        "go", "navigate", "drag",
        "add", "remove", "delete", "remind", "schedule",
        "send", "email", "text", "message",
        "tell", "give", "compose", "draft", "translate", "rephrase",
        "generate", "produce", "save", "upload", "download",
        "ask", "fetch", "read", "list", "check", "toggle", "mute", "unmute",
        "type", "click", "press", "scroll", "copy", "paste", "say", "answer",
    )
    def looks_multistep(text: str) -> bool:
        low = (text or "").lower()
        seq = low.count(" then ") + low.count(" and ") + low.count(", ")
        verbs = sum(1 for v in _ACTION_VERB_STARTS if (v + " ") in low)
        return ((seq >= 2 and verbs >= 2)
                or verbs >= 3
                or (verbs >= 2 and (" and " in low or " then " in low
                                    or "; " in low))
                or len(text) > 160)

    cases: list[tuple[str, bool, str]] = [
        ("write me a haiku about coffee and add a task to drink some", True,
         "Bug C20: 2-verb 'and' must trigger"),
        ("add a task: review PR, set volume to 30, and tell me the weather",
         True, "Bug C19: 3-clause comma+and"),
        ("write a haiku", False, "single verb single task"),
        ("open chrome", False, "single command"),
        ("set volume to 30", False, "single command"),
        ("open Notepad and type 'hi'", True, "2 verbs and"),
        ("send an email to vesko and add a calendar event", True,
         "send + add via 'and'"),
        ("create a doc, then open it", True, "then triggers"),
        ("just chatting", False, "no verbs"),
    ]
    passed = 0
    fails: list[str] = []
    for text, expected, why in cases:
        got = looks_multistep(text)
        if got == expected:
            passed += 1
        else:
            fails.append(f"  [{why}] {text!r}: expected {expected}, got {got}")
    return passed, len(cases), fails


# Layer 0: simulate command_router skip rules without instantiating the
# heavy VoiceCommandProcessor. Checks BOTH:
#   - marker / pronoun-regex defers (local model, 'open it')
#   - looks_multi_action defer (multi-clause prompts like C19/C20)
def _check_router_defers() -> tuple[int, int, list[str]]:
    from hgr.live_api.command_router import (
        _LOCAL_MODEL_MARKERS, _PRONOUN_OPEN_RE,
    )
    from hgr.live_api.planner.triggers import looks_multi_action
    passed = 0
    fails: list[str] = []
    for text, why in ROUTER_DEFER_CASES:
        lower = text.lower()
        deferred = (any(m in lower for m in _LOCAL_MODEL_MARKERS)
                    or bool(_PRONOUN_OPEN_RE.match(text))
                    or looks_multi_action(text))
        if deferred:
            passed += 1
        else:
            fails.append(f"  [{why}] {text!r} should defer but didn't")
    return passed, len(ROUTER_DEFER_CASES), fails


# ms_mail_send should reject bare names + placeholder domains BEFORE
# hitting Graph (which 202s them silently and the user's Sent folder
# stays empty). Mirrors the validation block in ms365_connector.execute.
def _check_mail_send_validation() -> tuple[int, int, list[str]]:
    def validates(to: str) -> str | None:
        """Returns None if accepted, an error string if rejected."""
        if not to:
            return "'to' is required"
        if "@" not in to or "." not in to.split("@", 1)[1]:
            return "needs full email address"
        if to.lower().endswith(("@example.com", "@example.org",
                                 "@test.com", "@placeholder.com")):
            return "placeholder domain"
        return None
    cases: list[tuple[str, bool, str]] = [
        # (to, should_be_accepted, why)
        ("dani@mangollc.org", True, "real address accepted"),
        ("vesko", False, "bare name rejected"),
        ("vesko@example.com", False, "placeholder domain rejected"),
        ("vesko@example.org", False, "placeholder .org rejected"),
        ("foo@test.com", False, "test.com rejected"),
        ("noatsign", False, "no @ rejected"),
        ("user@", False, "no domain rejected"),
        ("user@domain", False, "no TLD rejected"),
        ("", False, "empty rejected"),
        ("a.b.c@sub.example-real.io", True, "complex real-looking accepted"),
    ]
    passed = 0
    fails: list[str] = []
    for to, should_accept, why in cases:
        err = validates(to)
        accepted = err is None
        if accepted == should_accept:
            passed += 1
        else:
            fails.append(f"  [{why}] to={to!r}: expected "
                         f"{'accept' if should_accept else 'reject'}, "
                         f"got {'accept' if accepted else f'reject ({err})'}")
    return passed, len(cases), fails


# Volume connector should accept the keys/shapes the Tier-2 LLM tends to
# emit — not just strict {'percent': int}. The C19 plan errored
# "percent must be an integer 0-100" because the LLM passed the value
# under 'level' / 'value' / with a '%' suffix.
def _check_volume_set_tolerance() -> tuple[int, int, list[str]]:
    # Test arg-parsing without needing the real audio controller. Mirrors
    # the parsing block in volume_connector.execute().
    def parse_percent(args: dict) -> int | None:
        raw = (args.get("percent") if args.get("percent") is not None
               else args.get("level") if args.get("level") is not None
               else args.get("value") if args.get("value") is not None
               else args.get("volume"))
        if isinstance(raw, str):
            raw = raw.strip().rstrip("%").strip()
        try:
            return int(round(float(raw)))
        except (TypeError, ValueError):
            return None
    cases: list[tuple[dict, int | None, str]] = [
        ({"percent": 30}, 30, "exact key"),
        ({"percent": "30"}, 30, "string percent"),
        ({"percent": "30%"}, 30, "string with %"),
        ({"percent": 30.5}, 30, "float (rounds)"),
        ({"level": 50}, 50, "alias: level"),
        ({"value": 75}, 75, "alias: value"),
        ({"volume": 20}, 20, "alias: volume"),
        ({"level": "60%"}, 60, "alias + suffix"),
        ({}, None, "empty: rejects"),
        ({"percent": "loud"}, None, "non-numeric: rejects"),
    ]
    passed = 0
    fails: list[str] = []
    for args, expected, why in cases:
        got = parse_percent(args)
        if got == expected:
            passed += 1
        else:
            fails.append(f"  [{why}] {args!r}: expected {expected}, got {got}")
    return passed, len(cases), fails


# Validate looks_multi_action directly — it's the trigger Layer 0 calls.
# Bug from live test C19/C20: 'add', 'set', 'tell' weren't in its verb list,
# so multi-action prompts with those verbs were classified single-action.
def _check_looks_multi_action() -> tuple[int, int, list[str]]:
    from hgr.live_api.planner.triggers import looks_multi_action
    cases: list[tuple[str, bool, str]] = [
        # MUST detect as multi-action
        ("add a task: review PR, set volume to 30, and tell me the weather",
         True, "C19 three-clause"),
        ("write me a haiku about coffee and add a task to drink some",
         True, "C20 two-verb 'and'"),
        ("get the weather in Sofia and email it to vesko", True,
         "D25 cross-app pipe (was failing — _AND_VERB_RE raw-string bug)"),
        ("send an email to vesko and add a calendar event", True,
         "send + add"),
        ("open notepad and type 'hi'", True, "open + type"),
        ("set volume to 50 and mute discord", True,
         "set + mute"),
        ("create a doc, then open it", True, "create + then"),
        ("remove that task and add a new one", True, "remove + add"),
        # MUST NOT false-positive (single-action prompts)
        ("open chrome", False, "single action"),
        ("set volume to 50", False, "single command"),
        ("add a task: buy milk", False,
         "single todo, no second action"),
        ("tell me Dani's email", False,
         "single lookup — must not catch on 'tell' alone"),
        ("what's the weather", False, "question"),
        ("play music on spotify", False, "single action"),
        ("set the table", False, "'set' as non-command"),
    ]
    passed = 0
    fails: list[str] = []
    for text, expected, why in cases:
        got = looks_multi_action(text)
        if got == expected:
            passed += 1
        else:
            fails.append(f"  [{why}] {text!r}: expected {expected}, got {got}")
    return passed, len(cases), fails


def main() -> int:
    passed = 0
    failed: list[tuple[str, str | None, str | None, str]] = []
    for text, expected, why in CASES:
        got = name_of(clf.classify(text))
        if expected == "ANY_BUT_OLLAMA":
            ok = (got is not None) and (got != "ollama_generate")
        else:
            ok = got == expected
        if ok:
            passed += 1
        else:
            failed.append((text, expected, got, why))

    total = len(CASES)
    print(f"\n=== Classifier routing: {passed}/{total} pass ===")
    if failed:
        print("FAILURES:")
        for text, expected, got, why in failed:
            print(f"  [{why}]")
            print(f"    input:    {text!r}")
            print(f"    expected: {expected}")
            print(f"    got:      {got}")

    r_pass, r_total, r_fails = _check_router_defers()
    print(f"\n=== Layer-0 defers: {r_pass}/{r_total} pass ===")
    for f in r_fails:
        print(f)

    a_pass, a_total, a_fails = _check_executor_aliases()
    print(f"\n=== Executor key aliases: {a_pass}/{a_total} pass ===")
    for f in a_fails:
        print(f)

    m_pass, m_total, m_fails = _check_multistep_heuristic()
    print(f"\n=== Multistep heuristic: {m_pass}/{m_total} pass ===")
    for f in m_fails:
        print(f)

    l_pass, l_total, l_fails = _check_looks_multi_action()
    print(f"\n=== looks_multi_action: {l_pass}/{l_total} pass ===")
    for f in l_fails:
        print(f)

    v_pass, v_total, v_fails = _check_volume_set_tolerance()
    print(f"\n=== volume_set arg tolerance: {v_pass}/{v_total} pass ===")
    for f in v_fails:
        print(f)

    s_pass, s_total, s_fails = _check_mail_send_validation()
    print(f"\n=== ms_mail_send recipient validation: {s_pass}/{s_total} pass ===")
    for f in s_fails:
        print(f)

    all_pass = (len(failed) == 0 and not r_fails and not a_fails
                and not m_fails and not l_fails and not v_fails
                and not s_fails)
    grand_total = (total + r_total + a_total + m_total + l_total
                   + v_total + s_total)
    grand_pass = (passed + r_pass + a_pass + m_pass + l_pass
                  + v_pass + s_pass)
    print(f"\n{'='*60}")
    print(f"OVERALL: {grand_pass}/{grand_total} "
          + ("[ALL GREEN]" if all_pass else "[FAILURES]"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
