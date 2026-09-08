"""Comprehensive phrasing-matrix test for the deterministic classifier's
Google Docs / Sheets / Slides create flows.

What this checks
----------------
For each phrasing below we run `Classifier().classify_chain(text)` and
verify:

  * the expected tool (gdocs_create / sheets_create / slides_create) appears
    in the produced step list, AND
  * the body / content / header arg is populated (non-empty) when the
    phrasing supplies one, AND
  * the title was NOT polluted by the full body sentence.

PASS criteria are intentionally strict so we surface real misses (the
classifier returning `None` and falling back to the LLM is treated as a
FAIL for these utterances — they SHOULD be deterministic).

Run:
    cd "c:/HGR App v1.0.0"
    python scripts/test_classifier_phrasings.py

Exit code: 0 always (this is a diagnostic, not a CI gate). Pass / fail
counts are printed at the bottom and the failing phrasings are listed
individually with the reason they failed.

Author: phrasing-matrix harness (no production code modified).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Make `src/` importable without requiring an editable install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hgr.live_api.planner.classifier import Classifier  # noqa: E402
from hgr.live_api.planner.plan import Step  # noqa: E402


# --------------------------------------------------------------------------- #
# Test matrix                                                                 #
# --------------------------------------------------------------------------- #
#
# Each entry: (phrase, expected_tool, body_kind)
#   expected_tool  -- one of 'gdocs_create' / 'sheets_create' / 'slides_create'
#   body_kind      -- which arg key on the created step should be non-empty:
#                       'text'    -> gdocs body
#                       'headers' -> sheets header row (any of headers/header/
#                                    columns/data/first_row args)
#                       'content' -> slides content / first-slide text
#                       None      -> only need the tool to match (title-only)
#
# CATEGORY: gdocs body phrasings (9 phrasings -- covers all required forms)
# CATEGORY: sheets header phrasings (5+ phrasings -- headers/columns/etc.)
# CATEGORY: slides content phrasings (3+ phrasings)
# CATEGORY: multi-action (each of the above suffixed with " and open it")
# CATEGORY: plurals -- "google slides" vs "slideshow"
# CATEGORY: picky-edge-case -- known tricky cases that are currently broken
# --------------------------------------------------------------------------- #


@dataclass
class Case:
    phrase: str
    expected_tool: str
    body_kind: Optional[str]  # 'text' / 'headers' / 'content' / None
    category: str


CASES: List[Case] = [
    # ---- gdocs body ------------------------------------------------------ #
    Case("make a doc titled Sprint Notes with body hello world",
         "gdocs_create", "text", "gdocs-body"),
    Case("create a doc titled Sprint Notes saying hello world",
         "gdocs_create", "text", "gdocs-body"),
    Case("make a google doc titled Sprint Notes containing hello world",
         "gdocs_create", "text", "gdocs-body"),
    Case("create a doc titled Sprint Notes with hello world written in it",
         "gdocs_create", "text", "gdocs-body"),
    Case("make a doc titled Sprint Notes and write hello world in it",
         "gdocs_create", "text", "gdocs-body"),
    Case("create a doc titled Sprint Notes with the text hello world",
         "gdocs_create", "text", "gdocs-body"),
    Case("make a doc titled Sprint Notes with hello world as the body",
         "gdocs_create", "text", "gdocs-body"),
    Case("create a doc titled Sprint Notes that says hello world",
         "gdocs_create", "text", "gdocs-body"),
    Case("make a doc titled Sprint Notes with hello world in it",
         "gdocs_create", "text", "gdocs-body"),

    # ---- sheets header / columns / data ---------------------------------- #
    Case("create a sheet titled Q4 Planning with header Name, Email, Phone",
         "sheets_create", "headers", "sheets-headers"),
    Case("make a spreadsheet titled Q4 Planning with headers Name, Email, Phone",
         "sheets_create", "headers", "sheets-headers"),
    Case("create a sheet titled Q4 Planning with columns Name, Email, Phone",
         "sheets_create", "headers", "sheets-headers"),
    Case("make a sheet titled Q4 Planning with first row Name, Email, Phone",
         "sheets_create", "headers", "sheets-headers"),
    Case("create a spreadsheet titled Q4 Planning with the data Name, Email, Phone",
         "sheets_create", "headers", "sheets-headers"),

    # ---- slides content -------------------------------------------------- #
    Case("create a slideshow titled Kickoff with first slide saying welcome team",
         "slides_create", "content", "slides-content"),
    Case("make a presentation titled Kickoff with content welcome team",
         "slides_create", "content", "slides-content"),
    Case("create a slideshow titled Kickoff with title slide welcome team",
         "slides_create", "content", "slides-content"),
    Case("make a google slides titled Kickoff with first slide saying welcome team",
         "slides_create", "content", "plurals"),

    # ---- plurals --------------------------------------------------------- #
    Case("create a google slides titled Roadmap",
         "slides_create", None, "plurals"),
    Case("make a slideshow titled Roadmap",
         "slides_create", None, "plurals"),
    Case("create a google slideshow titled Roadmap",
         "slides_create", None, "plurals"),
    Case("create a deck titled Roadmap",
         "slides_create", None, "plurals"),
    Case("create a presentation titled Roadmap",
         "slides_create", None, "plurals"),

    # ---- multi-action: chain "and open it" ------------------------------- #
    Case("make a doc titled Sprint Notes with body hello world and open it",
         "gdocs_create", "text", "multi-action"),
    Case("create a doc titled Sprint Notes saying hello world and open it",
         "gdocs_create", "text", "multi-action"),
    Case("make a doc titled Sprint Notes containing hello world and open it",
         "gdocs_create", "text", "multi-action"),
    Case("create a doc titled Sprint Notes that says hello world and open it",
         "gdocs_create", "text", "multi-action"),
    Case("make a doc titled Sprint Notes and write hello world in it and open it",
         "gdocs_create", "text", "multi-action"),
    Case("make a sheet titled Q4 Planning with headers Name, Email, Phone and open it",
         "sheets_create", "headers", "multi-action"),
    Case("create a sheet titled Q4 Planning with columns Name, Email, Phone and open it",
         "sheets_create", "headers", "multi-action"),
    Case("create a slideshow titled Kickoff with first slide saying welcome team and open it",
         "slides_create", "content", "multi-action"),
    Case("make a presentation titled Kickoff with content welcome team and open it",
         "slides_create", "content", "multi-action"),

    # ---- picky-edge-case: phrasings known to be tricky ------------------- #
    # Title-only with no body should still succeed (title='Sprint Notes').
    Case("create a doc titled Sprint Notes",
         "gdocs_create", None, "picky-edge-case"),
    # "google doc" pluralisation
    Case("make a google document titled Sprint Notes",
         "gdocs_create", None, "picky-edge-case"),
    # Quoted title
    Case('create a doc titled "Sprint Notes" with body hello world',
         "gdocs_create", "text", "picky-edge-case"),

    # ---- email-verbless: "email <recipient> <body>" without saying/that/about
    Case("email vesko hello from iris",
         "outlook_compose", "email_body", "email-verbless"),
    Case("email vesko@example.com hi",
         "outlook_compose", "email_body", "email-verbless"),
    Case("email O'Brien hi",
         "outlook_compose", "email_body", "email-verbless"),
    Case("email mary-anne the meeting is at 3",
         "outlook_compose", "email_body", "email-verbless"),
    Case('email vesko "hello world"',
         "outlook_compose", "email_body", "email-verbless"),
    Case("please email konstantin hello there",
         "outlook_compose", "email_body", "email-verbless"),
    Case("send email to vesko hello",
         "email_send", "email_body", "email-verbless"),
    Case("send an email vesko@x.com running late",
         "email_send", "email_body", "email-verbless"),
    Case("email vesko hi",
         "outlook_compose", "email_body", "email-verbless"),
    Case("e-mail vesko hello there",
         "outlook_compose", "email_body", "email-verbless"),
    # Connective forms — should keep matching the richer existing parser.
    Case("email vesko saying hello from iris",
         "outlook_compose", "email_body", "email-verbless"),
    Case("email vesko that the weather is bad",
         "outlook_compose", "email_body", "email-verbless"),

    # ---- sheets cell-write (Tier-1: 'add X in A2 in <name> sheet') ----
    # `body_kind` is the union 'headers' which accepts the `values` arg,
    # which is the populated matrix the Tier-1 emits.
    Case("add Test 1 into A2 and Test 2 into B2 in the Q4 plan sheet",
         "sheets_update_range", "headers", "sheets-update"),
    Case("put hello in cell A1 in the budget sheet",
         "sheets_update_range", "headers", "sheets-update"),
    Case("write 42 into B3 in the metrics spreadsheet",
         "sheets_update_range", "headers", "sheets-update"),
    Case("set 'Done' in C2 in the Q4 plan sheet",
         "sheets_update_range", "headers", "sheets-update"),
]


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #

# Keys we accept as "the body arg" for each tool. The current classifier
# only sets one of these (e.g. gdocs uses `text`), but other args are
# accepted so this test still passes after future improvements add the
# canonical key (e.g. `headers` for sheets, `content` for slides).
_BODY_KEYS = {
    "text": ("text", "body"),
    "headers": ("headers", "header", "columns", "data", "first_row",
                "rows", "values"),
    "content": ("content", "first_slide", "first_slide_text",
                "slide_content", "body", "text", "heading"),
    "email_body": ("body", "text", "message"),
}


def _steps_to_summary(steps: List[Step]) -> str:
    """Compact one-line summary of a step list for output."""
    if not steps:
        return "<no steps>"
    parts: List[str] = []
    for s in steps:
        # Truncate long args so the table stays readable.
        arg_strs: List[str] = []
        for k, v in s.args.items():
            vs = repr(v)
            if len(vs) > 60:
                vs = vs[:57] + "..."
            arg_strs.append(f"{k}={vs}")
        parts.append(f"{s.tool}({', '.join(arg_strs)})")
    return " -> ".join(parts)


def _find_step(steps: List[Step], tool_name: str) -> Optional[Step]:
    for s in steps:
        if s.tool == tool_name:
            return s
    return None


def _body_value(step: Step, body_kind: str) -> Optional[Any]:
    keys = _BODY_KEYS.get(body_kind, ())
    for k in keys:
        if k in step.args and step.args[k] not in (None, "", [], {}):
            return step.args[k]
    return None


def _title_polluted(step: Step, phrase: str) -> bool:
    """Catch the failure mode where the entire body got crammed into the
    title arg (because the body-prefix regex didn't fire). We flag when:
      * title contains a 'body marker' word like 'saying' / 'containing'
        / 'with body' / 'with the text' / 'that says' / 'and write', OR
      * title length exceeds ~50 chars (real titles are short).
    """
    title = step.args.get("title")
    if not isinstance(title, str):
        return False
    t = title.lower()
    body_markers = (
        "saying", "containing", "that says", "with body", "with the text",
        "with the body", "with the content", "and write", "as the body",
        "in it", "with first slide", "with content", "with title slide",
        "with header", "with headers", "with columns", "with first row",
        "with the data",
    )
    if any(marker in t for marker in body_markers):
        return True
    if len(title) > 50:
        return True
    return False


def evaluate(case: Case) -> Tuple[bool, str, Optional[List[Step]]]:
    """Run the classifier on `case.phrase` and return (passed, reason, steps)."""
    clf = Classifier()
    steps = clf.classify_chain(case.phrase)
    if steps is None:
        return False, "classifier returned None (no match)", None
    create_step = _find_step(steps, case.expected_tool)
    if create_step is None:
        actual = ", ".join(s.tool for s in steps) or "<empty>"
        return (
            False,
            f"expected tool {case.expected_tool!r} not in produced steps "
            f"(got: {actual})",
            steps,
        )
    if _title_polluted(create_step, case.phrase):
        return (
            False,
            f"title polluted: {create_step.args.get('title')!r}",
            steps,
        )
    if case.body_kind is not None:
        val = _body_value(create_step, case.body_kind)
        if val in (None, "", [], {}):
            return (
                False,
                f"expected {case.body_kind!r} arg populated, got "
                f"args={create_step.args!r}",
                steps,
            )
    return True, "ok", steps


def main() -> int:
    total = len(CASES)
    passed = 0
    failures: List[Tuple[Case, str, Optional[List[Step]]]] = []

    print("=" * 100)
    print(f"Classifier phrasing-matrix test  ({total} phrasings)")
    print("=" * 100)

    by_cat: Dict[str, List[Case]] = {}
    for c in CASES:
        by_cat.setdefault(c.category, []).append(c)

    for cat, cases in by_cat.items():
        print()
        print(f"--- category: {cat}  ({len(cases)} cases) ---")
        for case in cases:
            ok, reason, steps = evaluate(case)
            status = "PASS" if ok else "FAIL"
            summary = _steps_to_summary(steps) if steps is not None else "<None>"
            print(f"[{status}] {case.phrase!r}")
            print(f"        steps: {summary}")
            if not ok:
                print(f"        reason: {reason}")
                failures.append((case, reason, steps))
            if ok:
                passed += 1

    print()
    print("=" * 100)
    print(f"SUMMARY: {passed}/{total} PASS   {total - passed} FAIL")
    print("=" * 100)

    if failures:
        print()
        print("FAILING PHRASINGS:")
        for case, reason, steps in failures:
            print(f"  [{case.category}] {case.phrase!r}")
            print(f"      -> {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
