"""Smoke-test for the new Google connectors (contacts / forms / identity
/ photos / tasks) and a phrase-routing sanity check against the
deterministic classifier.

What this verifies
------------------
A. For each new connector class:
   1. Imports cleanly.
   2. Instantiates with a mocked GoogleClient (where `.service(...)`
      returns a fake that returns predictable dicts; `.ready()`->True).
   3. `.tools()` returns a non-empty list of valid OpenAI tool schemas
      (each entry has type=='function', a name, and a parameters obj).
   4. For each tool exposed, `.execute(name, sample_args)` returns a
      dict with at least a "status" field.
B. Pool-watchdog: for ONE tool per connector, the underlying API call
   is replaced with `time.sleep(30)`. We assert the connector returns
   a dict with code=='timeout' within ~26s of the connector's own
   `_CALL_TIMEOUT_SEC` ceiling (25s + a small grace).
C. Phrase-routing: for each connector, take ONE voice example from
   the classifier design and assert `Classifier().classify_chain()`
   produces a step whose `.tool` matches the expected tool name. Uses
   the same harness shape as `scripts/test_classifier_phrasings.py`.

Run
---
    cd "c:/HGR App v1.0.0"
    python scripts/test_new_google_connectors.py

Exit code: 0 always (diagnostic, not a CI gate). Pass/fail counts
are printed per connector and per phase at the bottom.

This script never hits a real Google API — every service object is a
unittest.mock.MagicMock.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

# Make `src/` importable without requiring an editable install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hgr.live_api.connectors.contacts_connector import ContactsConnector  # noqa: E402
from hgr.live_api.connectors.forms_connector import FormsConnector  # noqa: E402
from hgr.live_api.connectors.identity_connector import IdentityConnector  # noqa: E402
from hgr.live_api.connectors.photos_connector import GooglePhotosConnector  # noqa: E402
from hgr.live_api.connectors.tasks_connector import TasksConnector  # noqa: E402


# --------------------------------------------------------------------------- #
# Mock GoogleClient + service factories                                       #
# --------------------------------------------------------------------------- #

def _exec_returns(value):
    """Build a MagicMock chain whose terminal `.execute()` returns `value`.

    googleapiclient services are chained: svc.people().get(...).execute()
    A MagicMock auto-chains attribute / call access, so we only need to
    pin `.execute.return_value` on every leaf — easiest by returning a
    MagicMock that maps any call to a node whose execute() yields `value`.
    """
    leaf = MagicMock()
    leaf.execute.return_value = value

    # Make EVERY method call on the chain return the same leaf so
    # svc.X().Y(...).execute() and svc.X.Y().execute() both work.
    chain = MagicMock()
    chain.return_value = leaf  # for `svc()` style
    chain.__call__ = lambda *a, **k: leaf  # paranoia
    # By default, attribute access on a MagicMock returns a new MagicMock;
    # we override __getattr__ behavior via `side_effect` on the parent.
    return leaf, chain


class _FakeService:
    """Tiny chainable fake — every attribute access + call returns self,
    and `.execute()` returns the dict registered for the last leaf
    'method name' the caller asked for.

    Usage:
        svc = _FakeService({
            "people.searchContacts": {"results": [...]},
            "people.createContact": {"resourceName": "people/c1"},
            ...
        })
        svc.people().searchContacts(query="x").execute()  # -> {"results": [...]}
    """

    def __init__(self, table: Dict[str, Any]) -> None:
        self._table = table
        self._chain: List[str] = []
        # An optional callable invoked instead of returning the table
        # value — used by the timeout test to inject time.sleep().
        self._before_exec: Optional[Callable[[List[str]], None]] = None

    def __getattr__(self, name: str) -> "_FakeService":
        if name.startswith("_"):
            raise AttributeError(name)
        # Create a sibling node carrying the extended chain.
        sib = _FakeService(self._table)
        sib._chain = self._chain + [name]
        sib._before_exec = self._before_exec
        return sib

    def __call__(self, *args: Any, **kwargs: Any) -> "_FakeService":
        # Calling a node (e.g. .people()) returns the same node so the
        # next attribute access (e.g. .searchContacts) appends correctly.
        return self

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        if self._before_exec is not None:
            self._before_exec(self._chain)
        # Look up by progressively shorter chain suffixes so the test
        # author can register either "people.searchContacts" (specific)
        # or just "searchContacts" (loose).
        for joiner in (".", "/"):
            for start in range(len(self._chain)):
                key = joiner.join(self._chain[start:])
                if key in self._table:
                    return self._table[key]
        # Fall back to last segment alone.
        if self._chain:
            last = self._chain[-1]
            if last in self._table:
                return self._table[last]
        # Default empty response so a connector never crashes on a
        # method we forgot to register.
        return {}


def _mock_client(svc_table: Dict[str, Any]) -> MagicMock:
    """Build a GoogleClient-shaped MagicMock backed by a single fake
    service whose method-leaf calls return values from `svc_table`."""
    fake_svc = _FakeService(svc_table)
    client = MagicMock()
    client.ready.return_value = True
    client.has_scope.return_value = True
    client.service.return_value = fake_svc
    # Photos connector reaches past .service() into ._load_creds() and
    # also into private build() helpers — give it a creds stub so the
    # connector's _svc() at least returns a non-None object.
    creds = MagicMock()
    creds.token = "fake-bearer-token"
    creds.valid = True
    client._load_creds.return_value = creds
    # Also expose ._service-like fallback path some connectors take.
    client._fake_svc = fake_svc
    return client


# --------------------------------------------------------------------------- #
# Connector test catalogue                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class ToolCase:
    """One tool invocation under test."""
    tool: str
    args: Dict[str, Any]
    # Service table mocking what googleapiclient returns for this call.
    table: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectorCase:
    label: str
    connector_cls: type
    # Tools that exist for this connector + concrete sample call args.
    tools: List[ToolCase]
    # Tool to use for the watchdog test (must be one of `tools`).
    timeout_tool: str
    # Voice phrase + expected tool for the phrase-routing test.
    voice_phrase: str
    voice_expected_tool: str


# Use realistic, classifier-friendly phrases pulled from the connector
# design docs (matching the patterns in classifier.py).
CASES: List[ConnectorCase] = [
    ConnectorCase(
        label="contacts",
        connector_cls=ContactsConnector,
        tools=[
            ToolCase(
                tool="contacts_search",
                args={"query": "Dani"},
                table={
                    "people.searchContacts": {
                        "results": [
                            {"person": {
                                "resourceName": "people/c1",
                                "names": [{"displayName": "Dani Test",
                                           "givenName": "Dani",
                                           "familyName": "Test"}],
                                "emailAddresses": [{"value": "dani@test.com",
                                                    "type": "work"}],
                                "phoneNumbers": [{"value": "+1 555-1234",
                                                  "type": "mobile"}],
                                "organizations": [{"name": "Touchless Inc"}],
                            }}
                        ]
                    },
                },
            ),
            ToolCase(
                tool="contacts_create",
                args={"given_name": "John", "family_name": "Doe",
                      "email": "john@example.com"},
                table={
                    "people.createContact": {
                        "resourceName": "people/c2",
                        "names": [{"displayName": "John Doe"}],
                    },
                },
            ),
        ],
        timeout_tool="contacts_search",
        voice_phrase="find Dani in my contacts",
        voice_expected_tool="contacts_search",
    ),
    ConnectorCase(
        label="forms",
        connector_cls=FormsConnector,
        tools=[
            ToolCase(
                tool="forms_create",
                args={"title": "Beta Signup",
                      "questions": [
                          {"title": "What's your name?",
                           "type": "short_answer"}]},
                table={
                    "forms.create": {"formId": "form123",
                                     "responderUri":
                                     "https://docs.google.com/forms/d/e/form123/viewform"},
                    "forms.batchUpdate": {"form": {"formId": "form123"}},
                },
            ),
            ToolCase(
                tool="forms_responses",
                args={"form_id": "form123", "include_answers": False},
                table={
                    "forms.responses.list": {
                        "responses": [
                            {"responseId": "r1",
                             "lastSubmittedTime": "2026-06-09T12:00:00Z"}
                        ]
                    },
                    "forms.get": {"info": {"title": "Beta Signup"},
                                  "items": []},
                },
            ),
        ],
        timeout_tool="forms_responses",
        voice_phrase="how many responses on the beta signup form",
        voice_expected_tool="forms_responses",
    ),
    ConnectorCase(
        label="identity",
        connector_cls=IdentityConnector,
        tools=[
            ToolCase(
                tool="google_whoami",
                args={},
                table={
                    "oauth2.userinfo.get": {
                        "name": "Dani Markov",
                        "given_name": "Dani",
                        "family_name": "Markov",
                        "email": "dani@example.com",
                        "picture": "https://example.com/pic.jpg",
                        "locale": "en",
                        "verified_email": True,
                    },
                },
            ),
            ToolCase(
                tool="google_my_birthday",
                args={},
                table={
                    "people.get": {
                        "birthdays": [
                            {"metadata": {"primary": True},
                             "date": {"month": 3, "day": 14, "year": 1990}}
                        ]
                    },
                },
            ),
        ],
        timeout_tool="google_whoami",
        voice_phrase="what's my email",
        voice_expected_tool="google_whoami",
    ),
    ConnectorCase(
        label="photos",
        connector_cls=GooglePhotosConnector,
        tools=[
            ToolCase(
                tool="photos_upload",
                # path will be patched per-test to an actual temp file
                # because the connector pre-validates `os.path.isfile`.
                args={"path": "__TEMP_FILE__", "description": "test"},
                table={
                    "photoslibrary.mediaItems.batchCreate": {
                        "newMediaItemResults": [
                            {"status": {"code": 0},
                             "mediaItem": {
                                 "id": "media1",
                                 "productUrl":
                                 "https://photos.google.com/lr/photo/media1",
                                 "mimeType": "image/jpeg"}}
                        ]
                    },
                },
            ),
        ],
        timeout_tool="photos_upload",
        voice_phrase="upload C:/tmp/screenshot.png to my photos",
        voice_expected_tool="photos_upload",
    ),
    ConnectorCase(
        label="tasks",
        connector_cls=TasksConnector,
        tools=[
            ToolCase(
                tool="tasks_list",
                args={"include_completed": False},
                table={
                    "tasks.list": {"items": [
                        {"id": "t1", "title": "buy milk",
                         "status": "needsAction"}]},
                    "tasklists.get": {"title": "My Tasks"},
                },
            ),
            ToolCase(
                tool="tasks_add",
                args={"title": "buy milk"},
                table={
                    "tasks.insert": {"id": "t2", "title": "buy milk"},
                    "tasklists.get": {"title": "My Tasks"},
                },
            ),
            ToolCase(
                tool="tasks_complete",
                args={"title_match": "buy milk"},
                table={
                    "tasks.list": {"items": [
                        {"id": "t1", "title": "buy milk",
                         "status": "needsAction"}]},
                    "tasks.patch": {"id": "t1", "status": "completed"},
                    "tasks.get": {"id": "t1", "title": "buy milk"},
                    "tasklists.get": {"title": "My Tasks"},
                },
            ),
            ToolCase(
                tool="tasks_delete",
                args={"title_match": "buy milk"},
                table={
                    "tasks.list": {"items": [
                        {"id": "t1", "title": "buy milk",
                         "status": "needsAction"}]},
                    "tasks.delete": {},
                    "tasks.get": {"id": "t1", "title": "buy milk"},
                    "tasklists.get": {"title": "My Tasks"},
                },
            ),
        ],
        timeout_tool="tasks_list",
        voice_phrase="what's on my tasks",
        voice_expected_tool="tasks_list",
    ),
]


# --------------------------------------------------------------------------- #
# A. Per-connector schema + execute() smoke tests                             #
# --------------------------------------------------------------------------- #

def _is_valid_schema(schema: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(schema, dict):
        return False, "schema not a dict"
    if schema.get("type") != "function":
        return False, f"type != function (got {schema.get('type')!r})"
    if not schema.get("name"):
        return False, "missing 'name'"
    params = schema.get("parameters")
    if not isinstance(params, dict):
        return False, "'parameters' missing or not a dict"
    if params.get("type") != "object":
        return False, "parameters.type != object"
    return True, "ok"


def _make_temp_image(tmp_dir: str) -> str:
    """Create a tiny 1x1 PNG so photos_upload's os.path.isfile passes."""
    path = os.path.join(tmp_dir, "smoke.png")
    # Minimal valid PNG (8-byte header + IHDR + IDAT + IEND); cheap.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    with open(path, "wb") as fh:
        fh.write(png_bytes)
    return path


def run_connector_tests(case: ConnectorCase, tmp_dir: str) -> Dict[str, Any]:
    """Returns {passed, failed, schema_count, details:[...]}"""
    details: List[str] = []
    passed = 0
    failed = 0

    # --- instantiate with mocked client (table-less; per-tool tables set
    #     in each subtest by swapping the fake service) ---
    client = _mock_client(svc_table={})
    try:
        if case.connector_cls is GooglePhotosConnector:
            # Photos's _svc() goes around .service() and uses build()
            # directly. Patch the connector's _svc method to return our
            # fake_svc so the tool exercise stays hermetic.
            connector = case.connector_cls(client=client)
            connector._svc = lambda: client._fake_svc  # type: ignore[assignment]
            # Also patch the upload HTTP POST (photos uses requests.post
            # directly for the raw bytes upload).
            import hgr.live_api.connectors.photos_connector as _p
            _orig_post = None
            try:
                import requests
                _orig_post = requests.post
                def _fake_post(*_a, **_kw):
                    resp = MagicMock()
                    resp.status_code = 200
                    resp.text = "upload-token-xyz"
                    return resp
                requests.post = _fake_post  # type: ignore[assignment]
                connector._patched_requests_post = (_orig_post,)  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            connector = case.connector_cls(client=client)
    except Exception as exc:
        return {"passed": 0, "failed": 1, "schema_count": 0,
                "details": [f"FAIL instantiate: {type(exc).__name__}: {exc}"]}

    # --- tools() ---
    try:
        schemas = connector.tools()
    except Exception as exc:
        return {"passed": passed, "failed": failed + 1, "schema_count": 0,
                "details": [f"FAIL tools(): {type(exc).__name__}: {exc}"]}
    if not isinstance(schemas, list) or not schemas:
        details.append(f"FAIL tools() returned empty or non-list: {schemas!r}")
        failed += 1
    else:
        passed += 1
        details.append(f"PASS tools() returned {len(schemas)} schema(s)")
        # Validate each
        bad = []
        for s in schemas:
            ok, reason = _is_valid_schema(s)
            if not ok:
                bad.append(f"{s.get('name', '<noname>')}: {reason}")
        if bad:
            failed += 1
            details.append("FAIL invalid schemas: " + "; ".join(bad))
        else:
            passed += 1
            details.append(f"PASS all {len(schemas)} schemas valid")

    # --- execute(name, sample_args) per ToolCase ---
    for tc in case.tools:
        # Swap the fake service to one carrying THIS tool's mock table.
        client._fake_svc = _FakeService(tc.table)
        client.service.return_value = client._fake_svc
        if case.connector_cls is GooglePhotosConnector:
            connector._svc = lambda: client._fake_svc  # type: ignore[assignment]

        args = dict(tc.args)
        # Resolve placeholder paths to real temp files for photos.
        if args.get("path") == "__TEMP_FILE__":
            args["path"] = _make_temp_image(tmp_dir)

        try:
            result = connector.execute(tc.tool, args)
        except Exception as exc:
            failed += 1
            details.append(
                f"FAIL execute({tc.tool!r}) raised: "
                f"{type(exc).__name__}: {exc}")
            continue
        if not isinstance(result, dict):
            failed += 1
            details.append(
                f"FAIL execute({tc.tool!r}) returned non-dict: {result!r}")
            continue
        if "status" not in result:
            failed += 1
            details.append(
                f"FAIL execute({tc.tool!r}) result has no 'status' key: "
                f"{result!r}")
            continue
        passed += 1
        details.append(
            f"PASS execute({tc.tool!r}) -> status={result.get('status')!r}"
            + (f"  err={result.get('error')!r}"
               if result.get("status") == "error" else ""))

    # Restore requests.post if we patched it.
    if case.connector_cls is GooglePhotosConnector:
        try:
            import requests
            if hasattr(connector, "_patched_requests_post"):
                requests.post = connector._patched_requests_post[0]  # type: ignore[assignment]
        except Exception:
            pass

    return {"passed": passed, "failed": failed,
            "schema_count": len(schemas) if isinstance(schemas, list) else 0,
            "details": details}


# --------------------------------------------------------------------------- #
# B. Pool-watchdog test                                                       #
# --------------------------------------------------------------------------- #

def run_watchdog_test(case: ConnectorCase, tmp_dir: str) -> Tuple[bool, str]:
    """Replace the underlying API call for ONE tool with time.sleep(30)
    and assert the connector returns code=='timeout' within ~26s."""
    # Build a slow fake service: any .execute() blocks for 30s.
    slow_table: Dict[str, Any] = {}
    fake_svc = _FakeService(slow_table)

    def slow_exec(_chain: List[str]) -> None:
        time.sleep(30.0)

    fake_svc._before_exec = slow_exec  # noqa: SLF001

    client = _mock_client(svc_table={})
    client._fake_svc = fake_svc
    client.service.return_value = fake_svc

    if case.connector_cls is GooglePhotosConnector:
        connector = case.connector_cls(client=client)
        connector._svc = lambda: fake_svc  # type: ignore[assignment]
        # Make the requests.post slow instead (photos timeout fires INSIDE
        # the executor too) — use both for belt-and-braces.
        try:
            import requests
            def _slow_post(*_a, **_kw):
                time.sleep(30.0)
                resp = MagicMock()
                resp.status_code = 200
                resp.text = "tok"
                return resp
            connector._orig_post = requests.post  # type: ignore[attr-defined]
            requests.post = _slow_post  # type: ignore[assignment]
        except Exception:
            pass
    else:
        connector = case.connector_cls(client=client)

    # Find the right sample args.
    tc = next(t for t in case.tools if t.tool == case.timeout_tool)
    args = dict(tc.args)
    if args.get("path") == "__TEMP_FILE__":
        args["path"] = _make_temp_image(tmp_dir)

    t0 = time.monotonic()
    try:
        result = connector.execute(case.timeout_tool, args)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return False, (f"watchdog raised {type(exc).__name__}: {exc} "
                       f"after {elapsed:.1f}s")
    elapsed = time.monotonic() - t0

    # Restore requests.post if we patched it.
    if case.connector_cls is GooglePhotosConnector:
        try:
            import requests
            if hasattr(connector, "_orig_post"):
                requests.post = connector._orig_post  # type: ignore[assignment]
        except Exception:
            pass

    # Each connector ceiling is 25s; allow up to 28s wall to absorb
    # thread-pool scheduling jitter on Windows.
    if elapsed > 28.0:
        return False, (f"too slow: {elapsed:.1f}s "
                       f"(result={result!r})")
    if not isinstance(result, dict):
        return False, f"non-dict result: {result!r}"
    if result.get("code") != "timeout":
        return False, (f"missing code=='timeout' (got code="
                       f"{result.get('code')!r}, status="
                       f"{result.get('status')!r}, result={result!r}) "
                       f"after {elapsed:.1f}s")
    return True, f"timeout fired at {elapsed:.1f}s (code='timeout')"


# --------------------------------------------------------------------------- #
# C. Phrase-routing smoke test                                                #
# --------------------------------------------------------------------------- #

def run_phrase_routing(case: ConnectorCase) -> Tuple[bool, str]:
    """Call Classifier().classify_chain() with the connector's voice
    phrase and assert at least one step has the expected tool name."""
    from hgr.live_api.planner.classifier import Classifier
    clf = Classifier()
    try:
        steps = clf.classify_chain(case.voice_phrase)
    except Exception as exc:
        return False, f"classify_chain raised: {type(exc).__name__}: {exc}"
    if not steps:
        return False, f"classify_chain returned None/empty for {case.voice_phrase!r}"
    tool_names = [s.tool for s in steps]
    if case.voice_expected_tool not in tool_names:
        return False, (f"expected tool {case.voice_expected_tool!r} not in "
                       f"steps; got {tool_names!r}")
    return True, f"matched -> {tool_names!r}"


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def main() -> int:
    import tempfile

    print("=" * 100)
    print("New Google connectors smoke-test  (contacts / forms / identity "
          "/ photos / tasks)")
    print("=" * 100)

    tmp_dir = tempfile.mkdtemp(prefix="hgr_smoke_")

    # Phase A + B per connector
    per_connector_summary: List[Tuple[str, int, int, int, str, str]] = []
    for case in CASES:
        print()
        print(f"--- connector: {case.label} ---")
        a = run_connector_tests(case, tmp_dir)
        for line in a["details"]:
            print(f"    {line}")
        print(f"    -- watchdog (slow {case.timeout_tool!r}, expect ~25s "
              "timeout):")
        ok, msg = run_watchdog_test(case, tmp_dir)
        wd_status = "PASS" if ok else "FAIL"
        print(f"    [{wd_status}] {msg}")
        per_connector_summary.append(
            (case.label, a["passed"], a["failed"],
             a["schema_count"], wd_status, msg))

    # Phase C
    print()
    print("=" * 100)
    print("Phrase-routing smoke-test  (classify_chain per connector)")
    print("=" * 100)
    phrase_results: List[Tuple[str, str, bool, str]] = []
    for case in CASES:
        ok, msg = run_phrase_routing(case)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case.label:<10} phrase={case.voice_phrase!r}")
        print(f"        expected tool: {case.voice_expected_tool!r}")
        print(f"        result:        {msg}")
        phrase_results.append((case.label, case.voice_phrase, ok, msg))

    # ---- summary -------------------------------------------------------- #
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    grand_pass = 0
    grand_fail = 0
    for label, p, f, n_sch, wd, _ in per_connector_summary:
        wd_pass = 1 if wd == "PASS" else 0
        wd_fail = 0 if wd == "PASS" else 1
        grand_pass += p + wd_pass
        grand_fail += f + wd_fail
        print(f"  [{label:<10}] schemas={n_sch}  schema+execute "
              f"PASS={p} FAIL={f}  watchdog={wd}")
    p_pass = sum(1 for _, _, ok, _ in phrase_results if ok)
    p_fail = sum(1 for _, _, ok, _ in phrase_results if not ok)
    print()
    print(f"  phrase-routing PASS={p_pass}/{len(phrase_results)}  "
          f"FAIL={p_fail}/{len(phrase_results)}")
    print()
    print(f"  TOTAL connector tests  PASS={grand_pass}  FAIL={grand_fail}")
    print(f"  TOTAL phrase-routing   PASS={p_pass}  FAIL={p_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
