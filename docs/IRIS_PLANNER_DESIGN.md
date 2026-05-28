# Iris Planner — design & build plan ("JARVIS-level" decision engine)

Status: **DESIGN (recorded before implementation)**. Author: Konstantin Markov.

Goal: make iris decide *the cheapest reliable way* to do each request and run
multi-step tasks with the **fewest possible model calls**, so it's fast and
cost-efficient instead of taking one rate-limited gpt-realtime turn per action.

---

## 1. The core problem (why we're building this)

Today gpt-realtime drives everything **turn-by-turn** (ReAct style): it emits
one tool call, we run it, it emits the next, etc. Every turn re-processes ~80
tool schemas + history, and **every turn — even a connector call — costs a
model turn**, so multi-step tasks pause repeatedly on the realtime
tokens-per-minute limit.

Key facts that shape the design:
- **Layer 0 (deterministic command_router)** uses **no model** → truly free.
- **Connectors** are deterministic functions — they can be **called directly
  without the model** once we know the tool + args.
- The expensive thing is the **model**, especially **gpt-realtime** (low TPM).
- A **cheap text LLM** (Chat Completions, e.g. a mini model) has far higher
  rate limits and ~100x lower cost than realtime — ideal for *planning* and
  *summarizing*, reserving gpt-realtime for actual live voice conversation.

## 2. The chosen architecture: deterministic-first + Plan-and-Execute

Validated against the known agent patterns:

| Pattern | LLM calls for an N-step task | Fit |
|---|---|---|
| **ReAct** (today) | ~N (one per step) + overhead | simple but expensive — what hurts now |
| **Plan-and-Execute** | 1 plan + 1 synthesis (+ optional replans) | far fewer calls |
| **LLMCompiler** | 1 plan (a DAG) + parallel execution + 1 synthesis | most efficient for parallelizable tasks |
| **Cost cascade** | try cheapest layer first, escalate | minimizes *which* model is used |

**Our engine = cost-cascade routing + Plan-and-Execute with an LLMCompiler-style
step graph + a final synthesis step.** Concretely, for each request:

1. **Normalize** the text (lowercase, strip fillers, keep entities/URLs/names).
2. **Layer 0 — deterministic** (`command_router`). If it matches a known
   command → run it, **done, 0 model tokens** (open app, play/pause, volume…).
3. **Cheap classify** — a fast local classifier maps obvious single-intent
   requests to a connector + slots (e.g. "set volume 30" → `volume_set`). If
   high-confidence and slots are unambiguous → **call the connector directly,
   no model**.
4. **Plan** — for anything else, **one cheap-LLM call** produces a structured
   **Plan**: an ordered/▸parallel list of steps, each `{tool, args, layer,
   depends_on, needs_result}`. The plan can reference earlier steps' outputs.
5. **Execute the plan** — the **planner runs the steps itself**:
   - deterministic/connector/Touchless steps → call directly (free/cheap),
   - independent steps in **parallel**,
   - screen reads via the cached unified `ScreenReader` (no re-OCR if unchanged),
   - **no model turn per step.**
6. **Synthesize** — if the user needs a natural-language answer over gathered
   data (e.g. "summarize these search results"), **one cheap-LLM call** turns
   the collected step outputs into the answer.
7. **gpt-realtime** is used **only** for live, conversational/multimodal voice
   mode — it's no longer the per-action executor.

Net: an N-step task costs **~1 plan + ~1 synthesis** cheap-LLM calls (not N
realtime turns), plus free deterministic execution. That is the efficiency win.

## 3. Worked examples (the decision flow in action)

- **"open spotify"** → Layer 0 → `open_app`. *0 model calls.*
- **"set volume to 30"** → cheap classify → `volume_set(30)`. *0 model calls.*
- **"email dani@x saying hi"** → classify (has recipient+body) → `outlook_compose`
  directly, or 1 plan call if ambiguous. *0–1 calls.*
- **"read/summarize my unread emails"** → plan: [open Outlook (Touchless),
  read_screen(scroll) (local OCR — free), synthesize summary (1 cheap-LLM
  call)]. *1 model call total, no realtime.*
- **"search latest AI news and put a summary in a OneNote called Demo"** → plan:
  1. web search (Google Custom Search API → results+links) — *free/cheap*,
  2. (optional) iris reads the top articles via the page — *local*,
  3. `onenote_create('Demo')` (connector) — *free, runs even while step 4 waits*,
  4. **synthesize** the summary with sources — *1 cheap-LLM call*,
  5. paste summary into the OneNote page (connector) — *free*.
  Steps 3 runs immediately; 4 is the only model call; 1–2 fetch content. The
  page is prepped while the model step is queued — exactly the "do connector
  prep while rate-limited" behavior.
- **"message Vesselin on Teams saying hi"** (personal) → plan: [open Teams
  (Touchless), read_screen (local), click_type send (local), confirm]. *0–1
  model calls.*

## 4. Rate-aware scheduling (work *around* the limit)

- The scheduler runs **free/local/connector steps immediately**, regardless of
  model rate state (open the app, create the doc, set the title, fetch text).
- **Model steps (plan / synthesis)** go through a small queue with
  backoff that respects the API's `rate_limits` signals; if rate-limited, free
  steps keep going and the model step fires when budget returns.
- **Never block a free step on a paused model step.** This is the user's
  insight: prep with connectors while the model is cooling down.

## 5. Models used (and why)

- **Planner + Synthesizer → a cheap TEXT LLM** (Chat Completions, mini model).
  High TPM, ~100x cheaper than realtime, no audio. Behind the existing
  iris-backend broker or the user's key.
- **gpt-realtime → live voice conversation only** (assistant_mode/realtime_mode),
  invoked by the planner when the task is genuinely conversational.
- **Local-only fallback**: the existing `command_router` + classifier handle
  the common commands with **no LLM at all**; if no model is available, iris
  still does deterministic + connector + screen-read tasks.

## 6. Modules (extend `live_api/`, do NOT fork a parallel tree)

Reuse the working systems; add a thin orchestration layer:

```
src/hgr/live_api/planner/
    __init__.py
    normalizer.py        # text normalization (reuse command_router helpers)
    classifier.py        # deterministic + cheap intent->tool+slots
    plan.py              # Plan / Step / StepResult dataclasses
    planner_llm.py       # one cheap-LLM call -> structured Plan (JSON schema)
    executor.py          # runs the Plan: tools direct, parallel, cached screen
    scheduler.py         # rate-aware step scheduling + model-call queue
    synthesizer.py       # one cheap-LLM call -> natural-language answer
    orchestrator.py      # ties it together; the public entrypoint
```
Reuses: `tool_registry`/`connectors` (execute tools), `cost_policy` (levels),
`screen_reader` (cached unified ScreenContext), `command_router` (Layer 0),
`safety` (confirmations). gpt-realtime stays in `realtime_client`.

## 7. Data model (sketch)

```
Step:   id, tool, args(dict, may ref {step:N.output}), layer(cost level),
        depends_on[list], needs_user_confirm(bool), description
Plan:   goal, steps[list], final("synthesize"|"return"|"speak")
StepResult: step_id, status, output, cost_level, error
```
Plan is produced as **strict JSON** (function-call/JSON-schema constrained) so
it's machine-executable, never free-text.

## 8. Safety

`safety.py` gates irreversible/outward steps (send email/message, delete,
purchase, shell that modifies, share private data) with a single confirmation
*before* executing that step — reusing the existing confirm callback. Safe
routine steps (open app, read, set volume, click visible button) run without
prompting.

## 9. Integration & rollout (incremental, behind a flag)

1. **Feature flag** `TOUCHLESS_IRIS_PLANNER` (default off at first) so current
   behavior is untouched until proven.
2. Text/command path routes through the **orchestrator**; realtime voice stays
   as-is initially.
3. The orchestrator emits the same `tool_event` + `routing_decision` logs so the
   UI badges (⚡/🔌/👁) and cost log keep working — plus a new "PLAN" pill.
4. Preserve every existing behavior; the planner is additive.

## 10. Phased build (each phase shippable + tested)

- **Phase 1 — foundations:** `plan.py` data model, `classifier.py`
  (deterministic + slot extraction for the common connectors), wire Layer 0 →
  classify → direct connector for high-confidence single intents. Tests:
  routing decisions, no model used for known commands. *Biggest immediate win,
  lowest risk.*
- **Phase 2 — LLM planner:** `planner_llm.py` (cheap-LLM → JSON Plan) +
  `executor.py` (run steps, deps, parallel, cached screen). Tests with a mocked
  LLM producing canned plans; assert correct tool sequence, no per-step model
  calls.
- **Phase 3 — synthesis + scheduler:** `synthesizer.py` + rate-aware
  `scheduler.py` (free steps proceed; model steps queue/backoff). Tests for
  ordering + rate handling with a fake clock.
- **Phase 4 — integration:** `orchestrator.py` + manager wiring behind the
  flag; the search→summarize→OneNote and read-email flows end-to-end.
- **Phase 5 — web search:** Google Custom Search API connector for results +
  iris reading/summarizing the article pages (per user's choice).

## 11. Acceptance tests (carried from the JARVIS spec)

1. `open spotify`, `pause`, `volume 30` → **0 model calls** (Layer 0/connector).
2. `summarize my unread emails` → open Outlook + read_screen + **1** synthesis
   call; no realtime; no per-step pauses.
3. `search X and save a summary to OneNote` → connector prep runs while the
   single synthesis model-step is queued; result pasted by connector.
4. Multi-step tasks use **≤2** model calls (plan + synthesize), never N.
5. Risky steps (send/delete) confirm once before running.
6. Every step logs tool/cost_level/why; a PLAN is logged for multi-step tasks.
7. Feature flag off ⇒ behavior identical to today.

## 12. Risks & mitigations

- **Bad plans / wrong tools** → strict JSON schema + a validation pass + the
  ability to **replan once** on step failure; fall back to today's
  realtime-driven path if planning fails.
- **Latency of an extra planning call** → only for non-deterministic requests;
  single commands skip it entirely (Layer 0/classify).
- **Cheap-LLM availability/key** → broker via iris-backend; local fallback keeps
  deterministic + connector + screen tasks working with no LLM.
- **Scope creep / breaking working flows** → feature flag + phase gating + tests
  per phase; never remove the existing path until the planner is proven.

## 13. Why this is the efficient choice (double-check)

- **Deterministic-first** = the majority of commands cost **0 tokens**.
- **Plan-and-Execute (LLMCompiler-style)** = multi-step tasks cost **~2 cheap
  calls**, not N realtime turns — the documented win over ReAct.
- **Cheap text LLM for plan/synthesis, realtime only for voice** = the heavy
  rate limit (realtime TPM) is hit far less.
- **Rate-aware scheduling** = free work proceeds during model cooldown.
- **Caching + direct tool execution** = no redundant screenshots or per-step
  model turns.
This matches state-of-the-art efficient tool-agent design while reusing all the
connectors/screen/cost infrastructure already built.
