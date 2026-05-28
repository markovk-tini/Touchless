# Iris — Product Requirements Document

> **Audience:** Konstantin (product owner / lead dev), AI agents
> contributing to the codebase, future collaborators.
> **Status:** living document; phases 1-5 of the planner are SHIPPED,
> memory + realtime-sync are NEXT.
> **Related docs:** `docs/IRIS_PLANNER_DESIGN.md` (engineering design),
> `CLAUDE.md` (working notes for AI agents).

---

## 1. Problem statement

Touchless users want to operate their Windows desktop, web browser, and
connected services (Microsoft 365, Google Workspace, Discord, system
controls) through natural language and gesture — without the cost,
latency, and rate-limit pauses of routing everything through one large
model.

Pre-planner state:
- Every command burned **gpt-realtime** tokens (~5–20× more expensive
  than cheap text models).
- Realtime rate-limit pauses interrupted multi-step flows mid-execution.
- Common requests like "set volume to 30" took 2–5 s through realtime
  when they could take ~100 ms locally.
- The agent had no memory: every turn started from zero.

---

## 2. Goals

| # | Goal | Measure |
|---|------|---------|
| G1 | **Fast common path** | p95 < 200 ms for known single intents |
| G2 | **Affordable mid path** | < $0.005 per multi-step request |
| G3 | **Realtime reserved** | < 20% of turns reach gpt-realtime |
| G4 | **No silent rate-limit pauses** | 429 on one lane auto-routes to another within the same turn |
| G5 | **Brain-like continuity** | Iris remembers who/what across turns and sessions |
| G6 | **Safe by default** | Destructive / outbound actions confirm before running |

---

## 3. Non-goals

- Replacing realtime entirely — voice + ambiguous queries still need it.
- Full autonomous agency — risky operations always confirm.
- Multi-user shared state — single-user, single-machine.
- PDDL-style provable correctness — best-effort with safeguards is fine.
- Cloud-sync of memory or telemetry beyond what API calls require.

---

## 4. Personas

- **Power user (Konstantin):** wants Iris to handle complex chains
  fast. Tolerates occasional confirms for safety. Expects memory of
  prior actions, preferences, and people.
- **New user:** doesn't know what Iris can do. Expects natural language
  to "just work" — discovers capabilities by trying.
- **Voice-while-busy user:** giving commands while gaming, on a call,
  or coding. Hates rate-limit pauses and modal dialogs.

---

## 5. Use cases — current status and target

| # | Use case | Status (after Phase 1-5) | Target |
|---|---|---|---|
| U1 | "Set volume to 30" | ✅ Tier 1, ~100 ms, 0 tokens | — |
| U2 | "Email dani@x saying hi" | ✅ Tier 1, ~200 ms, 0 tokens | — |
| U3 | "Find Dani's email and send him hi" | ✅ Tier 2, ~2 s, ~1K cheap tokens | — |
| U4 | "Search for AI news and summarize the top result" | ✅ Tier 2 (web_search + navigate + text + synth), ~3 s | — |
| U5 | "Send him a thank you too" (follow-up to U3) | ❌ Realtime has no context; doesn't know who "him" is | **Multi-turn coherent** (realtime-sync) |
| U6 | "What did I email Sarah about last week?" | ❌ No memory | **Memory recall** |
| U7 | "Do my morning briefing" | Realtime improvises each time | **Stored skill**, instant |
| U8 | "Send the usual email to the team" | Realtime guesses recipient + body | **Memory-resolved** |
| U9 | "Mute Discord" while voice-throttled | ✅ Tier 1 (auto-bypasses 429) | — |
| U10 | "Delete the file I just made" | Risky → confirm | ✅ Tier 1 + confirm |

---

## 6. Functional requirements

### 6.1 Routing architecture (SHIPPED — Phases 1-5)

Five tiers, walked cheapest-first:

| Tier | Implementation | Triggers | Cost |
|---|---|---|---|
| **0** command_router | Regex match against known voice commands | Direct verb match | 0 tokens |
| **1** Classifier | Regex → connector call | Strict intent + NOT multi-action | 0 tokens |
| **2** LLM planner + Executor | One cheap-LLM call → JSON Plan; local execution | Multi-action heuristic, opt-in flag, OR realtime throttled | ~1K cheap |
| **3** Synthesizer | Cheap-LLM writes the reply from step outputs | Plan.final = "synthesize" | ~500 cheap |
| **R** Realtime fallback | gpt-realtime handles everything else | None of the above caught it | 5–20K realtime |

### 6.2 Cross-cutting (SHIPPED)

- **Rate-aware scheduler** (`planner/scheduler.py`): sliding-window
  memory of 429s per lane; realtime 429 → next turn auto-opens Tier 2.
- **Plan cache** (`planner/plan_cache.py`): normalized-goal LRU with
  10-min TTL; same goal twice skips the LLM call.
- **Whole-plan confirm-gate**: any plan touching `RISKY_TOOLS` surfaces
  ONE summary dialog before running.
- **Multi-action heuristic** (`planner/triggers.py`): chain words / and-
  verb / two-verbs-with-connector → Tier 2 (also guards Tier 1 from
  silently dropping the second half of a chained request).
- **`web_search` built-in** (`web_search.py`): Google CSE → DDG
  fallback, structured `{title, url, snippet}` results.

### 6.3 Needed next (THIS PRD's roadmap)

| Priority | Feature | Why | Effort |
|---|---|---|---|
| **P0** | Realtime-session sync | U5 broken without it; one-line addition to manager | ~half day |
| **P0** | Memory layer (episodic + semantic) | U6, U8 impossible without it; biggest "brain" win | ~2 days |
| **P1** | Skills / user macros | U7; can build on top of memory's plan library | ~half day |
| **P1** | Preconditions / plan validation | Borrow from PDDL: registry says "can I run this now?" | ~half day |
| **P2** | Auto-skill discovery | If a plan recurs 3+ times → offer to save as skill | ~half day |
| **P2** | Memory UI | User can review / edit what Iris remembers | ~1 day |

### 6.4 Memory layer — design sketch (P0)

- **Storage:** SQLite at `%LOCALAPPDATA%\Touchless\memory.db`. One
  table for episodic (one row per interaction), one for semantic facts.
- **Embeddings:** OpenAI `text-embedding-3-small` (~$0.02/M tokens —
  effectively free). Cached locally in the SQLite blob.
- **Episodic schema:** `(id, ts, user_text, plan_json, outcome_summary,
  embedding)`.
- **Semantic schema:** `(id, ts, kind, key, value, source)` — e.g.
  `(person, "Dani", "dani@mangollc.org", "from-prior-email-compose")`.
- **Retrieval:** at Tier 2 planning time, cosine-search the top-3
  episodic memories matching the goal AND inject as system context.
- **Write:** after every successful turn (any tier), extract entities +
  store the interaction. Async — never blocks the user's reply.
- **Privacy:** plaintext at rest under user's profile dir. User can
  inspect/delete via the memory UI (P2). No cloud sync.

### 6.5 Realtime-session sync — design sketch (P0)

When Tier 0/1/2 handles a turn, inject a `conversation.item.create`
event into the realtime session:

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "message",
    "role": "system",
    "content": [{"type": "input_text",
                 "text": "(User said \"find Dani's email and send him hi\". Planner handled it: looked up dani@mangollc.org, drafted email to Dani.)"}]
  }
}
```

So when the user says "send him a thank you too," realtime can resolve
"him" from the injected note. Tokens added: ~50–100 per planner-handled
turn (cheap compared to the 5–20K realtime would have spent).

---

## 7. Non-functional requirements

- **Latency:**
  - Tier 0/1: p95 < 200 ms
  - Tier 2: p95 < 3 s end-to-end (plan + execute + reply)
  - Tier R: unchanged from baseline realtime
- **Cost per request:**
  - Tier 0/1: $0
  - Tier 2 + 3: < $0.005
  - Tier R: realtime billing (reserved for voice + ambiguous)
- **Reliability:**
  - One lane 429s → other lane handles within the same turn
  - ≥ 95% of common requests handled below Tier R
- **Safety:**
  - All RISKY_TOOLS operations require a confirm
  - Confirm dialog summarizes the full plan, not per-step

---

## 8. Acceptance criteria

Phase milestones for the **next** roadmap items:

### P0: realtime-session sync
- [ ] After any Tier 0/1/2 success, realtime receives a structured note
- [ ] U5 ("send him a thank you too" after a Tier 2 email lookup) works
- [ ] Session note ≤ 200 chars per turn
- [ ] Unit test: mock realtime client receives expected event

### P0: memory layer
- [ ] Episodic table populated after each successful turn
- [ ] Semantic facts extracted: people→email, common recipients,
      common file paths
- [ ] U6 ("what did I email Sarah about last week?") returns relevant
      excerpts
- [ ] U8 ("send the usual email to the team") resolves recipient + body
      from memory without LLM-from-scratch reasoning
- [ ] Memory writes never block the user-visible response
- [ ] All memory tests run offline (no live embedding calls in CI)

### P1: skills
- [ ] User can define a skill via voice ("save this as 'morning
      briefing'") or via UI
- [ ] Skill matches at Tier 0.5 (between command_router and classifier)
- [ ] Skill catalog persists across restarts

### P1: preconditions
- [ ] Registry exposes `tool.is_available()` for each connector tool
- [ ] Tier 2 validates plan steps before executing; failures explain
      which precondition was unmet ("Outlook not connected for the
      active account")

---

## 9. Open questions

- **Memory recall scope:** how aggressive should recall be? Always
  inject top-3 episodic, or only when the goal explicitly references
  past actions ("the email I sent yesterday")? — Tentative: always
  inject if cosine-sim > threshold, gated by token budget.
- **Memory privacy:** encrypt at rest, or rely on Windows file-system
  ACLs under `%LOCALAPPDATA%`? — Tentative: ACLs only; revisit if user
  installs Touchless on a shared machine.
- **Voice multi-turn:** how long does a realtime session note "live"
  in the model's context window? — Likely OK for ~10 turns at current
  prompt sizes; verify empirically.
- **Skills UX:** voice-define via "save that as <name>" vs UI list? —
  Tentative: voice for power users, UI for discoverability.
- **Auto-skill threshold:** how many recurrences before suggesting a
  skill? — Tentative: 3 within 7 days.

---

## 10. Out of scope (for the foreseeable future)

- Multi-user shared brain
- Cloud sync of memory
- Mobile / web client
- LLM fine-tuning
- PDDL-formal planning
- Replacing realtime entirely

---

## 11. Risks

- **Memory bloat:** episodic table grows unbounded. Mitigation: cap at
  N=10,000 rows, FIFO eviction; user-visible "clear memory" button.
- **Stale facts:** semantic memory says "Dani = X@Y" but Dani's email
  changed. Mitigation: timestamp facts; prefer recent; let user
  override.
- **Privacy leak via injected memory:** realtime model sees prior
  interactions. Mitigation: redact sensitive fields (passwords, OAuth
  tokens) before injection; offer per-session memory-off toggle.
- **Cheap-LLM model deprecation:** `gpt-5-mini` model name changes.
  Mitigation: `TOUCHLESS_PLANNER_MODEL` override env var (already
  in place); document fallback model.

---

## 12. Glossary

- **Tier 0–R:** the routing tiers in §6.1.
- **RISKY_TOOLS:** the set of tools that trigger the confirm-gate.
- **Connector:** an OpenClaw-style API-first adapter (Google, Microsoft
  365, etc.) registered with the ToolRegistry.
- **Realtime:** gpt-realtime, the OpenAI websocket model used for
  voice + ambiguous queries.
- **Cheap-LLM:** the text-only Chat Completions model used for plan +
  synthesis (default `gpt-5-mini`).

---

*Author: Konstantin Markov. Drafted with Claude.*
