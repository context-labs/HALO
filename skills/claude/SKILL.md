---
name: halo-loop
description: Use HALO Engine as a diagnostic aid to iteratively improve an agent harness. Engine surfaces patterns from traces; Claude maps findings to code, makes the smallest sensible change, re-runs, and measures.
---

# HALO Loop — diagnostic-led harness improvement

## When to invoke this skill

Use this skill when:

- The user is iterating on an agent harness that emits HALO-shaped JSONL trace files (e.g. an OpenAI Agents SDK app wired with `tracing.py`, an AppWorld run, or any harness that follows `docs/integrations/openai-agents-sdk.md`).
- The user wants to find and fix harness-level issues (prompts, tool descriptions, retry logic, error handling, configuration) — *not* model-level issues.
- The user explicitly says "use HALO", "improve the harness", "look at the traces and figure out what's wrong", or similar.

Do **not** invoke this skill for:

- General agent-development questions unrelated to a specific trace dataset.
- Cases where there are no captured traces yet — you'd need to set tracing up first (see *Setup* below) and run the harness.
- Cases where the user wants you to fix an LLM model (HALO is for harness, not model weights).

## Mental model — what HALO Engine is and isn't

**Halo Engine is a trace-exploration runtime.** It's an LLM agent that has tools to read a JSONL trace dataset and answer questions about it. It is *not* a code-modification tool, *not* a fix proposer, and *not* a verifier.

The engine's hard-coded root system prompt (`engine/agents/prompt_templates.py`) starts with:

> "You are the root agent in the HALO engine. You explore OTel trace data using the tools the runtime provides."

Treat its output as **trace evidence**, not as a directive. It can identify patterns, cite trace_ids, surface error strings, count failure modes — that's what it's for. It cannot see your code; if it names a file path or claims a constraint is missing, **verify before acting**.

The HALO loop is therefore a two-actor system:

| Actor | Role | Knows about |
|---|---|---|
| **HALO Engine** | Diagnostic — answers questions about traces. | The traces only. |
| **You (Claude)** | Executor — maps findings to actual code, makes the change, verifies, measures. | The repo, the engine's findings. |

You should never ask HALO Engine to propose code changes. You should ask it questions whose answers help you decide where to look in the code.

## Setup — getting HALO Engine ready to use

Prereqs:

- A clone of `https://github.com/inference-net/HALO` somewhere on disk. The path is referred to below as `$HALO`.
- Python 3.10+, `uv` for environment management.
- An `OPENAI_API_KEY` env var (the engine itself uses an LLM to drive tool calls).

Install the engine + CLI once:

```bash
cd $HALO
uv sync                       # creates .venv with halo-engine (CLI bundled)
```

That's it for the engine side. The CLI entry point is `halo` (registered by `pyproject.toml`).

## Trace-format prerequisites — what the harness must produce

The engine reads a single JSONL file. Each line is one OTel-shaped span as a JSON object. The format is documented at `$HALO/docs/integrations/openai-agents-sdk.md`. Required top-level fields:

```
trace_id, span_id, parent_span_id, name, kind, start_time, end_time,
status: { code, message },
resource: { attributes: {...} },
scope: { name, version },
attributes: { ... }
```

Required `attributes.*` keys for HALO indexing to work:

- `inference.export.schema_version` (always `"1"`)
- `inference.project_id` — used by the engine's index/filter
- `inference.observation_kind` — one of `AGENT`, `LLM`, `TOOL`, `CHAIN`, `GUARDRAIL`, `SPAN`

Optional but valuable for richer analysis:

- `inference.llm.model_name`, `inference.llm.input_tokens`, `inference.llm.output_tokens` (LLM spans)
- `tool.name`, `input.value`, `output.value` (TOOL spans)
- `agent.name` (AGENT root span)

**Important format constraints:**

1. **Plain JSONL only — not gzip.** The engine opens the file with `Path.open("rb")` and parses one span per line. Gzip files won't work; `gunzip` first.
2. **Append mode accumulation:** If the harness uses HALO's vendored `tracing.py`, the file is opened in append mode. Repeated runs *stack* in the same file. Before a fresh diagnostic pass, clean stale outputs (e.g. `task clean:run-outputs` in the AppWorld demo, or just delete the file and re-run the harness).
3. **Parallel runs produce per-process files** (`traces-p0.jsonl`, `traces-p1.jsonl`, ...). Merge them into a single `traces.jsonl` before invoking the engine — the AppWorld demo has a `task traces:merge` task that does this; for ad-hoc cases `cat traces-p*.jsonl > traces.jsonl` works fine since the engine indexes by `trace_id` and ignores order.

If the user has a harness that doesn't yet emit traces, the canonical setup pattern is in `$HALO/demo/openai-agents-sdk-demo/`:

- Copy `tracing.py` verbatim into the harness's source tree.
- At startup, before any `Agent(...)` is constructed, call:
  ```python
  import agents
  agents.set_trace_processors([])           # clear SDK default OpenAI uploader
  processor = setup_tracing(service_name="myapp", project_id="myproj")
  ```
- Call `processor.shutdown()` in a `finally` block before exit so the file flushes.

For non-SDK harnesses, refer to the schema in `$HALO/engine/traces/models/canonical_span.py` — the engine accepts any JSONL that matches that shape.

**Don't accidentally break the mechanism:**

- Don't replace `set_trace_processors([])` with `set_tracing_disabled(True)` — that disables HALO's processor too.
- Don't strip the `inference.*` keys from `tracing.py` — those are what the engine indexes on. If a span loses `inference.project_id`, the engine still parses it (the model is `extra="allow"`) but filters and overviews silently drop it.
- Don't use `OPENAI_AGENTS_DISABLE_TRACING=1` — same effect as `set_tracing_disabled`. The engine's default OpenAI-dashboard uploader is what we want to disable, *not* tracing as a whole. The pattern is "clear default, add HALO's processor on top".

For deeper format details: `$HALO/docs/integrations/openai-agents-sdk.md` and `$HALO/engine/traces/models/canonical_span.py`.

## The HALO loop — the workflow you should follow

Once the harness emits valid traces:

### Step 1 — Capture baseline

Run the harness once and capture eval results. Note the score so you can measure improvement later. If running an existing demo (e.g. AppWorld), the run command typically also produces an evaluation report.

### Step 2 — Diagnose with halo

Invoke the CLI with a **question**, not a command:

```bash
cd $HALO
uv run halo /absolute/path/to/traces.jsonl \
    --prompt "What are the most common failure modes across the failed traces? For each, give me trace_id evidence and the precise error string." \
    --model gpt-4.1-2025-04-14 \
    --max-turns 15
```

Important parameters:

- **`TRACE_PATH`** (positional, required): absolute path to a JSONL trace file. Must exist; must be plain JSONL.
- **`--prompt` / `-p`** (required): the question. Frame as a *diagnostic question*, never as "propose a fix".
- **`--model` / `-m`** (default `gpt-5.4-mini`): the engine's driver model. Bump to `gpt-4.1-2025-04-14` (1M context) for large traces — multiple runs at default model OOM on AppWorld-sized data even with attribute truncation. The engine still hits context limits when conversation history accumulates across many tool calls; bigger model = more headroom.
- **`--max-turns`** (default `8`): how many reasoning turns the root agent gets. 12-25 is reasonable for deep analyses; lower numbers force tighter answers.
- **`--max-depth`** (default `1`): subagent recursion depth. Default is fine in most cases.
- **`--max-parallel`** (default `2`): concurrent subagent cap. Default is fine.
- **`--instructions`**: override the default tool-discipline instructions. Don't override unless you're sure — the defaults already steer toward `get_dataset_overview` → `query_traces` → `search_trace` → `view_spans` rather than fabricating trace_ids or dumping huge `view_trace` results.

### Step 3 — Verify the engine's claims against the actual repo

The engine's tool descriptions and instructions don't let it touch your filesystem. **Anything it says about a file path, line number, or "this constraint is missing" is dead reckoning from convention** — verify before treating it as truth.

After a diagnostic answer comes back, do one or more of:

```bash
# Find the file the engine alluded to:
rg -l "page_limit" $HALO/demo/appworld/src/appworld/apps/spotify/

# Check whether the constraint is actually missing:
rg "le=20|maximum.*20" $HALO/demo/appworld/src/appworld/apps/spotify/apis.py

# Check whether the prompt says what the engine claimed:
rg "page_limit" $HALO/demo/appworld/experiments/prompts/function_calling_agent/
```

If the engine said file `X.json` exists and `rg`/`Read` show it doesn't, that's a hallucinated path — re-grep based on the *concept* (the tool name, the error string) rather than the engine's path guess.

### Step 4 — Form a hypothesis, make the smallest change

Based on verified evidence:

- Identify the precise file(s) involved.
- Choose the minimal-blast-radius edit (description text, a single prompt line, a config value, an error-recovery branch).
- Make the change. **Do not** make sweeping refactors based on a single diagnostic pass.

### Step 5 — Re-run + re-diagnose

Re-run the harness, capture new traces, ask the same diagnostic question to compare. Look at the eval delta numerically; look at whether the cited error pattern reduced in frequency.

If it did, ship. If it didn't, you misread the evidence — go back to step 2 with a sharper question.

## Diagnostic-mode discipline — questions to ask vs. avoid

**Good prompts** (HALO answers these well):

- "How many traces had at least one TOOL span with `status.code == STATUS_CODE_ERROR`? List the top 5 by error count and cite trace_ids."
- "Across the failed traces, what's the most frequent literal error string in `output.value` of TOOL spans? Group near-duplicates."
- "Look at trace `<id>`. What was the tool sequence? Did the agent ever recover after a tool error?"
- "Which tool names appear in `function.*` spans most often, and which of those are most associated with error outcomes?"
- "Compare two specific traces — one that completed cleanly and one that hit max-turns. What did the failing one do differently after turn 5?"
- "Are there traces where the same tool was called more than 3 times with similar arguments? Cite trace_ids."

**Bad prompts** (HALO will answer but the result is unreliable):

- "What change should I make to fix this?" — engine has no repo access; will hallucinate file paths.
- "Write a patch for `prompts/instructions.txt`." — same, plus you should write the patch yourself once you've verified the diagnosis.
- "Refactor the harness to be more robust." — too vague; engine will give generic advice.
- "Is the API predictor under-fetching?" — leading; the engine will agree because the question implies it. Ask "How often does the trace contain a tool call to a tool that wasn't in the predicted list?" instead.

The pattern: ask about the **trace data**, not about the **code**. Use the answer to navigate the code yourself.

## Halo Engine tool inventory — what the engine can do internally

Knowing the engine's own tool surface helps you frame prompts that route to the cheapest tools:

| Tool | What it does | When the engine uses it |
|---|---|---|
| `get_dataset_overview` | High-level stats: total_traces, total_spans, model names, agent names, first 20 trace_ids. | Always called first. |
| `query_traces` | Paginated `TraceSummary` listing with filters (`has_errors`, model names, etc.). | When the engine needs more trace_ids than the overview's sample. |
| `count_traces` | Cheap count under filter, no materialization. | Quick "how many failed?" type questions. |
| `view_trace(trace_id)` | All spans of one trace, with per-attribute payloads head-capped at ~4KB and noisy OpenInference flat projections (`llm.input_messages.<i>.*`, etc.) dropped. **If the truncated total still exceeds ~150K chars, the spans are dropped and an `oversized` summary is returned instead** (counts, span_size min/median/max, `top_span_names`, `error_span_count`, an explicit recommendation to switch tools). | For small traces (≤ ~50 spans). On `oversized` response, switch to `search_trace` + `view_spans`. |
| `view_spans(trace_id, span_ids)` | Read only specific spans (same truncation as `view_trace`). | After `search_trace` has located interesting spans in a long trace. |
| `search_trace(trace_id, pattern)` | Substring match against the raw on-disk JSON; returns matching spans (truncated). Pattern can target attribute keys, error strings, tool names. | For surgical introspection of large traces. |
| `synthesize_traces` | LLM-driven cross-trace summary tool. | Sometimes called for higher-level rollups. |
| `get_context_item` / `run_code` / `call_subagent` | Less common; depth-1 sub-agent spawning, code execution in a sandbox. | Mostly for advanced workflows. |

If the engine OOMs on a `view_trace` of a long trace, that's a sign the trace is unusually big (heavy `mcp_tools` listings, very long agent runs). Use `search_trace` + `view_spans` instead — explicitly nudge in your prompt.

## Common gotchas

- **Trace file is gzipped.** Engine errors out reading. Run `gunzip traces.jsonl.gz` first.
- **Trace file accumulated multiple runs.** `get_dataset_overview` reports `total_traces` higher than expected. Clean and re-run the harness.
- **Engine OOMs on a giant trace.** First, check whether `view_trace` on a single trace was the cause — if so, the engine should now have received an `oversized` summary and routed to `search_trace`/`view_spans` automatically. If OOM persists despite that, conversation history (many tool calls accumulating) is the cause; switch to `gpt-4.1-2025-04-14` or split the question into multiple smaller CLI invocations. The default `gpt-5.4-mini` is acceptable for normal datasets but tight for AppWorld-sized data.

- **`view_trace` returned `oversized` instead of `spans`.** This is intentional and means the trace would have exceeded the per-call ~150K char budget. The summary contains everything needed to plan the next call: `top_span_names` (which span names appeared and how often), `span_size_min/median/max`, `error_span_count`, and an explicit recommendation. Pick a span name or substring from `top_span_names` and call `search_trace`, then `view_spans` on the matched span ids. Never retry `view_trace` on the same id expecting a different result.
- **Engine returns a file path you can't find.** Hallucinated; treat as a *direction* not a destination. Grep the repo for the *concept*.
- **Engine's tool result includes a `__halo_dropped_flat_projections` marker.** That just means the OpenInference per-message flat projections were dropped to keep the span size bounded. The same content is still available in `llm.input_messages` / `llm.output_messages` JSON-blob attributes (head-capped at 4KB).
- **Engine reasoning starts circular** (calling the same tool with similar arguments). Reduce `--max-turns`, simplify the prompt, or split the question into two CLI calls.
- **Trace has no `inference.project_id`.** Engine still parses but indexing/filtering becomes lossy. Fix the harness's `setup_tracing` call to pass an explicit `project_id`.

## Where to find more information

- `$HALO/README.md` — top-level project overview.
- `$HALO/docs/integrations/openai-agents-sdk.md` — canonical trace format spec; how to wire a harness from scratch.
- `$HALO/docs/engine-architecture-plan.md` — engine's design (input/output contracts, package layout).
- `$HALO/engine/tools/trace_tools.py` — tool class definitions (with up-to-date descriptions visible to the engine LLM).
- `$HALO/engine/traces/models/canonical_span.py` — `SpanRecord` schema.
- `$HALO/engine/traces/models/trace_query_models.py` — argument and result models for every trace tool (useful for understanding what each tool can take/return).
- `$HALO/halo_cli/main.py` — full set of CLI options and the default agent instructions.
- `$HALO/demo/openai-agents-sdk-demo/` — minimal working harness reference (single-agent, three file tools).
- `$HALO/demo/appworld/` — a real benchmark integration with parallel runs, eval reports, and `HALO_PATCH.md` documenting the per-harness changes needed; useful as a worked example of the loop.
- `$HALO/demo/appworld/HALO_PATCH.md` — concrete patches a harness might need to emit valid traces.

## Quick reference

```bash
# One-time install
cd $HALO && uv sync

# Diagnose an existing trace file
cd $HALO
uv run halo /abs/path/traces.jsonl \
    --prompt "Your diagnostic question, framed as data inquiry." \
    --model gpt-4.1-2025-04-14 \
    --max-turns 15

# After answer comes back: verify in the repo
rg "<the literal string the engine cited>" /path/to/harness
# Read the actual file. Form a hypothesis. Make the minimal change.
# Re-run harness, re-run halo, compare.
```

Stay diagnostic in your prompts. Stay surgical in your edits. Verify before you patch.
