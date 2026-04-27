---
name: halo-engine-probes
description: Use when probing the HALO engine for behavioral correctness via
  scripted SDK event streams. Provides a FakeRunner seam, event builders,
  and a thin run wrapper for deterministically driving the engine through
  specific state-machine pathways. Produces a freeform report of pathways
  probed, results, and surprises. Not for unit tests; not for live-LLM tests.
---

# HALO Engine Probes

You are probing the HALO engine for behavioral correctness across pathways
that fixed-test suites tend to miss: state-machine edges, error
classification, stream lifecycle, ordering invariants, depth enforcement,
compaction triggers. You are not writing pytest cases — you are running
ad-hoc Python scripts that drive the engine through a scripted SDK event
stream and report what you observed.

**You have wide latitude.** Read the engine deeply, think about what could
go wrong, design probes that match what the code actually does. The
"Areas worth probing" list below is a seed for ideas, not a checklist —
the engine's surface is much larger than that list. Your job is to
*think*, then drive the engine with `probe_kit` primitives and report
honestly. The probes themselves are likely to be deleted after this run;
the report is the durable artifact.

## When to use

- A reviewer asked for behavioral verification of a recent change.
- You suspect a state-machine bug you can't easily express as a unit test.
- You're validating a fix landed correctly across the public surface.
- You're exploring uncovered pathways before a release.

**Do not use this kit for:**

- Plain assertions about a single pure function — write a unit test under
  `engine/tests/unit/`.
- Verifying real-LLM behavior — write a live test under
  `engine/tests/integration/` (requires `OPENAI_API_KEY`).
- Anything that doesn't go through `stream_engine_async` /
  `run_engine_async` — that's the only public surface this kit exercises.

## What FakeRunner can and cannot do

`FakeRunner` substitutes for the OpenAI Agents SDK's `Runner`. It controls
the **LLM-side** event stream — messages, tool-call requests, tool-call
*outputs as if a tool had run*, text deltas. It does **not** invoke the
real `FunctionTool`s the engine registers.

**Consequence — the seam stops at the LLM.** Anything that requires the SDK
to actually invoke a registered tool function is unreachable from a plain
`run_with_fake` call. Two cases this matters:

1. **Trace tools** (`get_dataset_overview`, `query_traces`, etc.) — the
   `tool_call_item` event lands in the engine, but the engine never calls
   `GetDatasetOverviewTool.run`. To probe a tool's reaction to engine state,
   call the tool's `.run(...)` method directly with a constructed
   `ToolContext`.
2. **Subagents** — `call_subagent` is a `FunctionTool` whose
   `on_invoke_tool` would normally fire a fresh `Runner.run_streamed` for
   the child agent. With `FakeRunner` in place that never happens. To probe
   subagent lifecycle: call `_build_subagent_as_tool(...).on_invoke_tool`
   directly with a fake `run_state.runner` set up to emit the child's
   events. That is closer to a unit test than an end-to-end probe — and
   that's fine, just be honest about what you're probing.

If you find yourself needing FakeRunner to reach further (e.g. "I want it
to invoke registered tools"), **stop and report that**. Do not reach for
`unittest.mock` to bridge the gap. The seam being too narrow is a finding,
not a problem to silently route around.

## Hard rules

You **MUST**:

1. **Run every script you write.** No claiming "this should pass" without
   evidence. Run it. Read the output. Report what actually happened.
2. **Use `probe_kit` primitives.** `FakeRunner`, the `make_*` event builders,
   `make_default_config`, `run_with_fake`. They compose. If you think you
   need to mock something the kit doesn't provide, the seam is wrong — stop
   and report that, don't reach for `unittest.mock`.
3. **Stay read-only on `engine/engine/**`.** You are testing, not fixing.
   When a probe surfaces a bug, write down what you observed and move on.
   Do not modify production code in this pass.
4. **Print one line per check** in the format
   `PASS: <description>` or `FAIL: <description> — observed: <fact>`.
   Track failures, `sys.exit(1)` if any.
5. **Treat `TimeoutError` from `run_with_fake` as a deadlock signal.**
   It almost always means the engine hit an exception path that doesn't
   close the output bus. Report it as such, don't paper over it.
6. **Compose, don't copy.** Each new probe goes in its own
   `probe_<topic>.py` script under this directory. Do not edit any
   `example_*.py` file — those are committed exemplars.
7. **Probes are ephemeral, the report is the deliverable.** Your
   `probe_*.py` files will likely be deleted after this run. The user
   keeps a probe as an `example_*` only when it is exceptionally well
   written or surfaced a particularly subtle bug. Optimize your effort
   accordingly — the report you produce is what matters.

You **MUST NOT**:

- Skip running scripts you wrote ("looks right" is not evidence).
- Wrap engine calls in `try/except` to "make tests pass". A `FAIL` is the
  correct outcome when behavior is wrong.
- Pull in `unittest.mock`, `pytest-mock`, or any other mocking library.
- Invent new helpers when the existing primitives compose. If you find
  yourself wanting `make_subagent_response`, you probably want
  `make_tool_call(name="call_subagent", ...)` followed by
  `make_tool_output(call_id=..., output=...)`.
- Touch `tests/fixtures/`. Use `isolated_trace_copy()` so the index file
  lands in a tempdir.

## Process

1. **Research first.** This step is the most important and the one most
   likely to be skipped. Read at least:
   - `engine/main.py` (the public entrypoint and `_drive` lifecycle)
   - `engine/agents/openai_agent_runner.py` (the loop, retries, circuit
     breaker)
   - `engine/agents/openai_event_mapper.py` (what each SDK event becomes)
   - `engine/agents/agent_context.py` (compaction logic)
   - `engine/tools/subagent_tool_factory.py` (depth + subagent
     construction)
   - any module relevant to the area you're considering probing.

   Then read every `example_*.py` here — they are committed exemplars
   that show how to structure probes, how to compose `probe_kit`
   primitives, how to phrase `_check` descriptions, how to inspect
   `runner.calls`, and how to use `make_run_state` for internal probes.
   Steal patterns liberally. **They are not coverage indicators** — many
   pathways they touch are not exhaustively covered, and many areas are
   not touched at all.

2. **Think about what could go wrong.** Before writing code, list 3–5
   ways the engine could misbehave in the area you're considering. Edge
   cases nobody usually thinks of: empty inputs, `None` content, unicode
   in deltas, two tool calls in one assistant message, sentinel mid-text,
   zero-length streams, exceptions raised between events vs at the start.
   Prefer probes that target the most surprising or fragile behavior —
   the boring happy path is usually already covered by unit tests.

3. **Pick 1–3 pathways and write `probe_<topic>.py`** in this directory.
   One topic per file. Mimic an `example_*.py` structure (`_check`,
   `probe_*` functions, `main` that runs them and exits).

4. **Run it:**
   `cd engine && uv run python -m tests.probes.probe_<topic>`

5. **Read the output.** If a check failed, ask whether the FAIL is a
   true bug or a flaw in your probe. Be honest. A `FAIL` is never
   "expected" — the engine should behave correctly, and any failure is
   a bug to report.

6. **Loop.** Move to the next pathway, or stop when you've covered what
   you set out to cover (or you've found ≥3 distinct issues worth
   reporting).

## Areas worth probing

Each line points at one pathway. The agent picks freely; this list is not
exhaustive.

- **Streaming contract.** Sequencing of `AgentOutputItem` and
  `AgentTextDelta`; deltas filtered out by `run_engine_async`; deadlock
  when the driver task raises before closing the bus.
- **Final sentinel.** `<final/>` stripped from root assistant text and
  marked `final=True`; subagent's `<final/>` does **not** flag final;
  multiple sentinel-bearing messages.
- **Circuit breaker.** Retriable error → retry → success; ten consecutive
  retriable errors → `EngineAgentExhaustedError`; non-retriable error
  propagates immediately.
- **Maximum turns.** `AgentConfig.maximum_turns` should be forwarded as
  `max_turns` to `Runner.run_streamed`. Probe by inspecting
  `runner.calls[i]` — `FakeRunner` records every kwarg the engine passes.
  If `max_turns` is missing or wrong, that's a bug; report it with a
  proposed fix.
- **Compaction.** Per-turn trigger; eligibility split between text vs
  tool items; system message never compacted; rendered output uses the
  compaction summary.
- **Depth enforcement.** `maximum_depth=0` → root has no `call_subagent`
  tool; `maximum_depth=1` → depth-1 subagent has no `call_subagent`.
  Use `await make_run_state(cfg)` then call `_child_tools_for_depth(...)`
  directly — no need for a full `run_with_fake`.
- **Subagent lifecycle (unit-style only).** Cannot be probed end-to-end
  through `run_with_fake` — see "What FakeRunner can and cannot do" above.
  To probe: build a state via `make_run_state`, install a `FakeRunner` for
  the child, call `_build_subagent_as_tool(...).on_invoke_tool(None, "{}")`
  directly. Probes: failure → `SubagentToolResult(answer="Subagent failed:
  ...")`; success → answer extracted from child's final assistant message.
- **Tool dispatch (unit-style only).** Same FakeRunner constraint as
  subagents. Construct a `ToolContext`, instantiate the tool class, call
  its `.run(...)` directly. Probe each trace tool's behavior on real
  `TraceStore` state (use `make_run_state`). Probe error paths too —
  what happens when a tool's required field on `ToolContext` is missing,
  what happens with unknown trace_ids, what happens with empty filters.
- **AgentContext input handling.** No system message → engine prepends
  rendered prompt; system message at front → passed through unchanged;
  multi-message continuation preserves stable `item_id`s. Inspect via
  `runner.calls[0]["input"]`.
- **Sandbox / `run_code`.** Default venv path resolution from an installed
  vs editable install; timeout / stdout-cap / stderr-cap; network and
  filesystem-write denials. (These need `bwrap` available, and per the
  constraint above, can only be probed by calling `RunCodeTool.run`
  directly — `FakeRunner` won't dispatch the SDK tool.)

## Common assertion patterns

- **What did the engine send to the SDK?** `runner.calls[i]["input"]`,
  `runner.calls[i]["context"]`, `runner.calls[i]["starting_agent"]`. Use
  for input-shape assertions and for "was kwarg X forwarded?" checks.
- **How many times did the SDK get called?** `len(runner.calls)`. Use for
  retry-count and circuit-breaker assertions.
- **Did the right items reach the consumer?** `result.output_items`
  filtered by `.depth`, `.final`, or `.item.role`.
- **Did the engine deadlock?** `isinstance(result.error, TimeoutError)`.
  Always means the bus was never closed on an exception path. Always
  worth reporting.
- **Did the engine raise the right exception?** `isinstance(result.error,
  EngineAgentExhaustedError)` etc. (And if you see `TimeoutError` instead
  of your expected error type, that's the deadlock — see above.)

## How to write a probe

Skeleton (mimic the example):

```python
from __future__ import annotations
import asyncio
import sys

from tests.probes.probe_kit import (
    FakeRunner, make_assistant_text, make_tool_call, make_tool_output,
    run_with_fake, isolated_trace_copy, make_run_state,
)

_FAILURES: list[str] = []

def _check(condition: bool, description: str, observed: str = "") -> None:
    if condition:
        print(f"PASS: {description}")
    else:
        suffix = f" — observed: {observed}" if observed else ""
        print(f"FAIL: {description}{suffix}")
        _FAILURES.append(description)

async def probe_<thing>() -> None:
    runner = FakeRunner(...)  # script the events
    result = await run_with_fake(runner)
    _check(result.error is None, "<thing>: completed without error",
           observed=f"error={type(result.error).__name__ if result.error else None}")
    # more checks...

async def main() -> int:
    await probe_<thing>()
    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) failed:")
        for d in _FAILURES:
            print(f"  - {d}")
        return 1
    print("\nAll checks passed.")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

Conventions for `_check` descriptions: `"<probe topic>: <observation>"`.
Keep them short and grep-able. The `observed` argument is for the *fact*
that contradicted your expectation — type names, counts, sequence numbers.

For probes that need to inspect engine internals (tool list construction,
direct tool/subagent calls), skip `run_with_fake` and use:

```python
state = await make_run_state(make_default_config(maximum_depth=1))
state.runner = FakeRunner(...)  # if needed
# now call _child_tools_for_depth(...), or
# _build_subagent_as_tool(...).on_invoke_tool(...), etc.
```

## Reporting back

Your report exists so a developer can turn it into commits. Structure it
so each failure carries everything needed to act on it.

### Section 1 — What I did

One short paragraph. Pathways probed (bullet them), probe filenames,
totals (`N PASS / M FAIL / K not-reachable across all probes`).

### Section 2 — Findings

For **every FAIL**, one structured entry in this exact shape:

```
### FAIL: <one-line symptom>
- **Where:** `engine/path/to/file.py:LINE` (or `LINE-LINE` for a range)
- **What I observed:** the actual behavior — error type, count, sequence,
  output value. Cite the probe's `_check` description if helpful.
- **Why it's wrong:** one or two sentences pointing at the engine's
  intended behavior (the spec, a sibling code path that does it right,
  or "this contradicts the docstring at file:line").
- **Suggested fix:** describe the change in 1–3 sentences. Name the
  function, the line, and the operation ("wrap `runner.run(...)` in
  `try/finally` so `output_bus.close()` always runs"). If you can write
  the fix as a 5-line diff sketch, do that. Don't paste the full new
  function.
- **Confidence:** high / medium / low. Be honest. "Low" is acceptable
  and useful — it tells the reader to verify before applying.
```

For **PASS** and **SURPRISE** items, one-line bullets are fine. Group
them under `### PASS notes` and `### Surprises` if there's anything worth
calling out (e.g. behavior that passed but seems fragile).

### Section 3 — Framework feedback (optional)

Anything in `probe_kit` or this README that got in your way: missing
helper, unclear instruction, a pathway you tried to probe but couldn't
reach with the current seam. One bullet per item.

### Hard rules for the report

- **Do not edit production code as part of this pass.** Your job is to
  surface bugs with enough information that someone else can fix them in a
  separate pass.
- **Do not silently swallow failures.** Every `FAIL` from your scripts
  appears as a structured entry above. If you delete a check, say so.
- **Do not invent severity ratings.** Confidence is a verifiable claim
  about your own analysis; severity is for the reader to assign.
- **Do not soften findings to be polite.** If `<final/>` is broken,
  write "broken", not "could be improved".

## Red flags

| Thought | Reality |
|---|---|
| "I can predict the outcome from reading the code" | Run it. Code reading misses bugs that running surfaces. |
| "FakeRunner doesn't expose this — let me patch the engine" | If `FakeRunner` can't drive the pathway, the seam is wrong. Stop and report that as a finding. |
| "I'll wrap this in `try/except` to make it pass" | A `FAIL` is the correct outcome when the engine misbehaves. Report it. |
| "This is fast — let me put it in `tests/unit/`" | Wrong directory. `tests/unit/` is for pure unit tests of individual modules. Probes go *here*. |
| "I'll mock the OpenAI client too" | Don't. The seam is `RunnerProtocol`. If you need finer control, ask whether the seam should expose more. |
| "I see `loguru` warnings — let me silence them" | Leave them. They show what the engine is doing. |
| "I don't need `isolated_trace_copy` — I'll just point at the fixture" | The index file gets written next to the trace. Pointing at `tests/fixtures/` pollutes the repo. Always use `isolated_trace_copy`. |

## When the kit can't reach a pathway

You will find pathways that `FakeRunner` cannot drive on its own (subagent
lifecycle, real tool dispatch, sandbox execution under `run_code`). The
right responses are:

1. **Drop the sub-check.** Don't try to fake your way around the gap.
2. **Report the limitation as a finding** in your final report — it's as
   valuable as a `FAIL`. "Pathway X not reachable through current seam"
   tells the next reader something real.
3. **Optionally, write a unit-style probe** by calling the relevant
   internal directly (`make_run_state` → `_child_tools_for_depth`,
   `_build_subagent_as_tool().on_invoke_tool`, `Tool.run(...)`). Be honest
   in the probe's name and docstring that it's unit-style, not end-to-end.

Do **not**: pull in `unittest.mock`, build a parallel fake of the SDK
beyond what `probe_kit` exposes, or modify production code to "expose more
seam". If the seam needs widening, that's a design decision for the user,
not the probe author.

## Examples

These are committed exemplars. They show the shape of good probes and
demonstrate the `probe_kit` primitives in use. **Do not edit any of
them.** They do not represent coverage — many other pathways remain
unprobed.

- `example_circuit_breaker.py` — `OpenAiAgentRunner` retry classification:
  baseline success, retriable error → retry → success, ten consecutive
  retriable errors → `EngineAgentExhaustedError`, non-retriable error
  propagates immediately. Demonstrates `FakeRunner(*programs)` with mixed
  exception/event lists.
- `example_streaming_contract.py` — `AgentTextDelta` vs `AgentOutputItem`
  separation, monotonic `sequence`, `run_engine_async` filtering deltas,
  and a probe that surfaces the deadlock when the driver raises before
  closing the bus.
- `example_final_sentinel.py` — `<final/>` stripping and `final=True`
  flagging across single-message, multi-message, mid-text, sentinel-only,
  and double-sentinel cases.
- `example_agent_context_input.py` — `AgentContext.from_input_messages`
  behavior across no-system-message, caller-supplied system, multi-turn
  continuation, and system-only edge case. Inspects what the engine
  actually sends to the SDK via `runner.calls[0]["input"]`.
- `example_depth_enforcement.py` — `_child_tools_for_depth` construction
  at `maximum_depth=0/1/2`, plus an end-to-end check via
  `runner.calls[0]["starting_agent"].tools`. Demonstrates `make_run_state`
  for direct internal inspection without `run_with_fake`.
