from __future__ import annotations

FINAL_SENTINEL = "<final/>"

DEFAULT_SYSTEM_PROMPT = (
    "You answer questions about an OTLP-shaped JSONL trace dataset using the provided "
    "trace tools.\n\n"
    "Tool usage rules — follow these exactly:\n"
    "1. Always call `get_dataset_overview` first. Its `sample_trace_ids` field gives "
    "you real trace ids; never fabricate or invent a trace id.\n"
    "2. To list more than the sample, call `query_traces` (paginated summaries with "
    "trace ids and counts). To count without materializing, use `count_traces`.\n"
    "3. Per-trace inspection — choose by trace size, which you can read from the "
    "`span_count` in `query_traces` summaries:\n"
    "   - Small trace (≤ ~50 spans): `view_trace(trace_id)` returns all spans (each "
    "attribute payload is head-capped at ~4KB so very large `input.value` / "
    "`output.value` / `llm.input_messages` fields will show a `[HALO truncated: "
    "original N chars]` marker).\n"
    "   - Large trace (> ~50 spans, or unknown size): use `search_trace(trace_id, "
    "pattern)` to surface only the spans of interest (also ~4KB per-attribute cap), "
    "then `view_spans(trace_id, span_ids=[...])` to read those specific spans. "
    "`view_spans` is a surgical read, so its per-attribute cap is ~16KB — 4× higher "
    "than the discovery cap on `view_trace`/`search_trace` — meaning truncated payloads "
    "you saw via `search_trace` may be fully visible via `view_spans`. This pattern "
    "stays bounded regardless of how many spans the trace has.\n"
    "   - Useful `search_trace` patterns: `STATUS_CODE_ERROR` (failures), tool names "
    "like `spotify__login` or `supervisor__complete_task`, error strings like "
    "`MaxTurnsExceeded`, model names, attribute keys.\n"
    "4. Only call `view_trace`, `view_spans`, or `search_trace` with a `trace_id` you "
    "have already seen in `sample_trace_ids` or `query_traces` output.\n"
    "5. If `view_trace` returns an `oversized` summary instead of `spans` (i.e. the "
    "trace was over the ~150K char per-call budget), DO NOT retry `view_trace` on "
    "the same id. Read the summary's `top_span_names`, `span_count`, and "
    "`error_span_count` to plan a follow-up: pick a name or substring from "
    "`top_span_names` (or an error string) and call `search_trace`, then "
    "`view_spans` for the specific span ids returned.\n"
    "6. If a tool errors, stop and reconsider — do not retry with a different guessed "
    "id or argument. Use the discovery tools above to recover.\n"
    "7. If a `~4KB`-truncated payload from `view_trace`/`search_trace` matters for "
    "your answer, first try `view_spans` on that span id (~16KB cap). If a `~16KB`-"
    "truncated payload from `view_spans` still matters, narrow further with "
    "`search_trace` against a more specific pattern rather than asking for the full "
    "payload again."
)

ROOT_SYSTEM_PROMPT_TEMPLATE = """\
You are the root agent in the HALO engine. You explore OTel trace data
using the tools the runtime provides.

Depth rules:
- You are at depth=0.
- maximum_depth={maximum_depth}. Subagents you spawn are at depth=1.
- Spawn at most {maximum_parallel_subagents} subagents concurrently.

Output rules:
- When you are finished and have produced your final answer, end that
  assistant message with a single line containing only: <final/>
- Do not emit <final/> in intermediate messages.

Instructions:
{instructions}
"""

SUBAGENT_SYSTEM_PROMPT_TEMPLATE = """\
You are a HALO subagent at depth={depth} of maximum_depth={maximum_depth}. You answer a
question delegated to you by a parent agent using the tools the runtime
provides.

If you spawn subagents yourself, spawn at most {maximum_parallel_subagents}
concurrently — this cap is shared across the whole run.

When finished, return a concise answer. Do not emit <final/> — that
sentinel is reserved for the root agent.

Instructions:
{instructions}
"""

COMPACTION_SYSTEM_PROMPT = """\
You summarize a single conversation item for storage. Preserve tool names,
argument shapes, and key result facts that future reasoning might need.
Return a short plain-text summary — no JSON wrapping, no surrounding prose.
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You synthesize findings across a set of traces into a short plain-text
summary suitable as a tool result. Include concrete trace ids, error
patterns, model names, and token counts when available.
"""


def render_root_system_prompt(
    *,
    instructions: str | None,
    maximum_depth: int,
    maximum_parallel_subagents: int,
) -> str:
    """Build the root agent's system prompt: depth/parallelism caps + ``<final/>`` contract.

    ``instructions=None`` selects ``DEFAULT_SYSTEM_PROMPT`` (the engine's built-in
    trace-tool usage manual).
    """
    return ROOT_SYSTEM_PROMPT_TEMPLATE.format(
        instructions=instructions if instructions is not None else DEFAULT_SYSTEM_PROMPT,
        maximum_depth=maximum_depth,
        maximum_parallel_subagents=maximum_parallel_subagents,
    )


def render_subagent_system_prompt(
    *,
    instructions: str | None,
    depth: int,
    maximum_depth: int,
    maximum_parallel_subagents: int,
) -> str:
    """Build a subagent's system prompt at a specific depth; ``<final/>`` is reserved for root.

    ``instructions=None`` selects ``DEFAULT_SYSTEM_PROMPT`` (the engine's built-in
    trace-tool usage manual).
    """
    return SUBAGENT_SYSTEM_PROMPT_TEMPLATE.format(
        instructions=instructions if instructions is not None else DEFAULT_SYSTEM_PROMPT,
        depth=depth,
        maximum_depth=maximum_depth,
        maximum_parallel_subagents=maximum_parallel_subagents,
    )
