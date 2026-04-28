from __future__ import annotations

FINAL_SENTINEL = "<final/>"

ROOT_SYSTEM_PROMPT_TEMPLATE = """\
You are the root agent in the HALO engine. You explore OTel trace data
using the provided tools: dataset overview, query/count/view/search traces,
get_context_item, synthesize_traces, run_code, and call_subagent.

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
question delegated to you by a parent agent. You have trace tools and, if
your depth permits, a call_subagent tool.

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
    instructions: str,
    maximum_depth: int,
    maximum_parallel_subagents: int,
) -> str:
    """Build the root agent's system prompt: depth/parallelism caps + ``<final/>`` contract."""
    return ROOT_SYSTEM_PROMPT_TEMPLATE.format(
        instructions=instructions,
        maximum_depth=maximum_depth,
        maximum_parallel_subagents=maximum_parallel_subagents,
    )


def render_subagent_system_prompt(
    *,
    instructions: str,
    depth: int,
    maximum_depth: int,
    maximum_parallel_subagents: int,
) -> str:
    """Build a subagent's system prompt at a specific depth; ``<final/>`` is reserved for root."""
    return SUBAGENT_SYSTEM_PROMPT_TEMPLATE.format(
        instructions=instructions,
        depth=depth,
        maximum_depth=maximum_depth,
        maximum_parallel_subagents=maximum_parallel_subagents,
    )
