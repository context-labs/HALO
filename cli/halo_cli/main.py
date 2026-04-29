"""`halo-engine` CLI: stream the HALO engine over a JSONL trace file."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.rule import Rule

from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.main import stream_engine_async
from engine.model_config import ModelConfig
from engine.models.engine_output import AgentOutputItem, AgentTextDelta
from engine.models.messages import AgentMessage

console = Console()

DEFAULT_INSTRUCTIONS = (
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
    "pattern)` to surface only the spans of interest, then `view_spans(trace_id, "
    "span_ids=[...])` to read those specific spans (same truncation cap). This "
    "pattern stays bounded regardless of how many spans the trace has.\n"
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
    "7. If a truncation marker appears in a span attribute and the truncated content "
    "matters for your answer, narrow further with `search_trace` against a more "
    "specific pattern rather than asking for the full payload again.\n\n"
    "End your final reply with a line containing only <final/>."
)


def _make_config(
    model: str,
    max_depth: int,
    max_turns: int,
    max_parallel: int,
    instructions: str,
) -> EngineConfig:
    agent = AgentConfig(
        name="root",
        instructions=instructions,
        model=ModelConfig(name=model),
        maximum_turns=max_turns,
    )
    return EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub"}),
        synthesis_model=ModelConfig(name=model),
        compaction_model=ModelConfig(name=model),
        maximum_depth=max_depth,
        maximum_parallel_subagents=max_parallel,
    )


async def _stream(trace_path: Path, prompt: str, cfg: EngineConfig) -> None:
    msgs = [AgentMessage(role="user", content=prompt)]
    async for ev in stream_engine_async(msgs, cfg, trace_path):
        if isinstance(ev, AgentTextDelta):
            console.print(ev.text_delta, end="", soft_wrap=True)
        elif isinstance(ev, AgentOutputItem):
            console.print()
            console.print(Rule(f"{ev.agent_name} (depth={ev.depth}, final={ev.final})"))
            console.print(ev.item)


def _run(
    trace_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="JSONL trace file (e.g. engine/tests/fixtures/realistic_traces.jsonl).",
    ),
    prompt: str = typer.Option(..., "--prompt", "-p", help="User prompt to send to the root agent."),
    model: str = typer.Option("gpt-5.4-mini", "--model", "-m"),
    max_depth: int = typer.Option(1, "--max-depth", min=0),
    max_turns: int = typer.Option(8, "--max-turns", min=1),
    max_parallel: int = typer.Option(2, "--max-parallel", min=1),
    instructions: str = typer.Option(DEFAULT_INSTRUCTIONS, "--instructions"),
) -> None:
    """Run the HALO engine against TRACE_PATH and stream output to stdout."""
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set; the engine needs real LLM access.")
        raise typer.Exit(1)
    cfg = _make_config(model, max_depth, max_turns, max_parallel, instructions)
    asyncio.run(_stream(trace_path, prompt, cfg))


def app() -> None:
    """Entry point bound to `halo-engine` in pyproject.toml."""
    typer.run(_run)


if __name__ == "__main__":
    app()
