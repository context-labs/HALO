"""`halo` CLI: stream the HALO engine over a JSONL trace file."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import get_args

import typer
from pydantic import TypeAdapter, ValidationError
from rich.console import Console
from rich.rule import Rule

from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.main import stream_engine_async
from engine.model_config import ModelConfig, ReasoningEffort
from engine.models.engine_output import AgentOutputItem, AgentTextDelta
from engine.models.messages import AgentMessage

console = Console()

REASONING_EFFORT_CHOICES: tuple[str, ...] = get_args(ReasoningEffort)
_REASONING_EFFORT_ADAPTER: TypeAdapter[ReasoningEffort | None] = TypeAdapter(ReasoningEffort | None)


def _parse_reasoning_effort(value: str | None) -> ReasoningEffort | None:
    """Validate the CLI string against the canonical ``ReasoningEffort`` literal.

    Pydantic owns the list of valid values; this just routes its error
    through typer's CLI surface so users see ``Invalid value for
    '--reasoning-effort'`` instead of a stack trace.
    """
    try:
        return _REASONING_EFFORT_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc), param_hint="--reasoning-effort") from exc


def _make_config(
    model: str,
    max_depth: int,
    max_turns: int,
    max_parallel: int,
    instructions: str | None,
    reasoning_effort: ReasoningEffort | None,
) -> EngineConfig:
    # One ModelConfig per role so each is independently tunable. Compaction
    # intentionally skips reasoning_effort — it's a deterministic summarizer.
    root_model = ModelConfig(name=model, reasoning_effort=reasoning_effort)
    subagent_model = ModelConfig(name=model, reasoning_effort=reasoning_effort)
    synthesis_model = ModelConfig(name=model, reasoning_effort=reasoning_effort)
    compaction_model = ModelConfig(name=model)

    root_agent = AgentConfig(
        name="root",
        instructions=instructions,
        model=root_model,
        maximum_turns=max_turns,
    )
    subagent = AgentConfig(
        name="sub",
        instructions=instructions,
        model=subagent_model,
        maximum_turns=max_turns,
    )

    return EngineConfig(
        root_agent=root_agent,
        subagent=subagent,
        synthesis_model=synthesis_model,
        compaction_model=compaction_model,
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
        help="JSONL trace file (e.g. tests/fixtures/realistic_traces.jsonl).",
    ),
    prompt: str = typer.Option(
        ..., "--prompt", "-p", help="User prompt to send to the root agent."
    ),
    model: str = typer.Option("gpt-5.4-mini", "--model", "-m"),
    max_depth: int = typer.Option(2, "--max-depth", min=0),
    max_turns: int = typer.Option(20, "--max-turns", min=1),
    max_parallel: int = typer.Option(2, "--max-parallel", min=1),
    instructions: str | None = typer.Option(
        None,
        "--instructions",
        help="Override the engine's default trace-tool agent instructions.",
    ),
    reasoning_effort: str | None = typer.Option(
        None,
        "--reasoning-effort",
        help=(
            "Reasoning effort forwarded to the model on root, subagent, and "
            f"synthesis calls (compaction never uses reasoning). One of: "
            f"{', '.join(REASONING_EFFORT_CHOICES)}. Omit to use the model "
            "family's documented max for known reasoning models, or the "
            "provider default for non-reasoning models."
        ),
    ),
) -> None:
    """Run the HALO engine against TRACE_PATH and stream output to stdout."""
    if not os.environ.get("OPENAI_API_KEY"):
        typer.echo("OPENAI_API_KEY not set; the engine needs real LLM access.", err=True)
        raise typer.Exit(1)
    cfg = _make_config(
        model,
        max_depth,
        max_turns,
        max_parallel,
        instructions,
        _parse_reasoning_effort(reasoning_effort),
    )
    asyncio.run(_stream(trace_path, prompt, cfg))


def app() -> None:
    """Entry point bound to `halo` in pyproject.toml."""
    typer.run(_run)


if __name__ == "__main__":
    app()
