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


def _make_config(
    model: str,
    max_depth: int,
    max_turns: int,
    max_parallel: int,
    instructions: str | None,
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
    instructions: str | None = typer.Option(
        None,
        "--instructions",
        help="Override the engine's default trace-tool agent instructions.",
    ),
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
