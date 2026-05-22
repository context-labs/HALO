"""Shared CLI helpers for running the HALO engine."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO, get_args

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


def parse_reasoning_effort(value: str | None) -> ReasoningEffort | None:
    """Validate the CLI string against the canonical ``ReasoningEffort`` literal."""
    try:
        return _REASONING_EFFORT_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc), param_hint="--reasoning-effort") from exc


def make_config(
    model: str,
    max_depth: int,
    max_turns: int,
    max_parallel: int,
    reasoning_effort: ReasoningEffort | None,
    refusal_retries: int,
    trace_detail_tools_enabled: bool = True,
    run_code_enabled: bool = True,
) -> EngineConfig:
    # One ModelConfig per role so each is independently tunable. Compaction
    # intentionally skips reasoning_effort because it is a deterministic summarizer.
    root_model = ModelConfig(name=model, reasoning_effort=reasoning_effort)
    subagent_model = ModelConfig(name=model, reasoning_effort=reasoning_effort)
    synthesis_model = ModelConfig(name=model, reasoning_effort=reasoning_effort)
    compaction_model = ModelConfig(name=model)

    root_agent = AgentConfig(
        name="root",
        model=root_model,
        maximum_turns=max_turns,
        refusal_retries=refusal_retries,
    )
    subagent = AgentConfig(
        name="sub",
        model=subagent_model,
        maximum_turns=max_turns,
        refusal_retries=refusal_retries,
    )

    return EngineConfig(
        root_agent=root_agent,
        subagent=subagent,
        synthesis_model=synthesis_model,
        compaction_model=compaction_model,
        maximum_depth=max_depth,
        maximum_parallel_subagents=max_parallel,
        trace_detail_tools_enabled=trace_detail_tools_enabled,
        run_code_enabled=run_code_enabled,
    )


async def stream_to_console(
    trace_path: Path,
    prompt: str,
    cfg: EngineConfig,
    *,
    telemetry: bool = False,
    output_path: Path | None = None,
    events_path: Path | None = None,
) -> str | None:
    msgs = [AgentMessage(role="user", content=prompt)]
    final_answer: str | None = None
    with _open_optional_text(events_path) as events_file:
        async for ev in stream_engine_async(msgs, cfg, trace_path, telemetry=telemetry):
            if events_file is not None and isinstance(ev, AgentOutputItem):
                events_file.write(ev.model_dump_json() + "\n")
                events_file.flush()
            if isinstance(ev, AgentTextDelta):
                console.print(ev.text_delta, end="", soft_wrap=True)
            elif isinstance(ev, AgentOutputItem):
                console.print()
                console.print(Rule(f"{ev.agent_name} (depth={ev.depth}, final={ev.final})"))
                console.print(ev.item)
                if ev.final:
                    final_answer = _message_content_to_text(ev.item.content)
    if output_path is not None:
        _write_text_artifact(output_path, final_answer or "")
        typer.echo(f"Wrote final answer to {output_path}")
    return final_answer


def _message_content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, indent=2, sort_keys=True)


def _write_text_artifact(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n")


@contextmanager
def _open_optional_text(path: Path | None) -> Iterator[TextIO | None]:
    if path is None:
        yield None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yield fh


def run_trace(
    *,
    trace_path: Path,
    prompt: str,
    model: str,
    max_depth: int,
    max_turns: int,
    max_parallel: int,
    refusal_retries: int,
    reasoning_effort: str | None,
    telemetry: bool,
    trace_detail_tools_enabled: bool = True,
    run_code_enabled: bool = True,
    timeout_seconds: int | None = None,
    output_path: Path | None = None,
    events_path: Path | None = None,
) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        typer.echo("OPENAI_API_KEY not set; the engine needs real LLM access.", err=True)
        raise typer.Exit(1)
    cfg = make_config(
        model,
        max_depth,
        max_turns,
        max_parallel,
        parse_reasoning_effort(reasoning_effort),
        refusal_retries,
        trace_detail_tools_enabled,
        run_code_enabled,
    )
    task = stream_to_console(
        trace_path,
        prompt,
        cfg,
        telemetry=telemetry,
        output_path=output_path,
        events_path=events_path,
    )
    if timeout_seconds is None:
        asyncio.run(task)
    else:
        asyncio.run(asyncio.wait_for(task, timeout=timeout_seconds))
