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
from engine.code.code_repo import find_ripgrep
from engine.engine_config import EngineConfig
from engine.main import stream_engine_async
from engine.model_config import ModelConfig, ReasoningEffort
from engine.model_provider_config import ModelProviderConfig
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


def _parse_headers(values: list[str] | None) -> dict[str, str] | None:
    headers: dict[str, str] = {}
    for raw in values or []:
        name, separator, value = raw.partition(":")
        if separator == "" or name.strip() == "":
            raise typer.BadParameter(
                "Expected NAME: VALUE.",
                param_hint="--header",
            )
        headers[name.strip()] = value.strip()
    return headers or None


def _make_config(
    *,
    model: str,
    synthesis_model: str | None,
    compaction_model: str | None,
    max_depth: int,
    max_turns: int,
    max_parallel: int,
    temperature: float | None,
    max_output_tokens: int | None,
    parallel_tool_calls: bool,
    reasoning_effort: ReasoningEffort | None,
    refusal_retries: int,
    base_url: str | None,
    api_key: str | None,
    default_headers: dict[str, str] | None,
    repo_path: Path | None,
) -> EngineConfig:
    def make_model_config(name: str, reasoning_effort: ReasoningEffort | None) -> ModelConfig:
        return ModelConfig(
            name=name,
            temperature=temperature,
            maximum_output_tokens=max_output_tokens,
            parallel_tool_calls=parallel_tool_calls,
            reasoning_effort=reasoning_effort,
        )

    # One ModelConfig per role so each is independently tunable. Synthesis
    # and compaction intentionally skip reasoning_effort — both are plain
    # summarizers, and --reasoning-effort targets the agents' model; their
    # models resolve their own family default via
    # ``effective_reasoning_effort`` instead. They fall back to the agent
    # model (never a hardcoded name) so a plain --model run stays on one
    # provider; --synthesis-model / --compaction-model point them at a
    # cheaper model.
    root_model = make_model_config(model, reasoning_effort)
    subagent_model = make_model_config(model, reasoning_effort)
    synthesis = make_model_config(synthesis_model or model, None)
    compaction = make_model_config(compaction_model or model, None)

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
        synthesis_model=synthesis,
        compaction_model=compaction,
        model_provider=ModelProviderConfig(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
        ),
        maximum_depth=max_depth,
        maximum_parallel_subagents=max_parallel,
        repo_path=repo_path,
    )


async def _stream(
    trace_path: Path, prompt: str, cfg: EngineConfig, *, telemetry: bool = False
) -> None:
    msgs = [AgentMessage(role="user", content=prompt)]
    async for ev in stream_engine_async(msgs, cfg, trace_path, telemetry=telemetry):
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
    synthesis_model: str | None = typer.Option(
        None,
        "--synthesis-model",
        help=(
            "Model for synthesis calls (trace summarization). Defaults to "
            "--model. A small, cheap model your provider serves (e.g. "
            "gpt-4.1-nano on OpenAI) is recommended."
        ),
    ),
    compaction_model: str | None = typer.Option(
        None,
        "--compaction-model",
        help=(
            "Model for compaction calls (context summarization) — the "
            "biggest token consumer in large runs. Defaults to --model. "
            "A small, cheap model your provider serves (e.g. gpt-4.1-nano "
            "on OpenAI) is recommended."
        ),
    ),
    max_depth: int = typer.Option(2, "--max-depth", min=0),
    max_turns: int = typer.Option(20, "--max-turns", min=1),
    max_parallel: int = typer.Option(10, "--max-parallel", min=1),
    repo_path: Path | None = typer.Option(
        None,
        "--repo-path",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help=(
            "Local checkout of the agent/harness source code that produced the "
            "traces. Enables the read-only code tools (glob_files/grep_files/"
            "read_file) so the analysis can cross-reference findings with code "
            "and cite file:line. Omit to disable."
        ),
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help=(
            "OpenAI-compatible API base URL. Omit to use OPENAI_BASE_URL "
            "or https://api.openai.com/v1."
        ),
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="Provider API key. Omit to use OPENAI_API_KEY.",
    ),
    headers: list[str] | None = typer.Option(
        None,
        "--header",
        "-H",
        help="Provider header as NAME: VALUE. May be repeated.",
    ),
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        min=0.0,
        max=2.0,
        help="Sampling temperature forwarded to the model.",
    ),
    max_output_tokens: int | None = typer.Option(
        None,
        "--max-output-tokens",
        min=1,
        help="Maximum output tokens forwarded to the model.",
    ),
    parallel_tool_calls: bool = typer.Option(
        True,
        "--parallel-tool-calls/--no-parallel-tool-calls",
        help="Allow models to issue parallel tool calls.",
    ),
    refusal_retries: int = typer.Option(
        0,
        "--refusal-retries",
        min=0,
        help="Retry an agent model request this many times when the model refuses.",
    ),
    reasoning_effort: str | None = typer.Option(
        None,
        "--reasoning-effort",
        help=(
            "Reasoning effort forwarded to the model on root and subagent "
            f"calls (synthesis and compaction never use reasoning). One of: "
            f"{', '.join(REASONING_EFFORT_CHOICES)}. Omit to use the model "
            "family's documented max for known reasoning models, or the "
            "provider default for non-reasoning models."
        ),
    ),
    telemetry: bool = typer.Option(
        False,
        "--telemetry",
        help=(
            "Emit OpenInference traces of HALO's own LLM/tool/agent "
            "activity. If CATALYST_OTLP_TOKEN is set, spans go to "
            "inference.net Catalyst; otherwise to "
            "$HALO_TELEMETRY_PATH (default: ./halo-telemetry-{run_id}.jsonl)."
        ),
    ),
) -> None:
    """Run the HALO engine against TRACE_PATH and stream output to stdout."""
    if api_key is None and not os.environ.get("OPENAI_API_KEY"):
        typer.echo(
            "OPENAI_API_KEY not set; pass --api-key or export OPENAI_API_KEY.",
            err=True,
        )
        raise typer.Exit(1)
    if repo_path is not None and find_ripgrep() is None:
        typer.echo(
            "--repo-path requires ripgrep (rg), which was not found on PATH. "
            "Install it (`brew install ripgrep`, `apt-get install ripgrep`, or "
            "`pip install ripgrep`) and re-run.",
            err=True,
        )
        raise typer.Exit(1)
    cfg = _make_config(
        model=model,
        synthesis_model=synthesis_model,
        compaction_model=compaction_model,
        max_depth=max_depth,
        max_turns=max_turns,
        max_parallel=max_parallel,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        parallel_tool_calls=parallel_tool_calls,
        reasoning_effort=_parse_reasoning_effort(reasoning_effort),
        refusal_retries=refusal_retries,
        base_url=base_url,
        api_key=api_key,
        default_headers=_parse_headers(headers),
        repo_path=repo_path,
    )
    asyncio.run(_stream(trace_path, prompt, cfg, telemetry=telemetry))


cli = typer.Typer(add_completion=False, rich_markup_mode=None)
cli.command()(_run)


def app() -> None:
    """Entry point bound to `halo` in pyproject.toml."""
    cli()


if __name__ == "__main__":
    app()
