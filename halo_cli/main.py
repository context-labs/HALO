"""`halo` CLI: stream the HALO engine over a JSONL trace file."""

from __future__ import annotations

from pathlib import Path

import typer

from halo_cli import engine_runner

REASONING_EFFORT_CHOICES = engine_runner.REASONING_EFFORT_CHOICES
_make_config = engine_runner.make_config
_parse_reasoning_effort = engine_runner.parse_reasoning_effort
_stream = engine_runner.stream_to_console


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
            "Reasoning effort forwarded to the model on root, subagent, and "
            f"synthesis calls (compaction never uses reasoning). One of: "
            f"{', '.join(REASONING_EFFORT_CHOICES)}. Omit to use the model "
            "family's documented max for known reasoning models, or the "
            "provider default for non-reasoning models."
        ),
    ),
    telemetry: bool = typer.Option(
        False,
        "--telemetry/--no-telemetry",
        help=(
            "Emit OpenInference traces of HALO's own LLM/tool/agent "
            "activity. If CATALYST_OTLP_TOKEN is set, spans go to "
            "inference.net Catalyst; otherwise to "
            "$HALO_TELEMETRY_PATH (default: ./halo-telemetry-{run_id}.jsonl)."
        ),
    ),
    timeout_seconds: int | None = typer.Option(
        None,
        "--timeout-seconds",
        min=1,
        help="Abort the run after this many seconds.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        dir_okay=False,
        writable=True,
        help="Write the final root answer to this file.",
    ),
    events_path: Path | None = typer.Option(
        None,
        "--events-jsonl",
        dir_okay=False,
        writable=True,
        help="Write durable HALO agent events to this JSONL file.",
    ),
) -> None:
    """Run the HALO engine against TRACE_PATH and stream output to stdout."""
    engine_runner.run_trace(
        trace_path=trace_path,
        prompt=prompt,
        model=model,
        max_depth=max_depth,
        max_turns=max_turns,
        max_parallel=max_parallel,
        refusal_retries=refusal_retries,
        reasoning_effort=reasoning_effort,
        telemetry=telemetry,
        timeout_seconds=timeout_seconds,
        output_path=output_path,
        events_path=events_path,
    )


def app() -> None:
    """Entry point bound to `halo` in pyproject.toml."""
    typer.run(_run)


if __name__ == "__main__":
    app()
