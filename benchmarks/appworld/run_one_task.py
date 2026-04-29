"""Run a single AppWorld task through the OpenAI Agents SDK and emit traces.

Usage::

    uv run run_one_task.py fac291d_1               # dev task by ID
    uv run run_one_task.py fac291d_1 --model gpt-4.1-mini --max-steps 30

Outputs a single ``traces/<task_id>.jsonl`` file per run, in the
inference.net JSONL format the HALO Engine reads directly. The trace
file is self-contained — upload it to HuggingFace as a fixture so others
can replay the agent's behaviour without re-running the harness.

Requires:
  - OPENAI_API_KEY in the environment / .env (uses the Responses API by default)
  - AppWorld data prepared at ``APPWORLD_ROOT`` (defaults to ./.data)
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from agents import Runner
from appworld import AppWorld
from dotenv import load_dotenv
from rich import print as rprint

from agent import AgentContext, build_agent, render_instructions
from tracing import setup_tracing

DEFAULT_DATA_ROOT = Path(__file__).parent / ".data"
DEFAULT_TRACES_DIR = Path(__file__).parent / "traces"
DEFAULT_EXPERIMENT = "halo-bench"


def main(
    task_id: str = typer.Argument(..., help="AppWorld task ID, e.g. 'fac291d_1'."),
    model: str = typer.Option("gpt-4.1-mini", "--model", help="OpenAI model name."),
    max_steps: int = typer.Option(
        20, "--max-steps", help="Hard cap on agent turns; AppWorld also enforces."
    ),
    experiment_name: str = typer.Option(
        DEFAULT_EXPERIMENT,
        "--experiment-name",
        help="Subdir under ./runs/ for AppWorld artifacts.",
    ),
    appworld_root: Path = typer.Option(
        DEFAULT_DATA_ROOT,
        "--appworld-root",
        help="Path containing AppWorld's ./data directory.",
    ),
    traces_dir: Path = typer.Option(
        DEFAULT_TRACES_DIR,
        "--traces-dir",
        help="Where to write `<task_id>.jsonl.gz`. Created if missing.",
    ),
    service_name: str = typer.Option(
        "halo-appworld-bench",
        "--service-name",
        help="Stamped onto every exported span as `service.name`.",
    ),
    project_id: str = typer.Option(
        "halo-appworld",
        "--project-id",
        help="Stamped onto every exported span as `inference.project_id`.",
    ),
) -> None:
    load_dotenv()

    # AppWorld reads its data path from APPWORLD_ROOT.
    os.environ["APPWORLD_ROOT"] = str(appworld_root.resolve())

    # Tracing must be installed before the first Agent(...) construction.
    # `setup_tracing` reads the output path from the HALO_TRACES_PATH env var
    # — set it before the call so each task gets its own file.
    traces_dir.mkdir(parents=True, exist_ok=True)
    output_path = traces_dir / f"{task_id}.jsonl"
    os.environ["HALO_TRACES_PATH"] = str(output_path)
    processor = setup_tracing(
        service_name=service_name,
        project_id=project_id,
    )
    rprint(f"[dim]Writing traces to: {output_path}[/]")

    agent = build_agent(model=model)

    rprint(f"[bold cyan]Loading AppWorld task[/]: {task_id}")
    try:
        # ``timeout_seconds=None`` disables AppWorld's per-execution
        # ``signal.SIGALRM`` watchdog. The OpenAI Agents SDK dispatches
        # sync ``@function_tool`` callbacks through ``asyncio.to_thread``,
        # which lands ``world.execute`` on a worker thread; ``signal.alarm``
        # only works on the main thread, so the default 100s timeout
        # raises ``signal only works in main thread of the main interpreter``
        # on every tool call. Off-thread execution is fine for our use case
        # — agent step caps still bound the run.
        with AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            timeout_seconds=None,
        ) as world:
            agent.instructions = render_instructions(world)

            rprint(
                f"[bold]Instruction:[/] {world.task.instruction}\n"
                f"[bold]Supervisor:[/] {world.task.supervisor['first_name']} "
                f"{world.task.supervisor['last_name']}\n"
            )

            ctx = AgentContext(world=world)
            # Use async Runner.run instead of run_sync. AppWorld's IPython
            # shell uses ``signal.alarm`` for execution timeouts, which only
            # works on the main thread; ``run_sync`` dispatches tool callbacks
            # off-thread, which trips ``signal only works in main thread of
            # the main interpreter``. ``asyncio.run`` keeps the tool callbacks
            # on the main thread.
            result = asyncio.run(
                Runner.run(
                    agent,
                    input="Begin.",
                    context=ctx,
                    max_turns=max_steps,
                )
            )

            rprint(f"\n[bold]Final agent message:[/]\n{result.final_output}")

            completed = world.task_completed()
            rprint(f"\n[bold]Task completed (called complete_task):[/] {completed}")

            if completed:
                tracker = world.evaluate()
                rprint(
                    f"[bold]Evaluation:[/] {tracker.pass_count}/{tracker.total_count} "
                    f"tests passed (success={tracker.success})"
                )
                if tracker.failures:
                    rprint("[bold red]Failures:[/]")
                    for f in tracker.failures:
                        rprint(f"  - {f.requirement}")
            else:
                rprint("[yellow]Agent did not call complete_task; no eval run.[/]")

            rprint(
                f"\n[dim]AppWorld artifacts: "
                f"{world.output_directory}[/]"
            )
    finally:
        # Always flush the gzip stream, even if the run raised. This guarantees
        # the trace file is well-formed and uploadable to HuggingFace.
        processor.shutdown()
        rprint(f"[dim]Trace flushed: {output_path}[/]")


if __name__ == "__main__":
    typer.run(main)
