"""Run a batch of AppWorld tasks in parallel and write per-task JSONL traces.

Usage::

    # Default: train(90) + dev(10) = 100 tasks, 8 workers, gpt-4.1-mini, max_steps=30
    uv run run_batch.py

    # Or specify your own list:
    uv run run_batch.py --task-ids fac291d_1 fac291d_2 50e1ac9_1

Outputs:
  - ``./traces/<task_id>.jsonl`` per task (inference.net format)
  - ``./traces/index.jsonl`` summary line per task with success/fail
    metadata, suitable as a HuggingFace dataset index

Implementation note: each task runs in its own subprocess (``uv run``)
so AppWorld's per-process global state and the ``setup_tracing`` call
are isolated; concurrency is via ``ProcessPoolExecutor``.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich import print as rprint
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

ROOT = Path(__file__).parent
DEFAULT_DATA_ROOT = ROOT / ".data"
DEFAULT_TRACES_DIR = ROOT / "traces"
DEFAULT_EXPERIMENT = "halo-bench"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_STEPS = 30
DEFAULT_WORKERS = 8


def _default_task_ids(data_root: Path) -> list[str]:
    """Return train(90) + dev(10), task-id-sorted, total 100."""
    train = (data_root / "data" / "datasets" / "train.txt").read_text().split()
    dev = (data_root / "data" / "datasets" / "dev.txt").read_text().split()
    return sorted(train) + sorted(dev)[:10]


def _read_trace_metadata(trace_path: Path) -> dict[str, Any]:
    """Aggregate per-trace metadata from a JSONL trace file."""
    if not trace_path.exists():
        return {"trace_present": False}
    spans: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    by_kind: dict[str, int] = {}
    trace_id: str | None = None
    final_message: str | None = None
    for s in spans:
        kind = s.get("attributes", {}).get("inference.observation_kind", "?")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        if not trace_id:
            trace_id = s.get("trace_id")
        # Look for final assistant text on the last `response`/`generation` span
        if kind == "LLM":
            for k, v in (s.get("attributes") or {}).items():
                if k.endswith(".message.content") and "output_messages" in k:
                    final_message = v if isinstance(v, str) else json.dumps(v)
    return {
        "trace_present": True,
        "trace_id": trace_id,
        "span_count": len(spans),
        "by_kind": by_kind,
        "final_message": final_message,
    }


def _run_one(args: tuple[str, str, int, str, str, str, str]) -> dict[str, Any]:
    """Subprocess invocation of run_one_task.py for one task. Pickle-safe."""
    (
        task_id,
        model,
        max_steps,
        appworld_root,
        traces_dir,
        experiment_name,
        python_unbuffered,
    ) = args
    start = time.monotonic()
    cmd = [
        "uv",
        "run",
        "python",
        "run_one_task.py",
        task_id,
        "--model",
        model,
        "--max-steps",
        str(max_steps),
        "--appworld-root",
        appworld_root,
        "--traces-dir",
        traces_dir,
        "--experiment-name",
        experiment_name,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", python_unbuffered)
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    duration = time.monotonic() - start
    log_lines = (completed.stdout + "\n" + completed.stderr).splitlines()
    # Persist per-task stdout for debugging (overwrites on re-run).
    log_path = Path(traces_dir) / f"{task_id}.log"
    log_path.write_text(
        f"=== exit_code={completed.returncode} duration={duration:.1f}s ===\n"
        f"=== STDOUT ===\n{completed.stdout}\n"
        f"=== STDERR ===\n{completed.stderr}\n"
    )
    final_message = None
    task_completed = False
    eval_pass = None
    eval_total = None
    eval_success = None
    for line in log_lines:
        if line.startswith("Final agent message:"):
            # The very next non-empty line is the message
            idx = log_lines.index(line)
            for cand in log_lines[idx + 1 : idx + 6]:
                cand = cand.strip()
                if cand:
                    final_message = cand
                    break
        if "Task completed (called complete_task):" in line:
            task_completed = "True" in line
        if line.startswith("Evaluation:"):
            # "Evaluation: 5/5 tests passed (success=True)"
            try:
                left, right = line.split(":", 1)[1].strip().split(" tests")[0].split("/")
                eval_pass = int(left.strip())
                eval_total = int(right.strip())
                eval_success = "success=True" in line
            except (ValueError, IndexError):
                pass
    trace_path = Path(traces_dir) / f"{task_id}.jsonl"
    trace_meta = _read_trace_metadata(trace_path)
    return {
        "task_id": task_id,
        "exit_code": completed.returncode,
        "duration_seconds": round(duration, 1),
        "task_completed": task_completed,
        "eval_pass": eval_pass,
        "eval_total": eval_total,
        "eval_success": eval_success,
        "final_message": final_message,
        "stdout_tail": "\n".join(log_lines[-5:])[:500] if completed.returncode != 0 else None,
        **trace_meta,
    }


def main(
    task_ids: list[str] = typer.Option(
        None,
        "--task-id",
        "-t",
        help=(
            "Specific task IDs (repeat the flag, e.g. -t a -t b). "
            "Defaults to train(90) + dev(10) = 100."
        ),
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model"),
    max_steps: int = typer.Option(DEFAULT_MAX_STEPS, "--max-steps"),
    workers: int = typer.Option(DEFAULT_WORKERS, "--workers", help="Parallel subprocess count."),
    appworld_root: Path = typer.Option(DEFAULT_DATA_ROOT, "--appworld-root"),
    traces_dir: Path = typer.Option(DEFAULT_TRACES_DIR, "--traces-dir"),
    experiment_name: str = typer.Option(DEFAULT_EXPERIMENT, "--experiment-name"),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip tasks whose trace JSONL is already present (resumable runs).",
    ),
) -> None:
    load_dotenv()
    appworld_root = appworld_root.resolve()
    traces_dir = traces_dir.resolve()
    traces_dir.mkdir(parents=True, exist_ok=True)

    ids = task_ids or _default_task_ids(appworld_root)
    if skip_existing:
        before = len(ids)
        ids = [i for i in ids if not (traces_dir / f"{i}.jsonl").exists()]
        skipped = before - len(ids)
        if skipped:
            rprint(f"[yellow]Skipping {skipped} task(s) with existing traces[/]")

    if not ids:
        rprint("[green]Nothing to run — all traces already present.[/]")
        _write_index(traces_dir)
        return

    rprint(
        f"[bold]Running {len(ids)} task(s)[/] with model={model}, "
        f"max_steps={max_steps}, workers={workers}"
    )
    rprint(f"[dim]Traces dir: {traces_dir}[/]")

    args_iter = [
        (
            task_id,
            model,
            max_steps,
            str(appworld_root),
            str(traces_dir),
            experiment_name,
            "1",
        )
        for task_id in ids
    ]

    results: list[dict[str, Any]] = []
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[last_id]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        bar = progress.add_task("AppWorld batch", total=len(ids), last_id="—")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, a): a[0] for a in args_iter}
            for fut in as_completed(futures):
                task_id = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {
                        "task_id": task_id,
                        "exit_code": -1,
                        "error": repr(e),
                        "trace_present": False,
                    }
                results.append(res)
                progress.update(bar, advance=1, last_id=task_id)
                # Append to index incrementally so a crash doesn't lose progress
                _append_index_entry(traces_dir, res)

    # Summary
    total = len(results)
    completed_count = sum(1 for r in results if r.get("task_completed"))
    success_count = sum(1 for r in results if r.get("eval_success"))
    failed_runs = sum(1 for r in results if r.get("exit_code") != 0)
    rprint(
        f"\n[bold]Summary:[/] "
        f"{completed_count}/{total} called complete_task, "
        f"{success_count}/{total} eval-passed, "
        f"{failed_runs}/{total} subprocess failures"
    )
    rprint(f"[dim]Index: {traces_dir / 'index.jsonl'}[/]")


def _append_index_entry(traces_dir: Path, entry: dict[str, Any]) -> None:
    """Append a row to traces_dir/index.jsonl, creating it if needed."""
    entry["recorded_at"] = datetime.now(timezone.utc).isoformat()
    index_path = traces_dir / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False))
        fh.write("\n")


def _write_index(traces_dir: Path) -> None:
    """Rebuild index.jsonl by scanning every present trace file (idempotent)."""
    index_path = traces_dir / "index.jsonl"
    rows: list[dict[str, Any]] = []
    for trace_file in sorted(traces_dir.glob("*.jsonl")):
        if trace_file.name == "index.jsonl":
            continue
        task_id = trace_file.stem
        meta = _read_trace_metadata(trace_file)
        rows.append({"task_id": task_id, **meta})
    with index_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    rprint(f"[dim]Rewrote index ({len(rows)} rows): {index_path}[/]")


if __name__ == "__main__":
    typer.run(main)
