"""CLI entry point for the HALO RLM harness.

Subcommands:

* ``halo ingest``    — one-shot: sniff a JSONL, write a descriptor to
  ``catalog/``, build the summary index. The zero-keystroke path for new
  datasets; no hand-written Python required.
* ``halo datasets``  — list registered datasets.
* ``halo index``     — rebuild the summary index for a registered dataset
  (only needed after you edit its descriptor).
* ``halo ask``       — run the agent against a single question.
* ``halo serve``     — start the FastAPI + React UI.
* ``halo batch``     — run the agent against every seed question of a dataset.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="halo",
    help="HALO RLM — agent harness + UI for exploring large trace datasets.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    no_args_is_help=True,
)


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Path to the trace JSONL file."),
    id: str | None = typer.Option(None, help="URL-safe dataset id; defaults to the filename stem."),
    name: str | None = typer.Option(None, help="Display name; defaults to the filename stem."),
    source_model: str | None = typer.Option(None, help="Model that produced the traces (informational)."),
    description: str | None = typer.Option(None, help="One-line description for the UI."),
    sample: int = typer.Option(200, help="Records to sniff (regex-only fallback) or hand to the LLM (default)."),
    skip_index: bool = typer.Option(False, "--skip-index", help="Write the descriptor but don't build the index yet."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing descriptor file with the same id."),
    workers: int = typer.Option(0, help="Parse workers for the index build."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the LLM; use the regex-only sniffer only."),
    llm_model: str = typer.Option(
        "claude-opus-4-7",
        "--llm-model",
        help="Model used to design the descriptor (default: claude-opus-4-7).",
    ),
) -> None:
    """Design ``catalog/<id>.py`` and build its index from a trace JSONL.

    The default path feeds a sample of records to an LLM (Opus 4.7 1M)
    that designs the whole descriptor end-to-end — format, primary
    metric + direction, labels with display names, seed questions.
    Falls back to the fast regex sniffer when ``--no-llm`` is set or
    the LLM call fails.
    """
    from rich import print
    from rich.table import Table

    from dataset.autodetect import descriptor_to_python, infer_descriptor
    from dataset.indexer import build_index
    from inference.config import InferenceConfig

    path = path.resolve()
    if not path.exists():
        raise typer.BadParameter(f"source JSONL not found: {path}")

    descriptor = None
    report_notes: list[str] = []
    if not no_llm:
        print(f"[cyan]designing[/] {path}  (LLM: {llm_model})")
        try:
            from dataset.autodetect_llm import infer_descriptor_with_llm
            descriptor, spec = infer_descriptor_with_llm(
                path, model=llm_model, sample_size=min(sample, 60),
            )
            if id:
                descriptor.id = id
            if name:
                descriptor.name = name
            if source_model:
                descriptor.source_model = source_model
            if description:
                descriptor.description = description
            report_notes.append(f"LLM chose format={descriptor.format}")
            if descriptor.primary_metric:
                pm = descriptor.primary_metric
                report_notes.append(
                    f"primary_metric={pm.name} (source={pm.source}, "
                    f"higher_is_better={pm.higher_is_better})"
                )
            label_names = ", ".join(lbl.name for lbl in descriptor.labels) or "∅"
            report_notes.append(f"labels=\\[{label_names}]")
            report_notes.append(f"seed_questions={len(descriptor.seed_questions)}")
        except Exception as e:
            print(f"[yellow]LLM design failed ({e!r}); falling back to regex sniffer.[/]")
            descriptor = None

    if descriptor is None:
        print(f"[cyan]sniffing[/]  {path}  ({sample} records, regex)")
        descriptor, report = infer_descriptor(
            path,
            dataset_id=id,
            name=name,
            source_model=source_model,
            description=description,
            sample_size=sample,
        )
        table = Table(title=f"inferred descriptor for '{descriptor.id}'")
        table.add_column("field")
        table.add_column("mapping")
        for k, v in report.fields.items():
            table.add_row(k, "—" if v is None or v == [] else str(v))
        print(table)
        for note in report.notes:
            print(f"[yellow]note[/] {note}")
    else:
        for note in report_notes:
            print(f"[dim]•[/] {note}")

    catalog_dir = Path(__file__).resolve().parent / "catalog"
    descriptor_path = catalog_dir / f"{descriptor.id}.py"
    if descriptor_path.exists() and not force:
        raise typer.BadParameter(
            f"{descriptor_path} already exists. Pass --force to overwrite, "
            f"or pick a different --id."
        )
    descriptor_path.write_text(descriptor_to_python(descriptor))
    print(f"[green]wrote[/]    {descriptor_path}")

    if skip_index:
        print("[yellow]skipping index build (--skip-index)[/]")
        return

    config = InferenceConfig()
    config.init()
    index_path = descriptor.default_index_path(config.index_dir)
    print(f"[cyan]indexing[/]  -> {index_path}")
    n = build_index(descriptor.source_path, descriptor, index_path, workers=workers)
    print(f"[green]done[/]     indexed {n:,} records; available as "
          f"`uv run halo ask '...' --dataset {descriptor.id}`")


@app.command()
def datasets() -> None:
    """List every descriptor auto-registered from the ``catalog/`` directory."""
    from rich.table import Table
    from rich import print

    from inference.config import InferenceConfig
    from registry import build_registry

    config = InferenceConfig()
    config.init()
    registry = build_registry(config.index_dir)
    table = Table(title="halo datasets")
    table.add_column("id")
    table.add_column("name")
    table.add_column("source_model")
    table.add_column("indexed?")
    table.add_column("source_path")
    for did, entry in registry.entries.items():
        table.add_row(
            did,
            entry.descriptor.name,
            entry.descriptor.source_model or "—",
            "yes" if entry.is_indexed else "no",
            str(entry.descriptor.source_path),
        )
    print(table)
    if not registry.entries:
        print("[yellow]No datasets registered. Drop a descriptor file into catalog/.[/]")


@app.command()
def index(
    dataset: str | None = typer.Option(None, help="Dataset id (defaults to default_dataset_id)."),
    limit: int | None = typer.Option(None, help="Stop after this many records (for debugging)."),
    workers: int = typer.Option(0, help="Parse workers; 0 = single process."),
) -> None:
    """Build the summary index over a dataset's JSONL."""
    from rich import print

    from dataset.indexer import build_index
    from inference.config import InferenceConfig
    from registry import build_registry

    config = InferenceConfig()
    config.init()
    registry = build_registry(config.index_dir)

    did = dataset or config.default_dataset_id
    if did not in registry.entries:
        raise typer.BadParameter(
            f"dataset '{did}' not registered. Available: {', '.join(registry.ids()) or '(none)'}"
        )
    entry = registry.entries[did]
    descriptor = entry.descriptor

    if not descriptor.source_path.exists():
        raise typer.BadParameter(f"source JSONL not found: {descriptor.source_path}")

    print(f"[cyan]dataset[/]  {did}")
    print(f"[cyan]source [/]  {descriptor.source_path}")
    print(f"[cyan]index  [/]  {entry.index_path}")
    print(f"[cyan]workers[/]  {workers}")
    n = build_index(
        descriptor.source_path, descriptor, entry.index_path,
        max_records=limit, workers=workers,
    )
    print(f"[green]done[/] indexed {n:,} records to {entry.index_path}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask the agent."),
    dataset: str | None = typer.Option(None, help="Dataset id (defaults to default_dataset_id)."),
    model: str | None = typer.Option(None, help="Override the agent's model."),
    max_turns: int | None = typer.Option(None, help="Override max tool-calling turns."),
    json_out: bool = typer.Option(False, "--json", help="Emit the full event log as JSON."),
) -> None:
    """Run the agent once against a question and print the result."""
    import json as _json

    from rich import print
    from rich.markdown import Markdown

    from inference.config import InferenceConfig
    from inference.harness import run_agent
    from registry import build_registry

    config = InferenceConfig()
    config.init()
    if model:
        config.model = model
    if max_turns:
        config.max_turns = max_turns

    registry = build_registry(config.index_dir)
    did = dataset or config.default_dataset_id
    if did not in registry.entries:
        raise typer.BadParameter(
            f"dataset '{did}' not registered. Run `uv run halo datasets` to list."
        )
    entry = registry.entries[did]
    if not entry.is_indexed:
        raise typer.BadParameter(
            f"dataset '{did}' is not indexed. Run `uv run halo index --dataset {did}`."
        )
    store = registry.load_store(did)

    final = None
    events_out: list[dict] = []
    for ev in run_agent(question, config, descriptor=entry.descriptor, store=store):
        events_out.append(ev.to_dict())
        if json_out:
            continue
        if ev.kind == "tool_call":
            print(f"[orange3]→ {ev.data['name']}[/]  {ev.data['arguments']}")
        elif ev.kind == "tool_result":
            print(f"[dim]←[/] {ev.data['name']}")
        elif ev.kind == "thinking":
            if ev.data.get("content"):
                print(f"[cyan]· {ev.data['content'][:200]}[/]")
        elif ev.kind == "usage":
            t = ev.data.get("tokens") or {}
            print(f"[dim]   turn {ev.data['turn']} · {t.get('total')} tok · ${ev.data.get('cost', 0):.6f}[/]")
        elif ev.kind == "final":
            final = ev.data
        elif ev.kind == "error":
            print(f"[red]! {ev.data['message']}[/]")

    if json_out:
        print(_json.dumps({
            "dataset_id": did, "question": question, "final": final, "events": events_out
        }, ensure_ascii=False, indent=2))
        return

    if final:
        print()
        print(Markdown(f"## Final answer\n\n{final.get('content') or '(empty)'}"))
        print(
            f"\n[dim]{final.get('turns_used')} turns · "
            f"{final.get('tool_calls_made')} tool calls · "
            f"${final.get('total_cost', 0):.6f}[/]"
        )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host for uvicorn."),
    port: int = typer.Option(8000, help="Bind port for uvicorn."),
) -> None:
    """Launch the FastAPI server + static React UI."""
    from web.server import run
    run(host=host, port=port)


@app.command()
def batch(
    dataset: str | None = typer.Option(None, help="Dataset id (defaults to default_dataset_id).")
) -> None:
    """Run the agent against every seed question of a dataset."""
    from inference.run import run as batch_run
    batch_run(dataset_id=dataset)


if __name__ == "__main__":
    app()
