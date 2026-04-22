"""Batch-run the HALO RLM agent over every seed question of a dataset."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from rich import print

from dataset import TraceReader
from inference.config import InferenceConfig
from inference.harness import run_agent
from registry import build_registry


def run(dataset_id: str | None = None) -> None:
    config = InferenceConfig()
    config.init()
    registry = build_registry(config.index_dir)

    did = dataset_id or config.default_dataset_id
    if did not in registry.entries:
        raise SystemExit(
            f"Dataset '{did}' not found. Available: {', '.join(registry.ids()) or '(none)'}"
        )
    entry = registry.entries[did]
    store = registry.load_store(did)
    descriptor = entry.descriptor

    out_dir = Path("data/answers") / did
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = descriptor.seed_questions
    with TraceReader(descriptor.source_path) as reader:
        for i, question in enumerate(questions, start=1):
            logger.info("[{}/{}] {}", i, len(questions), question)
            events: list[dict] = []
            final = None
            for ev in run_agent(question, config, descriptor=descriptor,
                                store=store, reader=reader):
                events.append(ev.to_dict())
                if ev.kind == "final":
                    final = ev.data
                elif ev.kind == "error":
                    final = {"error": ev.data}
            out_path = out_dir / f"{i:02d}.json"
            out_path.write_text(json.dumps({
                "dataset_id": did,
                "question": question,
                "final": final,
                "events": events,
            }, ensure_ascii=False, indent=2))
            print(f"[green]saved[/] {out_path}")


if __name__ == "__main__":
    run()
