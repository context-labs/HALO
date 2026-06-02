#!/usr/bin/env python3
"""Structured local bridge for running HALO from the desktop app.

The Bun backend owns queueing, persistence, cancellation, and live updates. This
script only imports the local HALO engine, streams events, and writes JSON lines
to stdout so the desktop app has a stable wire shape.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def emit(event_type: str, payload: dict[str, Any]) -> None:
    print(json.dumps({"type": event_type, **payload}, default=str), flush=True)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_imports(halo_path: str) -> None:
    path = str(Path(halo_path).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


async def run(config: dict[str, Any]) -> None:
    configure_imports(config["haloPath"])

    from engine.agents.agent_config import AgentConfig
    from engine.engine_config import EngineConfig
    from engine.main import stream_engine_async
    from engine.model_config import ModelConfig
    from engine.model_provider_config import ModelProviderConfig
    from engine.models.engine_output import AgentOutputItem, AgentTextDelta
    from engine.models.messages import AgentMessage

    provider = config["provider"]
    if provider.get("apiKey"):
        os.environ["OPENAI_API_KEY"] = provider["apiKey"]
    if provider.get("baseUrl"):
        os.environ["OPENAI_BASE_URL"] = provider["baseUrl"]

    model_name = config["model"]
    model_config = ModelConfig(name=model_name)
    root_agent = AgentConfig(
        name="root",
        model=model_config,
        maximum_turns=int(config.get("maxTurns", 8)),
    )
    engine_config = EngineConfig(
        root_agent=root_agent,
        subagent=root_agent.model_copy(update={"name": "sub"}),
        synthesis_model=model_config,
        compaction_model=model_config,
        model_provider=ModelProviderConfig(
            base_url=provider.get("baseUrl"),
            api_key=provider.get("apiKey"),
            default_headers=provider.get("headers") or None,
        ),
        maximum_depth=int(config.get("maxDepth", 1)),
        maximum_parallel_subagents=int(config.get("maxParallel", 2)),
    )

    messages = [AgentMessage(role="user", content=config["prompt"])]
    emit("started", {"runId": config["runId"]})

    final_answer: str | None = None
    final_answer_source: str | None = None
    items: list[dict[str, Any]] = []

    async for event in stream_engine_async(
        messages,
        engine_config,
        Path(config["tracePath"]),
    ):
        if isinstance(event, AgentTextDelta):
            emit("delta", event.model_dump(mode="json"))
            continue
        if isinstance(event, AgentOutputItem):
            dumped = event.model_dump(mode="json")
            items.append(dumped)
            emit("agent_step", dumped)
            if event.final and event.item.role == "assistant":
                final_answer = content_to_text(event.item.content)
                final_answer_source = "final"

    if final_answer is None:
        for item in reversed(items):
            message = item.get("item") or {}
            if message.get("role") == "assistant":
                final_answer = content_to_text(message.get("content"))
                final_answer_source = "last_assistant"
                break

    has_answer = bool((final_answer or "").strip())
    emit(
        "completed" if has_answer else "incomplete",
        {
            "finalAnswer": final_answer or "",
            "finalAnswerSource": final_answer_source or "none",
            "itemCount": len(items),
        },
    )


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def main() -> int:
    if len(sys.argv) != 2:
        emit("failed", {"error": "Usage: halo-local-runner.py <config.json>"})
        return 2
    try:
        asyncio.run(run(load_config(Path(sys.argv[1]))))
        return 0
    except KeyboardInterrupt:
        emit("cancelled", {"error": "Run cancelled."})
        return 130
    except Exception as exc:  # noqa: BLE001 - boundary script reports all errors.
        emit(
            "failed",
            {
                "error": str(exc),
                "traceback": traceback.format_exc(limit=20),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
