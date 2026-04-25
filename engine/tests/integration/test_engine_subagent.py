from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.main import run_engine_async
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage


E2E_MODEL = os.environ.get("HALO_E2E_MODEL", "gpt-5.4-mini")
E2E_TIMEOUT_SECONDS = float(os.environ.get("HALO_E2E_TIMEOUT", "120"))


@pytest.mark.asyncio
async def test_root_delegates_to_subagent(tmp_path: Path, fixtures_dir: Path) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; E2E requires real LLM access")

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())

    agent = AgentConfig(
        name="root",
        instructions="Answer briefly.",
        model=ModelConfig(name=E2E_MODEL),
        maximum_turns=6,
    )
    cfg = EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub"}),
        synthesis_model=ModelConfig(name=E2E_MODEL),
        compaction_model=ModelConfig(name=E2E_MODEL),
        maximum_depth=1,
        maximum_parallel_subagents=2,
    )

    messages = [AgentMessage(
        role="user",
        content=(
            "Delegate this question to a subagent via call_subagent: "
            "'How many traces have errors? Use count_traces with has_errors=true.' "
            "Then report the subagent's answer to me and end with <final/>."
        ),
    )]

    async with asyncio.timeout(E2E_TIMEOUT_SECONDS):
        results = await run_engine_async(messages, cfg, trace_path)

    subagent_calls = [
        item for item in results
        if item.depth == 0 and item.item.tool_calls and any(
            tc.function.name == "call_subagent" for tc in item.item.tool_calls
        )
    ]
    assert subagent_calls, "root did not call call_subagent"

    child_items = [item for item in results if item.depth == 1]
    assert child_items, "no depth=1 items streamed — child stream not forwarded to bus"

    assert any(item.final for item in results)
