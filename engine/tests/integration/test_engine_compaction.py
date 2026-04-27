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
E2E_TIMEOUT_SECONDS = float(os.environ.get("HALO_E2E_TIMEOUT", "90"))


@pytest.mark.asyncio
async def test_engine_compaction_fires_without_crash(tmp_path: Path, fixtures_dir: Path) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; E2E requires real LLM access")

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())

    agent = AgentConfig(
        name="root",
        instructions="Answer concisely.",
        model=ModelConfig(name=E2E_MODEL),
        maximum_turns=4,
    )
    cfg = EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub"}),
        synthesis_model=ModelConfig(name=E2E_MODEL),
        compaction_model=ModelConfig(name=E2E_MODEL),
        # Force compaction to trigger even on small conversations
        text_message_compaction_keep_last_messages=1,
        tool_call_compaction_keep_last_turns=1,
        maximum_depth=0,
        maximum_parallel_subagents=1,
    )

    messages = [AgentMessage(
        role="user",
        content=(
            "Call get_dataset_overview, then count_traces with has_errors=true, "
            "then tell me the two numbers and end with <final/>."
        ),
    )]

    async with asyncio.timeout(E2E_TIMEOUT_SECONDS):
        results = await run_engine_async(messages, cfg, trace_path)

    assert any(item.final for item in results)
