from __future__ import annotations

import os
from pathlib import Path

import pytest

from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.main import run_engine_async
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage


@pytest.mark.asyncio
async def test_engine_runs_on_tiny_fixture(tmp_path: Path, fixtures_dir: Path) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; E2E requires real LLM access")

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())

    agent = AgentConfig(
        name="root",
        instructions="Answer briefly.",
        model=ModelConfig(name="gpt-5.4-mini"),
        maximum_turns=6,
    )
    cfg = EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub"}),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        maximum_depth=1,
        maximum_parallel_subagents=2,
    )

    messages = [AgentMessage(
        role="user",
        content="Use dataset_overview to report how many traces exist.",
    )]

    results = await run_engine_async(messages, cfg, trace_path)
    assert len(results) >= 1
    assert any(item.final for item in results)
