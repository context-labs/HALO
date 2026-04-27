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
E2E_TIMEOUT_SECONDS = float(os.environ.get("HALO_E2E_TIMEOUT", "60"))


@pytest.mark.asyncio
async def test_engine_runs_on_tiny_fixture(tmp_path: Path, fixtures_dir: Path) -> None:
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
        maximum_depth=0,
        maximum_parallel_subagents=2,
    )

    messages = [
        AgentMessage(
            role="user",
            content=(
                "Use get_dataset_overview to tell me how many traces are in the dataset. "
                "Then end your reply with a line containing only <final/>."
            ),
        )
    ]

    async with asyncio.timeout(E2E_TIMEOUT_SECONDS):
        results = await run_engine_async(messages, cfg, trace_path)

    assert len(results) >= 1
    assert any(item.final for item in results), "no AgentOutputItem with final=True emitted"
    tool_calls = [item for item in results if item.item.tool_calls]
    assert any(
        tc.function.name == "get_dataset_overview" for item in tool_calls for tc in (item.item.tool_calls or [])
    ), "expected root agent to call get_dataset_overview"
