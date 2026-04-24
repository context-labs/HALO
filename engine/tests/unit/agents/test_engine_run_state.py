from __future__ import annotations

from pathlib import Path

import pytest

from engine.agents.agent_config import AgentConfig
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


def _cfg() -> EngineConfig:
    ac = AgentConfig(
        name="root", instructions="",
        model=ModelConfig(name="claude-sonnet-4-5"), maximum_turns=10,
    )
    return EngineConfig(
        root_agent=ac, subagent=ac,
        synthesis_model=ModelConfig(name="claude-haiku-4-5"),
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
    )


@pytest.mark.asyncio
async def test_run_state_holds_registries(tmp_path: Path, fixtures_dir: Path) -> None:
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(trace_path=trace_path, config=TraceIndexConfig())
    store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    state = EngineRunState(trace_store=store, output_bus=EngineOutputBus(), config=_cfg())

    exec_ = AgentExecution(
        agent_id="root", agent_name="root", depth=0,
        parent_agent_id=None, parent_tool_call_id=None,
    )
    state.register(exec_)
    assert state.get_by_agent_id("root") is exec_

    child = AgentExecution(
        agent_id="sub1", agent_name="sub", depth=1,
        parent_agent_id="root", parent_tool_call_id="call_xyz",
    )
    state.register(child)
    assert state.get_by_tool_call_id("call_xyz") is child
