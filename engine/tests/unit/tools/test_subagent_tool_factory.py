from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from engine.agents.agent_config import AgentConfig
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig
from engine.tools.subagent_tool_factory import _child_tools_for_depth


def _engine_config(max_depth: int) -> EngineConfig:
    agent = AgentConfig(
        name="a", instructions="",
        model=ModelConfig(name="claude-sonnet-4-5"), maximum_turns=10,
    )
    return EngineConfig(
        root_agent=agent, subagent=agent,
        synthesis_model=ModelConfig(name="claude-haiku-4-5"),
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        maximum_depth=max_depth,
    )


def test_child_tools_at_max_depth_omits_subagent_tool() -> None:
    cfg = _engine_config(max_depth=2)
    run_state = MagicMock(spec=EngineRunState)
    run_state.config = cfg
    run_state.output_bus = EngineOutputBus()
    run_state.trace_store = MagicMock()
    tools = _child_tools_for_depth(depth=2, run_state=run_state, semaphore=None)
    names = {t.name for t in tools}
    assert "call_subagent" not in names


def test_child_tools_below_max_depth_includes_subagent_tool() -> None:
    cfg = _engine_config(max_depth=2)
    run_state = MagicMock(spec=EngineRunState)
    run_state.config = cfg
    run_state.output_bus = EngineOutputBus()
    run_state.trace_store = MagicMock()
    import asyncio
    sem = asyncio.Semaphore(4)
    tools = _child_tools_for_depth(depth=1, run_state=run_state, semaphore=sem)
    names = {t.name for t in tools}
    assert "call_subagent" in names


@pytest.mark.asyncio
async def test_semaphore_wrapper_limits_parallelism() -> None:
    from engine.tools.subagent_tool_factory import _wrap_with_semaphore
    import asyncio

    in_flight = 0
    peak = 0

    async def fake_tool(ctx, args):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return "ok"

    sem = asyncio.Semaphore(2)
    wrapped = _wrap_with_semaphore(fake_tool, sem)

    await asyncio.gather(*[wrapped(None, "{}") for _ in range(6)])
    assert peak <= 2


@pytest.mark.asyncio
async def test_guarded_invoke_returns_failure_on_exception() -> None:
    import asyncio
    from unittest.mock import MagicMock

    from engine.agents.engine_output_bus import EngineOutputBus
    from engine.agents.engine_run_state import EngineRunState
    from engine.agents.agent_config import AgentConfig
    from engine.engine_config import EngineConfig
    from engine.model_config import ModelConfig
    from engine.tools.subagent_result import SubagentToolResult
    from engine.tools.subagent_tool_factory import _build_subagent_as_tool
    from engine.traces.trace_store import TraceStore
    from engine.tools import subagent_tool_factory as mod

    cfg = EngineConfig(
        root_agent=AgentConfig(name="r", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3),
        subagent=AgentConfig(name="s", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        maximum_depth=1,
    )
    fake_store = MagicMock(spec=TraceStore)
    run_state = EngineRunState(trace_store=fake_store, output_bus=EngineOutputBus(), config=cfg)

    sem = asyncio.Semaphore(1)
    tool = _build_subagent_as_tool(run_state=run_state, child_depth=1, semaphore=sem)

    def _raise(*args, **kwargs):
        raise RuntimeError("SDK exploded")

    orig = mod.Runner.run_streamed
    mod.Runner.run_streamed = _raise
    try:
        result_json = await tool.on_invoke_tool(None, "{}")
    finally:
        mod.Runner.run_streamed = orig

    result = SubagentToolResult.model_validate_json(result_json)
    assert "SDK exploded" in result.answer
