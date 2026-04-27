from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from agents.tool_context import ToolContext as SdkToolContext

from engine.agents.agent_config import AgentConfig
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig
from engine.tools.subagent_result import SubagentToolResult
from engine.tools.subagent_tool_factory import (
    _build_subagent_as_tool,
    _child_tools_for_depth,
)
from engine.traces.trace_store import TraceStore


def _engine_config(max_depth: int) -> EngineConfig:
    agent = AgentConfig(
        name="a",
        instructions="",
        model=ModelConfig(name="claude-sonnet-4-5"),
        maximum_turns=10,
    )
    return EngineConfig(
        root_agent=agent,
        subagent=agent,
        synthesis_model=ModelConfig(name="claude-haiku-4-5"),
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        maximum_depth=max_depth,
    )


def _fake_parent() -> AgentExecution:
    return AgentExecution(
        agent_id="parent-x",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )


def _fake_tool_ctx(tool_call_id: str = "parent-call-x") -> SdkToolContext:
    return SdkToolContext(
        context=None,
        tool_name="call_subagent",
        tool_call_id=tool_call_id,
        tool_arguments="{}",
    )


def test_child_tools_at_max_depth_omits_subagent_tool() -> None:
    cfg = _engine_config(max_depth=2)
    run_state = MagicMock(spec=EngineRunState)
    run_state.config = cfg
    run_state.output_bus = EngineOutputBus()
    run_state.trace_store = MagicMock()
    sem = asyncio.Semaphore(1)
    tools = _child_tools_for_depth(
        depth=2,
        run_state=run_state,
        semaphore=sem,
        parent_execution=_fake_parent(),
    )
    names = {t.name for t in tools}
    assert "call_subagent" not in names


def test_child_tools_below_max_depth_includes_subagent_tool() -> None:
    cfg = _engine_config(max_depth=2)
    run_state = MagicMock(spec=EngineRunState)
    run_state.config = cfg
    run_state.output_bus = EngineOutputBus()
    run_state.trace_store = MagicMock()
    sem = asyncio.Semaphore(4)
    tools = _child_tools_for_depth(
        depth=1,
        run_state=run_state,
        semaphore=sem,
        parent_execution=_fake_parent(),
    )
    names = {t.name for t in tools}
    assert "call_subagent" in names


@pytest.mark.asyncio
async def test_semaphore_wrapper_limits_parallelism() -> None:
    from engine.tools.subagent_tool_factory import _wrap_with_semaphore

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
    cfg = EngineConfig(
        root_agent=AgentConfig(name="r", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3),
        subagent=AgentConfig(name="s", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        maximum_depth=1,
    )
    fake_store = MagicMock(spec=TraceStore)
    run_state = EngineRunState(trace_store=fake_store, output_bus=EngineOutputBus(), config=cfg)

    class _ExplodingRunner:
        @staticmethod
        def run_streamed(*args, **kwargs):
            raise RuntimeError("SDK exploded")

    run_state.runner = _ExplodingRunner

    sem = asyncio.Semaphore(1)
    tool = _build_subagent_as_tool(
        run_state=run_state,
        child_depth=1,
        semaphore=sem,
        parent_execution=_fake_parent(),
    )

    result_json = await tool.on_invoke_tool(_fake_tool_ctx(), "{}")
    result = SubagentToolResult.model_validate_json(result_json)
    assert "SDK exploded" in result.answer


@pytest.mark.asyncio
async def test_guarded_invoke_counts_turns_and_tool_calls(monkeypatch) -> None:
    cfg = EngineConfig(
        root_agent=AgentConfig(name="r", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3),
        subagent=AgentConfig(name="s", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        maximum_depth=1,
    )
    fake_store = MagicMock(spec=TraceStore)
    run_state = EngineRunState(trace_store=fake_store, output_bus=EngineOutputBus(), config=cfg)

    events = [
        SimpleNamespace(
            type="run_item_stream_event",
            item=SimpleNamespace(
                type="tool_call_item",
                raw_item=SimpleNamespace(call_id="c1", id="c1", name="query_traces", arguments="{}"),
            ),
        ),
        SimpleNamespace(
            type="run_item_stream_event",
            item=SimpleNamespace(
                type="message_output_item",
                raw_item=SimpleNamespace(
                    id="m1", role="assistant", content=[SimpleNamespace(type="output_text", text="done")]
                ),
            ),
        ),
    ]

    class _Stream:
        new_items: list = []

        async def stream_events(self):
            for e in events:
                yield e

        async def wait_for_final_output(self):
            return self

    class _FakeRunner:
        @staticmethod
        def run_streamed(*args, **kwargs):
            return _Stream()

    run_state.runner = _FakeRunner

    sem = asyncio.Semaphore(1)
    tool = _build_subagent_as_tool(
        run_state=run_state,
        child_depth=1,
        semaphore=sem,
        parent_execution=_fake_parent(),
    )
    result_json = await tool.on_invoke_tool(_fake_tool_ctx(), "{}")
    result = SubagentToolResult.model_validate_json(result_json)
    assert result.turns_used == 1
    assert result.tool_calls_made == 1


@pytest.mark.asyncio
async def test_guarded_invoke_extracts_child_answer_from_raw_item(monkeypatch) -> None:
    cfg = EngineConfig(
        root_agent=AgentConfig(name="r", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3),
        subagent=AgentConfig(name="s", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        maximum_depth=1,
    )
    fake_store = MagicMock(spec=TraceStore)
    run_state = EngineRunState(trace_store=fake_store, output_bus=EngineOutputBus(), config=cfg)

    message_item = SimpleNamespace(
        type="message_output_item",
        raw_item=SimpleNamespace(
            id="m1",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text="child says 42")],
        ),
    )
    stream_event = SimpleNamespace(type="run_item_stream_event", item=message_item)

    class _Stream:
        new_items = [message_item]

        async def stream_events(self):
            yield stream_event

        async def wait_for_final_output(self):
            return self

    class _FakeRunner:
        @staticmethod
        def run_streamed(*args, **kwargs):
            return _Stream()

    run_state.runner = _FakeRunner

    sem = asyncio.Semaphore(1)
    tool = _build_subagent_as_tool(
        run_state=run_state,
        child_depth=1,
        semaphore=sem,
        parent_execution=_fake_parent(),
    )
    result_json = await tool.on_invoke_tool(_fake_tool_ctx(), "{}")
    result = SubagentToolResult.model_validate_json(result_json)
    assert result.answer == "child says 42"
