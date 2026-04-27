from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from agents.tool_context import ToolContext as SdkToolContext

from engine.agents.agent_config import AgentConfig
from engine.agents.agent_context import AgentContext
from engine.agents.agent_context_items import AgentContextItem
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


def _fake_parent_context() -> AgentContext:
    return AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=2,
        tool_call_compaction_keep_last_turns=2,
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
        parent_context=_fake_parent_context(),
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
        parent_context=_fake_parent_context(),
    )
    names = {t.name for t in tools}
    assert "call_subagent" in names


@pytest.mark.asyncio
async def test_guarded_invoke_raises_when_child_depth_exceeds_maximum() -> None:
    """Defense-in-depth: structural guard in ``_child_tools_for_depth`` keeps this
    unreachable in normal flow, but the runtime check must still fire if a future
    refactor bypasses the structural enforcement. Constructed directly to exercise it."""
    from engine.errors import EngineMaxDepthExceededError

    cfg = _engine_config(max_depth=2)
    fake_store = MagicMock(spec=TraceStore)
    run_state = EngineRunState(trace_store=fake_store, output_bus=EngineOutputBus(), config=cfg)
    run_state.runner = MagicMock()

    sem = asyncio.Semaphore(1)
    tool = _build_subagent_as_tool(
        run_state=run_state,
        child_depth=cfg.maximum_depth + 1,
        semaphore=sem,
        parent_execution=_fake_parent(),
    )

    with pytest.raises(EngineMaxDepthExceededError):
        await tool.on_invoke_tool(_fake_tool_ctx(), '{"input": "ask child"}')


@pytest.mark.asyncio
async def test_get_context_item_resolves_through_wired_agent_context() -> None:
    """``make_ctx`` must populate ``ToolContext.agent_context`` so ``get_context_item``
    can resolve item ids against the calling agent's stored items."""
    cfg = _engine_config(max_depth=2)
    run_state = MagicMock(spec=EngineRunState)
    run_state.config = cfg
    run_state.output_bus = EngineOutputBus()
    run_state.trace_store = MagicMock()

    parent_context = _fake_parent_context()
    parent_context.append(AgentContextItem(item_id="ctx-42", role="user", content="stored content"))

    sem = asyncio.Semaphore(1)
    tools = _child_tools_for_depth(
        depth=0,
        run_state=run_state,
        semaphore=sem,
        parent_execution=_fake_parent(),
        parent_context=parent_context,
    )

    get_context_tool = next(t for t in tools if t.name == "get_context_item")
    result_json = await get_context_tool.on_invoke_tool(
        MagicMock(spec=SdkToolContext), '{"item_id": "ctx-42"}'
    )
    assert "stored content" in result_json
    assert "ctx-42" in result_json


@pytest.mark.asyncio
async def test_guarded_invoke_returns_failure_on_exception() -> None:
    cfg = EngineConfig(
        root_agent=AgentConfig(
            name="r", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3
        ),
        subagent=AgentConfig(
            name="s", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3
        ),
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

    result_json = await tool.on_invoke_tool(_fake_tool_ctx(), '{"input": "ask child"}')
    result = SubagentToolResult.model_validate_json(result_json)
    assert "SDK exploded" in result.answer


@pytest.mark.asyncio
async def test_guarded_invoke_counts_turns_and_tool_calls(monkeypatch) -> None:
    cfg = EngineConfig(
        root_agent=AgentConfig(
            name="r", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3
        ),
        subagent=AgentConfig(
            name="s", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3
        ),
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
                raw_item=SimpleNamespace(
                    call_id="c1", id="c1", name="query_traces", arguments="{}"
                ),
            ),
        ),
        SimpleNamespace(
            type="run_item_stream_event",
            item=SimpleNamespace(
                type="message_output_item",
                raw_item=SimpleNamespace(
                    id="m1",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="done")],
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
    result_json = await tool.on_invoke_tool(_fake_tool_ctx(), '{"input": "ask child"}')
    result = SubagentToolResult.model_validate_json(result_json)
    assert result.turns_used == 1
    assert result.tool_calls_made == 1


@pytest.mark.asyncio
async def test_guarded_invoke_passes_parsed_input_not_raw_json() -> None:
    """The subagent's first user message must be ``params["input"]``, not the JSON wrapper."""
    cfg = EngineConfig(
        root_agent=AgentConfig(
            name="r", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3
        ),
        subagent=AgentConfig(
            name="s", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3
        ),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        maximum_depth=1,
    )
    fake_store = MagicMock(spec=TraceStore)
    run_state = EngineRunState(trace_store=fake_store, output_bus=EngineOutputBus(), config=cfg)

    captured_inputs: list[list[dict]] = []

    class _EmptyStream:
        async def stream_events(self):
            return
            yield  # pragma: no cover - makes this an async generator

        async def wait_for_final_output(self):
            return self

    class _CapturingRunner:
        @staticmethod
        def run_streamed(*, starting_agent, input, context, max_turns):
            captured_inputs.append(list(input))
            return _EmptyStream()

    run_state.runner = _CapturingRunner

    sem = asyncio.Semaphore(1)
    tool = _build_subagent_as_tool(
        run_state=run_state,
        child_depth=1,
        semaphore=sem,
        parent_execution=_fake_parent(),
    )

    await tool.on_invoke_tool(_fake_tool_ctx(), '{"input": "what is the failure rate?"}')

    assert len(captured_inputs) == 1
    user_messages = [m for m in captured_inputs[0] if m.get("role") == "user"]
    assert len(user_messages) == 1
    assert user_messages[0]["content"] == "what is the failure rate?"


@pytest.mark.asyncio
async def test_guarded_invoke_extracts_child_answer_from_raw_item(monkeypatch) -> None:
    cfg = EngineConfig(
        root_agent=AgentConfig(
            name="r", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3
        ),
        subagent=AgentConfig(
            name="s", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3
        ),
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
    result_json = await tool.on_invoke_tool(_fake_tool_ctx(), '{"input": "ask child"}')
    result = SubagentToolResult.model_validate_json(result_json)
    assert result.answer == "child says 42"
