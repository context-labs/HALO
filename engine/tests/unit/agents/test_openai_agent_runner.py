from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from engine.agents.agent_context import AgentContext
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.openai_agent_runner import OpenAiAgentRunner
from engine.errors import EngineAgentExhaustedError
from engine.model_config import ModelConfig


def _assistant_event(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="run_item_stream_event",
        item=SimpleNamespace(
            type="message_output_item",
            raw_item=SimpleNamespace(
                id="m1",
                role="assistant",
                content=[SimpleNamespace(type="output_text", text=text)],
            ),
        ),
    )


class _FakeStream:
    def __init__(self, events: list) -> None:
        self._events = events

    async def stream_events(self):
        for e in self._events:
            yield e


def _context() -> AgentContext:
    return AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=2,
        tool_call_compaction_keep_last_messages=2,
    )


@pytest.mark.asyncio
async def test_runner_emits_final_output_and_updates_context() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root", agent_name="root", depth=0,
        parent_agent_id=None, parent_tool_call_id=None,
    )

    async def fake_run_streamed(*, agent, input, context):
        return _FakeStream([_assistant_event("answer\n<final/>")])

    compact_calls: list[int] = []

    async def fake_compactor(item):
        compact_calls.append(1)
        return "sum"

    runner = OpenAiAgentRunner(
        run_streamed=fake_run_streamed,
        compactor_factory=lambda _: fake_compactor,
    )

    await runner.run(
        sdk_agent=object(),
        agent_context=ctx,
        agent_execution=execution,
        output_bus=bus,
        is_root=True,
    )

    await bus.close()
    events = [e async for e in bus.stream()]
    assert any(getattr(e, "final", False) for e in events)
    assert any(item.role == "assistant" for item in ctx.items)


@pytest.mark.asyncio
async def test_runner_circuit_breaker() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root", agent_name="root", depth=0,
        parent_agent_id=None, parent_tool_call_id=None,
    )

    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    async def always_fail(*, agent, input, context):
        raise APIConnectionError(request=fake_request)

    async def noop_compactor(_):
        return ""

    runner = OpenAiAgentRunner(
        run_streamed=always_fail,
        compactor_factory=lambda _: noop_compactor,
    )

    with pytest.raises(EngineAgentExhaustedError):
        await runner.run(
            sdk_agent=object(),
            agent_context=ctx,
            agent_execution=execution,
            output_bus=bus,
            is_root=True,
        )


@pytest.mark.asyncio
async def test_runner_does_not_retry_on_bad_request() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root", agent_name="root", depth=0,
        parent_agent_id=None, parent_tool_call_id=None,
    )

    call_count = 0
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fake_response = httpx.Response(400, request=fake_request)

    async def raise_400(*, agent, input, context):
        nonlocal call_count
        call_count += 1
        raise BadRequestError(
            message="bad field",
            response=fake_response,
            body={"error": {"message": "bad field"}},
        )

    async def noop_compactor(_):
        return ""

    runner = OpenAiAgentRunner(
        run_streamed=raise_400,
        compactor_factory=lambda _: noop_compactor,
    )

    with pytest.raises(BadRequestError):
        await runner.run(
            sdk_agent=object(),
            agent_context=ctx,
            agent_execution=execution,
            output_bus=bus,
            is_root=True,
        )
    assert call_count == 1


@pytest.mark.asyncio
async def test_runner_retries_on_connection_error_then_fails() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root", agent_name="root", depth=0,
        parent_agent_id=None, parent_tool_call_id=None,
    )

    call_count = 0
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    async def raise_connection(*, agent, input, context):
        nonlocal call_count
        call_count += 1
        raise APIConnectionError(request=fake_request)

    async def noop_compactor(_):
        return ""

    runner = OpenAiAgentRunner(
        run_streamed=raise_connection,
        compactor_factory=lambda _: noop_compactor,
    )

    with pytest.raises(EngineAgentExhaustedError):
        await runner.run(
            sdk_agent=object(),
            agent_context=ctx,
            agent_execution=execution,
            output_bus=bus,
            is_root=True,
        )
    assert call_count == 10
