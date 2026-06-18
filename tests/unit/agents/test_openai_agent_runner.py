from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError, APIError, AsyncOpenAI, BadRequestError

from engine.agents.agent_context import AgentContext
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.openai_agent_runner import OpenAiAgentRunner
from engine.errors import EngineAgentExhaustedError, EngineAgentRefusedError
from engine.model_config import ModelConfig
from tests._sdk_events import (
    assistant_message_event,
    assistant_refusal_event,
    tool_call_event,
    tool_output_event,
)

_DUMMY_CLIENT = AsyncOpenAI(api_key="test")


def _assistant_event(text: str):
    """Local alias keeping the test bodies short."""
    return assistant_message_event(item_id="m1", text=text)


def _refusal_event(text: str):
    return assistant_refusal_event(item_id="r1", refusal=text)


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
        tool_call_compaction_keep_last_turns=2,
    )


@pytest.mark.asyncio
async def test_runner_emits_final_output_and_updates_context() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )

    async def fake_run_streamed(*, agent, input, context):
        return _FakeStream([_assistant_event("answer\n<final/>")])

    runner = OpenAiAgentRunner(
        run_streamed=fake_run_streamed,
        client=_DUMMY_CLIENT,
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
async def test_runner_retries_refusal_without_emitting_refusal() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    calls: list[list[dict]] = []

    async def fake_run_streamed(*, agent, input, context):
        calls.append(input)
        if len(calls) == 1:
            return _FakeStream(
                [_refusal_event("I'm sorry, but I cannot assist with that request.")]
            )
        return _FakeStream([_assistant_event("answer\n<final/>")])

    runner = OpenAiAgentRunner(
        run_streamed=fake_run_streamed,
        client=_DUMMY_CLIENT,
        refusal_retries=1,
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
    assert len(calls) == 2
    assert calls[0] == []
    assert calls[1] == [
        {
            "role": "user",
            "content": "Continue.",
        }
    ]
    assert len(events) == 1
    assert events[0].item.content == "answer"
    assert events[0].final is True
    assert [item.content for item in ctx.items] == ["answer"]


@pytest.mark.asyncio
async def test_runner_keeps_refusal_retry_prompt_after_transient_retry_call_failure() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    calls: list[list[dict]] = []

    async def fake_run_streamed(*, agent, input, context):
        calls.append(input)
        if len(calls) == 1:
            return _FakeStream(
                [_refusal_event("I'm sorry, but I cannot assist with that request.")]
            )
        if len(calls) == 2:
            raise APIConnectionError(request=fake_request)
        return _FakeStream([_assistant_event("answer\n<final/>")])

    runner = OpenAiAgentRunner(
        run_streamed=fake_run_streamed,
        client=_DUMMY_CLIENT,
        refusal_retries=1,
        retry_backoff_base=0.0,
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
    retry_message = {
        "role": "user",
        "content": "Continue.",
    }
    assert len(calls) == 3
    assert calls[1] == [retry_message]
    assert calls[2] == [retry_message]
    assert len(events) == 1
    assert events[0].item.content == "answer"


@pytest.mark.asyncio
async def test_runner_does_not_retry_when_refusal_is_not_last_message() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    call_count = 0

    async def fake_run_streamed(*, agent, input, context):
        nonlocal call_count
        call_count += 1
        return _FakeStream(
            [
                _refusal_event("I'm sorry, but I cannot assist with that request."),
                _assistant_event("answer\n<final/>"),
            ]
        )

    runner = OpenAiAgentRunner(
        run_streamed=fake_run_streamed,
        client=_DUMMY_CLIENT,
        refusal_retries=1,
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
    assert call_count == 1
    assert len(events) == 1
    assert events[0].item.content == "answer"
    assert events[0].final is True
    assert [item.content for item in ctx.items] == ["answer"]


@pytest.mark.asyncio
async def test_runner_raises_after_refusal_retries_exhausted() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    call_count = 0

    async def fake_run_streamed(*, agent, input, context):
        nonlocal call_count
        call_count += 1
        return _FakeStream([_refusal_event("I'm sorry, but I cannot assist with that request.")])

    runner = OpenAiAgentRunner(
        run_streamed=fake_run_streamed,
        client=_DUMMY_CLIENT,
        refusal_retries=1,
    )

    with pytest.raises(EngineAgentRefusedError):
        await runner.run(
            sdk_agent=object(),
            agent_context=ctx,
            agent_execution=execution,
            output_bus=bus,
            is_root=True,
        )
    assert call_count == 2
    assert ctx.items == []


@pytest.mark.asyncio
async def test_runner_retries_refusal_after_tool_result_without_replaying_tool_output() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    calls: list[list[dict]] = []

    async def fake_run_streamed(*, agent, input, context):
        calls.append(input)
        if len(calls) == 1:
            return _FakeStream(
                [
                    tool_call_event(call_id="call_1", name="query_traces", arguments='{"q":"x"}'),
                    tool_output_event(call_id="call_1", output="trace result"),
                    _refusal_event("I'm sorry, but I cannot assist with that request."),
                ]
            )
        return _FakeStream([_assistant_event("answer\n<final/>")])

    runner = OpenAiAgentRunner(
        run_streamed=fake_run_streamed,
        client=_DUMMY_CLIENT,
        refusal_retries=1,
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
    assert len(calls) == 2
    assert calls[1] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "query_traces", "arguments": '{"q":"x"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": "trace result",
            "tool_call_id": "call_1",
            "name": "query_traces",
        },
        {
            "role": "user",
            "content": "Continue.",
        },
    ]
    assert [event.item.role for event in events] == ["assistant", "tool", "assistant"]
    assert events[-1].item.content == "answer"
    assert execution.tool_calls_made == 1
    assert execution.turns_used == 1


@pytest.mark.asyncio
async def test_runner_circuit_breaker() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )

    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    async def always_fail(*, agent, input, context):
        raise APIConnectionError(request=fake_request)

    runner = OpenAiAgentRunner(
        run_streamed=always_fail,
        client=_DUMMY_CLIENT,
        retry_backoff_base=0.0,
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
async def test_runner_does_not_retry_on_terminal_400() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )

    call_count = 0
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fake_response = httpx.Response(400, request=fake_request)

    async def raise_400(*, agent, input, context):
        nonlocal call_count
        call_count += 1
        raise BadRequestError(
            message="too many tokens",
            response=fake_response,
            body={"message": "too many tokens", "code": "context_length_exceeded"},
        )

    runner = OpenAiAgentRunner(
        run_streamed=raise_400,
        client=_DUMMY_CLIENT,
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
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )

    call_count = 0
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    async def raise_connection(*, agent, input, context):
        nonlocal call_count
        call_count += 1
        raise APIConnectionError(request=fake_request)

    runner = OpenAiAgentRunner(
        run_streamed=raise_connection,
        client=_DUMMY_CLIENT,
        retry_backoff_base=0.0,
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


@pytest.mark.asyncio
async def test_runner_retries_plain_api_error_from_backend() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )

    call_count = 0
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    async def fail_then_recover(*, agent, input, context):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise APIError(
                message=(
                    "Backend returned unexpected response. Please contact Microsoft for help."
                ),
                request=fake_request,
                body=None,
            )
        return _FakeStream([_assistant_event("answer\n<final/>")])

    runner = OpenAiAgentRunner(
        run_streamed=fail_then_recover,
        client=_DUMMY_CLIENT,
        retry_backoff_base=0.0,
    )

    await runner.run(
        sdk_agent=object(),
        agent_context=ctx,
        agent_execution=execution,
        output_bus=bus,
        is_root=True,
    )

    assert call_count == 2
    assert execution.consecutive_llm_failures == 0


class _RaisingStream:
    """Yields nothing and raises ``exc`` on the first iteration step."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def stream_events(self):
        raise self._exc
        yield  # pragma: no cover - makes this an async generator


@pytest.mark.asyncio
async def test_runner_retries_when_stream_iteration_raises_retriable() -> None:
    """Lazy LLM errors surface from ``stream_events()`` and must engage the breaker."""
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )

    call_count = 0
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    async def stream_that_raises(*, agent, input, context):
        nonlocal call_count
        call_count += 1
        return _RaisingStream(APIConnectionError(request=fake_request))

    runner = OpenAiAgentRunner(
        run_streamed=stream_that_raises,
        client=_DUMMY_CLIENT,
        retry_backoff_base=0.0,
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


@pytest.mark.asyncio
async def test_runner_propagates_non_retriable_stream_iteration_error() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )

    call_count = 0
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fake_response = httpx.Response(400, request=fake_request)

    async def stream_that_raises(*, agent, input, context):
        nonlocal call_count
        call_count += 1
        return _RaisingStream(
            BadRequestError(
                message="too many tokens",
                response=fake_response,
                body={"message": "too many tokens", "code": "context_length_exceeded"},
            )
        )

    runner = OpenAiAgentRunner(
        run_streamed=stream_that_raises,
        client=_DUMMY_CLIENT,
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


class _StreamYieldsThenRaises:
    """Yields ``events`` successfully, then raises ``exc`` on the next iteration step.

    Models a network drop after the LLM has streamed some tokens — the case where
    state has already been mutated and a retry would corrupt it.
    """

    def __init__(self, events: list, exc: BaseException) -> None:
        self._events = events
        self._exc = exc

    async def stream_events(self):
        for e in self._events:
            yield e
        raise self._exc


@pytest.mark.asyncio
async def test_runner_reruns_from_local_history_after_mid_stream_failure() -> None:
    """A retriable mid-stream failure no longer kills the run (INF-3504): the
    runner reruns from the local conversation history, preserving completed
    items from the failed attempt."""
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )

    calls: list[list[dict]] = []
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    async def stream_that_partially_succeeds(*, agent, input, context):
        calls.append(input)
        if len(calls) == 1:
            return _StreamYieldsThenRaises(
                [_assistant_event("partial answer")],
                APIConnectionError(request=fake_request),
            )
        return _FakeStream([_assistant_event("answer\n<final/>")])

    runner = OpenAiAgentRunner(
        run_streamed=stream_that_partially_succeeds,
        client=_DUMMY_CLIENT,
        retry_backoff_base=0.0,
    )

    await runner.run(
        sdk_agent=object(),
        agent_context=ctx,
        agent_execution=execution,
        output_bus=bus,
        is_root=True,
    )

    assert len(calls) == 2
    # The retry replays the completed assistant message from local history.
    assert calls[1] == [{"role": "assistant", "content": "partial answer"}]
    assert [item.content for item in ctx.items] == ["partial answer", "answer"]
    assert execution.consecutive_llm_failures == 0


@pytest.mark.asyncio
async def test_runner_trims_incomplete_tool_turn_before_mid_stream_retry() -> None:
    """When the stream dies after the model emitted tool_calls but before the
    matching results landed, the orphan tool-call turn is trimmed so the retried
    message array stays valid for the LLM API."""
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )

    calls: list[list[dict]] = []
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    async def stream_dies_mid_tool_turn(*, agent, input, context):
        calls.append(input)
        if len(calls) == 1:
            return _StreamYieldsThenRaises(
                [tool_call_event(call_id="call_1", name="query_traces", arguments='{"q":"x"}')],
                APIConnectionError(request=fake_request),
            )
        return _FakeStream([_assistant_event("answer\n<final/>")])

    runner = OpenAiAgentRunner(
        run_streamed=stream_dies_mid_tool_turn,
        client=_DUMMY_CLIENT,
        retry_backoff_base=0.0,
    )

    await runner.run(
        sdk_agent=object(),
        agent_context=ctx,
        agent_execution=execution,
        output_bus=bus,
        is_root=True,
    )

    assert len(calls) == 2
    # The orphan assistant tool-call item was trimmed before the retry.
    assert calls[1] == []
    assert [item.content for item in ctx.items] == ["answer"]
    assert execution.tool_calls_made == 0
    assert execution.consecutive_llm_failures == 0


@pytest.mark.asyncio
async def test_runner_retries_stale_response_state_400_mid_stream() -> None:
    """Stale ``rs_*`` Responses-state 400s (INF-3308) are retriable: the rerun
    rebuilds the request from local history and does not replay the stale ids."""
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )

    calls: list[list[dict]] = []
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fake_response = httpx.Response(400, request=fake_request)

    async def stream_hits_stale_reasoning_item(*, agent, input, context):
        calls.append(input)
        if len(calls) == 1:
            return _StreamYieldsThenRaises(
                [_assistant_event("worked a bit")],
                BadRequestError(
                    message="Item with id 'rs_0123abc' not found.",
                    response=fake_response,
                    body={"error": {"message": "Item with id 'rs_0123abc' not found."}},
                ),
            )
        return _FakeStream([_assistant_event("answer\n<final/>")])

    runner = OpenAiAgentRunner(
        run_streamed=stream_hits_stale_reasoning_item,
        client=_DUMMY_CLIENT,
        retry_backoff_base=0.0,
    )

    await runner.run(
        sdk_agent=object(),
        agent_context=ctx,
        agent_execution=execution,
        output_bus=bus,
        is_root=True,
    )

    assert len(calls) == 2
    assert [item.content for item in ctx.items] == ["worked a bit", "answer"]


@pytest.mark.asyncio
async def test_runner_propagates_terminal_400_mid_stream() -> None:
    """Terminal-code 400s mid-stream stay non-retriable — no clean rerun fixes
    e.g. an over-budget context, so an identical replay would fail again."""
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )

    call_count = 0
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fake_response = httpx.Response(400, request=fake_request)

    async def stream_bad_request(*, agent, input, context):
        nonlocal call_count
        call_count += 1
        return _StreamYieldsThenRaises(
            [_assistant_event("partial")],
            BadRequestError(
                message="too many tokens",
                response=fake_response,
                body={"message": "too many tokens", "code": "context_length_exceeded"},
            ),
        )

    runner = OpenAiAgentRunner(
        run_streamed=stream_bad_request,
        client=_DUMMY_CLIENT,
        retry_backoff_base=0.0,
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
