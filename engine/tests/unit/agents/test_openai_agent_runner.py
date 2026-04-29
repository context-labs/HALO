from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from engine.agents import openai_agent_runner as runner_mod
from engine.agents.agent_context import AgentContext
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.openai_agent_runner import OpenAiAgentRunner, configure_default_sdk_client
from engine.errors import EngineAgentExhaustedError
from engine.model_config import ModelConfig
from engine.model_provider_config import ModelProviderConfig
from tests._sdk_events import assistant_message_event


def _assistant_event(text: str):
    """Local alias keeping the test bodies short."""
    return assistant_message_event(item_id="m1", text=text)


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
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
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

    async def noop_compactor(_):
        return ""

    runner = OpenAiAgentRunner(
        run_streamed=stream_that_raises,
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
                message="bad field",
                response=fake_response,
                body={"error": {"message": "bad field"}},
            )
        )

    async def noop_compactor(_):
        return ""

    runner = OpenAiAgentRunner(
        run_streamed=stream_that_raises,
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
async def test_runner_propagates_retriable_iteration_error_after_event_seen() -> None:
    """Once any event has been applied to context/counters/bus, a retriable mid-stream error
    must propagate rather than retry — replay would duplicate context items, double-count
    turns, and re-emit bus events. Turn-boundary recovery is the caller's job."""
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

    async def stream_that_partially_succeeds(*, agent, input, context):
        nonlocal call_count
        call_count += 1
        return _StreamYieldsThenRaises(
            [_assistant_event("partial answer")],
            APIConnectionError(request=fake_request),
        )

    async def noop_compactor(_):
        return ""

    runner = OpenAiAgentRunner(
        run_streamed=stream_that_partially_succeeds,
        compactor_factory=lambda _: noop_compactor,
    )

    with pytest.raises(APIConnectionError):
        await runner.run(
            sdk_agent=object(),
            agent_context=ctx,
            agent_execution=execution,
            output_bus=bus,
            is_root=True,
        )
    assert call_count == 1
    # The single partial assistant message stays in context exactly once.
    assistant_items = [i for i in ctx.items if i.role == "assistant"]
    assert len(assistant_items) == 1


def test_configure_default_sdk_client_noop_when_provider_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, dict]] = []
    monkeypatch.setattr(
        runner_mod, "set_default_openai_client", lambda c, **kw: calls.append((c, kw))
    )
    configure_default_sdk_client(ModelProviderConfig())
    assert calls == []


def test_configure_default_sdk_client_sets_when_base_url_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, dict]] = []
    monkeypatch.setattr(
        runner_mod, "set_default_openai_client", lambda c, **kw: calls.append((c, kw))
    )
    configure_default_sdk_client(
        ModelProviderConfig(base_url="https://example.com/v1/", api_key="sk-x")
    )
    assert len(calls) == 1
    client, kwargs = calls[0]
    assert str(client.base_url).startswith("https://example.com/v1")
    # Tracing must NOT be redirected to the model endpoint — see
    # configure_default_sdk_client docstring.
    assert kwargs == {"use_for_tracing": False}
