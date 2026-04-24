from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.agents.agent_context import AgentContext
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.openai_agent_runner import OpenAiAgentRunner
from engine.errors import EngineAgentExhaustedError
from engine.model_config import ModelConfig


def _assistant_event(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="message_output_item",
        message=SimpleNamespace(
            id="m1",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text=text)],
            tool_calls=None,
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

    async def always_fail(*, agent, input, context):
        raise RuntimeError("provider 500")

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
