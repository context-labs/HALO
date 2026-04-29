from __future__ import annotations

import pytest

from engine.agents.engine_output_bus import EngineOutputBus
from engine.models.engine_output import AgentOutputItem, AgentTextDelta
from engine.models.messages import AgentMessage


def _msg(agent: str = "root", text: str = "hi") -> AgentOutputItem:
    return AgentOutputItem(
        sequence=0,
        agent_id=agent,
        parent_agent_id=None,
        parent_tool_call_id=None,
        agent_name=agent,
        depth=0,
        item=AgentMessage(role="assistant", content=text),
    )


@pytest.mark.asyncio
async def test_bus_assigns_monotonic_sequences() -> None:
    bus = EngineOutputBus()
    a = await bus.emit(_msg(text="a"))
    b = await bus.emit(_msg(text="b"))
    assert a.sequence == 0
    assert b.sequence == 1


@pytest.mark.asyncio
async def test_bus_stream_emits_and_closes() -> None:
    bus = EngineOutputBus()
    await bus.emit(_msg(text="a"))
    await bus.close()
    collected = [item async for item in bus.stream()]
    assert len(collected) == 1


@pytest.mark.asyncio
async def test_bus_fail_propagates_after_drain() -> None:
    bus = EngineOutputBus()
    await bus.emit(_msg(text="a"))
    await bus.fail(RuntimeError("boom"))
    events: list = []
    with pytest.raises(RuntimeError, match="boom"):
        async for ev in bus.stream():
            events.append(ev)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_bus_handles_deltas_and_items() -> None:
    bus = EngineOutputBus()
    await bus.emit(_msg(text="full"))
    await bus.emit(
        AgentTextDelta(
            sequence=0,
            agent_id="root",
            parent_agent_id=None,
            parent_tool_call_id=None,
            depth=0,
            item_id="x",
            text_delta="par",
        )
    )
    await bus.close()
    events = [ev async for ev in bus.stream()]
    assert [type(ev).__name__ for ev in events] == ["AgentOutputItem", "AgentTextDelta"]
    assert events[1].sequence == 1
