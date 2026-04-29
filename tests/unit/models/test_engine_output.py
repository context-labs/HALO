from __future__ import annotations

from engine.models.engine_output import AgentOutputItem, AgentTextDelta, EngineStreamEvent
from engine.models.messages import AgentMessage


def test_output_item_defaults() -> None:
    item = AgentOutputItem(
        sequence=0,
        agent_id="root",
        parent_agent_id=None,
        parent_tool_call_id=None,
        agent_name="root",
        depth=0,
        item=AgentMessage(role="assistant", content="hi"),
    )
    assert item.final is False


def test_delta_requires_text() -> None:
    delta = AgentTextDelta(
        sequence=1,
        agent_id="root",
        parent_agent_id=None,
        parent_tool_call_id=None,
        depth=0,
        item_id="msg_1",
        text_delta="par",
    )
    assert delta.text_delta == "par"


def test_stream_event_union_accepts_both() -> None:
    events: list[EngineStreamEvent] = [
        AgentOutputItem(
            sequence=0,
            agent_id="root",
            parent_agent_id=None,
            parent_tool_call_id=None,
            agent_name="root",
            depth=0,
            item=AgentMessage(role="assistant", content="hi"),
        ),
        AgentTextDelta(
            sequence=1,
            agent_id="root",
            parent_agent_id=None,
            parent_tool_call_id=None,
            depth=0,
            item_id="msg_1",
            text_delta="x",
        ),
    ]
    assert len(events) == 2
