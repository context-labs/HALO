from __future__ import annotations

from engine.agents.agent_context_items import AgentContextItem
from engine.models.messages import AgentToolCall, AgentToolFunction


def test_user_item() -> None:
    item = AgentContextItem(item_id="msg_1", role="user", content="hi")
    assert item.is_compacted is False
    assert item.agent_id is None


def test_assistant_tool_call_item_with_lineage() -> None:
    item = AgentContextItem(
        item_id="msg_2",
        role="assistant",
        content=None,
        tool_calls=[
            AgentToolCall(id="c1", function=AgentToolFunction(name="x", arguments="{}"))
        ],
        agent_id="root",
        parent_agent_id=None,
    )
    assert item.tool_calls is not None


def test_compacted_item() -> None:
    item = AgentContextItem(
        item_id="msg_3", role="user", content="hi",
        is_compacted=True, compaction_summary="User said hi.",
    )
    assert item.is_compacted is True
    assert item.compaction_summary == "User said hi."
