from __future__ import annotations

import pytest

from engine.agents.agent_context import AgentContext
from engine.agents.agent_context_items import AgentContextItem
from engine.model_config import ModelConfig
from engine.models.messages import AgentToolCall, AgentToolFunction


def _ctx() -> AgentContext:
    return AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=2,
        tool_call_compaction_keep_last_messages=2,
    )


def test_append_and_get_item() -> None:
    ctx = _ctx()
    ctx.append(AgentContextItem(item_id="1", role="user", content="hi"))
    assert ctx.get_item("1").content == "hi"


def test_get_item_missing_raises() -> None:
    ctx = _ctx()
    with pytest.raises(KeyError):
        ctx.get_item("nope")


def test_to_messages_array_uncompacted_user() -> None:
    ctx = _ctx()
    ctx.append(AgentContextItem(item_id="1", role="user", content="hi"))
    msgs = ctx.to_messages_array()
    assert len(msgs) == 1
    assert msgs[0].role == "user"
    assert msgs[0].content == "hi"


def test_to_messages_array_assistant_tool_call_item() -> None:
    ctx = _ctx()
    ctx.append(AgentContextItem(
        item_id="2",
        role="assistant",
        content=None,
        tool_calls=[AgentToolCall(id="c1", function=AgentToolFunction(name="x", arguments="{}"))],
    ))
    ctx.append(AgentContextItem(
        item_id="3",
        role="tool",
        content="ok",
        tool_call_id="c1",
        name="x",
    ))
    msgs = ctx.to_messages_array()
    assert msgs[0].role == "assistant" and msgs[0].tool_calls is not None
    assert msgs[1].role == "tool" and msgs[1].tool_call_id == "c1"
