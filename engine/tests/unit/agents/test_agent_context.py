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


from engine.agents.agent_context import AgentContext
from engine.agents.agent_context_items import AgentContextItem


class _StubCompactor:
    def __init__(self) -> None:
        self.calls: list[AgentContextItem] = []

    async def compact(self, item: AgentContextItem) -> str:
        self.calls.append(item)
        return f"SUMMARY({item.item_id})"


@pytest.mark.asyncio
async def test_compact_old_items_only_touches_eligible_text() -> None:
    ctx = AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=2,
        tool_call_compaction_keep_last_messages=2,
    )
    for i in range(4):
        ctx.append(AgentContextItem(item_id=f"t{i}", role="user", content=f"msg {i}"))

    stub = _StubCompactor()
    await ctx.compact_old_items(compactor=stub.compact)

    ids_compacted = {call.item_id for call in stub.calls}
    assert ids_compacted == {"t0", "t1"}
    assert ctx.get_item("t0").is_compacted is True
    assert ctx.get_item("t0").compaction_summary == "SUMMARY(t0)"
    assert ctx.get_item("t3").is_compacted is False


@pytest.mark.asyncio
async def test_compact_old_items_separate_thresholds_for_tools() -> None:
    ctx = AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=10,
        tool_call_compaction_keep_last_messages=1,
    )
    for i in range(3):
        ctx.append(AgentContextItem(
            item_id=f"a{i}",
            role="assistant",
            content=None,
            tool_calls=[AgentToolCall(id=f"c{i}", function=AgentToolFunction(name="x", arguments="{}"))],
        ))
        ctx.append(AgentContextItem(
            item_id=f"r{i}",
            role="tool",
            content="ok",
            tool_call_id=f"c{i}",
            name="x",
        ))

    stub = _StubCompactor()
    await ctx.compact_old_items(compactor=stub.compact)

    ids_compacted = {call.item_id for call in stub.calls}
    assert ids_compacted == {"a0", "r0", "a1", "r1", "a2"}


@pytest.mark.asyncio
async def test_compact_old_items_skips_system_and_already_compacted() -> None:
    ctx = AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=0,
        tool_call_compaction_keep_last_messages=0,
    )
    ctx.append(AgentContextItem(item_id="s", role="system", content="sys"))
    ctx.append(AgentContextItem(item_id="u1", role="user", content="hi", is_compacted=True, compaction_summary="x"))
    ctx.append(AgentContextItem(item_id="u2", role="user", content="hello"))
    stub = _StubCompactor()
    await ctx.compact_old_items(compactor=stub.compact)
    compacted_ids = {c.item_id for c in stub.calls}
    assert compacted_ids == {"u2"}
    assert ctx.get_item("s").is_compacted is False
