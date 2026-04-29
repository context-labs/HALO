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
        tool_call_compaction_keep_last_turns=2,
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
    ctx.append(
        AgentContextItem(
            item_id="2",
            role="assistant",
            content=None,
            tool_calls=[
                AgentToolCall(id="c1", function=AgentToolFunction(name="x", arguments="{}"))
            ],
        )
    )
    ctx.append(
        AgentContextItem(
            item_id="3",
            role="tool",
            content="ok",
            tool_call_id="c1",
            name="x",
        )
    )
    msgs = ctx.to_messages_array()
    assert msgs[0].role == "assistant" and msgs[0].tool_calls is not None
    assert msgs[1].role == "tool" and msgs[1].tool_call_id == "c1"


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
        tool_call_compaction_keep_last_turns=2,
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
        tool_call_compaction_keep_last_turns=1,
    )
    for i in range(3):
        ctx.append(
            AgentContextItem(
                item_id=f"a{i}",
                role="assistant",
                content=None,
                tool_calls=[
                    AgentToolCall(id=f"c{i}", function=AgentToolFunction(name="x", arguments="{}"))
                ],
            )
        )
        ctx.append(
            AgentContextItem(
                item_id=f"r{i}",
                role="tool",
                content="ok",
                tool_call_id=f"c{i}",
                name="x",
            )
        )

    stub = _StubCompactor()
    await ctx.compact_old_items(compactor=stub.compact)

    ids_compacted = {call.item_id for call in stub.calls}
    # 3 tool turns; keep_last_turns=1 → 2 oldest turns (4 items) compacted.
    # The third turn (a2 + r2) stays uncompacted as a unit.
    assert ids_compacted == {"a0", "r0", "a1", "r1"}


@pytest.mark.asyncio
async def test_compact_old_items_skips_system_and_already_compacted() -> None:
    ctx = AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=0,
        tool_call_compaction_keep_last_turns=0,
    )
    ctx.append(AgentContextItem(item_id="s", role="system", content="sys"))
    ctx.append(
        AgentContextItem(
            item_id="u1", role="user", content="hi", is_compacted=True, compaction_summary="x"
        )
    )
    ctx.append(AgentContextItem(item_id="u2", role="user", content="hello"))
    stub = _StubCompactor()
    await ctx.compact_old_items(compactor=stub.compact)
    compacted_ids = {c.item_id for c in stub.calls}
    assert compacted_ids == {"u2"}
    assert ctx.get_item("s").is_compacted is False


# --- from_input_messages -----------------------------------------------------

from engine.agents.agent_config import AgentConfig
from engine.agents.prompt_templates import render_root_system_prompt
from engine.engine_config import EngineConfig
from engine.models.messages import AgentMessage


def _engine_config(instructions: str = "Be brief.") -> EngineConfig:
    agent = AgentConfig(
        name="root",
        instructions=instructions,
        model=ModelConfig(name="gpt-5.4-mini"),
        maximum_turns=4,
    )
    return EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub"}),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
    )


def _expected_system(cfg: EngineConfig) -> str:
    return render_root_system_prompt(
        instructions=cfg.root_agent.instructions,
        maximum_depth=cfg.maximum_depth,
        maximum_parallel_subagents=cfg.maximum_parallel_subagents,
    )


def test_from_input_messages_no_system_prepends() -> None:
    cfg = _engine_config()
    messages = [AgentMessage(role="user", content="Find errors")]
    ctx = AgentContext.from_input_messages(messages=messages, engine_config=cfg)
    assert ctx.items[0].role == "system"
    assert ctx.items[0].content == _expected_system(cfg)
    assert ctx.items[0].item_id == "sys-0"
    assert ctx.items[1].role == "user"
    assert ctx.items[1].content == "Find errors"


def test_from_input_messages_continuation_passes_through() -> None:
    cfg = _engine_config()
    sys_text = _expected_system(cfg)
    messages = [
        AgentMessage(role="system", content=sys_text),
        AgentMessage(role="user", content="Original Q"),
        AgentMessage(role="assistant", content="Original A"),
        AgentMessage(role="user", content="Follow-up Q"),
    ]
    ctx = AgentContext.from_input_messages(messages=messages, engine_config=cfg)
    systems = [i for i in ctx.items if i.role == "system"]
    assert len(systems) == 1
    assert systems[0].content == sys_text
    assert [(i.role, i.content) for i in ctx.items[1:]] == [
        ("user", "Original Q"),
        ("assistant", "Original A"),
        ("user", "Follow-up Q"),
    ]


def test_from_input_messages_caller_system_left_alone() -> None:
    cfg = _engine_config()
    custom_system = "You are a pirate. Speak in rhymes."
    messages = [
        AgentMessage(role="system", content=custom_system),
        AgentMessage(role="user", content="Hi"),
    ]
    ctx = AgentContext.from_input_messages(messages=messages, engine_config=cfg)
    # Caller's system preserved verbatim, engine does NOT replace it
    assert ctx.items[0].role == "system"
    assert ctx.items[0].content == custom_system
    roles = [i.role for i in ctx.items]
    assert roles == ["system", "user"]
    assert ctx.items[1].content == "Hi"
