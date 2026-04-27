from __future__ import annotations

import pytest

from engine.agents.agent_context import AgentContext
from engine.agents.agent_context_items import AgentContextItem
from engine.model_config import ModelConfig
from engine.tools.agent_context_tools import (
    GetContextItemArguments,
    GetContextItemTool,
)
from engine.tools.tool_protocol import ToolContext


@pytest.mark.asyncio
async def test_get_context_item_returns_full_stored_item() -> None:
    agent_context = AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=2,
        tool_call_compaction_keep_last_turns=2,
    )
    agent_context.append(
        AgentContextItem(
            item_id="m1",
            role="user",
            content="hi",
            is_compacted=True,
            compaction_summary="user said hi",
        )
    )
    ctx = ToolContext.model_construct(agent_context=agent_context)

    tool = GetContextItemTool()
    result = await tool.run(ctx, GetContextItemArguments(item_id="m1"))
    assert result.item.item_id == "m1"
    assert result.item.content == "hi"
    assert result.item.compaction_summary == "user said hi"
