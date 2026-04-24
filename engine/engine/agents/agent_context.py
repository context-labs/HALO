from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeAlias

from engine.agents.agent_context_items import AgentContextItem
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage

Compactor: TypeAlias = Callable[[AgentContextItem], Awaitable[str]]


class AgentContext:
    def __init__(
        self,
        items: list[AgentContextItem],
        compaction_model: ModelConfig,
        text_message_compaction_keep_last_messages: int,
        tool_call_compaction_keep_last_messages: int,
    ) -> None:
        self.items = list(items)
        self.compaction_model = compaction_model
        self.text_message_compaction_keep_last_messages = text_message_compaction_keep_last_messages
        self.tool_call_compaction_keep_last_messages = tool_call_compaction_keep_last_messages
        self._index: dict[str, AgentContextItem] = {item.item_id: item for item in items}

    def append(self, item: AgentContextItem) -> None:
        self.items.append(item)
        self._index[item.item_id] = item

    def get_item(self, item_id: str) -> AgentContextItem:
        return self._index[item_id]

    def to_messages_array(self) -> list[AgentMessage]:
        return [_render_item(item) for item in self.items]


def _render_item(item: AgentContextItem) -> AgentMessage:
    if not item.is_compacted:
        return AgentMessage(
            role=item.role,
            content=item.content,
            tool_calls=item.tool_calls,
            tool_call_id=item.tool_call_id,
            name=item.name,
        )

    summary = item.compaction_summary or ""
    if item.role == "user":
        return AgentMessage(role="user", content=f"Compacted message (id: {item.item_id}): {summary}")
    if item.role == "assistant":
        return AgentMessage(
            role="assistant",
            content=f"Compacted tool calls (id: {item.item_id}): {summary}",
        )
    if item.role == "tool":
        tool_name = item.name or "tool"
        return AgentMessage(
            role="assistant",
            content=f"Compacted tool result (id: {item.item_id}, tool: {tool_name}): {summary}",
        )
    return AgentMessage(role=item.role, content=item.content)
