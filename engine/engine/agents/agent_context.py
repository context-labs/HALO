from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeAlias

from engine.agents.agent_context_items import AgentContextItem
from engine.agents.prompt_templates import render_root_system_prompt
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage

if TYPE_CHECKING:
    from engine.engine_config import EngineConfig

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

    @classmethod
    def from_input_messages(
        cls,
        messages: list[AgentMessage],
        engine_config: "EngineConfig",
    ) -> "AgentContext":
        """Build a root AgentContext from caller-supplied messages.

        Two cases:
          1. No system message at front: prepend the engine-rendered system prompt.
          2. Front message is already a system message: pass through unchanged.
             The caller is responsible for whatever it contains. This supports
             continuations and lets users supply their own system prompts.
        """
        has_system = bool(messages) and messages[0].role == "system"

        if has_system:
            sys_item = AgentContextItem(
                item_id="sys-0",
                role="system",
                content=messages[0].content,
            )
            body = messages[1:]
        else:
            rendered = render_root_system_prompt(
                instructions=engine_config.root_agent.instructions,
                maximum_depth=engine_config.maximum_depth,
                maximum_parallel_subagents=engine_config.maximum_parallel_subagents,
            )
            sys_item = AgentContextItem(item_id="sys-0", role="system", content=rendered)
            body = messages

        items: list[AgentContextItem] = [sys_item]
        for i, msg in enumerate(body):
            items.append(AgentContextItem(
                item_id=f"in-{i}",
                role=msg.role,
                content=msg.content,
                tool_calls=msg.tool_calls,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            ))

        return cls(
            items=items,
            compaction_model=engine_config.compaction_model,
            text_message_compaction_keep_last_messages=engine_config.text_message_compaction_keep_last_messages,
            tool_call_compaction_keep_last_messages=engine_config.tool_call_compaction_keep_last_messages,
        )

    def append(self, item: AgentContextItem) -> None:
        self.items.append(item)
        self._index[item.item_id] = item

    def get_item(self, item_id: str) -> AgentContextItem:
        return self._index[item_id]

    def to_messages_array(self) -> list[AgentMessage]:
        return [_render_item(item) for item in self.items]

    async def compact_old_items(self, compactor: "Compactor") -> None:
        text_positions: list[int] = []
        tool_positions: list[int] = []
        for idx, item in enumerate(self.items):
            if item.is_compacted or item.role == "system":
                continue
            if _is_tool_related(item):
                tool_positions.append(idx)
            else:
                text_positions.append(idx)

        eligible: list[int] = []
        if len(text_positions) > self.text_message_compaction_keep_last_messages:
            cutoff = len(text_positions) - self.text_message_compaction_keep_last_messages
            eligible.extend(text_positions[:cutoff])
        if len(tool_positions) > self.tool_call_compaction_keep_last_messages:
            cutoff = len(tool_positions) - self.tool_call_compaction_keep_last_messages
            eligible.extend(tool_positions[:cutoff])

        for idx in sorted(eligible):
            item = self.items[idx]
            summary = await compactor(item)
            self.items[idx] = item.model_copy(update={"is_compacted": True, "compaction_summary": summary})
            self._index[item.item_id] = self.items[idx]


def _is_tool_related(item: AgentContextItem) -> bool:
    if item.role == "tool":
        return True
    if item.role == "assistant" and item.tool_calls:
        return True
    return False


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
