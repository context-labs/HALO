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
    """One agent's conversation memory, with compaction-aware rendering to AgentMessage.

    Each agent (root and every subagent) owns its own AgentContext. Stored items keep
    their original fields plus compaction state; ``to_messages_array`` renders them
    into provider-compatible messages, substituting summaries for items where
    ``is_compacted=True``. Tool turns are compacted as a unit (assistant tool_calls
    plus matching role=tool results) so message arrays stay valid for the LLM API.
    """

    def __init__(
        self,
        items: list[AgentContextItem],
        compaction_model: ModelConfig,
        text_message_compaction_keep_last_messages: int,
        tool_call_compaction_keep_last_turns: int,
    ) -> None:
        self.items = list(items)
        self.compaction_model = compaction_model
        self.text_message_compaction_keep_last_messages = text_message_compaction_keep_last_messages
        self.tool_call_compaction_keep_last_turns = tool_call_compaction_keep_last_turns
        self._index: dict[str, AgentContextItem] = {item.item_id: item for item in items}

    @classmethod
    def from_input_messages(
        cls,
        *,
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
            items.append(
                AgentContextItem(
                    item_id=f"in-{i}",
                    role=msg.role,
                    content=msg.content,
                    tool_calls=msg.tool_calls,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            )

        return cls(
            items=items,
            compaction_model=engine_config.compaction_model,
            text_message_compaction_keep_last_messages=engine_config.text_message_compaction_keep_last_messages,
            tool_call_compaction_keep_last_turns=engine_config.tool_call_compaction_keep_last_turns,
        )

    def append(self, item: AgentContextItem) -> None:
        """Append a new context item and index it by ``item_id`` for ``get_item`` lookups."""
        self.items.append(item)
        self._index[item.item_id] = item

    def get_item(self, item_id: str) -> AgentContextItem:
        """Return the full stored item (including original fields and any compaction summary)."""
        return self._index[item_id]

    def to_messages_array(self) -> list[AgentMessage]:
        """Render stored items into provider-compatible messages, swapping in summaries for compacted items."""
        return [_render_item(item) for item in self.items]

    async def compact_old_items(self, compactor: "Compactor") -> None:
        """Compact eligible older items in place using two independent keep-last thresholds.

        Text messages and tool turns are tracked separately; tool turns (assistant
        tool_calls + matching results) are always compacted together so the rendered
        message array stays valid for the LLM API. System messages are never touched.
        """
        text_positions: list[int] = []
        tool_groups = _build_tool_groups(self.items)

        for idx, item in enumerate(self.items):
            if item.is_compacted or item.role == "system":
                continue
            if not _is_tool_related(item):
                text_positions.append(idx)

        eligible: list[int] = []
        if len(text_positions) > self.text_message_compaction_keep_last_messages:
            cutoff = len(text_positions) - self.text_message_compaction_keep_last_messages
            eligible.extend(text_positions[:cutoff])
        if len(tool_groups) > self.tool_call_compaction_keep_last_turns:
            cutoff = len(tool_groups) - self.tool_call_compaction_keep_last_turns
            for group in tool_groups[:cutoff]:
                eligible.extend(group)

        for idx in sorted(set(eligible)):
            item = self.items[idx]
            summary = await compactor(item)
            self.items[idx] = item.model_copy(
                update={"is_compacted": True, "compaction_summary": summary}
            )
            self._index[item.item_id] = self.items[idx]


def _is_tool_related(item: AgentContextItem) -> bool:
    """True for assistant messages with tool_calls and for role=tool results."""
    if item.role == "tool":
        return True
    if item.role == "assistant" and item.tool_calls:
        return True
    return False


def _build_tool_groups(items: list[AgentContextItem]) -> list[list[int]]:
    """Group tool-related items into conversational turns.

    A turn = one assistant message that emitted tool_calls, paired with the
    role=tool result messages whose tool_call_id matches one of those calls.

    The returned list preserves turn order (oldest first) and contains item
    indices into ``items``. Already-compacted items are skipped so a second
    pass cannot re-compact them or accidentally pull a still-uncompacted
    result into a turn whose assistant has already been collapsed.

    A tool result whose matching assistant tool_call is not present in the
    context (or has already been compacted) is dropped from grouping —
    standalone, it cannot form a coherent turn for OpenAI's API anyway.
    """
    groups: list[list[int]] = []
    call_id_to_group: dict[str, int] = {}
    for idx, item in enumerate(items):
        if item.is_compacted or item.role == "system":
            continue
        if item.role == "assistant" and item.tool_calls:
            groups.append([idx])
            for tc in item.tool_calls:
                call_id_to_group[tc.id] = len(groups) - 1
        elif item.role == "tool" and item.tool_call_id is not None:
            group_index = call_id_to_group.get(item.tool_call_id)
            if group_index is not None:
                groups[group_index].append(idx)
    return groups


def _render_item(item: AgentContextItem) -> AgentMessage:
    """Convert one stored item to an AgentMessage, swapping in the compaction summary if compacted.

    Compacted assistant tool-call items and compacted tool-result items both render
    as plain assistant messages so the resulting message array never has an orphan
    ``role="tool"`` without its matching ``tool_calls`` parent.
    """
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
        return AgentMessage(
            role="user", content=f"Compacted message (id: {item.item_id}): {summary}"
        )
    if item.role == "assistant":
        # Three valid assistant shapes: text only, tool_calls only, or both.
        # Label accurately so the model gets the right hint about what was
        # compacted; the summary itself carries the substantive content.
        if item.tool_calls and item.content:
            label = f"Compacted message with tool calls (id: {item.item_id})"
        elif item.tool_calls:
            label = f"Compacted tool calls (id: {item.item_id})"
        else:
            label = f"Compacted message (id: {item.item_id})"
        return AgentMessage(role="assistant", content=f"{label}: {summary}")
    if item.role == "tool":
        tool_name = item.name or "tool"
        return AgentMessage(
            role="assistant",
            content=f"Compacted tool result (id: {item.item_id}, tool: {tool_name}): {summary}",
        )
    return AgentMessage(role=item.role, content=item.content)
