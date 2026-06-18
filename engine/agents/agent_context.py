from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from engine.agents.agent_context_items import AgentContextItem
from engine.agents.compactor import compact
from engine.agents.prompt_templates import render_root_system_prompt
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage

if TYPE_CHECKING:
    from engine.code.code_repo import CodeRepo
    from engine.engine_config import EngineConfig
    from engine.git.git_repo import GitRepo

logger = logging.getLogger(__name__)


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
        messages: list[AgentMessage],
        engine_config: "EngineConfig",
        code_repo: "CodeRepo | None",
        git_repo: "GitRepo | None",
    ) -> "AgentContext":
        """Build a root AgentContext from caller-supplied messages.

        Two cases:
          1. No system message at front: prepend the engine-rendered system prompt
             (including the code-repository section when ``code_repo`` is set and
             the git-history section when ``git_repo`` is set).
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
                maximum_depth=engine_config.maximum_depth,
                maximum_parallel_subagents=engine_config.maximum_parallel_subagents,
                code_repo=code_repo,
                git_repo=git_repo,
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

    def trim_incomplete_tool_turn(self, *, min_items: int = 0) -> list[AgentContextItem]:
        """Drop a trailing inconsistent tool turn so the rendered message array
        stays valid for the LLM API after a mid-stream failure.

        Scans backwards for the last non-compacted assistant item carrying
        ``tool_calls`` and validates the tail after it both ways: every call
        id needs a matching ``role=tool`` result, AND every trailing ``tool``
        row must reference one of that turn's call ids. A missing result OR an
        orphan tool row (referencing a call id from a previously trimmed
        attempt) removes the whole turn — the next attempt regenerates it.
        If no assistant tool-call turn exists in the mutable range, trailing
        ``tool`` rows that don't belong to the nearest preceding tool-call
        turn are orphans and are trimmed from the first one onward. Earlier
        turns are complete by construction, so a single check suffices. Never
        trims below ``min_items`` (items that existed before the failed
        attempt are consistent already). Returns the removed items in their
        original order.
        """
        floor = max(min_items, 0)
        trim_from = self._tool_turn_trim_index(floor)
        if trim_from is None:
            return []
        removed = self.items[trim_from:]
        del self.items[trim_from:]
        for item in removed:
            self._index.pop(item.item_id, None)
        return removed

    def _tool_turn_trim_index(self, floor: int) -> int | None:
        """Index to trim from so the trailing tool turn is consistent, or ``None``."""
        for idx in range(len(self.items) - 1, floor - 1, -1):
            item = self.items[idx]
            if item.role == "assistant" and item.tool_calls and not item.is_compacted:
                call_ids = {tc.id for tc in item.tool_calls}
                result_ids = {
                    later.tool_call_id for later in self.items[idx + 1 :] if later.role == "tool"
                }
                # Incomplete (missing results) or polluted (orphan results
                # for other call ids): rerun the whole turn.
                if call_ids != result_ids:
                    return idx
                return None
        # No assistant tool-call turn in the mutable range. Tool rows at or
        # after ``floor`` are only valid if they belong to the nearest
        # preceding (protected) tool-call turn; anything else is an orphan.
        prior_call_ids: set[str] = set()
        for idx in range(floor - 1, -1, -1):
            item = self.items[idx]
            if item.role == "assistant" and item.tool_calls and not item.is_compacted:
                prior_call_ids = {tc.id for tc in item.tool_calls}
                break
            if item.role != "tool":
                break
        for idx in range(floor, len(self.items)):
            item = self.items[idx]
            if item.role == "tool" and item.tool_call_id not in prior_call_ids:
                return idx
        return None

    async def compact_old_items(self, client: AsyncOpenAI) -> None:
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
            try:
                summary = await compact(
                    client=client, compaction_model=self.compaction_model, item=item
                )
            except Exception:
                # Compaction is an optimization — a failed summarization call
                # (after its own retries) must never take down the run. Leave
                # the item uncompacted; the next turn's pass retries it.
                logger.error(
                    "Compaction failed for item %s; leaving uncompacted",
                    item.item_id,
                    exc_info=True,
                )
                continue
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
