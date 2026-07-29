from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from engine.agents.agent_context_items import AgentContextItem
from engine.agents.compactor import compact
from engine.agents.prompt_templates import render_root_system_prompt
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage
from engine.telemetry.spans import engine_span

if TYPE_CHECKING:
    from engine.code.code_repo import CodeRepo
    from engine.engine_config import EngineConfig
    from engine.git.git_repo import GitRepo

logger = logging.getLogger(__name__)

# Parallel in-flight compaction calls. Matches the default for
# ``maximum_parallel_subagents``: enough to collapse the serial chain, low
# enough that a big compaction pass doesn't trip provider rate limits and
# hand the latency back as retry backoff.
COMPACTION_CONCURRENCY = 4


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
                dataset_context=engine_config.dataset_context,
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

        One forward pass tracks unresolved tool-call ids and cuts at the last
        point where the prefix renders validly — no assistant ``tool_calls``
        left unanswered and no orphan ``role=tool`` row. This keeps completed
        turns (including a parallel turn split across several assistant rows,
        one per call) and drops only the trailing junk. Never cuts below
        ``min_items`` (items that existed before the failed attempt are
        consistent already). Returns the removed items in original order.
        """
        cut = self._consistent_prefix_length(max(min_items, 0))
        if cut is None:
            return []
        removed = self.items[cut:]
        del self.items[cut:]
        for item in removed:
            self._index.pop(item.item_id, None)
        return removed

    def _consistent_prefix_length(self, floor: int) -> int | None:
        """Largest ``p >= floor`` where ``items[:p]`` renders to a valid message
        array, or ``None`` when that's the whole history (nothing to trim).

        Valid = every assistant ``tool_calls`` answered by matching ``role=tool``
        results and no orphan tool row. Compacted items render as plain
        summaries, so they don't participate in call/result pairing.
        """
        open_call_ids: set[str] = set()
        saw_orphan = False
        cut: int | None = None
        for pos, item in enumerate([*self.items, None]):
            if pos >= floor and not open_call_ids and not saw_orphan:
                cut = pos
            if item is None or item.is_compacted:
                continue
            if item.role == "assistant" and item.tool_calls:
                open_call_ids.update(tc.id for tc in item.tool_calls)
            elif item.role == "tool":
                if item.tool_call_id in open_call_ids:
                    open_call_ids.discard(item.tool_call_id)
                else:
                    saw_orphan = True
        if cut is None or cut == len(self.items):
            return None
        return cut

    async def compact_old_items(self, client: AsyncOpenAI) -> None:
        """Compact eligible older items in place using two independent keep-last thresholds.

        Text messages compact individually; a tool turn (its assistant tool_calls
        rows plus matching results) is one atomic unit so the rendered array never
        pairs a compacted summary with an uncompacted ``role=tool`` row. System
        messages are never touched.
        """
        text_positions = [
            idx
            for idx, item in enumerate(self.items)
            if not item.is_compacted and item.role != "system" and not _is_tool_related(item)
        ]
        tool_groups = _build_tool_groups(self.items)

        units: list[list[int]] = []
        if len(text_positions) > self.text_message_compaction_keep_last_messages:
            cutoff = len(text_positions) - self.text_message_compaction_keep_last_messages
            units.extend([pos] for pos in text_positions[:cutoff])
        if len(tool_groups) > self.tool_call_compaction_keep_last_turns:
            cutoff = len(tool_groups) - self.tool_call_compaction_keep_last_turns
            units.extend(tool_groups[:cutoff])

        if not units:
            return

        # Units are independent: each summarizes a disjoint set of item indices
        # and commits only its own. Awaiting them one at a time made compaction
        # cost the SUM of every unit's LLM round-trip, at the end of every agent
        # execution — root and each subagent.
        #
        # One semaphore, shared by every unit and held around the individual
        # `compact` calls. Bounding units instead would not actually bound
        # anything: a tool-turn unit holds several items, so per-unit fan-out
        # would peak at units x unit_size in-flight calls. The thing worth
        # limiting is concurrent requests to the provider, so that is what the
        # limit counts.
        #
        # The bound is per agent execution, not global: root and each subagent
        # build their own AgentContext, so several finishing at once each get
        # their own allowance.
        semaphore = asyncio.Semaphore(COMPACTION_CONCURRENCY)
        with engine_span(
            "halo.compaction",
            **{
                "compaction.units": len(units),
                "compaction.concurrency": COMPACTION_CONCURRENCY,
            },
        ):
            # ``return_exceptions=True`` is load-bearing for two separate
            # reasons. Compaction is an optimization that must never take down
            # the run, and ``_compact_unit``'s own guard only covers a failing
            # ``compact`` call — anything else it raised would escape to a
            # caller with no handler. Worse, a bare gather does not cancel
            # siblings when a child raises: the remaining units would keep
            # running unsupervised and mutate ``self.items`` long after the
            # turn had moved on. This waits for every unit either way.
            outcomes = await asyncio.gather(
                *(self._compact_unit(unit, client, semaphore) for unit in units),
                return_exceptions=True,
            )

        for outcome in outcomes:
            # Same split as inside a unit: process-teardown exceptions
            # propagate, ordinary failures are logged and contained.
            if isinstance(outcome, BaseException) and not isinstance(outcome, Exception):
                raise outcome
            if isinstance(outcome, Exception):
                logger.error(
                    "Compaction unit raised unexpectedly; leaving it uncompacted",
                    exc_info=outcome,
                )

    async def _compact_unit(
        self, indices: list[int], client: AsyncOpenAI, semaphore: asyncio.Semaphore
    ) -> None:
        """Summarize every item in one compaction unit, committing only if all
        succeed. A tool turn is a single unit, so it never renders half-compacted;
        any failed summary leaves the whole unit uncompacted for the next pass.
        """
        items = [(idx, self.items[idx]) for idx in indices]

        # Items within a unit are independent summarization calls; only the
        # commit below is atomic. `return_exceptions` keeps the all-or-nothing
        # contract intact — one failure still leaves the WHOLE unit
        # uncompacted — while paying one round-trip of latency instead of N.
        async def _summarize(item: AgentContextItem) -> str:
            async with semaphore:
                return await compact(
                    client=client, compaction_model=self.compaction_model, item=item
                )

        results = await asyncio.gather(
            *(_summarize(item) for _, item in items),
            return_exceptions=True,
        )

        summaries: list[tuple[int, AgentContextItem, str]] = []
        for (idx, item), result in zip(items, results, strict=True):
            # Cancellation and the other non-``Exception`` BaseExceptions
            # (KeyboardInterrupt, SystemExit) must propagate: they mean the
            # process is going away, not that a summary failed. Only a real
            # ``Exception`` is an ordinary compaction failure.
            if isinstance(result, BaseException) and not isinstance(result, Exception):
                raise result
            if isinstance(result, Exception):
                # Compaction is an optimization — a failed summarization call
                # (after its own retries) must never take down the run, and must
                # not partially compact a tool turn. Leave the whole unit
                # uncompacted; the next turn's pass retries it.
                logger.error(
                    "Compaction failed for unit containing item %s; leaving unit uncompacted",
                    item.item_id,
                    exc_info=result,
                )
                return
            summaries.append((idx, item, result))
        for idx, item, summary in summaries:
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
    """Group tool-related items into conversational turns (atomic compaction units).

    A turn = a maximal run of consecutive assistant ``tool_calls`` rows (the
    mapper emits one per call, so a parallel turn spans several) plus the
    ``role=tool`` results that follow them. Grouping the whole contiguous block
    together is what lets compaction stay all-or-nothing: the keep-last cutoff
    can never split a parallel turn, and a compacted turn renders as a valid
    block of summaries.

    The returned list preserves turn order (oldest first) and contains item
    indices into ``items``. Already-compacted items are skipped so a second pass
    cannot re-compact them. A tool result with no preceding tool-call row in its
    block (its call is absent or already compacted) is dropped from grouping —
    standalone, it cannot form a coherent turn for OpenAI's API anyway.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    awaiting_results = False  # have this block's results started arriving?
    for idx, item in enumerate(items):
        if item.is_compacted or item.role == "system":
            continue
        if item.role == "assistant" and item.tool_calls:
            if awaiting_results:
                groups.append(current)
                current = []
                awaiting_results = False
            current.append(idx)
        elif item.role == "tool" and item.tool_call_id is not None:
            if current:
                current.append(idx)
                awaiting_results = True
        else:
            if current:
                groups.append(current)
                current = []
                awaiting_results = False
    if current:
        groups.append(current)
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
