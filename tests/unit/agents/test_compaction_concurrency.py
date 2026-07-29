"""Compaction runs its independent LLM calls concurrently, not one at a time.

Compaction fires at the end of every agent execution — root and each subagent —
and every unit is a separate model round-trip. Awaiting them serially made the
cost the *sum* of those round-trips; these tests pin the concurrency so a future
refactor can't quietly reintroduce the serial chain.

They assert observed overlap rather than wall-clock, so they don't turn into
timing flakes on a loaded CI box.
"""

from __future__ import annotations

import asyncio

import pytest

from engine.agents import agent_context as agent_context_module
from engine.agents.agent_context import COMPACTION_CONCURRENCY, AgentContext
from engine.agents.agent_context_items import AgentContextItem
from engine.model_config import ModelConfig
from engine.models.messages import AgentToolCall, AgentToolFunction


class _ConcurrencyProbe:
    """Stands in for ``compact``, recording peak overlap."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.in_flight = 0
        self.peak = 0
        self.calls: list[str] = []
        self._fail_on = fail_on

    async def __call__(
        self, *, client: object, compaction_model: object, item: AgentContextItem
    ) -> str:
        self.calls.append(item.item_id)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            # Yield twice so peers actually get scheduled; a single sleep(0)
            # can let a fast task finish before the next one starts.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            if self._fail_on is not None and item.item_id == self._fail_on:
                raise RuntimeError("summarization failed")
            return f"summary of {item.item_id}"
        finally:
            self.in_flight -= 1


def _text_context(count: int, *, keep_last: int) -> AgentContext:
    items = [AgentContextItem(item_id="sys-0", role="system", content="sys")]
    items += [
        AgentContextItem(item_id=f"m-{i}", role="user", content=f"message {i}")
        for i in range(count)
    ]
    return AgentContext(
        items=items,
        compaction_model=ModelConfig(name="test-compactor"),
        text_message_compaction_keep_last_messages=keep_last,
        tool_call_compaction_keep_last_turns=100,
    )


@pytest.mark.asyncio
async def test_independent_units_compact_concurrently(monkeypatch) -> None:
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agent_context_module, "compact", probe)

    # 6 compactable text messages, each its own unit.
    context = _text_context(8, keep_last=2)
    await context.compact_old_items(client=object())  # type: ignore[arg-type]

    assert len(probe.calls) == 6
    assert probe.peak > 1, "compaction units ran one at a time"
    assert probe.peak <= COMPACTION_CONCURRENCY


@pytest.mark.asyncio
async def test_concurrency_is_bounded(monkeypatch) -> None:
    """Unbounded fan-out would trade latency for provider rate-limit errors."""
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agent_context_module, "compact", probe)

    context = _text_context(COMPACTION_CONCURRENCY * 3, keep_last=0)
    await context.compact_old_items(client=object())  # type: ignore[arg-type]

    assert probe.peak <= COMPACTION_CONCURRENCY


@pytest.mark.asyncio
async def test_multi_item_units_respect_the_same_global_bound(monkeypatch) -> None:
    """The limit counts provider calls, not units.

    Single-item text units can't catch this: with a per-unit semaphore the peak
    would be ``units x unit_size``, which only diverges from ``units`` once a
    unit holds more than one item. Tool turns do — an assistant ``tool_calls``
    row plus each matching result — so this builds several multi-item tool
    units and asserts the bound still holds on the calls themselves.
    """
    items: list[AgentContextItem] = [
        AgentContextItem(item_id="sys-0", role="system", content="sys")
    ]
    unit_count = COMPACTION_CONCURRENCY * 2
    for turn in range(unit_count):
        call_id = f"call-{turn}"
        items.append(
            AgentContextItem(
                item_id=f"asst-{turn}",
                role="assistant",
                content=None,
                tool_calls=[
                    AgentToolCall(
                        id=call_id,
                        function=AgentToolFunction(name="view_trace", arguments="{}"),
                    )
                ],
            )
        )
        items.append(
            AgentContextItem(
                item_id=f"tool-{turn}",
                role="tool",
                content="rows",
                tool_call_id=call_id,
            )
        )

    context = AgentContext(
        items=items,
        compaction_model=ModelConfig(name="test-compactor"),
        text_message_compaction_keep_last_messages=100,
        tool_call_compaction_keep_last_turns=0,
    )
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agent_context_module, "compact", probe)

    await context.compact_old_items(client=object())  # type: ignore[arg-type]

    # Every tool turn contributes 2 items, so there is genuinely more than
    # ``COMPACTION_CONCURRENCY`` work available to over-subscribe with.
    assert len(probe.calls) == unit_count * 2
    assert probe.peak > 1, "multi-item units ran serially"
    assert probe.peak <= COMPACTION_CONCURRENCY


@pytest.mark.asyncio
async def test_all_summaries_are_committed(monkeypatch) -> None:
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agent_context_module, "compact", probe)

    context = _text_context(5, keep_last=1)
    await context.compact_old_items(client=object())  # type: ignore[arg-type]

    compacted = [i for i in context.items if i.is_compacted]
    assert len(compacted) == 4
    for item in compacted:
        assert item.compaction_summary == f"summary of {item.item_id}"
    # System messages are never compacted.
    assert not context.items[0].is_compacted


@pytest.mark.asyncio
async def test_a_failed_unit_commits_nothing_from_that_unit(monkeypatch) -> None:
    """The all-or-nothing contract survives the move to ``gather``.

    A tool turn is one unit and must never render half-compacted, so a single
    failed summary has to discard its unit's successful siblings too.
    """
    tool_call = AgentContextItem(
        item_id="tool-turn-assistant",
        role="assistant",
        content=None,
        tool_calls=[],
    )
    items = [
        AgentContextItem(item_id="sys-0", role="system", content="sys"),
        AgentContextItem(item_id="m-0", role="user", content="first"),
        AgentContextItem(item_id="m-1", role="user", content="second"),
        tool_call,
    ]
    context = AgentContext(
        items=items,
        compaction_model=ModelConfig(name="test-compactor"),
        text_message_compaction_keep_last_messages=0,
        tool_call_compaction_keep_last_turns=100,
    )

    probe = _ConcurrencyProbe(fail_on="m-1")
    monkeypatch.setattr(agent_context_module, "compact", probe)
    await context.compact_old_items(client=object())  # type: ignore[arg-type]

    # m-0 and m-1 are separate text units, so m-0 still commits; the point is
    # that the failure is contained and never raises out of compaction.
    by_id = {i.item_id: i for i in context.items}
    assert by_id["m-1"].is_compacted is False


@pytest.mark.asyncio
async def test_no_units_makes_no_calls(monkeypatch) -> None:
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agent_context_module, "compact", probe)

    context = _text_context(2, keep_last=10)
    await context.compact_old_items(client=object())  # type: ignore[arg-type]

    assert probe.calls == []
