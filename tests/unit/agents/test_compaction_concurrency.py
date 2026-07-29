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
async def test_a_failed_item_discards_its_units_successful_siblings(
    monkeypatch,
) -> None:
    """The all-or-nothing contract survives the move to ``gather``.

    This is the invariant that actually matters, and it needs a MULTI-item
    unit to test: single-item text units pass trivially whether or not
    atomicity holds. A tool turn is one unit — the assistant ``tool_calls``
    row plus each matching result — and a half-compacted one renders a
    message array that the Responses API rejects with a deterministic 400,
    which burns the whole retry budget (run ba6fc95c).

    So: fail the *result* item and assert the assistant row, whose summary
    succeeded, is discarded along with it.
    """
    items = [
        AgentContextItem(item_id="sys-0", role="system", content="sys"),
        AgentContextItem(
            item_id="asst-0",
            role="assistant",
            content=None,
            tool_calls=[
                AgentToolCall(
                    id="call-0",
                    function=AgentToolFunction(name="view_trace", arguments="{}"),
                )
            ],
        ),
        AgentContextItem(
            item_id="tool-0",
            role="tool",
            content="rows",
            tool_call_id="call-0",
        ),
    ]
    context = AgentContext(
        items=items,
        compaction_model=ModelConfig(name="test-compactor"),
        text_message_compaction_keep_last_messages=100,
        tool_call_compaction_keep_last_turns=0,
    )

    probe = _ConcurrencyProbe(fail_on="tool-0")
    monkeypatch.setattr(agent_context_module, "compact", probe)
    await context.compact_old_items(client=object())  # type: ignore[arg-type]

    by_id = {i.item_id: i for i in context.items}
    # Both items were summarized concurrently and asst-0 succeeded...
    assert "asst-0" in probe.calls
    # ...but neither is committed, because they share a unit.
    assert by_id["asst-0"].is_compacted is False
    assert by_id["tool-0"].is_compacted is False


@pytest.mark.asyncio
async def test_a_failed_unit_does_not_block_independent_units(monkeypatch) -> None:
    """Containment: one unit's failure must not cost the others their work."""
    context = _text_context(4, keep_last=0)
    probe = _ConcurrencyProbe(fail_on="m-2")
    monkeypatch.setattr(agent_context_module, "compact", probe)

    await context.compact_old_items(client=object())  # type: ignore[arg-type]

    by_id = {i.item_id: i for i in context.items}
    assert by_id["m-2"].is_compacted is False
    for other in ("m-0", "m-1", "m-3"):
        assert by_id[other].is_compacted is True, f"{other} lost its summary"


@pytest.mark.asyncio
async def test_no_units_makes_no_calls(monkeypatch) -> None:
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agent_context_module, "compact", probe)

    context = _text_context(2, keep_last=10)
    await context.compact_old_items(client=object())  # type: ignore[arg-type]

    assert probe.calls == []


@pytest.mark.asyncio
async def test_a_unit_raising_outside_its_guard_never_escapes(monkeypatch) -> None:
    """Compaction must never take down the run, whatever a unit throws.

    ``_compact_unit`` only guards the ``compact`` call itself. Anything else it
    raises — a commit-phase bug, a future fallible step — would escape to
    ``openai_agent_runner``, which has no handler. A bare ``gather`` also does
    not cancel siblings when a child raises, so the surviving units would keep
    running unsupervised and mutate ``self.items`` after the turn moved on.
    """
    context = _text_context(4, keep_last=0)
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agent_context_module, "compact", probe)

    original = AgentContext._compact_unit
    exploded: list[int] = []

    async def _explode_on_first(self, indices, client, semaphore):  # type: ignore[no-untyped-def]
        if not exploded:
            exploded.append(1)
            raise RuntimeError("commit-phase bug")
        return await original(self, indices, client, semaphore)

    monkeypatch.setattr(AgentContext, "_compact_unit", _explode_on_first)

    # Does not raise...
    await context.compact_old_items(client=object())  # type: ignore[arg-type]

    # ...and the surviving units finished before the call returned, rather
    # than being left to mutate state later.
    assert probe.in_flight == 0
    compacted = [i.item_id for i in context.items if i.is_compacted]
    assert len(compacted) == 3


@pytest.mark.asyncio
async def test_cancellation_is_not_absorbed_as_a_failure(monkeypatch) -> None:
    """A cancelled run must cancel, not be filed as a compaction failure."""

    async def _cancel(*, client: object, compaction_model: object, item: object) -> str:
        raise asyncio.CancelledError()

    monkeypatch.setattr(agent_context_module, "compact", _cancel)
    context = _text_context(3, keep_last=0)

    with pytest.raises(asyncio.CancelledError):
        await context.compact_old_items(client=object())  # type: ignore[arg-type]
