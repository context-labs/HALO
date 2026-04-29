"""Engine streaming contract: deltas vs items, sequence ordering, run_engine_async filtering.

Drives ``stream_engine_async`` / ``run_engine_async`` through a deterministic
``FakeRunner`` to lock in the contract ``halo_cli`` and other consumers depend
on. Unit tests cover the bus and mapper in isolation; this verifies the
public engine API actually composes them correctly.
"""

from __future__ import annotations

import pytest

from engine.main import run_engine_async
from engine.models.engine_output import AgentOutputItem, AgentTextDelta
from tests.probes.probe_kit import (
    FakeRunner,
    isolated_trace_copy,
    make_assistant_text,
    make_default_config,
    make_default_messages,
    make_text_delta,
    run_with_fake,
)


@pytest.mark.asyncio
async def test_deltas_and_items_emitted_with_monotonic_sequences() -> None:
    """Streaming deltas precede the final item, every event has a strictly
    increasing sequence number, and delta/item types appear in the scripted
    order."""
    runner = FakeRunner(
        [
            make_text_delta(item_id="m1", delta="hel"),
            make_text_delta(item_id="m1", delta="lo "),
            make_text_delta(item_id="m1", delta="world"),
            make_assistant_text("hello world\n<final/>", item_id="m1"),
        ],
    )

    result = await run_with_fake(runner)

    assert result.error is None, type(result.error).__name__
    assert [d.text_delta for d in result.deltas] == ["hel", "lo ", "world"]
    assert len(result.output_items) == 1

    item = result.output_items[0]
    assert item.final is True
    assert item.item.content == "hello world"

    sequences = [ev.sequence for ev in result.all_events]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)

    types = [type(ev).__name__ for ev in result.all_events]
    assert types == ["AgentTextDelta", "AgentTextDelta", "AgentTextDelta", "AgentOutputItem"]


@pytest.mark.asyncio
async def test_run_engine_async_filters_text_deltas() -> None:
    """``run_engine_async`` materializes ``AgentOutputItem``s only — streaming
    text deltas are filtered out. Confirmed against a runner that emits both."""
    runner = FakeRunner(
        [
            make_text_delta(item_id="m1", delta="streamed"),
            make_text_delta(item_id="m1", delta=" text"),
            make_assistant_text("done\n<final/>", item_id="m1"),
        ],
    )

    items = await run_engine_async(
        make_default_messages(),
        make_default_config(),
        isolated_trace_copy(),
        runner=runner,
    )

    assert len(items) == 1
    assert isinstance(items[0], AgentOutputItem)
    assert not any(isinstance(it, AgentTextDelta) for it in items)
    assert items[0].item.content == "done"
    assert items[0].final is True
