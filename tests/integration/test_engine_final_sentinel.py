"""Engine ``<final/>`` sentinel handling end-to-end.

The sentinel is the only way the root agent signals "I'm done." Behavior
must be exact: stripped from the emitted item's text, and only the
sentinel-bearing message gets ``final=True``. Mapper-level unit tests
cover the stripping primitive; this test covers the engine-level contract
across multi-message turns.
"""

from __future__ import annotations

import pytest

from tests.probes.probe_kit import FakeRunner, make_assistant_text, run_with_fake


@pytest.mark.asyncio
async def test_sentinel_stripped_and_final_flagged_on_single_message() -> None:
    """Trailing ``<final/>`` is stripped from the item content and ``final=True``
    is set."""
    runner = FakeRunner([make_assistant_text("the answer\n<final/>", item_id="m1")])

    result = await run_with_fake(runner)

    assert result.error is None, type(result.error).__name__
    assert len(result.output_items) == 1
    item = result.output_items[0]
    assert item.final is True
    assert item.item.content == "the answer"


@pytest.mark.asyncio
async def test_only_sentinel_bearing_message_in_a_turn_is_final() -> None:
    """When the model emits two messages and only the second carries the
    sentinel, only the second is flagged ``final``. The engine should not
    short-circuit on the first message."""
    runner = FakeRunner(
        [
            make_assistant_text("thinking...", item_id="m1"),
            make_assistant_text("done\n<final/>", item_id="m2"),
        ],
    )

    result = await run_with_fake(runner)

    assert result.error is None, type(result.error).__name__
    assert len(result.output_items) == 2
    first, second = result.output_items
    assert first.final is False
    assert first.item.content == "thinking..."
    assert second.final is True
    assert second.item.content == "done"


@pytest.mark.asyncio
async def test_mid_text_sentinel_removed_and_final_flagged() -> None:
    """``<final/>`` in the middle of text is still removed; ``final=True`` is
    set regardless of position. The mapper uses replace + rstrip, so the
    surrounding whitespace is preserved on the left of the sentinel and
    stripped on the right."""
    runner = FakeRunner([make_assistant_text("answer is <final/> 42", item_id="m1")])

    result = await run_with_fake(runner)

    assert result.error is None, type(result.error).__name__
    assert len(result.output_items) == 1
    item = result.output_items[0]
    assert item.final is True
    assert item.item.content == "answer is  42"
