"""Probe: streaming contract.

Pathways probed:
  1. ``AgentTextDelta`` events are emitted distinctly from ``AgentOutputItem``s
     and interleaved with them in monotonic ``sequence`` order.
  2. ``run_engine_async`` filters out deltas (only items returned).
  3. If the driver task raises a non-retriable error mid-stream, does the
     engine close the output bus or deadlock? (TimeoutError signals deadlock.)
"""

from __future__ import annotations

import asyncio
import sys

import httpx
from openai import BadRequestError

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

_FAILURES: list[str] = []


def _check(condition: bool, description: str, observed: str = "") -> None:
    if condition:
        print(f"PASS: {description}")
    else:
        suffix = f" — observed: {observed}" if observed else ""
        print(f"FAIL: {description}{suffix}")
        _FAILURES.append(description)


async def probe_delta_then_message_ordering() -> None:
    """Script several text deltas followed by a final message. Deltas must
    be exposed as ``AgentTextDelta``; the final message as ``AgentOutputItem``.
    All events should have strictly increasing ``sequence`` values."""
    runner = FakeRunner(
        [
            make_text_delta(item_id="m1", delta="hel"),
            make_text_delta(item_id="m1", delta="lo "),
            make_text_delta(item_id="m1", delta="world"),
            make_assistant_text("hello world\n<final/>", item_id="m1"),
        ],
    )
    result = await run_with_fake(runner)

    _check(result.error is None, "streaming: run completes without error",
           observed=f"error={type(result.error).__name__ if result.error else None}")
    _check(len(result.deltas) == 3,
           "streaming: 3 AgentTextDelta events emitted",
           observed=f"deltas={len(result.deltas)}")
    _check(len(result.output_items) == 1,
           "streaming: 1 AgentOutputItem emitted",
           observed=f"items={len(result.output_items)}")

    sequences = [ev.sequence for ev in result.all_events]
    _check(sequences == sorted(sequences) and len(set(sequences)) == len(sequences),
           "streaming: all events have strictly increasing sequence",
           observed=f"sequences={sequences}")

    # First three should be deltas, last should be item:
    types = [type(ev).__name__ for ev in result.all_events]
    _check(types == ["AgentTextDelta", "AgentTextDelta", "AgentTextDelta", "AgentOutputItem"],
           "streaming: delta-delta-delta-item order preserved",
           observed=f"types={types}")


async def probe_run_engine_async_filters_deltas() -> None:
    """``run_engine_async`` is documented to filter out streaming deltas; only
    durable items should remain."""
    runner = FakeRunner(
        [
            make_text_delta(item_id="m1", delta="streamed"),
            make_text_delta(item_id="m1", delta=" text"),
            make_assistant_text("done\n<final/>", item_id="m1"),
        ],
    )
    cfg = make_default_config()
    msgs = make_default_messages()
    tp = isolated_trace_copy()

    items = await run_engine_async(msgs, cfg, tp, runner=runner)
    _check(all(isinstance(it, AgentOutputItem) for it in items),
           "filter: only AgentOutputItem returned by run_engine_async",
           observed=f"types={[type(it).__name__ for it in items]}")
    _check(not any(isinstance(it, AgentTextDelta) for it in items),
           "filter: no AgentTextDelta in run_engine_async output")


async def probe_non_retriable_error_does_not_deadlock() -> None:
    """If the driver raises a non-retriable error, the engine should propagate
    it through the bus rather than deadlock the consumer.

    In ``stream_engine_async`` (engine/main.py:77-100), ``_drive`` does
    ``output_bus.close()`` only on success. If ``runner.run`` raises, the bus
    is never closed; the consumer task `async for event in output_bus.stream()`
    waits forever. We test for that.
    """
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fake_response = httpx.Response(400, request=fake_request)
    runner = FakeRunner(
        BadRequestError(message="bad", response=fake_response, body={"error": {"message": "bad"}}),
    )
    result = await run_with_fake(runner, timeout_seconds=3.0)

    _check(not isinstance(result.error, asyncio.TimeoutError),
           "no-deadlock: engine does not deadlock on non-retriable error",
           observed=f"error={type(result.error).__name__ if result.error else None}")
    _check(isinstance(result.error, BadRequestError),
           "no-deadlock: BadRequestError propagates through stream",
           observed=f"error={type(result.error).__name__ if result.error else None}")


async def main() -> int:
    await probe_delta_then_message_ordering()
    await probe_run_engine_async_filters_deltas()
    await probe_non_retriable_error_does_not_deadlock()

    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) failed:")
        for desc in _FAILURES:
            print(f"  - {desc}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
