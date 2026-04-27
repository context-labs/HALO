"""Probe: <final/> sentinel handling.

Pathways probed:
  1. Root assistant message with ``<final/>`` → text stripped, item ``final=True``.
  2. Subagent assistant message with ``<final/>`` → ``final`` should NOT be
     true on the emitted item (only root's <final/> closes the run).
  3. Root has multiple messages, only the LAST contains ``<final/>``.
  4. Root has TWO sentinel-bearing messages — what happens? Both flagged?
"""

from __future__ import annotations

import asyncio
import sys

from tests.probes.probe_kit import (
    FakeRunner,
    make_assistant_text,
    make_default_config,
    make_tool_call,
    make_tool_output,
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


async def probe_root_sentinel_stripped() -> None:
    """Root <final/> stripped from text; ``final=True`` on the item."""
    runner = FakeRunner(
        [make_assistant_text("the answer\n<final/>", item_id="m1")],
    )
    result = await run_with_fake(runner)
    _check(
        result.error is None,
        "root-sentinel: completes without error",
        observed=f"error={type(result.error).__name__ if result.error else None}",
    )
    _check(
        len(result.output_items) == 1,
        "root-sentinel: exactly one output item",
        observed=f"items={len(result.output_items)}",
    )
    if result.output_items:
        item = result.output_items[0]
        _check(item.final is True, "root-sentinel: item.final=True", observed=f"final={item.final}")
        _check(
            "<final/>" not in (item.item.content or ""),
            "root-sentinel: <final/> stripped from content",
            observed=f"content={item.item.content!r}",
        )
        _check(
            (item.item.content or "").strip() == "the answer",
            "root-sentinel: surrounding text preserved",
            observed=f"content={item.item.content!r}",
        )


async def probe_root_multiple_messages_last_is_final() -> None:
    """Root emits one message without sentinel, then a second with sentinel.
    Engine should keep iterating until <final/>; only second item is final."""
    runner = FakeRunner(
        [
            make_assistant_text("thinking...", item_id="m1"),
            make_assistant_text("done\n<final/>", item_id="m2"),
        ],
    )
    result = await run_with_fake(runner)
    _check(
        result.error is None,
        "multi-msg: completes without error",
        observed=f"error={type(result.error).__name__ if result.error else None}",
    )
    _check(
        len(result.output_items) == 2,
        "multi-msg: two output items emitted",
        observed=f"items={len(result.output_items)}",
    )
    if len(result.output_items) >= 2:
        m1, m2 = result.output_items[0], result.output_items[1]
        _check(m1.final is False, "multi-msg: first item not final", observed=f"final={m1.final}")
        _check(m2.final is True, "multi-msg: second (sentinel-bearing) item is final", observed=f"final={m2.final}")


async def probe_sentinel_in_middle_of_text() -> None:
    """<final/> appears mid-text. Mapper uses ``replace`` + ``rstrip`` so the
    sentinel is removed wherever it sits, but trailing whitespace is stripped
    only from the right. Probe: leading text preserved, sentinel removed."""
    runner = FakeRunner(
        [make_assistant_text("answer is <final/> 42", item_id="m1")],
    )
    result = await run_with_fake(runner)
    _check(
        result.error is None,
        "midtext-sentinel: completes without error",
        observed=f"error={type(result.error).__name__ if result.error else None}",
    )
    if result.output_items:
        item = result.output_items[0]
        _check(item.final is True, "midtext-sentinel: item.final=True", observed=f"final={item.final}")
        _check(
            "<final/>" not in (item.item.content or ""),
            "midtext-sentinel: sentinel removed from content",
            observed=f"content={item.item.content!r}",
        )
        # rstrip means trailing whitespace from the now-removed sentinel
        # gets cleaned up; "answer is " + "" + " 42" → "answer is  42" (rstripped)
        _check(
            (item.item.content or "") == "answer is  42",
            "midtext-sentinel: content is 'answer is  42' (double space, rstripped)",
            observed=f"content={item.item.content!r}",
        )


async def probe_sentinel_only() -> None:
    """Message contains *only* <final/>. After replace+rstrip, content is empty.
    Engine should still emit the item with final=True. Quirk: empty content
    becomes None in the mapper (line 90: ``content: str | None = text or None``)."""
    runner = FakeRunner(
        [make_assistant_text("<final/>", item_id="m1")],
    )
    result = await run_with_fake(runner)
    _check(
        result.error is None,
        "empty-sentinel: completes without error",
        observed=f"error={type(result.error).__name__ if result.error else None}",
    )
    if result.output_items:
        item = result.output_items[0]
        # When the entire text is <final/>, after stripping text is "" → content=None.
        # But also: the "if is_root and text and FINAL_SENTINEL in text" guard requires
        # truthy text to even strip — so let's see what happens.
        _check(
            item.final is True,
            "empty-sentinel: item.final=True even with sentinel-only message",
            observed=f"final={item.final} content={item.item.content!r}",
        )


async def probe_two_sentinel_messages_in_one_turn() -> None:
    """A single turn emits two messages, both with <final/>. The first should
    flag final and (presumably) close the run; the second... unclear behavior.
    Probe what happens. This is exploratory — not a hard contract."""
    runner = FakeRunner(
        [
            make_assistant_text("first\n<final/>", item_id="m1"),
            make_assistant_text("second\n<final/>", item_id="m2"),
        ],
    )
    result = await run_with_fake(runner)
    _check(
        result.error is None,
        "double-sentinel: completes without error",
        observed=f"error={type(result.error).__name__ if result.error else None}",
    )
    final_items = [it for it in result.output_items if it.final]
    _check(
        len(final_items) == 2,
        "double-sentinel: both items get final=True (mapper-level, not stop signal)",
        observed=f"final_count={len(final_items)} of {len(result.output_items)}",
    )


async def main() -> int:
    await probe_root_sentinel_stripped()
    await probe_root_multiple_messages_last_is_final()
    await probe_sentinel_in_middle_of_text()
    await probe_sentinel_only()
    await probe_two_sentinel_messages_in_one_turn()

    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) failed:")
        for desc in _FAILURES:
            print(f"  - {desc}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
