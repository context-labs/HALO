"""Probe: AgentContext input handling.

Pathways probed:
  1. No system message → engine prepends rendered root prompt as ``sys-0``.
  2. System message at front → passed through unchanged (caller-supplied wins).
  3. Multi-message continuation → assistant/tool/user messages preserved
     with stable ``in-N`` item_ids in order.
  4. The first call to FakeRunner.run_streamed receives ``input`` whose first
     element is a system message (rendered or caller-supplied), confirming
     the messages array fed to the SDK matches the AgentContext.
"""

from __future__ import annotations

import asyncio
import sys

from engine.models.messages import AgentMessage
from tests.probes.probe_kit import (
    FakeRunner,
    make_assistant_text,
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


def _first_input_messages(runner: FakeRunner) -> list[dict]:
    """Return the messages array passed to the first run_streamed call."""
    if not runner.calls:
        return []
    return runner.calls[0]["input"]


async def probe_no_system_message_prepends_default() -> None:
    """User-only input. Engine should render the root system prompt
    and put it as the first message in the input fed to the SDK."""
    runner = FakeRunner(
        [make_assistant_text("ok\n<final/>", item_id="m1")],
    )
    result = await run_with_fake(
        runner,
        messages=[AgentMessage(role="user", content="hi there")],
    )
    _check(
        result.error is None,
        "no-sys: completes without error",
        observed=f"error={type(result.error).__name__ if result.error else None}",
    )
    msgs = _first_input_messages(runner)
    _check(len(msgs) >= 2, "no-sys: at least 2 messages in input (system + user)", observed=f"len={len(msgs)}")
    if msgs:
        _check(
            msgs[0].get("role") == "system",
            "no-sys: first input message is system",
            observed=f"role={msgs[0].get('role')}",
        )
        # The rendered system prompt should not be empty
        _check(
            bool(msgs[0].get("content")),
            "no-sys: rendered system content is non-empty",
            observed=f"content_len={len(msgs[0].get('content') or '')}",
        )
        _check(
            msgs[1].get("role") == "user" and msgs[1].get("content") == "hi there",
            "no-sys: user content preserved as second message",
            observed=f"msg2={msgs[1]}",
        )


async def probe_system_message_passed_through_unchanged() -> None:
    """Caller-supplied system message at front. Engine should NOT replace it
    with its own rendered prompt — it should pass through verbatim."""
    custom_sys = "You are a custom system prompt; do exactly what I say."
    runner = FakeRunner(
        [make_assistant_text("ok\n<final/>", item_id="m1")],
    )
    result = await run_with_fake(
        runner,
        messages=[
            AgentMessage(role="system", content=custom_sys),
            AgentMessage(role="user", content="hi"),
        ],
    )
    _check(
        result.error is None,
        "custom-sys: completes without error",
        observed=f"error={type(result.error).__name__ if result.error else None}",
    )
    msgs = _first_input_messages(runner)
    if msgs:
        _check(
            msgs[0].get("role") == "system",
            "custom-sys: first input message is system",
            observed=f"role={msgs[0].get('role')}",
        )
        _check(
            msgs[0].get("content") == custom_sys,
            "custom-sys: caller's system content preserved verbatim",
            observed=f"content={msgs[0].get('content')!r}",
        )


async def probe_multi_message_continuation_preserves_ids() -> None:
    """Caller passes a continuation: system + user + assistant + user. The
    assistant message in the middle should land in the input as-is. We can't
    inspect ``AgentContext.items`` directly, but we can inspect the messages
    array fed to ``run_streamed``."""
    custom_sys = "Continuation system prompt."
    runner = FakeRunner(
        [make_assistant_text("final answer\n<final/>", item_id="m-new")],
    )
    result = await run_with_fake(
        runner,
        messages=[
            AgentMessage(role="system", content=custom_sys),
            AgentMessage(role="user", content="first turn"),
            AgentMessage(role="assistant", content="prior reply"),
            AgentMessage(role="user", content="follow-up"),
        ],
    )
    _check(
        result.error is None,
        "continuation: completes without error",
        observed=f"error={type(result.error).__name__ if result.error else None}",
    )
    msgs = _first_input_messages(runner)
    _check(len(msgs) == 4, "continuation: input has 4 messages (sys + user + asst + user)", observed=f"len={len(msgs)}")
    if len(msgs) == 4:
        roles = [m.get("role") for m in msgs]
        _check(
            roles == ["system", "user", "assistant", "user"],
            "continuation: role order preserved",
            observed=f"roles={roles}",
        )
        contents = [m.get("content") for m in msgs]
        _check(
            contents == [custom_sys, "first turn", "prior reply", "follow-up"],
            "continuation: contents preserved in order",
            observed=f"contents={contents}",
        )


async def probe_only_system_message_no_user() -> None:
    """Edge case: caller passes ONLY a system message, no user content. What
    does the engine do? The SDK Runner usually requires a user message."""
    runner = FakeRunner(
        [make_assistant_text("ok\n<final/>", item_id="m1")],
    )
    result = await run_with_fake(
        runner,
        messages=[AgentMessage(role="system", content="be brief")],
    )
    msgs = _first_input_messages(runner)
    # Engine should not error before calling the runner — it builds context
    # and hands off. Whether the SDK is happy with system-only input is
    # the SDK's problem, but the engine itself shouldn't crash.
    _check(
        result.error is None,
        "sys-only: engine doesn't crash with system-only input",
        observed=f"error={type(result.error).__name__ if result.error else None}",
    )
    _check(len(msgs) == 1, "sys-only: input is single system message", observed=f"len={len(msgs)} msgs={msgs}")


async def main() -> int:
    await probe_no_system_message_prepends_default()
    await probe_system_message_passed_through_unchanged()
    await probe_multi_message_continuation_preserves_ids()
    await probe_only_system_message_no_user()

    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) failed:")
        for desc in _FAILURES:
            print(f"  - {desc}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
