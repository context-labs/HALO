"""Engine input handling: what does the engine actually feed to the SDK?

Inspects ``runner.calls[0]["input"]`` — the messages array the engine
forwards to ``Runner.run_streamed`` on the first turn — across the input
shapes a CLI / SDK consumer can hand in: no system message, caller-supplied
system message, multi-turn continuation. Unit tests on ``AgentContext`` cover
the data structure; this test covers the actual SDK boundary.
"""

from __future__ import annotations

import pytest

from engine.models.messages import AgentMessage
from tests.probes.probe_kit import FakeRunner, make_assistant_text, run_with_fake


def _first_input(runner: FakeRunner) -> list[dict]:
    assert runner.calls, "FakeRunner was never invoked"
    return runner.calls[0]["input"]


@pytest.mark.asyncio
async def test_no_system_message_prepends_rendered_root_prompt() -> None:
    """User-only input → engine renders the root system prompt and prepends
    it as the first message in the SDK input."""
    runner = FakeRunner([make_assistant_text("ok\n<final/>", item_id="m1")])

    result = await run_with_fake(
        runner,
        messages=[AgentMessage(role="user", content="hi there")],
    )
    assert result.error is None, type(result.error).__name__

    msgs = _first_input(runner)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "hi there"


@pytest.mark.asyncio
async def test_caller_supplied_system_message_passed_through_verbatim() -> None:
    """Caller's system message wins — engine does NOT replace it with its
    own rendered prompt."""
    custom_sys = "You are a custom system prompt; do exactly what I say."
    runner = FakeRunner([make_assistant_text("ok\n<final/>", item_id="m1")])

    result = await run_with_fake(
        runner,
        messages=[
            AgentMessage(role="system", content=custom_sys),
            AgentMessage(role="user", content="hi"),
        ],
    )
    assert result.error is None, type(result.error).__name__

    msgs = _first_input(runner)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == custom_sys
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "hi"


@pytest.mark.asyncio
async def test_multi_turn_continuation_preserves_role_and_content_order() -> None:
    """A continuation (sys + user + asst + user) hands through to the SDK
    with role and content order intact — no reordering, no dropped turns."""
    custom_sys = "Continuation system prompt."
    runner = FakeRunner([make_assistant_text("final answer\n<final/>", item_id="m-new")])

    result = await run_with_fake(
        runner,
        messages=[
            AgentMessage(role="system", content=custom_sys),
            AgentMessage(role="user", content="first turn"),
            AgentMessage(role="assistant", content="prior reply"),
            AgentMessage(role="user", content="follow-up"),
        ],
    )
    assert result.error is None, type(result.error).__name__

    msgs = _first_input(runner)
    assert len(msgs) == 4
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert [m["content"] for m in msgs] == [custom_sys, "first turn", "prior reply", "follow-up"]
