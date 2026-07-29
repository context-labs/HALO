"""Engine input handling: what does the engine actually feed to the SDK?

Inspects ``runner.calls[0]["input"]`` — the messages array the engine
forwards to ``Runner.run_streamed`` on the first turn — across the input
shapes a CLI / SDK consumer can hand in: no system message, caller-supplied
system message, multi-turn continuation. Unit tests on ``AgentContext`` cover
the data structure; this test covers the actual SDK boundary.
"""

from __future__ import annotations

import pytest

from engine.models.engine_output import RunCheckpoint
from engine.models.messages import AgentMessage
from tests.probes.probe_kit import (
    FakeRunner,
    item_text,
    make_assistant_text,
    make_default_config,
    make_final_answer,
    run_with_fake,
)


def _first_input(runner: FakeRunner) -> list[dict]:
    assert runner.calls, "FakeRunner was never invoked"
    return runner.calls[0]["input"]


@pytest.mark.asyncio
async def test_no_system_message_prepends_rendered_root_prompt() -> None:
    """User-only input → engine renders the root system prompt and prepends
    it as the first message in the SDK input."""
    runner = FakeRunner([*make_final_answer("ok")])

    result = await run_with_fake(
        runner,
        messages=[AgentMessage(role="user", content="hi there")],
    )
    assert result.error is None, type(result.error).__name__

    msgs = _first_input(runner)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert item_text(msgs[0])
    assert msgs[1]["role"] == "user"
    assert item_text(msgs[1]) == "hi there"


@pytest.mark.asyncio
async def test_caller_supplied_system_message_passed_through_verbatim() -> None:
    """Caller's system message wins — engine does NOT replace it with its
    own rendered prompt."""
    custom_sys = "You are a custom system prompt; do exactly what I say."
    runner = FakeRunner([*make_final_answer("ok")])

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
    assert item_text(msgs[0]) == custom_sys
    assert msgs[1]["role"] == "user"
    assert item_text(msgs[1]) == "hi"


@pytest.mark.asyncio
async def test_multi_turn_continuation_preserves_role_and_content_order() -> None:
    """A continuation (sys + user + asst + user) hands through to the SDK
    with role and content order intact — no reordering, no dropped turns."""
    custom_sys = "Continuation system prompt."
    runner = FakeRunner([*make_final_answer("final answer")])

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
    assert [item_text(m) for m in msgs] == [
        custom_sys,
        "first turn",
        "prior reply",
        "follow-up",
    ]


@pytest.mark.asyncio
async def test_root_run_emits_a_resumable_checkpoint_when_enabled() -> None:
    """With checkpoints on, a completed root run leaves state to resume from.

    The checkpoint is emitted after compaction, so it carries the smaller
    history a resumed run would actually replay rather than the raw one.
    """
    fake = FakeRunner([[make_assistant_text("looking"), make_final_answer("done")]])
    result = await run_with_fake(fake, config=make_default_config(emit_run_checkpoints=True))

    assert result.error is None
    checkpoints = [e for e in result.all_events if isinstance(e, RunCheckpoint)]
    assert len(checkpoints) == 1
    # It must be usable as-is: same shape `from_input_messages` consumes.
    assert checkpoints[0].messages
    assert checkpoints[0].messages[0].role == "system"


@pytest.mark.asyncio
async def test_no_checkpoints_by_default() -> None:
    """Off by default: widening the event union must not reach a host pinned to
    an engine that predates the variant."""
    fake = FakeRunner([[make_final_answer("done")]])
    result = await run_with_fake(fake)

    assert result.error is None
    assert not [e for e in result.all_events if isinstance(e, RunCheckpoint)]
