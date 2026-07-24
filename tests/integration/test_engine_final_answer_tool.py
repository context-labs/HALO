"""Engine ``final_answer`` tool-call finalization end-to-end.

The root agent now completes the run by calling the ``final_answer`` tool
(``StopAtTools`` on the root SDK agent); the mapper transforms that call
into the same final-flagged assistant message the legacy ``<final/>``
sentinel produced, and the runner reprompts once when a run ends without
finalizing. These tests cover the engine-level contract with a scripted
``FakeRunner``; ``tests/integration/test_engine_final_sentinel.py`` keeps
covering the legacy sentinel fallback.
"""

from __future__ import annotations

import pytest

from tests.probes.probe_kit import (
    FakeRunner,
    make_assistant_text,
    make_default_config,
    make_tool_call,
    make_tool_output,
    run_with_fake,
)


def _config_with_reprompts(reprompts: int):
    cfg = make_default_config()
    return cfg.model_copy(
        update={
            "root_agent": cfg.root_agent.model_copy(update={"final_answer_reprompts": reprompts})
        }
    )


@pytest.mark.asyncio
async def test_final_answer_call_yields_final_assistant_item() -> None:
    """A ``final_answer`` call becomes one final-flagged assistant message
    carrying the answer; the tool's acknowledgement output is swallowed."""
    runner = FakeRunner(
        [
            make_tool_call(
                name="final_answer",
                arguments='{"answer": "## Summary\\nAll good."}',
                call_id="call-final",
            ),
            make_tool_output(call_id="call-final", output='{"acknowledged": true}'),
        ],
    )

    result = await run_with_fake(runner)

    assert result.error is None, type(result.error).__name__
    assert len(result.output_items) == 1
    item = result.output_items[0]
    assert item.final is True
    assert item.item.role == "assistant"
    assert item.item.content == "## Summary\nAll good."
    assert item.item.tool_calls is None


@pytest.mark.asyncio
async def test_work_then_final_answer_flags_only_the_final_item() -> None:
    """Ordinary tool work stays unflagged; only the ``final_answer``-derived
    message is final."""
    runner = FakeRunner(
        [
            make_tool_call(name="get_dataset_overview", arguments="{}", call_id="call-1"),
            make_tool_output(call_id="call-1", output="{}"),
            make_assistant_text("looking...", item_id="m1"),
            make_tool_call(
                name="final_answer",
                arguments='{"answer": "done"}',
                call_id="call-2",
            ),
            make_tool_output(call_id="call-2", output='{"acknowledged": true}'),
        ],
    )

    result = await run_with_fake(runner)

    assert result.error is None, type(result.error).__name__
    finals = [item for item in result.output_items if item.final]
    assert len(finals) == 1
    assert finals[0].item.content == "done"
    # The ordinary tool call + output + interim text all still surface.
    assert len(result.output_items) == 4


@pytest.mark.asyncio
async def test_run_ending_without_finalization_is_reprompted_once() -> None:
    """A run that ends on a plain no-tool-call message (the DeepSeek
    narrate-and-stop flavor) gets one nudge and finalizes on the rerun."""
    runner = FakeRunner(
        [make_assistant_text("I'll explore the API differently...", item_id="m1")],
        [
            make_tool_call(
                name="final_answer",
                arguments='{"answer": "recovered"}',
                call_id="call-final",
            ),
            make_tool_output(call_id="call-final", output='{"acknowledged": true}'),
        ],
    )

    result = await run_with_fake(runner, config=_config_with_reprompts(1))

    assert result.error is None, type(result.error).__name__
    assert len(runner.calls) == 2
    reprompt_input = runner.calls[1]["input"]
    assert reprompt_input[-1]["role"] == "user"
    assert "final_answer" in reprompt_input[-1]["content"]
    finals = [item for item in result.output_items if item.final]
    assert len(finals) == 1
    assert finals[0].item.content == "recovered"


@pytest.mark.asyncio
async def test_malformed_final_answer_arguments_trigger_the_reprompt() -> None:
    """DeepSeek's empty-``{}``-arguments flavor: the malformed call maps as a
    plain unflagged tool call, so the reprompt fires and the rerun recovers."""
    runner = FakeRunner(
        [
            make_tool_call(name="final_answer", arguments="{}", call_id="call-bad"),
            make_tool_output(call_id="call-bad", output="error"),
        ],
        [
            make_tool_call(
                name="final_answer",
                arguments='{"answer": "second try"}',
                call_id="call-final",
            ),
            make_tool_output(call_id="call-final", output='{"acknowledged": true}'),
        ],
    )

    result = await run_with_fake(runner, config=_config_with_reprompts(1))

    assert result.error is None, type(result.error).__name__
    assert len(runner.calls) == 2
    finals = [item for item in result.output_items if item.final]
    assert len(finals) == 1
    assert finals[0].item.content == "second try"


@pytest.mark.asyncio
async def test_exhausted_reprompts_end_the_run_without_error() -> None:
    """When every reprompt is burned without finalization the run ends
    cleanly with no final item — the runtime shim's existing incomplete
    handling takes over, exactly as in the sentinel era."""
    runner = FakeRunner(
        [make_assistant_text("no tool call", item_id="m1")],
        [make_assistant_text("still no tool call", item_id="m2")],
    )

    result = await run_with_fake(runner, config=_config_with_reprompts(1))

    assert result.error is None, type(result.error).__name__
    assert len(runner.calls) == 2
    assert all(item.final is False for item in result.output_items)


@pytest.mark.asyncio
async def test_finalized_run_is_not_reprompted() -> None:
    """A run that finalizes properly never sees the nudge, even with the
    reprompt budget available."""
    runner = FakeRunner(
        [
            make_tool_call(
                name="final_answer",
                arguments='{"answer": "done"}',
                call_id="call-final",
            ),
            make_tool_output(call_id="call-final", output='{"acknowledged": true}'),
        ],
    )

    result = await run_with_fake(runner, config=_config_with_reprompts(1))

    assert result.error is None, type(result.error).__name__
    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_legacy_sentinel_still_finalizes_without_reprompt() -> None:
    """The ``<final/>`` text sentinel keeps working as a fallback and counts
    as finalization for the reprompt check."""
    runner = FakeRunner([make_assistant_text("the answer\n<final/>", item_id="m1")])

    result = await run_with_fake(runner, config=_config_with_reprompts(1))

    assert result.error is None, type(result.error).__name__
    assert len(runner.calls) == 1
    assert len(result.output_items) == 1
    assert result.output_items[0].final is True
    assert result.output_items[0].item.content == "the answer"
