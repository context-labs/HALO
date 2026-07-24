"""Engine ``final_answer`` tool-call finalization end-to-end.

The root agent completes the run by calling the ``final_answer`` tool
(``StopAtTools`` on the root SDK agent); the mapper transforms that call
into a final-flagged plain assistant message, and the runner reprompts
once when a run ends without finalizing. These tests cover the
engine-level contract with a scripted ``FakeRunner``.
"""

from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError

from tests.probes.probe_kit import (
    FakeRunner,
    make_assistant_text,
    make_default_config,
    make_final_answer,
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
async def test_root_text_containing_a_final_marker_string_does_not_finalize() -> None:
    """Text that merely *contains* a would-be sentinel string never finalizes —
    HALO analyzes traces of other agent systems, so quoted markers like
    ``<final/>`` must not terminate the run. Only the tool call can."""
    runner = FakeRunner(
        [make_assistant_text("the trace contains <final/> markers", item_id="m1")],
        [*make_final_answer("done")],
    )

    result = await run_with_fake(runner, config=_config_with_reprompts(1))

    assert result.error is None, type(result.error).__name__
    assert len(runner.calls) == 2
    finals = [item for item in result.output_items if item.final]
    assert len(finals) == 1
    assert finals[0].item.content == "done"
    assert result.output_items[0].final is False
    assert result.output_items[0].item.content == "the trace contains <final/> markers"


@pytest.mark.asyncio
async def test_reprompt_nudge_survives_a_transient_failure() -> None:
    """A retriable failure on the nudged attempt must not burn the only
    nudge: like the refusal retry, the pending reprompt stays set until the
    stream starts successfully, so the retry re-sends the nudge instead of
    rerunning without it."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    runner = FakeRunner(
        [make_assistant_text("no tool call", item_id="m1")],
        APIConnectionError(request=request),
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
    assert len(runner.calls) == 3
    # Both the failed nudged attempt and its retry carry the nudge message.
    for call in runner.calls[1:]:
        nudge = call["input"][-1]
        assert nudge["role"] == "user"
        assert "final_answer" in nudge["content"]
    finals = [item for item in result.output_items if item.final]
    assert len(finals) == 1
    assert finals[0].item.content == "recovered"
