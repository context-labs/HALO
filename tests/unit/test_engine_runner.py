from __future__ import annotations

import json

import pytest

from engine.models.engine_output import AgentOutputItem, AgentTextDelta
from engine.models.messages import AgentMessage
from halo_cli import engine_runner


@pytest.mark.asyncio
async def test_stream_to_console_writes_final_answer_and_events(tmp_path, monkeypatch) -> None:
    async def fake_stream_engine_async(_messages, _cfg, _trace_path, *, telemetry):
        assert telemetry is False
        yield AgentTextDelta(
            sequence=1,
            agent_id="root-1",
            parent_agent_id=None,
            parent_tool_call_id=None,
            depth=0,
            item_id="msg-1",
            text_delta="partial",
        )
        yield AgentOutputItem(
            sequence=2,
            agent_id="root-1",
            parent_agent_id=None,
            parent_tool_call_id=None,
            agent_name="root",
            depth=0,
            item=AgentMessage(role="assistant", content="intermediate"),
            final=False,
        )
        yield AgentOutputItem(
            sequence=3,
            agent_id="root-1",
            parent_agent_id=None,
            parent_tool_call_id=None,
            agent_name="root",
            depth=0,
            item=AgentMessage(role="assistant", content="final diagnosis\n<final/>"),
            final=True,
        )

    monkeypatch.setattr(engine_runner, "stream_engine_async", fake_stream_engine_async)

    output_path = tmp_path / "report.md"
    events_path = tmp_path / "events.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("")
    cfg = engine_runner.make_config(
        "gpt-test",
        0,
        4,
        1,
        None,
        0,
        trace_detail_tools_enabled=False,
        run_code_enabled=False,
    )

    final_answer = await engine_runner.stream_to_console(
        trace_path,
        "prompt",
        cfg,
        output_path=output_path,
        events_path=events_path,
    )

    assert final_answer == "final diagnosis\n<final/>"
    assert output_path.read_text() == "final diagnosis\n<final/>\n"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["sequence"] for event in events] == [2, 3]
    assert events[-1]["final"] is True
