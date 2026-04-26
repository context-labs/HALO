from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.agents.agent_execution import AgentExecution
from engine.agents.openai_event_mapper import OpenAiEventMapper
from engine.models.engine_output import AgentOutputItem


def _exec() -> AgentExecution:
    return AgentExecution(
        agent_id="root", agent_name="root", depth=0,
        parent_agent_id=None, parent_tool_call_id=None,
    )


def _wrap_item(item: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(type="run_item_stream_event", item=item)


def test_assistant_text_item_plain() -> None:
    mapper = OpenAiEventMapper()
    raw = _wrap_item(SimpleNamespace(
        type="message_output_item",
        raw_item=SimpleNamespace(
            id="msg_1",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text="Done.")],
        ),
    ))
    mapped = mapper.to_mapped_event(raw, execution=_exec(), is_root=True)
    assert mapped.context_item is not None
    assert mapped.context_item.role == "assistant"
    assert mapped.output_item is not None
    assert isinstance(mapped.output_item, AgentOutputItem)
    assert mapped.output_item.final is False


def test_root_assistant_final_sentinel_strips_and_sets_final() -> None:
    mapper = OpenAiEventMapper()
    raw = _wrap_item(SimpleNamespace(
        type="message_output_item",
        raw_item=SimpleNamespace(
            id="msg_2",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text="Final answer.\n<final/>")],
        ),
    ))
    mapped = mapper.to_mapped_event(raw, execution=_exec(), is_root=True)
    assert mapped.output_item is not None
    assert mapped.output_item.final is True
    assert mapped.output_item.item.content == "Final answer."
    assert mapped.context_item.content == "Final answer."


def test_subagent_assistant_final_sentinel_ignored() -> None:
    mapper = OpenAiEventMapper()
    raw = _wrap_item(SimpleNamespace(
        type="message_output_item",
        raw_item=SimpleNamespace(
            id="msg_3",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text="sub done <final/>")],
        ),
    ))
    execution = AgentExecution(
        agent_id="sub", agent_name="sub", depth=1,
        parent_agent_id="root", parent_tool_call_id="c1",
    )
    mapped = mapper.to_mapped_event(raw, execution=execution, is_root=False)
    assert mapped.output_item is not None
    assert mapped.output_item.final is False
    assert "sub done" in (mapped.output_item.item.content or "")


def test_tool_call_output_item() -> None:
    mapper = OpenAiEventMapper()
    raw = _wrap_item(SimpleNamespace(
        type="tool_call_item",
        raw_item=SimpleNamespace(
            call_id="call_1",
            id="call_1",
            name="query_traces",
            arguments="{}",
        ),
    ))
    mapped = mapper.to_mapped_event(raw, execution=_exec(), is_root=True)
    assert mapped.context_item is not None
    assert mapped.context_item.role == "assistant"
    assert mapped.context_item.tool_calls is not None
    assert mapped.context_item.tool_calls[0].function.name == "query_traces"


def test_tool_output_item() -> None:
    mapper = OpenAiEventMapper()
    raw = _wrap_item(SimpleNamespace(
        type="tool_call_output_item",
        raw_item=SimpleNamespace(call_id="call_1", id="call_1", name="query_traces"),
        output="ok",
    ))
    mapped = mapper.to_mapped_event(raw, execution=_exec(), is_root=True)
    assert mapped.context_item is not None
    assert mapped.context_item.role == "tool"
    assert mapped.context_item.tool_call_id == "call_1"


def test_raw_text_delta_produces_delta_only() -> None:
    mapper = OpenAiEventMapper()
    raw = SimpleNamespace(
        type="raw_response_event",
        data=SimpleNamespace(
            type="response.output_text.delta",
            delta="par",
            item_id="msg_1",
        ),
    )
    mapped = mapper.to_mapped_event(raw, execution=_exec(), is_root=True)
    assert mapped.context_item is None
    assert mapped.output_item is None
    assert mapped.delta is not None
    assert mapped.delta.text_delta == "par"
