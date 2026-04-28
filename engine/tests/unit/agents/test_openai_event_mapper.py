from __future__ import annotations

from engine.agents.agent_execution import AgentExecution
from engine.agents.openai_event_mapper import OpenAiEventMapper
from tests.unit.agents._sdk_events import (
    assistant_message_event,
    text_delta_event,
    tool_call_event,
    tool_output_event,
)


def _exec(*, depth: int = 0, parent_tool_call_id: str | None = None) -> AgentExecution:
    return AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=depth,
        parent_agent_id="root" if depth else None,
        parent_tool_call_id=parent_tool_call_id,
    )


def test_assistant_text_item_plain() -> None:
    mapper = OpenAiEventMapper()
    mapped = mapper.to_mapped_event(
        assistant_message_event(item_id="msg_1", text="Done."),
        execution=_exec(),
        is_root=True,
    )
    assert mapped.context_item is not None
    assert mapped.context_item.role == "assistant"
    assert mapped.output_item is not None
    assert mapped.output_item.final is False
    assert mapped.output_item.item.content == "Done."


def test_root_assistant_final_sentinel_strips_and_sets_final() -> None:
    mapper = OpenAiEventMapper()
    mapped = mapper.to_mapped_event(
        assistant_message_event(item_id="msg_2", text="Final answer.\n<final/>"),
        execution=_exec(),
        is_root=True,
    )
    assert mapped.output_item is not None
    assert mapped.context_item is not None
    assert mapped.output_item.final is True
    assert mapped.output_item.item.content == "Final answer."
    assert mapped.context_item.content == "Final answer."


def test_subagent_assistant_final_sentinel_ignored() -> None:
    mapper = OpenAiEventMapper()
    mapped = mapper.to_mapped_event(
        assistant_message_event(item_id="msg_3", text="sub done <final/>"),
        execution=_exec(depth=1, parent_tool_call_id="c1"),
        is_root=False,
    )
    assert mapped.output_item is not None
    assert mapped.output_item.final is False
    assert "sub done" in (mapped.output_item.item.content or "")


def test_tool_call_item_extracts_name_and_arguments() -> None:
    """Object-form ``raw_item`` (Responses-API streaming): SDK property surface works directly."""
    mapper = OpenAiEventMapper()
    mapped = mapper.to_mapped_event(
        tool_call_event(call_id="call_1", name="query_traces", arguments='{"q":"x"}'),
        execution=_exec(),
        is_root=True,
    )
    assert mapped.context_item is not None
    assert mapped.context_item.tool_calls is not None
    tc = mapped.context_item.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.function.name == "query_traces"
    assert tc.function.arguments == '{"q":"x"}'


def test_tool_output_item_uses_call_id_and_output() -> None:
    """Dict-form ``raw_item`` (Chat-Completions / replay path): the SDK ``call_id`` property still works."""
    mapper = OpenAiEventMapper()
    mapped = mapper.to_mapped_event(
        tool_output_event(call_id="call_1", output="ok"),
        execution=_exec(),
        is_root=True,
    )
    assert mapped.context_item is not None
    assert mapped.context_item.role == "tool"
    assert mapped.context_item.tool_call_id == "call_1"
    assert mapped.context_item.content == "ok"


def test_tool_output_inherits_name_from_preceding_tool_call() -> None:
    """The function name follows from the call to its result.

    Compactor output and Chat-Completions replay both read
    ``AgentContextItem.name`` on tool messages. The SDK's
    ``ToolCallOutputItem`` doesn't carry it (Responses-API
    ``FunctionCallOutput`` has no ``name`` field), so the mapper
    correlates by ``call_id`` from the preceding ``ToolCallItem``.
    """
    mapper = OpenAiEventMapper()
    mapper.to_mapped_event(
        tool_call_event(call_id="call_42", name="search_trace"),
        execution=_exec(),
        is_root=True,
    )
    output_mapped = mapper.to_mapped_event(
        tool_output_event(call_id="call_42", output="ok"),
        execution=_exec(),
        is_root=True,
    )
    assert output_mapped.context_item is not None
    assert output_mapped.context_item.name == "search_trace"
    assert output_mapped.output_item is not None
    assert output_mapped.output_item.item.name == "search_trace"


def test_tool_output_name_is_none_when_call_unseen() -> None:
    """Defensive: an orphan tool_output (call we never saw) keeps ``name=None``.

    Replay code already handles ``name`` being missing — the engine should
    not fabricate a name when it has no record of the matching call.
    """
    mapper = OpenAiEventMapper()
    mapped = mapper.to_mapped_event(
        tool_output_event(call_id="call_unseen", output="ok"),
        execution=_exec(),
        is_root=True,
    )
    assert mapped.context_item is not None
    assert mapped.context_item.name is None


def test_raw_text_delta_produces_delta_only() -> None:
    mapper = OpenAiEventMapper()
    mapped = mapper.to_mapped_event(
        text_delta_event(item_id="msg_1", delta="par"), execution=_exec(), is_root=True
    )
    assert mapped.context_item is None
    assert mapped.output_item is None
    assert mapped.delta is not None
    assert mapped.delta.text_delta == "par"
    assert mapped.delta.item_id == "msg_1"


def test_tool_call_and_output_have_distinct_item_ids() -> None:
    """``AgentContext._index`` keys by ``item_id``; the tool_call and its tool_output
    must not collide on the same key, otherwise ``get_context_item`` returns the
    wrong record. Synthetic ``tool-call-{call_id}`` / ``tool-result-{call_id}``
    naming guarantees they differ even when both events share the same call_id.
    """
    mapper = OpenAiEventMapper()
    call_mapped = mapper.to_mapped_event(
        tool_call_event(call_id="call_xyz", name="query_traces"),
        execution=_exec(),
        is_root=True,
    )
    output_mapped = mapper.to_mapped_event(
        tool_output_event(call_id="call_xyz", output="ok"),
        execution=_exec(),
        is_root=True,
    )
    assert call_mapped.context_item is not None
    assert output_mapped.context_item is not None
    assert call_mapped.context_item.item_id == "tool-call-call_xyz"
    assert output_mapped.context_item.item_id == "tool-result-call_xyz"
