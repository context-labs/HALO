"""Unit tests for the chat-shape → Responses-API input render boundary.

Regression coverage for the rerun-from-local-history failure: assistant
``tool_calls`` messages rendered verbatim into a Responses ``input`` array
400 with ``Missing required parameter: 'input[N].content'`` and exhausted the
runner's retry budget (``EngineAgentExhaustedError``).
"""

from __future__ import annotations

from engine.agents.responses_input import to_responses_input
from engine.models.messages import AgentMessage, AgentToolCall, AgentToolFunction


def _tool_call(call_id: str, name: str = "query_traces", arguments: str = "{}") -> AgentToolCall:
    return AgentToolCall(id=call_id, function=AgentToolFunction(name=name, arguments=arguments))


def test_plain_messages_render_as_message_items() -> None:
    items = to_responses_input(
        [
            AgentMessage(role="system", content="sys"),
            AgentMessage(role="user", content="hi"),
            AgentMessage(role="assistant", content="hello"),
        ]
    )
    assert items == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_tool_turn_renders_as_function_call_pair() -> None:
    """The chat-shaped tool turn must become function_call/function_call_output items —

    never an assistant message item without content (the Responses API rejects
    that with 400 ``Missing required parameter: 'input[N].content'``)."""
    items = to_responses_input(
        [
            AgentMessage(
                role="assistant", tool_calls=[_tool_call("call_1", arguments='{"q":"x"}')]
            ),
            AgentMessage(
                role="tool", content="trace result", tool_call_id="call_1", name="query_traces"
            ),
        ]
    )
    assert items == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "query_traces",
            "arguments": '{"q":"x"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "trace result",
        },
    ]


def test_assistant_text_with_tool_calls_renders_message_then_calls() -> None:
    items = to_responses_input(
        [
            AgentMessage(
                role="assistant",
                content="thinking out loud",
                tool_calls=[_tool_call("call_a"), _tool_call("call_b", name="view_trace")],
            )
        ]
    )
    assert items == [
        {"role": "assistant", "content": "thinking out loud"},
        {"type": "function_call", "call_id": "call_a", "name": "query_traces", "arguments": "{}"},
        {"type": "function_call", "call_id": "call_b", "name": "view_trace", "arguments": "{}"},
    ]


def test_none_content_becomes_empty_string() -> None:
    """Message items require ``content``; an empty-text assistant item must not
    round-trip as a content-less message."""
    items = to_responses_input([AgentMessage(role="assistant", content=None)])
    assert items == [{"role": "assistant", "content": ""}]


def test_tool_result_with_none_content_renders_empty_output() -> None:
    items = to_responses_input([AgentMessage(role="tool", content=None, tool_call_id="call_1")])
    assert items == [{"type": "function_call_output", "call_id": "call_1", "output": ""}]


def test_compacted_style_assistant_summary_stays_a_message() -> None:
    """Compacted tool turns render upstream as plain assistant text; they must
    pass through as ordinary message items."""
    items = to_responses_input(
        [AgentMessage(role="assistant", content="Compacted tool calls (id: t-1): summary")]
    )
    assert items == [{"role": "assistant", "content": "Compacted tool calls (id: t-1): summary"}]
