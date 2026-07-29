"""The wire shape sent to the Responses API when replaying local history.

Regression for the production failure where a mid-stream retry replayed
chat-completions-shaped history — an assistant item carrying only
``tool_calls`` (no ``content`` key after ``exclude_none`` dumping) — into a
Responses API ``input`` array. OpenAI/Azure reject that with
``400 missing_required_parameter: 'input[N].content'``, and because the
replay is rebuilt identically every attempt, the run burned its whole retry
budget on the same rejection (run ba6fc95c).

``to_responses_input`` renders tool turns as first-class Responses items
(``function_call`` / ``function_call_output``) so replayed histories are
valid by construction.
"""

from __future__ import annotations

from engine.agents.responses_input import to_responses_input
from engine.models.messages import AgentMessage, AgentToolCall, AgentToolFunction


def _tool_call(call_id: str, name: str = "query_traces") -> AgentToolCall:
    return AgentToolCall(id=call_id, function=AgentToolFunction(name=name, arguments='{"x":1}'))


def test_plain_messages_pass_through_as_message_items() -> None:
    items = to_responses_input(
        [
            AgentMessage(role="system", content="be helpful"),
            AgentMessage(role="user", content="hi"),
            AgentMessage(role="assistant", content="hello"),
        ]
    )
    assert items == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_tool_call_only_assistant_turn_renders_as_function_call_items() -> None:
    """The production crash shape: assistant with tool_calls and no content."""
    items = to_responses_input(
        [
            AgentMessage(role="assistant", tool_calls=[_tool_call("call-1")]),
            AgentMessage(role="tool", content="result", tool_call_id="call-1", name="query_traces"),
        ]
    )
    assert items == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "query_traces",
            "arguments": '{"x":1}',
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "result"},
    ]


def test_assistant_text_plus_tool_calls_emits_message_then_calls() -> None:
    items = to_responses_input(
        [
            AgentMessage(
                role="assistant",
                content="let me check",
                tool_calls=[_tool_call("call-1"), _tool_call("call-2", name="count_traces")],
            ),
        ]
    )
    assert items == [
        {"role": "assistant", "content": "let me check"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "query_traces",
            "arguments": '{"x":1}',
        },
        {
            "type": "function_call",
            "call_id": "call-2",
            "name": "count_traces",
            "arguments": '{"x":1}',
        },
    ]


def test_none_content_coerces_to_empty_string() -> None:
    """A content-less non-tool message must still carry a ``content`` key —
    dropping it is exactly what the API rejects."""
    items = to_responses_input(
        [
            AgentMessage(role="assistant"),
            AgentMessage(role="tool", tool_call_id="call-9"),
        ]
    )
    assert items == [
        {"role": "assistant", "content": ""},
        {"type": "function_call_output", "call_id": "call-9", "output": ""},
    ]


def test_every_item_is_wire_valid() -> None:
    """No rendered item may be a message lacking ``content`` — the invariant
    the Responses API enforces with ``missing_required_parameter``."""
    items = to_responses_input(
        [
            AgentMessage(role="system", content="s"),
            AgentMessage(role="user", content="u"),
            AgentMessage(role="assistant", tool_calls=[_tool_call("c1")]),
            AgentMessage(role="tool", content="r", tool_call_id="c1"),
            AgentMessage(role="assistant", content="done"),
        ]
    )
    for item in items:
        if "role" in item:
            assert "content" in item, f"message item missing content: {item}"
            assert "tool_calls" not in item, f"chat-only field leaked: {item}"
            assert item["role"] != "tool", f"chat-only tool role leaked: {item}"
