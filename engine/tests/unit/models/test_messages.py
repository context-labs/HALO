from __future__ import annotations

import pytest
from pydantic import ValidationError

from engine.models.messages import AgentMessage, AgentToolCall, AgentToolFunction


def test_user_message_minimum() -> None:
    msg = AgentMessage(role="user", content="hi")
    assert msg.role == "user"
    assert msg.content == "hi"
    assert msg.tool_calls is None


def test_assistant_tool_call_message() -> None:
    msg = AgentMessage(
        role="assistant",
        content=None,
        tool_calls=[
            AgentToolCall(
                id="call_1",
                function=AgentToolFunction(name="x", arguments="{}"),
            )
        ],
    )
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].function.name == "x"


def test_tool_role_requires_tool_call_id_when_used() -> None:
    msg = AgentMessage(
        role="tool",
        content="{}",
        tool_call_id="call_1",
        name="x",
    )
    assert msg.tool_call_id == "call_1"


def test_invalid_role_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentMessage(role="bogus", content="x")  # type: ignore[arg-type]


def test_roundtrip_json() -> None:
    msg = AgentMessage(role="user", content="hi")
    blob = msg.model_dump_json()
    restored = AgentMessage.model_validate_json(blob)
    assert restored == msg
