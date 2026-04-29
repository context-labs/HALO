from __future__ import annotations

from typing import Literal, TypeAlias

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, ConfigDict

MessageContent: TypeAlias = str | list[ChatCompletionContentPartParam] | None
"""OpenAI-compatible message content: plain text, structured parts, or absent."""


class AgentToolFunction(BaseModel):
    """The ``function`` block of an OpenAI tool call: tool name plus serialized JSON arguments."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: str


class AgentToolCall(BaseModel):
    """One tool call emitted by the assistant, in OpenAI's ``tool_calls`` array shape."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["function"] = "function"
    function: AgentToolFunction


class AgentMessage(BaseModel):
    """OpenAI/HF-compatible chat message; the on-the-wire shape sent to the model.

    The Engine deliberately uses the standard OpenAI fields without extras so message
    arrays remain valid for any compatible provider. Engine-specific metadata lives on
    ``AgentContextItem`` and ``AgentOutputItem`` instead of polluting this shape.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent = None
    tool_calls: list[AgentToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
