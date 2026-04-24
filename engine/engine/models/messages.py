from __future__ import annotations

from typing import Literal, TypeAlias

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, ConfigDict

MessageContent: TypeAlias = str | list[ChatCompletionContentPartParam] | None


class AgentToolFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: str


class AgentToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["function"] = "function"
    function: AgentToolFunction


class AgentMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent = None
    tool_calls: list[AgentToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
