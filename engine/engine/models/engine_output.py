from __future__ import annotations

from typing import TypeAlias

from pydantic import BaseModel, ConfigDict

from engine.models.messages import AgentMessage


class AgentOutputItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int
    agent_id: str
    parent_agent_id: str | None
    parent_tool_call_id: str | None
    agent_name: str
    depth: int
    item: AgentMessage
    final: bool = False


class AgentTextDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int
    agent_id: str
    parent_agent_id: str | None
    parent_tool_call_id: str | None
    depth: int
    item_id: str
    text_delta: str


EngineStreamEvent: TypeAlias = AgentOutputItem | AgentTextDelta
