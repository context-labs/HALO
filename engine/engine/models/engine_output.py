from __future__ import annotations

from typing import TypeAlias

from pydantic import BaseModel, ConfigDict

from engine.models.messages import AgentMessage


class AgentOutputItem(BaseModel):
    """Public, lineage-rich wrapper around one durable AgentMessage emitted by an agent.

    Tool calls and tool results live inside ``item`` (an AgentMessage), not as separate
    payload types — that keeps interleaved parallel-child output trivially groupable
    by lineage fields while preserving messages-array compatibility. ``final=True``
    marks the root agent's terminating assistant message.
    """

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
    """Incremental token-level delta emitted between durable AgentOutputItems while assistant text streams."""

    model_config = ConfigDict(extra="forbid")

    sequence: int
    agent_id: str
    parent_agent_id: str | None
    parent_tool_call_id: str | None
    depth: int
    item_id: str
    text_delta: str


EngineStreamEvent: TypeAlias = AgentOutputItem | AgentTextDelta
"""Anything the EngineOutputBus can yield: a durable item or a streaming text delta."""
