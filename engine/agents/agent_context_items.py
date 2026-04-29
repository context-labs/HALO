from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from engine.models.messages import AgentToolCall, MessageContent


class AgentContextItem(BaseModel):
    """Engine-internal stored representation of one conversation item.

    Superset of an OpenAI/HF ``AgentMessage``: keeps the original role/content/tool fields,
    plus lineage metadata (agent_id, parent_*) and compaction state. The original fields
    are preserved even after compaction so context tools can still inspect them.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent = None
    tool_calls: list[AgentToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    is_compacted: bool = False
    compaction_summary: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    parent_tool_call_id: str | None = None
