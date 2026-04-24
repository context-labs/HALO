from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SubagentToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    child_agent_id: str
    answer: str
    output_start_sequence: int
    output_end_sequence: int
    turns_used: int
    tool_calls_made: int
