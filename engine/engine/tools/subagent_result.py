from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SubagentToolResult(BaseModel):
    """Concise typed payload returned to a parent for one ``call_subagent`` tool call.

    ``answer`` is the child's final assistant text. The sequence range lets the parent
    correlate child output back to interleaved bus items. ``turns_used`` and
    ``tool_calls_made`` come straight off the child ``AgentExecution``. Serialized
    as JSON at the SDK tool-message boundary; the full child stream is delivered
    through ``EngineOutputBus``, not embedded here.
    """

    model_config = ConfigDict(extra="forbid")

    child_agent_id: str
    answer: str
    output_start_sequence: int
    output_end_sequence: int
    turns_used: int
    tool_calls_made: int
