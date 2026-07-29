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


class RunCheckpoint(BaseModel):
    """The root agent's full conversation state at a resumable boundary.

    A HALO run that dies at turn 40 of 50 currently restarts from turn zero and
    re-pays for all 40, because nothing about its conversation survives the
    process — ``EngineRunState`` is in-memory only. The engine can already
    *resume* from a message array (``AgentContext.from_input_messages`` passes a
    caller-supplied system message through unchanged, expressly to support
    continuations); what was missing is anything that durably emits one.

    ``messages`` is exactly what ``from_input_messages`` accepts, so a host can
    persist the latest checkpoint and hand it straight back to restart mid-run.

    Root only. Subagent state is transient — a resumed run re-derives it — and
    checkpointing every agent would multiply payload size for state nothing
    reads.

    Emitted only when ``EngineConfig.emit_run_checkpoints`` is set. It is off by
    default because this widens ``EngineStreamEvent``, and a host pinned to an
    older engine must not receive a variant it cannot parse.
    """

    model_config = ConfigDict(extra="forbid")

    sequence: int
    agent_id: str
    #: Turns the root agent has consumed at this boundary — lets a host tell
    #: checkpoint order without depending on bus sequence numbers.
    turns_used: int
    messages: list[AgentMessage]


EngineStreamEvent: TypeAlias = AgentOutputItem | AgentTextDelta | RunCheckpoint
"""Anything the EngineOutputBus can yield: a durable item, a streaming text
delta, or a resumable checkpoint of root conversation state."""
