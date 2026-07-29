"""Root runs emit resumable checkpoints; subagents and old hosts do not see them.

A HALO run that dies at turn 40 of 50 restarts from turn zero and re-pays for
all 40, because ``EngineRunState`` is in-memory only. The engine can already
*resume* — ``AgentContext.from_input_messages`` passes a caller-supplied system
message through unchanged, expressly for continuations — so the missing piece
was durably emitting the state to resume from.

The round-trip is the property that matters: what a checkpoint carries must be
accepted by ``from_input_messages`` without reshaping. These tests assert that
directly rather than trusting the two shapes to stay compatible by inspection.
"""

from __future__ import annotations

import pytest

from engine.agents.agent_context import AgentContext
from engine.agents.agent_context_items import AgentContextItem
from engine.agents.engine_output_bus import EngineOutputBus
from engine.model_config import ModelConfig
from engine.models.engine_output import RunCheckpoint
from engine.models.messages import AgentMessage
from tests.probes.probe_kit import make_default_config


def _context() -> AgentContext:
    return AgentContext(
        items=[
            AgentContextItem(item_id="sys-0", role="system", content="system prompt"),
            AgentContextItem(item_id="in-0", role="user", content="analyse these traces"),
            AgentContextItem(item_id="a-0", role="assistant", content="working on it"),
        ],
        compaction_model=ModelConfig(name="c"),
        text_message_compaction_keep_last_messages=100,
        tool_call_compaction_keep_last_turns=100,
    )


@pytest.mark.asyncio
async def test_the_bus_assigns_the_checkpoint_a_sequence() -> None:
    """Checkpoints interleave with output in the same monotonic ordering.

    A host reading the stream needs to know which items a checkpoint covers;
    that only works if the bus sequences it like everything else.
    """
    bus = EngineOutputBus()
    emitted = await bus.emit(
        RunCheckpoint(
            sequence=0,
            agent_id="root-1",
            turns_used=3,
            messages=_context().to_messages_array(),
        )
    )
    assert emitted.sequence == 0

    second = await bus.emit(RunCheckpoint(sequence=0, agent_id="root-1", turns_used=4, messages=[]))
    assert second.sequence == 1


def test_a_checkpoint_round_trips_into_a_resumable_context() -> None:
    """The whole point: hand a checkpoint back and get the same conversation.

    ``from_input_messages`` keeps a caller-supplied system message verbatim, so
    a resumed run must not acquire a second, engine-rendered one.
    """
    original = _context()
    checkpoint = RunCheckpoint(
        sequence=0,
        agent_id="root-1",
        turns_used=2,
        messages=original.to_messages_array(),
    )

    resumed = AgentContext.from_input_messages(
        checkpoint.messages,
        engine_config=make_default_config(),
        code_repo=None,
        git_repo=None,
    )

    assert [m.role for m in resumed.to_messages_array()] == [
        "system",
        "user",
        "assistant",
    ]
    assert resumed.to_messages_array() == original.to_messages_array()
    assert sum(1 for m in resumed.to_messages_array() if m.role == "system") == 1


def test_checkpoint_rejects_unknown_fields() -> None:
    """``extra="forbid"`` keeps a host from silently relying on a field the
    engine never promised."""
    with pytest.raises(Exception):
        RunCheckpoint(
            sequence=0,
            agent_id="root-1",
            turns_used=1,
            messages=[],
            not_a_real_field=True,  # type: ignore[call-arg]
        )


def test_messages_are_plain_agent_messages() -> None:
    """No bespoke wire type — the payload is exactly what the resume entry
    point already consumes."""
    checkpoint = RunCheckpoint(
        sequence=0,
        agent_id="root-1",
        turns_used=1,
        messages=_context().to_messages_array(),
    )
    assert all(isinstance(m, AgentMessage) for m in checkpoint.messages)
