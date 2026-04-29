from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agents import Agent
from agents.result import RunResultStreaming


@runtime_checkable
class RunnerProtocol(Protocol):
    """Minimal seam over the OpenAI Agents SDK Runner.

    The engine only depends on ``run_streamed``. ``agents.Runner`` from the
    openai-agents SDK satisfies this protocol structurally (its
    ``run_streamed`` is a staticmethod with a compatible signature). Tests
    can substitute a fake runner that returns a scripted stream of events;
    see ``engine/tests/probes/probe_kit.py`` for the canonical fake.

    Only the kwargs the engine actually forwards are declared here. Adding
    a new kwarg to the engine means adding it here too — that asymmetry is
    a feature, since it forces the engine to acknowledge any new SDK
    surface it depends on.
    """

    @staticmethod
    def run_streamed(
        *,
        starting_agent: Agent[Any],
        input: Any,
        context: Any = None,
        max_turns: int = 10,
    ) -> RunResultStreaming: ...
