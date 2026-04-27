from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RunnerProtocol(Protocol):
    """Minimal seam over the OpenAI Agents SDK Runner.

    The engine only depends on ``run_streamed``. ``agents.Runner`` from the
    openai-agents SDK satisfies this protocol structurally (its
    ``run_streamed`` is a staticmethod with a compatible signature). Tests can
    substitute a fake runner that returns a scripted stream of events; see
    ``engine/tests/probes/probe_kit.py`` for the canonical fake.

    The return value is an object exposing ``stream_events()`` as an async
    iterator. We type it as ``Any`` to avoid leaking SDK types into engine
    code.
    """

    @staticmethod
    def run_streamed(
        *,
        starting_agent: Any,
        input: Any,
        context: Any = None,
    ) -> Any: ...
