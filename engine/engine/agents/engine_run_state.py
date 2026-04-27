from __future__ import annotations

from dataclasses import dataclass, field

from agents import Runner

from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.runner_protocol import RunnerProtocol
from engine.engine_config import EngineConfig
from engine.sandbox.runtime_mounts import PythonRuntimeMounts
from engine.sandbox.sandbox_availability import SandboxStatus
from engine.traces.trace_store import TraceStore


@dataclass
class EngineRunState:
    """Shared mutable state for one Engine run.

    Holds the singleton TraceStore, output bus, and config, plus lookup tables for
    AgentExecutions by ``agent_id`` and by the ``tool_call_id`` that spawned them. The
    ``runner`` field is a test seam: production uses ``agents.Runner``, probes inject
    a fake (see ``RunnerProtocol``).

    ``sandbox_status`` and ``runtime_mounts`` are resolved once in ``main`` before
    agent construction. The tool factory consults ``sandbox_status.available`` to
    decide whether ``run_code`` is registered for the run; ``runtime_mounts`` is
    only used when sandboxing is available.
    """

    trace_store: TraceStore
    output_bus: EngineOutputBus
    config: EngineConfig
    sandbox_status: SandboxStatus
    runtime_mounts: PythonRuntimeMounts | None
    executions_by_agent_id: dict[str, AgentExecution] = field(default_factory=dict)
    executions_by_tool_call_id: dict[str, AgentExecution] = field(default_factory=dict)
    runner: RunnerProtocol = field(default_factory=lambda: Runner)

    def register(self, execution: AgentExecution) -> None:
        """Index a newly-created AgentExecution by agent_id, and by tool_call_id when subagent."""
        self.executions_by_agent_id[execution.agent_id] = execution
        if execution.parent_tool_call_id is not None:
            self.executions_by_tool_call_id[execution.parent_tool_call_id] = execution
