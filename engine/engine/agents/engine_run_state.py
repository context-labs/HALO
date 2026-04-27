from __future__ import annotations

from dataclasses import dataclass, field

from agents import Runner

from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.runner_protocol import RunnerProtocol
from engine.engine_config import EngineConfig
from engine.traces.trace_store import TraceStore


@dataclass
class EngineRunState:
    trace_store: TraceStore
    output_bus: EngineOutputBus
    config: EngineConfig
    executions_by_agent_id: dict[str, AgentExecution] = field(default_factory=dict)
    executions_by_tool_call_id: dict[str, AgentExecution] = field(default_factory=dict)
    runner: RunnerProtocol = field(default_factory=lambda: Runner)

    def register(self, execution: AgentExecution) -> None:
        self.executions_by_agent_id[execution.agent_id] = execution
        if execution.parent_tool_call_id is not None:
            self.executions_by_tool_call_id[execution.parent_tool_call_id] = execution
