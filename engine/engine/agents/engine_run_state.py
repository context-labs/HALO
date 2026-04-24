from __future__ import annotations

from dataclasses import dataclass, field

from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.engine_config import EngineConfig
from engine.traces.trace_store import TraceStore


@dataclass
class EngineRunState:
    trace_store: TraceStore
    output_bus: EngineOutputBus
    config: EngineConfig
    executions_by_agent_id: dict[str, AgentExecution] = field(default_factory=dict)
    executions_by_tool_call_id: dict[str, AgentExecution] = field(default_factory=dict)

    def register(self, execution: AgentExecution) -> None:
        self.executions_by_agent_id[execution.agent_id] = execution
        if execution.parent_tool_call_id is not None:
            self.executions_by_tool_call_id[execution.parent_tool_call_id] = execution

    def get_by_agent_id(self, agent_id: str) -> AgentExecution:
        return self.executions_by_agent_id[agent_id]

    def get_by_tool_call_id(self, tool_call_id: str) -> AgentExecution:
        return self.executions_by_tool_call_id[tool_call_id]
