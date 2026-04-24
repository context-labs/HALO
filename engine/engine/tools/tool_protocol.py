from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from engine.agents.agent_context import AgentContext
    from engine.agents.agent_execution import AgentExecution
    from engine.agents.engine_output_bus import EngineOutputBus
    from engine.agents.engine_run_state import EngineRunState
    from engine.sandbox.sandbox_runner import SandboxRunner
    from engine.traces.trace_store import TraceStore


class ToolContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    run_state: "EngineRunState | None" = None
    trace_store: "TraceStore | None" = None
    agent_context: "AgentContext | None" = None
    agent_execution: "AgentExecution | None" = None
    output_bus: "EngineOutputBus | None" = None
    sandbox_runner: "SandboxRunner | None" = None

    def require_trace_store(self) -> "TraceStore":
        if self.trace_store is None:
            raise RuntimeError("ToolContext.trace_store required")
        return self.trace_store

    def require_agent_context(self) -> "AgentContext":
        if self.agent_context is None:
            raise RuntimeError("ToolContext.agent_context required")
        return self.agent_context


@runtime_checkable
class EngineTool(Protocol):
    name: str
    description: str
    arguments_model: type[BaseModel]
    result_model: type[BaseModel]

    async def run(self, tool_context: ToolContext, arguments: Any) -> BaseModel: ...
