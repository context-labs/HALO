from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agents import FunctionTool, RunContextWrapper
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from engine.agents.agent_context import AgentContext
    from engine.agents.agent_execution import AgentExecution
    from engine.agents.engine_output_bus import EngineOutputBus
    from engine.agents.engine_run_state import EngineRunState
    from engine.sandbox.sandbox import Sandbox
    from engine.traces.trace_store import TraceStore


class ToolContext(BaseModel):
    """Per-invocation context handed to every EngineTool's ``run``.

    Holds references to the run-wide singletons (TraceStore, RunState,
    OutputBus, Sandbox) plus the calling agent's own AgentContext/Execution.
    Tools call ``require_*`` accessors to assert presence, since not every tool
    needs every dependency.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    run_state: "EngineRunState | None" = None
    trace_store: "TraceStore | None" = None
    agent_context: "AgentContext | None" = None
    agent_execution: "AgentExecution | None" = None
    output_bus: "EngineOutputBus | None" = None
    sandbox: "Sandbox | None" = None

    def require_trace_store(self) -> "TraceStore":
        """Return the TraceStore or raise — every trace tool needs it."""
        if self.trace_store is None:
            raise RuntimeError("ToolContext.trace_store required")
        return self.trace_store

    def require_agent_context(self) -> "AgentContext":
        """Return the calling agent's AgentContext or raise — needed by ``get_context_item``."""
        if self.agent_context is None:
            raise RuntimeError("ToolContext.agent_context required")
        return self.agent_context

    def require_sandbox(self) -> "Sandbox":
        """Return the run's Sandbox or raise — needed by ``run_code``."""
        if self.sandbox is None:
            raise RuntimeError("ToolContext.sandbox required")
        return self.sandbox


@runtime_checkable
class EngineTool(Protocol):
    """The unified Engine-side tool interface: a name, a description, typed argument/result models, and ``run``.

    Every tool exposes Pydantic models for arguments and results so the SDK boundary
    stays strongly typed and schemas are derivable for free via Pydantic.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def arguments_model(self) -> type[BaseModel]: ...

    @property
    def result_model(self) -> type[BaseModel]: ...

    async def run(self, tool_context: ToolContext, arguments: Any) -> BaseModel: ...


def to_sdk_function_tool(
    tool: EngineTool,
    *,
    context_factory: Callable[[RunContextWrapper[Any]], ToolContext],
) -> FunctionTool:
    """Adapt an EngineTool into an OpenAI Agents SDK ``FunctionTool``.

    Pulls the JSON schema from the tool's ``arguments_model``, parses raw arguments
    into the typed model on the way in, and serializes the typed result on the way
    out. ``context_factory`` builds a per-invocation ToolContext from the SDK's
    RunContextWrapper.
    """
    arguments_model = tool.arguments_model

    async def _invoke(ctx: RunContextWrapper[Any], raw_arguments: str) -> str:
        parsed = arguments_model.model_validate_json(raw_arguments or "{}")
        tool_context = context_factory(ctx)
        result = await tool.run(tool_context, parsed)
        return result.model_dump_json()

    schema = arguments_model.model_json_schema()
    return FunctionTool(
        name=tool.name,
        description=tool.description,
        params_json_schema=schema,
        on_invoke_tool=_invoke,
        strict_json_schema=False,
    )
