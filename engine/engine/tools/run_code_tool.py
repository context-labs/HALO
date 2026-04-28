from __future__ import annotations

from engine.sandbox.sandbox_availability import SandboxRuntime
from engine.sandbox.sandbox_config import (
    CodeExecutionResult,
    RunCodeArguments,
    SandboxConfig,
)
from engine.sandbox.sandbox_runner import SandboxRunner
from engine.tools.tool_protocol import ToolContext


class RunCodeTool:
    """Tool exposing the sandbox to agents for ad-hoc Python analysis over the trace dataset.

    Sandboxed code gets read-only TraceStore access plus numpy/pandas, no network,
    a writable temp dir, and a wall-clock timeout. The tool result is a typed
    ``CodeExecutionResult`` regardless of pass/fail/timeout, so the calling model
    can keep reasoning even when user code crashed.

    The constructor takes a ``SandboxRuntime`` directly: by the time the tool
    factory builds this tool, the sandbox is known to be ready. There is no
    representable "available but missing mounts" state to defend against.
    """

    name = "run_code"
    description = (
        "Execute Python code in a sandbox with read-only access to the trace dataset. "
        "numpy, pandas, and a preloaded trace_store variable are available."
    )
    arguments_model = RunCodeArguments
    result_model = CodeExecutionResult

    def __init__(self, *, sandbox_config: SandboxConfig, sandbox: SandboxRuntime) -> None:
        self._sandbox_config = sandbox_config
        self._sandbox = sandbox

    async def run(
        self, tool_context: ToolContext, arguments: RunCodeArguments
    ) -> CodeExecutionResult:
        """Run user code through ``SandboxRunner`` with the active TraceStore's trace/index paths."""
        runner = tool_context.sandbox_runner or SandboxRunner(sandbox=self._sandbox)
        store = tool_context.require_trace_store()
        return await runner.run_python(
            code=arguments.code,
            trace_path=store.trace_path,
            index_path=store.index_path,
            config=self._sandbox_config,
        )
