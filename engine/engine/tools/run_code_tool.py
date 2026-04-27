from __future__ import annotations

from engine.sandbox.runtime_mounts import PythonRuntimeMounts
from engine.sandbox.sandbox_availability import SandboxStatus
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
    """

    name = "run_code"
    description = (
        "Execute Python code in a sandbox with read-only access to the trace dataset. "
        "numpy, pandas, and a preloaded trace_store variable are available."
    )
    arguments_model = RunCodeArguments
    result_model = CodeExecutionResult

    def __init__(
        self,
        *,
        sandbox_config: SandboxConfig,
        sandbox_status: SandboxStatus,
        runtime_mounts: PythonRuntimeMounts,
    ) -> None:
        if not sandbox_status.available:
            raise RuntimeError(
                "RunCodeTool constructed with unavailable SandboxStatus; "
                "the tool factory must gate on status.available."
            )
        self._sandbox_config = sandbox_config
        self._sandbox_status = sandbox_status
        self._runtime_mounts = runtime_mounts

    async def run(
        self, tool_context: ToolContext, arguments: RunCodeArguments
    ) -> CodeExecutionResult:
        """Run user code through ``SandboxRunner`` with the active TraceStore's trace/index paths."""
        runner = tool_context.sandbox_runner or SandboxRunner(
            sandbox_status=self._sandbox_status,
            runtime_mounts=self._runtime_mounts,
        )
        store = tool_context.require_trace_store()
        return await runner.run_python(
            code=arguments.code,
            trace_path=store.trace_path,
            index_path=store.index_path,
            config=self._sandbox_config,
        )
