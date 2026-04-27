from __future__ import annotations

from pathlib import Path

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
        sandbox_config: SandboxConfig,
        sandbox_venv: Path | None = None,
    ) -> None:
        """Default ``sandbox_venv`` resolves to ``.sandbox-venv`` next to the engine package."""
        self._sandbox_config = sandbox_config
        self._default_venv = sandbox_venv or Path(__file__).resolve().parents[2] / ".sandbox-venv"

    async def run(
        self, tool_context: ToolContext, arguments: RunCodeArguments
    ) -> CodeExecutionResult:
        """Run user code through ``SandboxRunner`` with the active TraceStore's trace/index paths."""
        runner = tool_context.sandbox_runner or SandboxRunner(sandbox_venv=self._default_venv)
        store = tool_context.require_trace_store()
        return await runner.run_python(
            code=arguments.code,
            trace_path=store.trace_path,
            index_path=store.index_path,
            config=self._sandbox_config,
        )
