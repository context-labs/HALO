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
        self._sandbox_config = sandbox_config
        self._default_venv = sandbox_venv or Path(__file__).resolve().parents[2] / ".sandbox-venv"

    async def run(self, tool_context: ToolContext, arguments: RunCodeArguments) -> CodeExecutionResult:
        runner = tool_context.sandbox_runner or SandboxRunner(sandbox_venv=self._default_venv)
        store = tool_context.require_trace_store()
        return await runner.run_python(
            code=arguments.code,
            trace_path=store.trace_path,
            index_path=store.index_path,
            config=self._sandbox_config,
        )
