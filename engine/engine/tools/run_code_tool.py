from __future__ import annotations

from engine.sandbox.models import CodeExecutionResult, RunCodeArguments
from engine.tools.tool_protocol import ToolContext


class RunCodeTool:
    """Tool exposing the sandbox to agents for ad-hoc Python analysis over the trace dataset.

    Sandboxed code gets read-only TraceStore access plus numpy/pandas, no network,
    a writable temp dir, and a wall-clock timeout. The tool result is a typed
    ``CodeExecutionResult`` regardless of pass/fail/timeout, so the calling model
    can keep reasoning even when user code crashed.

    The tool itself is stateless: the live ``Sandbox`` comes through
    ``tool_context.sandbox`` (wired by the per-run ``make_ctx`` factory).
    Registration is gated upstream so ``tool_context.sandbox`` is always
    populated when this tool runs.
    """

    name = "run_code"
    description = (
        "Execute Python code in a sandbox with read-only access to the trace dataset. "
        "numpy, pandas, and a preloaded trace_store variable are available."
    )
    arguments_model = RunCodeArguments
    result_model = CodeExecutionResult

    async def run(
        self, tool_context: ToolContext, arguments: RunCodeArguments
    ) -> CodeExecutionResult:
        """Run user code through the run's ``Sandbox`` against the active TraceStore's paths."""
        sandbox = tool_context.require_sandbox()
        store = tool_context.require_trace_store()
        return await sandbox.run_python(
            code=arguments.code,
            trace_path=store.trace_path,
            index_path=store.index_path,
        )
