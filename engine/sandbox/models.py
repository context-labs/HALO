from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CodeExecutionResult(BaseModel):
    """Outcome of running code in the sandbox: capped stdout/stderr, exit code, timeout flag."""

    model_config = ConfigDict(extra="forbid")

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class RunCodeArguments(BaseModel):
    """Tool arguments for ``run_code``: a Python source string to execute in the sandbox.

    ``extra="ignore"`` is intentional — models periodically hallucinate
    extra keys onto the tool call (e.g., gpt-5.5 emits a ``timeout`` arg
    despite it not being in the schema). The engine governs per-call
    timeout via the module-level ``_TIMEOUT_SECONDS`` constant in
    ``engine.sandbox.sandbox``; no agent-supplied timeout is honored.
    With ``extra="forbid"`` those calls hard-fail at
    ``model_validate_json`` with ``extra_forbidden`` and the agent loses
    the tool result entirely (and the engine SIGTERMs the host run when
    the error propagates from the tool boundary); ignoring the unknown
    keys keeps the call moving while still validating the field we
    actually use.
    """

    model_config = ConfigDict(extra="ignore")

    code: str
