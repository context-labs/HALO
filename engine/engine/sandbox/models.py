from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SandboxConfig(BaseModel):
    """Caller-tunable knobs for ``run_code``: timeout and output caps.

    The Deno+Pyodide backend resolves its own runtime (Deno binary, Pyodide
    wheels) at probe time, so there's no per-config interpreter override.
    """

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=10.0, gt=0)
    maximum_stdout_bytes: int = Field(default=64_000, gt=0)
    maximum_stderr_bytes: int = Field(default=64_000, gt=0)


class CodeExecutionResult(BaseModel):
    """Outcome of running code in the sandbox: capped stdout/stderr, exit code, timeout flag."""

    model_config = ConfigDict(extra="forbid")

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class RunCodeArguments(BaseModel):
    """Tool arguments for ``run_code``: a Python source string to execute in the sandbox."""

    model_config = ConfigDict(extra="forbid")

    code: str
