from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SandboxConfig(BaseModel):
    """Caller-tunable knobs for ``run_code``: wall-clock timeout and stdout/stderr caps."""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=10.0, gt=0)
    maximum_stdout_bytes: int = Field(default=64_000, gt=0)
    maximum_stderr_bytes: int = Field(default=64_000, gt=0)
    python_executable: Path | None = None


class SandboxPolicy(BaseModel):
    """Read-only/writable path list passed to the platform-specific sandbox command builder.

    ``network_enabled`` is pinned to False at the type level — sandboxed code never
    gets network access. Path order is positional and consumed by the command
    builders (e.g. trace, index, venv).
    """

    model_config = ConfigDict(extra="forbid")

    readonly_paths: list[Path]
    writable_paths: list[Path]
    network_enabled: Literal[False] = False
    timeout_seconds: float = Field(gt=0)


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
