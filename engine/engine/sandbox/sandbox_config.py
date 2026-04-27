from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SandboxConfig(BaseModel):
    """Caller-tunable knobs for ``run_code``: wall-clock timeout and stdout/stderr caps.

    ``python_executable`` overrides the Python interpreter used inside the
    sandbox. When ``None`` (the default) the runtime falls back to
    ``sys.executable``. The chosen interpreter, its stdlib, and its
    site-packages are bound read-only into the sandbox via the runtime mount
    manifest computed at probe time.
    """

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=10.0, gt=0)
    maximum_stdout_bytes: int = Field(default=64_000, gt=0)
    maximum_stderr_bytes: int = Field(default=64_000, gt=0)
    python_executable: Path | None = None


class SandboxPolicy(BaseModel):
    """Path lists the platform-specific command builders consume.

    Path order is positional and meaningful. ``readonly_paths`` is structured as:
    ``[trace_path, index_path, *runtime_paths]``. ``library_paths`` carries
    individual shared library files (Linux only) bound at their original
    locations so the dynamic loader can resolve them without binding broad
    system directories. ``network_enabled`` is pinned to False at the type
    level — sandboxed code never gets network access.
    """

    model_config = ConfigDict(extra="forbid")

    python_executable: Path
    readonly_paths: list[Path]
    library_paths: list[Path]
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
