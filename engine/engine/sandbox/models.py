from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SandboxConfig(BaseModel):
    """Caller-tunable knobs for ``run_code``: timeout, output caps, optional Python override.

    ``python_executable`` overrides the interpreter used inside the sandbox.
    When ``None`` (the default), ``resolve_sandbox()`` falls back to
    ``sys.executable``. The chosen interpreter, its stdlib, and its
    site-packages are bound read-only into the sandbox via the runtime mount
    manifest computed at probe time.
    """

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=10.0, gt=0)
    maximum_stdout_bytes: int = Field(default=64_000, gt=0)
    maximum_stderr_bytes: int = Field(default=64_000, gt=0)
    python_executable: Path | None = None


class PythonRuntimeMounts(BaseModel):
    """The narrow set of host paths needed to execute Python inside the sandbox.

    Computed from the running interpreter (``sys`` / ``sysconfig`` / ``site``)
    plus ``/proc/self/maps`` on Linux, so the sandbox only sees the specific
    interpreter binary, stdlib, site-packages, and shared libraries that are
    already loaded by the host process. We deliberately avoid binding broad
    system roots like ``/usr`` or ``/lib``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    python_executable: Path
    runtime_paths: tuple[Path, ...]
    library_paths: tuple[Path, ...]


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
