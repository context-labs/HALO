from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SandboxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=10.0, gt=0)
    maximum_stdout_bytes: int = Field(default=64_000, gt=0)
    maximum_stderr_bytes: int = Field(default=64_000, gt=0)
    python_executable: Path | None = None


class SandboxPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readonly_paths: list[Path]
    writable_paths: list[Path]
    network_enabled: Literal[False] = False
    timeout_seconds: float = Field(gt=0)


class CodeExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class RunCodeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
