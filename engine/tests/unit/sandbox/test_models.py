from __future__ import annotations

from pathlib import Path

import pydantic
import pytest

from engine.sandbox.models import (
    CodeExecutionResult,
    PythonRuntimeMounts,
    RunCodeArguments,
    SandboxConfig,
)


def test_sandbox_config_defaults() -> None:
    cfg = SandboxConfig()
    assert cfg.timeout_seconds == 10.0
    assert cfg.maximum_stdout_bytes == 64_000
    assert cfg.maximum_stderr_bytes == 64_000
    assert cfg.python_executable is None


def test_sandbox_config_accepts_python_executable_override(tmp_path: Path) -> None:
    cfg = SandboxConfig(python_executable=tmp_path / "bin" / "python")
    assert cfg.python_executable == tmp_path / "bin" / "python"


def test_python_runtime_mounts_is_frozen(tmp_path: Path) -> None:
    mounts = PythonRuntimeMounts(
        python_executable=tmp_path / "python",
        runtime_paths=(tmp_path,),
        library_paths=(),
    )
    with pytest.raises(pydantic.ValidationError):
        mounts.python_executable = tmp_path / "other"  # type: ignore[misc]


def test_run_code_arguments() -> None:
    args = RunCodeArguments(code="print(1)")
    assert args.code == "print(1)"


def test_code_execution_result_shape() -> None:
    result = CodeExecutionResult(exit_code=0, stdout="ok", stderr="", timed_out=False)
    assert result.exit_code == 0
    assert result.timed_out is False
