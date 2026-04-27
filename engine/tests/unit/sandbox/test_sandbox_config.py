from __future__ import annotations

from pathlib import Path

from engine.sandbox.sandbox_config import (
    CodeExecutionResult,
    RunCodeArguments,
    SandboxConfig,
    SandboxPolicy,
)


def test_sandbox_config_defaults() -> None:
    cfg = SandboxConfig()
    assert cfg.timeout_seconds == 10.0
    assert cfg.maximum_stdout_bytes == 64_000
    assert cfg.maximum_stderr_bytes == 64_000
    assert cfg.python_executable is None


def test_sandbox_policy(tmp_path: Path) -> None:
    pol = SandboxPolicy(
        python_executable=tmp_path / "bin" / "python",
        readonly_paths=[tmp_path / "ro"],
        library_paths=[tmp_path / "libc.so.6"],
        writable_paths=[tmp_path / "w"],
        timeout_seconds=5.0,
    )
    assert pol.network_enabled is False
    assert pol.python_executable == tmp_path / "bin" / "python"
    assert pol.library_paths == [tmp_path / "libc.so.6"]


def test_result_shape() -> None:
    r = CodeExecutionResult(exit_code=0, stdout="ok", stderr="", timed_out=False)
    assert r.exit_code == 0


def test_run_code_arguments() -> None:
    args = RunCodeArguments(code="print(1)")
    assert args.code == "print(1)"
