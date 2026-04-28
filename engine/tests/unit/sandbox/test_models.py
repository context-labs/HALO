from __future__ import annotations

from engine.sandbox.models import (
    CodeExecutionResult,
    RunCodeArguments,
    SandboxConfig,
)


def test_sandbox_config_defaults() -> None:
    cfg = SandboxConfig()
    assert cfg.timeout_seconds == 10.0
    assert cfg.maximum_stdout_bytes == 64_000
    assert cfg.maximum_stderr_bytes == 64_000


def test_run_code_arguments() -> None:
    args = RunCodeArguments(code="print(1)")
    assert args.code == "print(1)"


def test_code_execution_result_shape() -> None:
    result = CodeExecutionResult(exit_code=0, stdout="ok", stderr="", timed_out=False)
    assert result.exit_code == 0
    assert result.timed_out is False
