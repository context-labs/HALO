from __future__ import annotations

from engine.sandbox.models import CodeExecutionResult, RunCodeArguments


def test_run_code_arguments() -> None:
    args = RunCodeArguments(code="print(1)")
    assert args.code == "print(1)"


def test_run_code_arguments_ignores_unknown_keys() -> None:
    """Unknown keys on the tool-call payload must be dropped, not rejected.

    Regression: gpt-5.5 (and other model versions) occasionally hallucinate
    a ``timeout`` argument that is not in the tool schema. With
    ``extra="forbid"`` Pydantic raised ``ValidationError`` at the tool
    boundary, the engine surfaced that as a tool failure, and Modal
    SIGTERM'd the host run. ``extra="ignore"`` keeps the call moving while
    the engine continues to govern the actual per-call timeout via the
    module-level ``_TIMEOUT_SECONDS`` constant.
    """
    # Both ``__init__`` and ``model_validate_json`` (the SDK adapter's
    # entry point) must tolerate extras; check both paths.
    args = RunCodeArguments(code="print(1)", timeout=30)  # type: ignore[call-arg]
    assert args.code == "print(1)"
    assert not hasattr(args, "timeout")

    parsed = RunCodeArguments.model_validate_json(
        '{"code": "print(2)", "timeout": 45, "other_extra": "x"}'
    )
    assert parsed.code == "print(2)"
    assert not hasattr(parsed, "timeout")


def test_code_execution_result_shape() -> None:
    result = CodeExecutionResult(exit_code=0, stdout="ok", stderr="", timed_out=False)
    assert result.exit_code == 0
    assert result.timed_out is False
