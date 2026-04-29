from __future__ import annotations

import pytest

from engine.errors import (
    EngineAgentExhaustedError,
    EngineError,
    EngineMaxDepthExceededError,
    EngineSandboxDeniedError,
    EngineToolError,
)


def test_hierarchy() -> None:
    for exc in (
        EngineAgentExhaustedError,
        EngineMaxDepthExceededError,
        EngineSandboxDeniedError,
        EngineToolError,
    ):
        assert issubclass(exc, EngineError)


def test_raise_and_message() -> None:
    with pytest.raises(EngineAgentExhaustedError) as ei:
        raise EngineAgentExhaustedError("10 consecutive llm failures")
    assert "10" in str(ei.value)
