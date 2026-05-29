from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel

from engine.tools.tool_protocol import to_sdk_function_tool


class _Args(BaseModel):
    value: str


class _Result(BaseModel):
    echoed: str


class _Echo:
    name = "echo"
    description = "Echo a value."
    arguments_model = _Args
    result_model = _Result

    async def run(self, tool_context, arguments: _Args) -> _Result:
        return _Result(echoed=arguments.value)


def test_adapter_produces_sdk_function_tool() -> None:
    from agents import FunctionTool

    sdk_tool = to_sdk_function_tool(
        _Echo(),
        run_state=MagicMock(),
        parent_context=MagicMock(),
    )
    assert isinstance(sdk_tool, FunctionTool)
    assert sdk_tool.name == "echo"
    assert "Echo" in (sdk_tool.description or "")
