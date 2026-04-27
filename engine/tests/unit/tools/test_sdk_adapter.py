from __future__ import annotations

from pydantic import BaseModel

from engine.tools.tool_protocol import ToolContext, to_sdk_function_tool


class _Args(BaseModel):
    value: str


class _Result(BaseModel):
    echoed: str


class _Echo:
    name = "echo"
    description = "Echo a value."
    arguments_model = _Args
    result_model = _Result

    async def run(self, tool_context: ToolContext, arguments: _Args) -> _Result:
        return _Result(echoed=arguments.value)


def test_adapter_produces_sdk_function_tool() -> None:
    from agents import FunctionTool

    sdk_tool = to_sdk_function_tool(
        _Echo(), context_factory=lambda ctx: ToolContext.model_construct()
    )
    assert isinstance(sdk_tool, FunctionTool)
    assert sdk_tool.name == "echo"
    assert "Echo" in (sdk_tool.description or "")
