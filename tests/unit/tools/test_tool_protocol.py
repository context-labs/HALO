from __future__ import annotations

from pydantic import BaseModel

from engine.tools.tool_protocol import EngineTool, ToolContext


class _EchoArgs(BaseModel):
    value: str


class _EchoResult(BaseModel):
    value: str


class _EchoTool:
    name = "echo"
    description = "Echo."
    arguments_model = _EchoArgs
    result_model = _EchoResult

    async def run(self, tool_context: ToolContext, arguments: _EchoArgs) -> _EchoResult:
        return _EchoResult(value=arguments.value)


async def test_engine_tool_runtime_conforms() -> None:
    tool: EngineTool = _EchoTool()
    ctx = ToolContext.model_construct()
    result = await tool.run(ctx, _EchoArgs(value="x"))
    assert isinstance(result, _EchoResult)
    assert result.value == "x"
