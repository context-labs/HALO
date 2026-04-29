import json
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any

from agents.exceptions import MaxTurnsExceeded
from agents.mcp import ToolFilterContext
from agents.mcp.server import MCPServerStdio, MCPServerStreamableHttp
from agents.result import RunResultStreaming
from rich import print

from appworld.apps import get_all_apps
from appworld.common.constants import DEFAULT_REMOTE_APIS_URL
from appworld.serve._mcp import build_mcp_config
from appworld_agents.code.common.logger import Logger


class AgentsMCP:
    def __init__(
        self,
        remote_apis_url: str = DEFAULT_REMOTE_APIS_URL,
        remote_mcp_url: str = "stdio",
        quiet: bool = True,
    ):
        self.server: Any = None
        self.remote_apis_url = remote_apis_url
        self.remote_mcp_url = remote_mcp_url
        if remote_mcp_url == "stdio":
            self.transport = "stdio"
        elif remote_mcp_url.startswith("http"):
            self.transport = "http"
        else:
            raise ValueError(
                f"Invalid remote_mcp_url: {remote_mcp_url}. It should be a valid URL or 'stdio'."
            )
        self.server_class = {"stdio": MCPServerStdio, "http": MCPServerStreamableHttp}[
            self.transport
        ]
        self._exit_stack = AsyncExitStack()
        self._app_names = get_all_apps(skip_admin=True, skip_api_docs=True)
        self._allowed_tools: set[str] | None = None
        self._quiet = quiet

    def set_allowed_tools(self, allowed_tools: set[str]) -> None:
        self._allowed_tools = allowed_tools

    async def connect(self) -> None:
        if not self._quiet:
            print("Connecting AppWorld MCP...")

        def tool_filter(context: ToolFilterContext, tool: Any) -> bool:
            allowed_tools = self._allowed_tools
            if allowed_tools is None:
                return True
            return tool.name in allowed_tools

        server = self.server_class(
            name="AppWorld MCP",
            params=build_mcp_config(
                transport=self.transport,
                app_names=self._app_names,
                remote_apis_url=self.remote_apis_url,
                remote_mcp_url=self.remote_mcp_url,
            ),
            tool_filter=tool_filter,
            cache_tools_list=False,
            use_structured_content=True,
        )
        self.server = server
        await self._exit_stack.enter_async_context(server)

    async def cleanup(self) -> None:
        if not self._quiet:
            print("Disconnecting AppWorld MCP...")
        await self._exit_stack.aclose()

    async def __aenter__(self) -> "AgentsMCP":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.cleanup()

    @classmethod
    async def stream(cls, streamer: RunResultStreaming, logger: Logger) -> None:
        try:
            async for event in streamer.stream_events():
                if event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        raw_data = event.item.raw_item.model_dump()
                        function_name = raw_data["name"]
                        arguments_str = raw_data["arguments"]
                        arguments_str = arguments_str.strip() or "{}"
                        arguments = json.loads(arguments_str)
                        message = f"{function_name}({', '.join(f'{k}={json.dumps(v)}' for k, v in arguments.items())})"
                        logger.show_message(role="agent", content=message)
                    if event.item.type == "tool_call_output_item":
                        message_ = json.loads(event.item.raw_item["output"])
                        message = json.dumps(message_, indent=4)
                        logger.show_message(role="environment", content=message)
                    if event.item.type == "message_output_item":
                        message = event.item.raw_item.content[0].text
                        logger.show_message(role="agent", content=message)
        except MaxTurnsExceeded:
            pass
