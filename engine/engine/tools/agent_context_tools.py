from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from engine.agents.agent_context_items import AgentContextItem
from engine.tools.tool_protocol import ToolContext


class GetContextItemArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str


class GetContextItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: AgentContextItem


class GetContextItemTool:
    name = "get_context_item"
    description = "Return the full stored context item by id, including original content and compaction summary."
    arguments_model = GetContextItemArguments
    result_model = GetContextItemResult

    async def run(self, tool_context: ToolContext, arguments: GetContextItemArguments) -> GetContextItemResult:
        agent_context = tool_context.require_agent_context()
        item = agent_context.get_item(arguments.item_id)
        return GetContextItemResult(item=item)
