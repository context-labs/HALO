from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from engine.agents.agent_context_items import AgentContextItem
from engine.tools.tool_protocol import ToolContext


class GetContextItemArguments(BaseModel):
    """Arguments for ``get_context_item``: the ``item_id`` of a stored AgentContextItem."""

    model_config = ConfigDict(extra="forbid")

    item_id: str


class GetContextItemResult(BaseModel):
    """Result for ``get_context_item``: the full stored item, including any compaction summary."""

    model_config = ConfigDict(extra="forbid")

    item: AgentContextItem


class GetContextItemTool:
    """Lets an agent fetch the full stored form of one of its own context items by id.

    Useful after compaction — the rendered message only contains a summary, but the
    original ``content``, ``tool_calls``, and lineage fields remain on the item.
    """

    name = "get_context_item"
    description = "Return the full stored context item by id, including original content and compaction summary."
    arguments_model = GetContextItemArguments
    result_model = GetContextItemResult

    async def run(
        self, tool_context: ToolContext, arguments: GetContextItemArguments
    ) -> GetContextItemResult:
        """Look up the item in the calling agent's AgentContext and return it verbatim."""
        agent_context = tool_context.require_agent_context()
        item = agent_context.get_item(arguments.item_id)
        return GetContextItemResult(item=item)
