from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


PART_OF_LITERAL = Literal["all", "p1", "p2", "p3", "p4"]  # priorities


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=False, validate_assignment=True
    )


class AgentInfo(StrictBaseModel):
    agent_name: str
    agent_humanized_name: str
    part_of: list[PART_OF_LITERAL]
    function_calling: bool


_AGENT_INFOS: list[dict[str, Any]] = [
    {
        "agent_name": "simplified_function_calling_agent",
        "agent_humanized_name": "Function Calling Agent",
        "part_of": ["p1"],
        "function_calling": True,
    },
    {
        "agent_name": "simplified_react_code_agent",
        "agent_humanized_name": "ReAct Code Agent",
        "part_of": ["p1"],
        "function_calling": False,
    },
    {
        "agent_name": "simplified_full_code_agent",
        "agent_humanized_name": "Full Code Agent",
        "part_of": ["p2"],
        "function_calling": False,
    },
    {
        "agent_name": "smolagents_code_agent",
        "agent_humanized_name": "Smolagents CodeAgent",
        "part_of": ["p3"],
        "function_calling": False,
    },
    {
        "agent_name": "smolagents_tool_calling_agent",
        "agent_humanized_name": "Smolagents ToolCallingAgent",
        "part_of": ["p3"],
        "function_calling": True,
    },
    {
        "agent_name": "openai_agents_mcp_agent",
        "agent_humanized_name": "OpenAI Agents MCP",
        "part_of": ["p4"],
        "function_calling": True,
    },
]

agent_names = [agent_info["agent_name"] for agent_info in _AGENT_INFOS]
if len(agent_names) != len(set(agent_names)):
    raise ValueError("Duplicate agent names found in AGENT_INFOS")

AGENT_INFOS: tuple[AgentInfo, ...] = ()
for agent_info_ in _AGENT_INFOS:
    if "all" not in agent_info_["part_of"]:
        agent_info_["part_of"].insert(0, "all")
    agent_info = AgentInfo(**agent_info_)
    AGENT_INFOS += (agent_info,)

AGENT_NAME_TO_INFO: dict[str, AgentInfo] = {
    agent_info.agent_name: agent_info for agent_info in AGENT_INFOS
}

__all__ = ["AGENT_INFOS", "AGENT_NAME_TO_INFO"]
