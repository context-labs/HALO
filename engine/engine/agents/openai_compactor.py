from __future__ import annotations

from collections.abc import Callable

from openai import AsyncOpenAI

from engine.agents.agent_context import Compactor
from engine.agents.agent_context_items import AgentContextItem
from engine.agents.agent_execution import AgentExecution
from engine.agents.prompt_templates import COMPACTION_SYSTEM_PROMPT
from engine.engine_config import EngineConfig

CompactorFactory = Callable[[AgentExecution], Compactor]


def build_openai_compactor_factory(
    engine_config: EngineConfig,
    client: AsyncOpenAI | None = None,
) -> CompactorFactory:
    """Returns a factory that produces a Compactor bound to an OpenAI client.

    The factory takes an AgentExecution (currently unused but reserved for
    future per-agent compaction policies) and returns a callable that the
    AgentContext can invoke per item it wants compacted.
    """
    openai_client = client

    def factory(_execution: AgentExecution) -> Compactor:
        async def compact(item: AgentContextItem) -> str:
            nonlocal openai_client
            if openai_client is None:
                openai_client = AsyncOpenAI()

            user_text = _item_as_prompt(item)
            response = await openai_client.chat.completions.create(
                model=engine_config.compaction_model.name,
                messages=[
                    {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=engine_config.compaction_model.temperature or 0.0,
            )
            return (response.choices[0].message.content or "").strip()

        return compact

    return factory


def _item_as_prompt(item: AgentContextItem) -> str:
    if item.role == "user":
        return f"USER MESSAGE:\n{item.content}"
    if item.role == "assistant":
        if item.tool_calls:
            calls = "\n".join(f"- {tc.function.name}({tc.function.arguments})" for tc in item.tool_calls)
            return f"ASSISTANT TOOL CALLS:\n{calls}"
        return f"ASSISTANT MESSAGE:\n{item.content}"
    if item.role == "tool":
        return f"TOOL RESULT (tool={item.name}, call={item.tool_call_id}):\n{item.content}"
    return str(item.content or "")
