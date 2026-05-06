from __future__ import annotations

from collections.abc import Callable

from openai import NOT_GIVEN, AsyncOpenAI

from engine.agents.agent_context import Compactor
from engine.agents.agent_context_items import AgentContextItem
from engine.agents.agent_execution import AgentExecution
from engine.agents.prompt_templates import COMPACTION_SYSTEM_PROMPT
from engine.engine_config import EngineConfig

CompactorFactory = Callable[[AgentExecution], Compactor]


def build_compactor_factory(
    engine_config: EngineConfig,
    client: AsyncOpenAI | None = None,
) -> CompactorFactory:
    """Returns a factory that produces a Compactor bound to an OpenAI-compatible client.

    The factory takes an AgentExecution (currently unused but reserved for
    future per-agent compaction policies) and returns a callable that the
    AgentContext can invoke per item it wants compacted. When ``client`` is
    not supplied, it is constructed from ``engine_config.model_provider`` so
    compaction routes through whichever OpenAI-compatible endpoint the run is
    configured for.
    """
    openai_client = client

    def factory(_execution: AgentExecution) -> Compactor:
        async def compact(item: AgentContextItem) -> str:
            nonlocal openai_client
            if openai_client is None:
                openai_client = AsyncOpenAI(
                    base_url=engine_config.model_provider.base_url,
                    api_key=engine_config.model_provider.api_key,
                )

            user_text = _item_as_prompt(item)
            # Frontier models (gpt-5.x, claude-opus-4-7+, …) reject
            # ``temperature`` as deprecated; only forward it when
            # explicitly set on the compaction model. ``NOT_GIVEN`` is
            # openai's typed sentinel for "omit from request" so the
            # SDK's signature stays honest.
            temperature = (
                engine_config.compaction_model.temperature
                if engine_config.compaction_model.temperature is not None
                else NOT_GIVEN
            )
            response = await openai_client.chat.completions.create(
                model=engine_config.compaction_model.name,
                messages=[
                    {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=temperature,
            )
            return (response.choices[0].message.content or "").strip()

        return compact

    return factory


def _item_as_prompt(item: AgentContextItem) -> str:
    if item.role == "user":
        return f"USER MESSAGE:\n{item.content}"
    if item.role == "assistant":
        if item.tool_calls:
            calls = "\n".join(
                f"- {tc.function.name}({tc.function.arguments})" for tc in item.tool_calls
            )
            return f"ASSISTANT TOOL CALLS:\n{calls}"
        return f"ASSISTANT MESSAGE:\n{item.content}"
    if item.role == "tool":
        return f"TOOL RESULT (tool={item.name}, call={item.tool_call_id}):\n{item.content}"
    return str(item.content or "")
