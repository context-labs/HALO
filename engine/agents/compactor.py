from __future__ import annotations

from openai import AsyncOpenAI, omit

from engine.agents.agent_context_items import AgentContextItem
from engine.agents.llm_retry import call_with_retries
from engine.agents.prompt_caching import as_cached_system_message
from engine.agents.prompt_templates import COMPACTION_SYSTEM_PROMPT
from engine.model_config import ModelConfig


async def compact(
    *,
    client: AsyncOpenAI,
    compaction_model: ModelConfig,
    item: AgentContextItem,
) -> str:
    """Summarize one ``AgentContextItem`` via ``client`` using ``compaction_model``.

    The system message is built with :func:`as_cached_system_message` so the
    byte-stable :data:`COMPACTION_SYSTEM_PROMPT` prefix is cacheable on
    Anthropic (read-back surfaces as ``cache_read_input_tokens``) without
    affecting OpenAI's automatic prefix cache. The per-item dynamic text
    goes in the trailing ``user`` message to keep the prefix byte-stable
    across calls.
    """
    user_text = _item_as_prompt(item)
    # Frontier models (gpt-5.x, claude-opus-4-7+, …) reject ``temperature``
    # as deprecated; only forward it when explicitly set on the compaction
    # model.
    temperature = compaction_model.temperature if compaction_model.temperature is not None else omit
    response = await call_with_retries(
        lambda: client.chat.completions.create(
            model=compaction_model.name,
            messages=[
                as_cached_system_message(COMPACTION_SYSTEM_PROMPT),
                {"role": "user", "content": user_text},
            ],
            temperature=temperature,
        ),
        description=f"compaction call (model={compaction_model.name}, item={item.item_id})",
    )
    return (response.choices[0].message.content or "").strip()


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
