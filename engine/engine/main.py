from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from engine.agents.agent_context import AgentContext
from engine.agents.agent_context_items import AgentContextItem
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.agents.openai_agent_runner import OpenAiAgentRunner
from engine.agents.prompt_templates import render_root_system_prompt
from engine.engine_config import EngineConfig
from engine.models.engine_output import AgentOutputItem, EngineStreamEvent
from engine.models.messages import AgentMessage
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


async def stream_engine_async(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
) -> AsyncIterator[EngineStreamEvent]:
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=engine_config.trace_index,
    )
    trace_store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    output_bus = EngineOutputBus()
    run_state = EngineRunState(
        trace_store=trace_store,
        output_bus=output_bus,
        config=engine_config,
    )

    root_execution = AgentExecution(
        agent_id=f"root-{uuid.uuid4().hex[:8]}",
        agent_name=engine_config.root_agent.name,
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    run_state.register(root_execution)

    root_context = _build_root_context(messages=messages, engine_config=engine_config)

    from engine.tools.subagent_tool_factory import build_root_sdk_agent

    sdk_agent = build_root_sdk_agent(
        engine_config=engine_config,
        run_state=run_state,
        agent_execution=root_execution,
    )

    async def _drive() -> None:
        from agents import Runner

        async def _run_streamed(*, agent, input, context):
            return Runner.run_streamed(starting_agent=agent, input=input, context=context)

        runner = OpenAiAgentRunner(
            run_streamed=_run_streamed,
            compactor_factory=_build_compactor_factory(engine_config),
        )
        await runner.run(
            sdk_agent=sdk_agent,
            agent_context=root_context,
            agent_execution=root_execution,
            output_bus=output_bus,
            is_root=True,
            run_context=run_state,
        )
        await output_bus.close()

    task = asyncio.create_task(_drive())

    try:
        async for event in output_bus.stream():
            yield event
        await task
    except BaseException:
        task.cancel()
        raise


async def run_engine_async(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
) -> list[AgentOutputItem]:
    out: list[AgentOutputItem] = []
    async for event in stream_engine_async(messages, engine_config, trace_path):
        if isinstance(event, AgentOutputItem):
            out.append(event)
    return out


def stream_engine(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
) -> list[EngineStreamEvent]:
    async def _collect() -> list[EngineStreamEvent]:
        out: list[EngineStreamEvent] = []
        async for ev in stream_engine_async(messages, engine_config, trace_path):
            out.append(ev)
        return out
    return asyncio.run(_collect())


def run_engine(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
) -> list[AgentOutputItem]:
    return asyncio.run(run_engine_async(messages, engine_config, trace_path))


def _build_root_context(
    *,
    messages: list[AgentMessage],
    engine_config: EngineConfig,
) -> AgentContext:
    first_user = next(
        (m for m in messages if m.role == "user" and isinstance(m.content, str)),
        None,
    )
    user_instructions = first_user.content if first_user else engine_config.root_agent.instructions

    system_prompt = render_root_system_prompt(
        user_instructions=user_instructions,
        maximum_depth=engine_config.maximum_depth,
        maximum_parallel_subagents=engine_config.maximum_parallel_subagents,
    )
    items: list[AgentContextItem] = [
        AgentContextItem(item_id="sys-0", role="system", content=system_prompt)
    ]
    for i, msg in enumerate(messages):
        items.append(AgentContextItem(
            item_id=f"in-{i}",
            role=msg.role,
            content=msg.content,
            tool_calls=msg.tool_calls,
            tool_call_id=msg.tool_call_id,
            name=msg.name,
        ))

    return AgentContext(
        items=items,
        compaction_model=engine_config.compaction_model,
        text_message_compaction_keep_last_messages=engine_config.text_message_compaction_keep_last_messages,
        tool_call_compaction_keep_last_messages=engine_config.tool_call_compaction_keep_last_messages,
    )


def _build_compactor_factory(engine_config: EngineConfig):
    from engine.agents.agent_context import Compactor
    from engine.agents.agent_context_items import AgentContextItem as _CI
    from engine.agents.prompt_templates import COMPACTION_SYSTEM_PROMPT
    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    def factory(_execution) -> Compactor:
        async def compact(item: _CI) -> str:
            user_text = _item_as_prompt(item)
            response = await client.chat.completions.create(
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
            calls = "\n".join(
                f"- {tc.function.name}({tc.function.arguments})"
                for tc in item.tool_calls
            )
            return f"ASSISTANT TOOL CALLS:\n{calls}"
        return f"ASSISTANT MESSAGE:\n{item.content}"
    if item.role == "tool":
        return f"TOOL RESULT (tool={item.name}, call={item.tool_call_id}):\n{item.content}"
    return str(item.content or "")
