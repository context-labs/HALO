from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from engine.model_config import ModelConfig
from engine.model_provider_config import ModelProviderConfig
from engine.tools.synthesis_tool import SynthesisTool, SynthesizeTracesArguments
from engine.tools.tool_protocol import ToolContext
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


@pytest_asyncio.fixture
async def ctx(tmp_path: Path, fixtures_dir: Path) -> ToolContext:
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    store = TraceStore.load(trace_path=trace_path, index_path=index_path)
    return ToolContext.model_construct(trace_store=store)


@pytest.mark.asyncio
async def test_synthesis_tool_calls_client_and_returns_summary(ctx: ToolContext) -> None:
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))]
                    )
                )
            )
        )
    )
    tool = SynthesisTool(
        model=ModelConfig(name="claude-haiku-4-5"),
        model_provider=ModelProviderConfig(),
        client=fake_client,
    )

    result = await tool.run(
        ctx, SynthesizeTracesArguments(trace_ids=["t-aaaa", "t-bbbb"], focus="errors")
    )
    assert result.summary == "summary"
    fake_client.chat.completions.create.assert_awaited_once()


def _stub_client_returning(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
                    )
                )
            )
        )
    )


@pytest.mark.asyncio
async def test_synthesis_tool_forwards_explicit_reasoning_effort(
    ctx: ToolContext,
) -> None:
    fake_client = _stub_client_returning("ok")
    tool = SynthesisTool(
        model=ModelConfig(name="gpt-5", reasoning_effort="low"),
        model_provider=ModelProviderConfig(),
        client=fake_client,
    )

    await tool.run(ctx, SynthesizeTracesArguments(trace_ids=["t-aaaa"]))

    call_kwargs = fake_client.chat.completions.create.await_args.kwargs
    assert call_kwargs["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_synthesis_tool_defaults_to_model_max_reasoning(ctx: ToolContext) -> None:
    fake_client = _stub_client_returning("ok")
    tool = SynthesisTool(
        model=ModelConfig(name="gpt-5.5"),
        model_provider=ModelProviderConfig(),
        client=fake_client,
    )

    await tool.run(ctx, SynthesizeTracesArguments(trace_ids=["t-aaaa"]))

    call_kwargs = fake_client.chat.completions.create.await_args.kwargs
    assert call_kwargs["reasoning_effort"] == "xhigh"


@pytest.mark.asyncio
async def test_synthesis_tool_omits_reasoning_for_non_reasoning_model(
    ctx: ToolContext,
) -> None:
    fake_client = _stub_client_returning("ok")
    tool = SynthesisTool(
        model=ModelConfig(name="claude-opus-4-7"),
        model_provider=ModelProviderConfig(),
        client=fake_client,
    )

    await tool.run(ctx, SynthesizeTracesArguments(trace_ids=["t-aaaa"]))

    from openai import Omit

    call_kwargs = fake_client.chat.completions.create.await_args.kwargs
    assert isinstance(call_kwargs["reasoning_effort"], Omit)
