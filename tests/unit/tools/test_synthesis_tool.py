from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

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
        model_name="claude-haiku-4-5",
        model_provider=ModelProviderConfig(),
        client=fake_client,
    )

    result = await tool.run(
        ctx, SynthesizeTracesArguments(trace_ids=["t-aaaa", "t-bbbb"], focus="errors")
    )
    assert result.summary == "summary"
    fake_client.chat.completions.create.assert_awaited_once()
