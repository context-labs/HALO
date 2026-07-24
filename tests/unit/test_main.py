from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI

import engine.agents.agent_context as agent_context_module
import engine.main as engine_main
from engine.agents.agent_config import AgentConfig
from engine.agents.agent_context_items import AgentContextItem
from engine.engine_config import EngineConfig
from engine.main import _drive_sync
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage
from engine.traces.models.trace_dataset_source import TraceDatasetSource
from tests._sdk_events import assistant_message_event
from tests.probes.probe_kit import FakeRunner, make_final_answer


async def _noop_compact(
    *,
    client: AsyncOpenAI,
    compaction_model: ModelConfig,
    item: AgentContextItem,
) -> str:
    del client, compaction_model, item
    return ""


def test_public_entrypoints_exist_and_are_async() -> None:
    assert inspect.isasyncgenfunction(engine_main.stream_engine_async)
    assert inspect.iscoroutinefunction(engine_main.run_engine_async)
    assert callable(engine_main.stream_engine)
    assert callable(engine_main.run_engine)


def test_async_signatures_match() -> None:
    for fn in (engine_main.stream_engine_async, engine_main.run_engine_async):
        params = list(inspect.signature(fn).parameters)
        assert params[:3] == ["messages", "engine_config", "trace_paths"]


def test_drive_sync_runs_finally_on_early_break() -> None:
    """Regression: early break must trigger the async generator's finally
    block so background tasks / telemetry handles get cleaned up."""
    cleaned_up: list[bool] = []

    async def _producer():
        try:
            for i in range(10):
                yield i
        finally:
            cleaned_up.append(True)

    seen: list[int] = []
    for value in _drive_sync(_producer()):
        seen.append(value)
        if value == 2:
            break

    assert seen == [0, 1, 2]
    assert cleaned_up == [True]


def test_drive_sync_runs_finally_on_consumer_exception() -> None:
    """Regression: an exception raised by the consumer must propagate
    through the sync generator AND trigger the async producer's finally."""
    cleaned_up: list[bool] = []

    async def _producer():
        try:
            for i in range(10):
                yield i
        finally:
            cleaned_up.append(True)

    class Boom(Exception):
        pass

    try:
        for value in _drive_sync(_producer()):
            if value == 1:
                raise Boom
    except Boom:
        pass

    assert cleaned_up == [True]


def test_drive_sync_runs_finally_on_full_consumption() -> None:
    cleaned_up: list[bool] = []

    async def _producer():
        try:
            yield 1
            yield 2
        finally:
            cleaned_up.append(True)

    assert list(_drive_sync(_producer())) == [1, 2]
    assert cleaned_up == [True]


def _assistant_text(text: str):
    return assistant_message_event(item_id="msg-1", text=text)


def _config() -> EngineConfig:
    agent = AgentConfig(
        name="root",
        model=ModelConfig(name="gpt-5.4-mini"),
        maximum_turns=4,
    )
    return EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub"}),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        text_message_compaction_keep_last_messages=0,
        tool_call_compaction_keep_last_turns=0,
        maximum_depth=0,
        maximum_parallel_subagents=1,
    )


@pytest.mark.asyncio
async def test_engine_wires_configured_client_via_run_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixtures_dir: Path,
) -> None:
    """Regression for the wired_tools-vs-stream_engine_async asymmetry.

    The engine must pass its ``AsyncOpenAI`` to the SDK via
    ``RunConfig.model_provider`` for every ``Runner.run_streamed`` call.
    Process-global ``set_default_openai_client`` is fragile: subagent
    tool factories invoked outside ``stream_engine_async`` (e.g. via
    ``tests/integration/tool_isolation_kit.wired_tools``) never see
    the default, and the SDK silently falls back to an ``AsyncOpenAI``
    built from env vars — losing ``default_headers`` and the
    deterministic close path. Wire it per-call instead.
    """
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())

    class _StubAsyncOpenAI:
        def __init__(
            self,
            *,
            base_url: str | None = None,
            api_key: str | None = None,
            default_headers: dict[str, str] | None = None,
        ) -> None:
            del base_url, api_key, default_headers
            self.close = AsyncMock()

    stub_client_instance: _StubAsyncOpenAI | None = None

    def _capture_client(
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> _StubAsyncOpenAI:
        nonlocal stub_client_instance
        stub_client_instance = _StubAsyncOpenAI(
            base_url=base_url, api_key=api_key, default_headers=default_headers
        )
        return stub_client_instance

    monkeypatch.setattr(engine_main, "AsyncOpenAI", _capture_client)
    monkeypatch.setattr(agent_context_module, "compact", _noop_compact)

    runner = FakeRunner([*make_final_answer("Final.")])
    monkeypatch.setattr("agents.Runner.run_streamed", runner.run_streamed)

    await engine_main.run_engine_async(
        [AgentMessage(role="user", content="hi")], _config(), trace_path
    )

    assert stub_client_instance is not None
    assert len(runner.calls) >= 1
    run_config = runner.calls[0]["run_config"]
    model_provider = run_config.model_provider
    assert isinstance(model_provider, engine_main.OpenAIProvider)
    # ``OpenAIProvider`` stores the passed client on ``_client`` and short-
    # circuits lazy construction; verify the engine's client is what the
    # SDK will use rather than an env-var-built fallback.
    assert model_provider._get_client() is stub_client_instance


@pytest.mark.asyncio
async def test_resolve_trace_sources_multi_file_derives_per_file_indexes(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """Each file in a multi-file dataset gets its own derived sidecar index."""
    from engine.main import _resolve_trace_sources
    from engine.traces.models.trace_index_config import TraceIndexConfig

    first = tmp_path / "conversations.jsonl"
    first.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    second = tmp_path / "evals.jsonl"
    second.write_bytes((fixtures_dir / "tiny_traces_second_file.jsonl").read_bytes())

    sources = await _resolve_trace_sources([first, second], config=TraceIndexConfig())

    assert sources == [
        TraceDatasetSource(trace_path=first, index_path=Path(str(first) + ".engine-index.jsonl")),
        TraceDatasetSource(trace_path=second, index_path=Path(str(second) + ".engine-index.jsonl")),
    ]


@pytest.mark.asyncio
async def test_resolve_trace_sources_index_dir_holds_every_file_index(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """A single ``index_dir`` serves a multi-file dataset — one index file per
    trace, named after it, all inside that directory."""
    from engine.main import _resolve_trace_sources
    from engine.traces.models.trace_index_config import TraceIndexConfig
    from engine.traces.trace_index_builder import sidecar_index_path

    first = tmp_path / "conversations.jsonl"
    first.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    second = tmp_path / "evals.jsonl"
    second.write_bytes((fixtures_dir / "tiny_traces_second_file.jsonl").read_bytes())
    index_dir = tmp_path / "indexes"

    sources = await _resolve_trace_sources(
        [first, second], config=TraceIndexConfig(index_dir=index_dir)
    )

    first_index = sidecar_index_path(first, index_dir)
    second_index = sidecar_index_path(second, index_dir)
    assert sources == [
        TraceDatasetSource(trace_path=first, index_path=first_index),
        TraceDatasetSource(trace_path=second, index_path=second_index),
    ]
    # Both indexes are distinct real files in the one directory.
    assert first_index != second_index
    assert first_index.parent == index_dir and second_index.parent == index_dir
    assert first_index.is_file() and second_index.is_file()


@pytest.mark.asyncio
async def test_resolve_trace_sources_requires_at_least_one_path() -> None:
    """An empty dataset is a caller bug — fail fast rather than load nothing."""
    from engine.main import _resolve_trace_sources
    from engine.traces.models.trace_index_config import TraceIndexConfig

    with pytest.raises(ValueError, match="at least one trace path"):
        await _resolve_trace_sources([], config=TraceIndexConfig())


def test_as_trace_path_list_wraps_a_bare_path() -> None:
    """The public-edge single-path convenience normalizes to a one-item list."""
    from engine.main import _as_trace_path_list

    one = Path("/data/traces.jsonl")
    assert _as_trace_path_list(one) == [one]


def test_as_trace_path_list_passes_a_sequence_through() -> None:
    """A list (or tuple) of paths comes back as a list unchanged."""
    from engine.main import _as_trace_path_list

    paths = [Path("/data/a.jsonl"), Path("/data/b.jsonl")]
    assert _as_trace_path_list(paths) == paths
    assert _as_trace_path_list(tuple(paths)) == paths
