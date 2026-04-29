from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.main import run_engine_async, stream_engine_async
from engine.model_config import ModelConfig
from engine.models.engine_output import AgentOutputItem, AgentTextDelta
from engine.models.messages import AgentMessage
from engine.sandbox.sandbox import Sandbox

E2E_MODEL = os.environ.get("HALO_E2E_MODEL", "gpt-5.4-mini")
E2E_TIMEOUT_SECONDS = float(os.environ.get("HALO_E2E_TIMEOUT", "60"))


def _engine_config(*, maximum_depth: int = 0, maximum_turns: int = 6) -> EngineConfig:
    """Live-model EngineConfig with the test's depth/turn knobs.

    All three model slots (root, synthesis, compaction) point at the same
    cheap model so a single API key + provider is enough for e2e.
    """
    agent = AgentConfig(
        name="root",
        instructions="Answer briefly.",
        model=ModelConfig(name=E2E_MODEL),
        maximum_turns=maximum_turns,
    )
    return EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub", "maximum_turns": 4}),
        synthesis_model=ModelConfig(name=E2E_MODEL),
        compaction_model=ModelConfig(name=E2E_MODEL),
        maximum_depth=maximum_depth,
        maximum_parallel_subagents=2,
    )


def _trace_path(tmp_path: Path, fixtures_dir: Path) -> Path:
    """Copy ``tiny_traces.jsonl`` into ``tmp_path`` so the engine can build the index next to it."""
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    return trace_path


def _tool_call_names(items: list[AgentOutputItem]) -> list[str]:
    """Flatten every ``tool_calls`` entry across the output stream into a name list."""
    return [tc.function.name for item in items for tc in (item.item.tool_calls or [])]


@pytest.mark.asyncio
async def test_engine_runs_on_tiny_fixture(tmp_path: Path, fixtures_dir: Path) -> None:
    """Smoke test: engine boots, root agent calls one tool, emits a final reply.

    Cheapest possible e2e — proves the OpenAI provider, tool resolution,
    and final-marker logic work end-to-end. Deeper capability coverage
    (subagents, streaming, run_code) lives in the targeted tests below.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; E2E requires real LLM access")

    trace_path = _trace_path(tmp_path, fixtures_dir)
    cfg = _engine_config(maximum_depth=0, maximum_turns=6)

    messages = [
        AgentMessage(
            role="user",
            content=(
                "Use get_dataset_overview to tell me how many traces are in the dataset. "
                "Then end your reply with a line containing only <final/>."
            ),
        )
    ]

    async with asyncio.timeout(E2E_TIMEOUT_SECONDS):
        results = await run_engine_async(messages, cfg, trace_path)

    assert len(results) >= 1
    assert any(item.final for item in results), "no AgentOutputItem with final=True emitted"
    assert "get_dataset_overview" in _tool_call_names(results), (
        "expected root agent to call get_dataset_overview"
    )


@pytest.mark.asyncio
async def test_engine_streams_subagent_chain(tmp_path: Path, fixtures_dir: Path) -> None:
    """End-to-end: prompt the root agent to delegate to a subagent and stream.

    Exercises three capabilities the smoke test skips:

    1. ``call_subagent`` — only registered when ``maximum_depth >= 1``.
       The prompt explicitly asks for delegation so the model picks it.
    2. Sub-agent execution path — once delegated, a depth-1 ``AgentExecution``
       runs its own tool loop (``view_trace`` here) and streams items back
       interleaved with the root.
    3. Streaming via ``stream_engine_async`` — the smoke test uses
       ``run_engine_async`` which discards ``AgentTextDelta`` events. The
       CLI relies on the streaming path; without an e2e here it could
       silently break.

    Asserts loose enough to allow alternate-but-correct tool choices: we
    just need at least one ``call_subagent``, at least one depth=1 item,
    at least one streaming text delta, and a final reply at depth=0.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; E2E requires real LLM access")

    trace_path = _trace_path(tmp_path, fixtures_dir)
    cfg = _engine_config(maximum_depth=1, maximum_turns=6)

    messages = [
        AgentMessage(
            role="user",
            content=(
                "There is a trace with id 't-bbbb' that errored. "
                "You MUST use the call_subagent tool to delegate the investigation: "
                "ask a subagent to look up trace t-bbbb (using view_trace) and report what went wrong. "
                "Wait for the subagent's reply, then summarize it in your own words. "
                "End your final reply with a line containing only <final/>."
            ),
        )
    ]

    items: list[AgentOutputItem] = []
    deltas: list[AgentTextDelta] = []

    async with asyncio.timeout(E2E_TIMEOUT_SECONDS * 2):
        async for event in stream_engine_async(messages, cfg, trace_path):
            if isinstance(event, AgentOutputItem):
                items.append(event)
            elif isinstance(event, AgentTextDelta):
                deltas.append(event)

    assert deltas, "stream_engine_async emitted no AgentTextDelta events"

    tool_calls = _tool_call_names(items)
    assert "call_subagent" in tool_calls, (
        f"expected root agent to call call_subagent; got tool calls: {tool_calls}"
    )

    depth_1_items = [item for item in items if item.depth == 1]
    assert depth_1_items, (
        "expected at least one AgentOutputItem at depth=1 (the subagent's run); "
        f"saw depths: {sorted({item.depth for item in items})}"
    )

    final_items = [item for item in items if item.final]
    assert final_items, "no AgentOutputItem with final=True emitted"
    assert all(item.depth == 0 for item in final_items), (
        f"final marker must be on the root (depth=0); got depths {[i.depth for i in final_items]}"
    )


@pytest.mark.asyncio
async def test_engine_run_code_executes_in_sandbox(tmp_path: Path, fixtures_dir: Path) -> None:
    """End-to-end: prompt forces ``run_code``; sandbox returns ``exit_code=0``.

    This is the only e2e that proves the WASM sandbox actually runs under
    a live LLM. The integration suite covers the SDK adapter and the
    sandbox itself, but neither exercises the full
    LLM → tool dispatch → subprocess → CodeExecutionResult chain.

    Skips when no Deno binary is available — there's no point asking the
    model to call ``run_code`` if the engine wouldn't have registered it.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; E2E requires real LLM access")
    if Sandbox.get() is None:
        pytest.skip("sandbox unavailable on this host; run_code is not registered without Deno")

    trace_path = _trace_path(tmp_path, fixtures_dir)
    cfg = _engine_config(maximum_depth=0, maximum_turns=6)

    messages = [
        AgentMessage(
            role="user",
            content=(
                "Use the run_code tool to compute the total number of traces in the dataset. "
                "The sandbox preloads a `trace_store` variable with a `.trace_count` property — "
                "call run_code with the exact code: print('total=', trace_store.trace_count) "
                "Then end your final reply with a line containing only <final/>."
            ),
        )
    ]

    async with asyncio.timeout(E2E_TIMEOUT_SECONDS * 2):
        results = await run_engine_async(messages, cfg, trace_path)

    # Pair each ``run_code`` tool_call with its tool-role result by
    # ``tool_call_id``. The mapper preserves call_id on both sides, so a
    # missing pair here means the engine boundary failed to surface the
    # sandbox's response — not a flaky LLM choice.
    run_code_call_ids = {
        tc.id
        for item in results
        for tc in (item.item.tool_calls or [])
        if tc.function.name == "run_code"
    }
    assert run_code_call_ids, (
        f"expected root agent to call run_code; got tool calls: {_tool_call_names(results)}"
    )
    run_code_results = [
        item
        for item in results
        if item.item.role == "tool" and item.item.tool_call_id in run_code_call_ids
    ]
    assert run_code_results, (
        f"expected a tool-role AgentOutputItem matching one of {run_code_call_ids}; "
        f"saw tool items={[(i.item.tool_call_id, i.item.name) for i in results if i.item.role == 'tool']}"
    )

    payload = json.loads(run_code_results[0].item.content or "{}")
    assert payload["exit_code"] == 0, (
        f"sandbox returned non-zero exit_code={payload['exit_code']}; "
        f"stdout: {payload.get('stdout')!r}; stderr: {payload.get('stderr')!r}"
    )
    assert payload["timed_out"] is False
    assert "total=" in payload["stdout"], (
        f"expected sandbox stdout to contain the printed count; got: {payload['stdout']!r}"
    )

    assert any(item.final for item in results), "no AgentOutputItem with final=True emitted"
