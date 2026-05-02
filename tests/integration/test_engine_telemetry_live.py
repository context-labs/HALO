"""Live end-to-end test: engine + local JSONL telemetry backend.

Uses a real LLM call (one tiny prompt, low max_turns) to drive the
openai-agents SDK's tracing layer, which is what fires the
InferenceOtlpFileProcessor. FakeRunner bypasses the SDK Runner entirely
and so cannot exercise this path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.main import run_engine_async
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage


@pytest.mark.live
@pytest.mark.asyncio
async def test_local_telemetry_backend_writes_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixtures_dir: Path
) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; live test requires LLM access")

    out_path = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("HALO_TELEMETRY_PATH", str(out_path))

    model = ModelConfig(name="gpt-5.4-mini")
    root = AgentConfig(name="root", model=model, maximum_turns=2)
    sub = AgentConfig(name="sub", model=model, maximum_turns=2)
    cfg = EngineConfig(
        root_agent=root,
        subagent=sub,
        synthesis_model=model,
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        maximum_depth=0,
        maximum_parallel_subagents=1,
    )

    messages = [AgentMessage(role="user", content="Reply with the word ok and stop.")]
    trace_path = fixtures_dir / "tiny_traces.jsonl"

    await run_engine_async(messages, cfg, trace_path, telemetry=True)

    assert out_path.exists(), "local telemetry file was not written"
    lines = out_path.read_text().splitlines()
    assert len(lines) > 0, "no spans were written"

    spans = [json.loads(line) for line in lines]
    assert all("trace_id" in s and "span_id" in s for s in spans)

    kinds = {s["attributes"].get("openinference.span.kind") for s in spans}
    assert "AGENT" in kinds, f"no AGENT span in {kinds}"
    assert any(k in kinds for k in ("LLM",)), f"no LLM span in {kinds}"
