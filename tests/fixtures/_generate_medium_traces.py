#!/usr/bin/env -S uv run --script
"""Regenerate medium_traces.jsonl: ~500 traces × 4 spans = 2000 lines, mixed shapes.

Run from the engine/ directory: ``uv run tests/fixtures/_generate_medium_traces.py``.
Deterministic via a fixed seed so the fixture is byte-stable across regenerations.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).parent / "medium_traces.jsonl"
SEED = 20260428
NUM_TRACES = 500
SPANS_PER_TRACE = 4

SERVICES = ["agent-a", "agent-b", "agent-c", "agent-d"]
MODELS = ["claude-sonnet-4-5", "claude-haiku-4-5", "gpt-5.4", "gpt-5-mini"]
PROVIDERS = ["anthropic", "openai"]


def _agent_span(rng: random.Random, trace_id: str, span_idx: int, t_base: int) -> dict:
    svc = rng.choice(SERVICES)
    has_error = rng.random() < 0.1
    return {
        "trace_id": trace_id,
        "span_id": f"s-{trace_id}-{span_idx}",
        "parent_span_id": "" if span_idx == 0 else f"s-{trace_id}-{span_idx - 1}",
        "trace_state": "",
        "name": "root" if span_idx == 0 else "agent.step",
        "kind": "SPAN_KIND_INTERNAL",
        "start_time": f"2026-04-23T05:{(t_base // 60) % 60:02d}:{t_base % 60:02d}.000000000Z",
        "end_time": f"2026-04-23T05:{((t_base + 1) // 60) % 60:02d}:{(t_base + 1) % 60:02d}.000000000Z",
        "status": {
            "code": "STATUS_CODE_ERROR" if has_error else "STATUS_CODE_OK",
            "message": "boom" if has_error else "",
        },
        "resource": {"attributes": {"service.name": svc, "service.version": "0.1.0"}},
        "scope": {"name": "@test/scope", "version": "0.0.1"},
        "attributes": {
            "openinference.span.kind": "AGENT",
            "inference.export.schema_version": 1,
            "inference.project_id": "prj_medium",
            "inference.observation_kind": "AGENT",
            "inference.agent_name": svc,
        },
    }


def _llm_span(rng: random.Random, trace_id: str, span_idx: int, t_base: int) -> dict:
    svc = rng.choice(SERVICES)
    model = rng.choice(MODELS)
    provider = rng.choice(PROVIDERS)
    in_tokens = rng.randint(50, 500)
    out_tokens = rng.randint(10, 200)
    has_error = rng.random() < 0.05
    return {
        "trace_id": trace_id,
        "span_id": f"s-{trace_id}-{span_idx}",
        "parent_span_id": f"s-{trace_id}-{span_idx - 1}",
        "trace_state": "",
        "name": f"{provider}.chat.completions.create",
        "kind": "SPAN_KIND_CLIENT",
        "start_time": f"2026-04-23T05:{(t_base // 60) % 60:02d}:{t_base % 60:02d}.100000000Z",
        "end_time": f"2026-04-23T05:{((t_base + 1) // 60) % 60:02d}:{(t_base + 1) % 60:02d}.500000000Z",
        "status": {
            "code": "STATUS_CODE_ERROR" if has_error else "STATUS_CODE_OK",
            "message": "rate limited" if has_error else "",
        },
        "resource": {"attributes": {"service.name": svc, "service.version": "0.1.0"}},
        "scope": {"name": "@test/scope", "version": "0.0.1"},
        "attributes": {
            "openinference.span.kind": "LLM",
            "inference.export.schema_version": 1,
            "inference.project_id": "prj_medium",
            "inference.observation_kind": "LLM",
            "inference.llm.provider": provider,
            "inference.llm.model_name": model,
            "inference.llm.input_tokens": in_tokens,
            "inference.llm.output_tokens": out_tokens,
            "inference.agent_name": svc,
        },
    }


def main() -> None:
    rng = random.Random(SEED)
    lines: list[str] = []
    for i in range(NUM_TRACES):
        trace_id = f"t-{i:05d}"
        t_base = i * 5
        for span_idx in range(SPANS_PER_TRACE):
            if span_idx == 0:
                span = _agent_span(rng, trace_id, span_idx, t_base)
            else:
                span = _llm_span(rng, trace_id, span_idx, t_base)
            lines.append(json.dumps(span, separators=(",", ":")))
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
