from __future__ import annotations

import json
from pathlib import Path

from engine.traces.models.canonical_span import SpanRecord


def test_parse_tiny_fixture_first_line(fixtures_dir: Path) -> None:
    raw = (fixtures_dir / "tiny_traces.jsonl").read_text().splitlines()[0]
    span = SpanRecord.model_validate_json(raw)
    assert span.trace_id == "t-aaaa"
    assert span.span_id == "s-aaaa-1"
    assert span.parent_span_id == ""
    assert span.resource.attributes["service.name"] == "agent-a"
    assert span.attributes["inference.export.schema_version"] == 1


def test_status_defaults_preserved() -> None:
    raw = {
        "trace_id": "t",
        "span_id": "s",
        "parent_span_id": "",
        "trace_state": "",
        "name": "x",
        "kind": "SPAN_KIND_INTERNAL",
        "start_time": "2026-04-23T00:00:00Z",
        "end_time": "2026-04-23T00:00:01Z",
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "resource": {"attributes": {}},
        "scope": {"name": "n", "version": "v"},
        "attributes": {},
    }
    span = SpanRecord.model_validate(raw)
    assert span.status.code == "STATUS_CODE_OK"
    assert json.loads(span.model_dump_json())["trace_id"] == "t"
