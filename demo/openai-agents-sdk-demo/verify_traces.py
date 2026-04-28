"""Verification script from openai-agents-sdk-span-conversion.md → 'Verifying the output'."""
import gzip
import json
import re
import sys


def verify(path: str) -> int:
    count = 0
    if path.endswith(".gz") or path.endswith(".gzip"):
        open_func = lambda p: gzip.open(p, "rt")
    else:
        open_func = lambda p: open(p, "rt")

    with open_func(path) as fh:
        for raw in fh:
            line = json.loads(raw)
            assert set(line) >= {
                "trace_id", "span_id", "parent_span_id", "trace_state",
                "name", "kind", "start_time", "end_time",
                "status", "resource", "scope", "attributes",
            }, f"missing top-level keys: {set(line)}"
            assert line["kind"].startswith("SPAN_KIND_"), line["kind"]
            assert line["status"]["code"].startswith("STATUS_CODE_"), line["status"]
            assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z", line["start_time"]), line["start_time"]
            assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z", line["end_time"]), line["end_time"]
            attrs = line["attributes"]
            assert attrs["inference.export.schema_version"] == 1, attrs.get("inference.export.schema_version")
            assert attrs["inference.project_id"], attrs.get("inference.project_id")
            assert attrs["inference.observation_kind"] in {
                "LLM", "TOOL", "AGENT", "CHAIN", "GUARDRAIL", "SPAN"
            }, attrs["inference.observation_kind"]
            count += 1
    return count


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "traces.jsonl.gz"
    n = verify(path)
    print(f"OK: {n} spans passed all spec assertions")
