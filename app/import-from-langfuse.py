#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LANGFUSE_PREFIX_SKIP = {"input", "output"}
TOOL_NAME_PREFIXES = ("tool_call_",)


def parse_time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=UTC)
    text = str(value).strip()
    if " " in text and "T" not in text:
        text = text.replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def format_nanos(dt: datetime | None) -> str:
    if dt is None:
        dt = datetime.now(UTC)
    dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond * 1000:09d}Z"


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compact_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def usage_value(row: dict[str, Any], key: str) -> int | None:
    direct = as_int(row.get(key))
    if direct is not None:
        return direct
    usage = compact_dict(row.get("usageDetails"))
    provided = compact_dict(row.get("providedUsageDetails"))
    aliases = {
        "inputUsage": ("input", "prompt"),
        "outputUsage": ("output", "completion"),
        "totalUsage": ("total",),
    }
    for usage_obj in (usage, provided):
        for alias in aliases.get(key, ()):
            found = as_int(usage_obj.get(alias))
            if found is not None:
                return found
    return None


def infer_provider(row: dict[str, Any]) -> str | None:
    metadata = compact_dict(row.get("metadata"))
    hidden = compact_dict(metadata.get("hidden_params"))
    model = (
        hidden.get("litellm_model_name")
        or metadata.get("litellm_model_name")
        or row.get("model")
        or row.get("modelId")
    )
    if model is None:
        return None
    text = str(model).lower()
    if "anthropic" in text or "claude" in text or "bedrock/" in text:
        return "anthropic"
    if "gpt" in text or "openai" in text:
        return "openai"
    return None


def observation_kind(row: dict[str, Any]) -> str:
    typ = str(row.get("type") or "").upper()
    name = str(row.get("name") or "")
    if typ in {"GENERATION", "LLM"}:
        return "LLM"
    if typ == "SPAN" and name.startswith(TOOL_NAME_PREFIXES):
        return "TOOL"
    return typ or "SPAN"


def otel_kind(kind: str) -> str:
    if kind == "LLM":
        return "SPAN_KIND_CLIENT"
    return "SPAN_KIND_INTERNAL"


def status(row: dict[str, Any]) -> dict[str, str]:
    level = str(row.get("level") or "").upper()
    message = str(row.get("statusMessage") or "")
    metadata = compact_dict(row.get("metadata"))
    success = metadata.get("success")
    failed = (
        level in {"ERROR", "WARNING"}
        or bool(message)
        or success is False
        or as_int(row.get("llm_error_count") or 0) not in (None, 0)
    )
    return {
        "code": "STATUS_CODE_ERROR" if failed else "STATUS_CODE_OK",
        "message": message,
    }


def langfuse_attrs(row: dict[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key, value in row.items():
        if key in LANGFUSE_PREFIX_SKIP or value is None:
            continue
        attrs[f"langfuse.{key}"] = value
    return attrs


def to_halo_span(row: dict[str, Any]) -> dict[str, Any]:
    kind = observation_kind(row)
    input_tokens = usage_value(row, "inputUsage")
    output_tokens = usage_value(row, "outputUsage")
    total_tokens = usage_value(row, "totalUsage")
    cost_total = as_float(row.get("totalCost"))
    provider = infer_provider(row)
    model = row.get("model") or row.get("modelId")
    trace_name = row.get("traceName") or row.get("name") or "langfuse-export"

    attrs = langfuse_attrs(row)
    attrs.update(
        {
            "openinference.span.kind": kind,
            "inference.export.schema_version": 1,
            "inference.project_id": row.get("projectId") or "",
            "inference.observation_kind": kind,
            "inference.agent_name": trace_name,
            "input.value": row.get("input"),
            "output.value": row.get("output"),
        }
    )

    if row.get("userId") is not None:
        attrs["inference.user_id"] = row.get("userId")
    if model is not None:
        attrs["llm.model_name"] = model
        if kind == "LLM":
            attrs["inference.llm.model_name"] = model
    if provider is not None:
        attrs["llm.provider"] = provider
        attrs["inference.llm.provider"] = provider
    if input_tokens is not None:
        attrs["llm.token_count.prompt"] = input_tokens
        attrs["inference.llm.input_tokens"] = input_tokens
    if output_tokens is not None:
        attrs["llm.token_count.completion"] = output_tokens
        attrs["inference.llm.output_tokens"] = output_tokens
    if total_tokens is not None:
        attrs["llm.token_count.total"] = total_tokens
    if cost_total is not None:
        attrs["inference.llm.cost.total"] = cost_total
    if kind == "LLM":
        attrs["llm.input_messages"] = row.get("input")
        attrs["llm.output_messages"] = row.get("output")
    if kind == "TOOL":
        metadata = compact_dict(row.get("metadata"))
        function_name = metadata.get("function_name")
        if not function_name and str(row.get("name") or "").startswith("tool_call_"):
            function_name = str(row["name"])[len("tool_call_") :]
        if function_name:
            attrs["tool.name"] = function_name

    return {
        "trace_id": row.get("traceId") or "",
        "span_id": row.get("id") or "",
        "parent_span_id": row.get("parentObservationId") or "",
        "trace_state": "",
        "name": row.get("name") or "",
        "kind": otel_kind(kind),
        "start_time": format_nanos(parse_time(row.get("startTime"))),
        "end_time": format_nanos(parse_time(row.get("endTime")) or parse_time(row.get("startTime"))),
        "status": status(row),
        "resource": {
            "attributes": {
                "service.name": trace_name,
                "deployment.environment": row.get("environment") or "",
            }
        },
        "scope": {"name": "langfuse-ui-export", "version": str(row.get("version") or "")},
        "attributes": attrs,
    }


def in_window(row: dict[str, Any], start: datetime | None, end: datetime | None) -> bool:
    if start is None and end is None:
        return True
    timestamp = parse_time(row.get("traceTimestamp")) or parse_time(row.get("startTime"))
    if timestamp is None:
        return False
    if start is not None and timestamp < start:
        return False
    if end is not None and timestamp >= end:
        return False
    return True


def completed_trace_ids(path: Path, start: datetime | None, end: datetime | None) -> set[str]:
    trace_has_window_row: set[str] = set()
    incomplete: set[str] = set()
    with path.open(errors="replace") as handle:
        for line in handle:
            row = json.loads(line)
            trace_id = row.get("traceId")
            if not trace_id:
                continue
            if in_window(row, start, end):
                trace_has_window_row.add(trace_id)
            if row.get("endTime") in (None, ""):
                incomplete.add(trace_id)
    return trace_has_window_row - incomplete


def convert(
    source: Path,
    output: Path,
    *,
    start: datetime | None,
    end: datetime | None,
    require_completed_traces: bool,
) -> dict[str, Any]:
    allowed = completed_trace_ids(source, start, end) if require_completed_traces else None
    counts: Counter[str] = Counter()
    trace_ids: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)

    with source.open(errors="replace") as inp, output.open("w") as out:
        for line in inp:
            counts["rows_in"] += 1
            row = json.loads(line)
            trace_id = row.get("traceId")
            if allowed is not None:
                if trace_id not in allowed or not in_window(row, start, end):
                    continue
            elif not in_window(row, start, end):
                continue

            span = to_halo_span(row)
            out.write(json.dumps(span, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts["rows_out"] += 1
            trace_ids.add(span["trace_id"])
            kind = span["attributes"].get("inference.observation_kind", "unknown").lower()
            counts[kind] += 1
            if span["status"]["code"] == "STATUS_CODE_ERROR":
                counts["error"] += 1

    return {
        "source": str(source),
        "output": str(output),
        "trace_count": len(trace_ids),
        **counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Langfuse observation JSONL to HALO JSONL.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", help="Inclusive UTC-ish start timestamp, e.g. 2026-04-19T00:00:00Z")
    parser.add_argument("--end", help="Exclusive UTC-ish end timestamp, e.g. 2026-05-20T00:00:00Z")
    parser.add_argument(
        "--require-completed-traces",
        action="store_true",
        help="Two-pass mode: only keep traces with no observations missing endTime.",
    )
    args = parser.parse_args()

    summary = convert(
        args.source,
        args.output,
        start=parse_time(args.start),
        end=parse_time(args.end),
        require_completed_traces=args.require_completed_traces,
    )
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
