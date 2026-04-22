"""Translate legacy HF flat-JSONL trace records into OpenInference OTLP
span trees.

Produces the same shape ``otel-interceptor``'s ``compact:traces`` step
emits — one JSON object per line with a ``spans`` array — so re-indexed
datasets look byte-for-byte identical to datasets that arrived via the
live interceptor path.

For each HF record we synthesize:

* one root ``AGENT`` span (carries query / final_answer / labels / outcome /
  ground_truth as attributes)
* one ``LLM`` span per assistant message — tokenized input_messages +
  output_messages attributes reconstructed from the prior context
* one ``TOOL`` span per ``tool_calls`` / ``tool`` message pair —
  ``tool.name`` / ``input.value`` / ``output.value``

The generated spans are not "real" (no wall-clock timestamps from the
original run), but they carry every bit of information the RLM harness
needs. Downstream tools operate on them identically to live-captured
OTLP.

CLI::

    uv run python -m dataset.translators.hf_to_openinference \\
        --descriptor catalog/grepfruit_pathrich.py \\
        --output /path/to/grepfruit_pathrich.oi.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from dataset._fastjson import loads as _fast_loads
from dataset.descriptor import DatasetDescriptor, HFMapping, get_nested


# ----------------------------------------------------------------------
# Span synthesis
# ----------------------------------------------------------------------

def _hex_id(*parts: str, length: int) -> str:
    """Deterministic hex span/trace id from a tuple of strings."""
    h = hashlib.sha256("/".join(parts).encode()).hexdigest()
    return h[:length]


def _attr(key: str, value: Any) -> dict:
    """Build an OTLP KeyValue with the right value-variant."""
    if isinstance(value, bool):
        v = {"boolValue": value}
    elif isinstance(value, int):
        # OTLP 64-bit ints come as JSON strings.
        v = {"intValue": str(value)}
    elif isinstance(value, float):
        v = {"doubleValue": value}
    elif isinstance(value, (list, tuple)):
        v = {"arrayValue": {"values": [_attr("", x)["value"] for x in value]}}
    elif isinstance(value, dict):
        v = {"kvlistValue": {"values": [_attr(k, x) for k, x in value.items()]}}
    else:
        v = {"stringValue": "" if value is None else str(value)}
    return {"key": key, "value": v}


def _messages_attrs(
    prefix: str,
    messages: list[dict[str, Any]],
) -> list[dict]:
    """Flatten a list of messages to OpenInference-convention attributes."""
    out: list[dict] = []
    for i, m in enumerate(messages):
        role = m.get("role") or ""
        content = m.get("content")
        if not isinstance(content, str):
            content = "" if content is None else json.dumps(content)
        out.append(_attr(f"{prefix}.{i}.message.role", role))
        if content:
            out.append(_attr(f"{prefix}.{i}.message.content", content))
        tcs = m.get("tool_calls") or []
        for j, tc in enumerate(tcs):
            fn = tc.get("function") or {}
            tc_id = tc.get("id") or ""
            out.append(_attr(
                f"{prefix}.{i}.message.tool_calls.{j}.tool_call.id", tc_id,
            ))
            if fn.get("name"):
                out.append(_attr(
                    f"{prefix}.{i}.message.tool_calls.{j}.tool_call.function.name",
                    fn.get("name"),
                ))
            if fn.get("arguments") is not None:
                out.append(_attr(
                    f"{prefix}.{i}.message.tool_calls.{j}.tool_call.function.arguments",
                    fn.get("arguments"),
                ))
        tcid = m.get("tool_call_id")
        if tcid:
            out.append(_attr(f"{prefix}.{i}.message.tool_call_id", tcid))
    return out


def _tool_result_for(messages: list[dict[str, Any]], tool_call_id: str) -> str | None:
    """Find the ``tool``-role message responding to a given tool_call_id."""
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id:
            c = m.get("content") or ""
            if not isinstance(c, str):
                c = json.dumps(c)
            return c
    return None


def _tool_result_positional(
    messages: list[dict[str, Any]],
    assistant_msg_index: int,
    tool_call_index: int,
) -> str | None:
    """Fallback: find the tool result by position.

    Some HF dumps carry ``tool``-role responses in order after the
    assistant message that made the calls, without ``tool_call_id``
    links. Walk forward from ``assistant_msg_index+1`` and return the
    N-th consecutive ``tool`` message (where N == ``tool_call_index``)
    before hitting a non-``tool`` role.
    """
    seen = 0
    for m in messages[assistant_msg_index + 1:]:
        if m.get("role") != "tool":
            return None
        if seen == tool_call_index:
            c = m.get("content") or ""
            if not isinstance(c, str):
                c = json.dumps(c)
            return c
        seen += 1
    return None


def translate_record(record: dict[str, Any], desc: DatasetDescriptor) -> dict[str, Any]:
    """Convert one HF record into a compact-traces-by-trace span tree.

    Requires an HF descriptor — translating an already-OI record is a
    no-op call-site bug.
    """
    if not isinstance(desc.mapping, HFMapping):
        raise TypeError(
            f"translate_record requires an HFMapping descriptor; got "
            f"{type(desc.mapping).__name__}"
        )
    m: HFMapping = desc.mapping

    trace_id_seed = str(get_nested(record, m.id_field) or hashlib.sha1(
        json.dumps(record, sort_keys=True, default=str).encode()
    ).hexdigest())
    trace_id = _hex_id(trace_id_seed, length=32)

    query = get_nested(record, m.query_field) or ""
    if not isinstance(query, str):
        query = json.dumps(query)
    messages = get_nested(record, m.messages_field) or []
    if not isinstance(messages, list):
        messages = []

    final_answer = (
        get_nested(record, m.final_answer_field) if m.final_answer_field else None
    )
    final_str = ""
    if isinstance(final_answer, dict):
        final_str = json.dumps(final_answer)
    elif isinstance(final_answer, str):
        final_str = final_answer

    # Root AGENT span ---------------------------------------------------
    root_id = _hex_id(trace_id_seed, "root", length=16)
    root_attrs = [
        _attr("openinference.span.kind", "AGENT"),
        _attr("input.value", query),
        _attr("output.value", final_str),
    ]
    # Labels as attributes — keyed by the label's ``source`` path so the
    # OI descriptor's Label.source string resolves back to the same value.
    for lbl in desc.labels:
        v = get_nested(record, lbl.source)
        if v is not None:
            root_attrs.append(_attr(lbl.source, v))
    # Primary metric + ground-truth as custom attributes.
    if desc.primary_metric is not None:
        v = get_nested(record, desc.primary_metric.source)
        if v is not None:
            try:
                root_attrs.append(_attr(desc.primary_metric.source, float(v)))
            except (TypeError, ValueError):
                root_attrs.append(_attr(desc.primary_metric.source, v))
    if desc.has_ground_truth:
        v = get_nested(record, desc.ground_truth_source)
        if v is not None:
            if isinstance(v, (list, tuple, dict)):
                root_attrs.append(_attr(desc.ground_truth_source, json.dumps(v)))
            else:
                root_attrs.append(_attr(desc.ground_truth_source, v))

    spans: list[dict] = [{
        "spanId": root_id,
        "parentSpanId": None,
        "name": "agent.run",
        "startTimeUnixNano": "0",
        "endTimeUnixNano": "0",
        "attributes": root_attrs,
        "resource": {"attributes": [_attr("service.name", desc.source_model or desc.id)]},
        "scope": {"name": "halo.hf_to_openinference"},
    }]

    # Walk messages: each assistant turn becomes one LLM span. Prior
    # messages form its input_messages; the assistant's output becomes
    # output_messages. Each tool_call spawns a TOOL span.
    history: list[dict[str, Any]] = []
    llm_counter = 0
    tool_counter = 0
    for msg_index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            # Build an LLM span for this assistant turn.
            llm_counter += 1
            llm_id = _hex_id(trace_id_seed, f"llm.{llm_counter}", length=16)
            llm_attrs = [
                _attr("openinference.span.kind", "LLM"),
                _attr("llm.model_name", desc.source_model or ""),
            ]
            llm_attrs.extend(_messages_attrs("llm.input_messages", history))
            llm_attrs.extend(_messages_attrs("llm.output_messages", [msg]))
            spans.append({
                "spanId": llm_id,
                "parentSpanId": root_id,
                "name": "llm.call",
                "startTimeUnixNano": "0",
                "endTimeUnixNano": "0",
                "attributes": llm_attrs,
                "resource": {"attributes": []},
                "scope": {"name": "halo.hf_to_openinference"},
            })
            # One TOOL span per tool_call, with its result paired by id
            # when available and by position otherwise (some HF dumps
            # omit ``tool_call_id`` on the tool-role response — we'd
            # silently drop results without the positional fallback).
            tcs = msg.get("tool_calls") or []
            for tc_pos, tc in enumerate(tcs):
                tool_counter += 1
                original_id = tc.get("id")
                tc_id = original_id or f"call_{tool_counter}"
                fn = tc.get("function") or {}
                result = None
                if original_id:
                    result = _tool_result_for(messages, original_id)
                if result is None:
                    result = _tool_result_positional(messages, msg_index, tc_pos)
                result = result or ""
                tool_span_id = _hex_id(trace_id_seed, f"tool.{tool_counter}", length=16)
                spans.append({
                    "spanId": tool_span_id,
                    "parentSpanId": llm_id,
                    "name": f"tool.{fn.get('name') or 'unknown'}",
                    "startTimeUnixNano": "0",
                    "endTimeUnixNano": "0",
                    "attributes": [
                        _attr("openinference.span.kind", "TOOL"),
                        _attr("tool.name", fn.get("name") or ""),
                        _attr("tool.call_id", tc_id),
                        _attr("input.value", fn.get("arguments") or ""),
                        _attr("output.value", result),
                    ],
                    "resource": {"attributes": []},
                    "scope": {"name": "halo.hf_to_openinference"},
                })
        history.append(msg)

    return {
        "traceId": trace_id,
        "spanCount": len(spans),
        "logCount": 0,
        "startTimeUnixNano": "0",
        "endTimeUnixNano": "0",
        "durationMs": 0.0,
        "services": [desc.source_model or desc.id],
        "spans": spans,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _load_descriptor(path: Path) -> DatasetDescriptor:
    spec = importlib.util.spec_from_file_location("_desc", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "DESCRIPTOR")


def translate_file(
    descriptor: DatasetDescriptor,
    output: Path,
    *,
    progress_every: int = 5000,
) -> int:
    """Translate ``descriptor.source_path`` into an OpenInference JSONL at ``output``.

    Returns the number of records written.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(descriptor.source_path, "rb") as src, output.open("w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                record = _fast_loads(line)
            except Exception:
                continue
            if not isinstance(record, dict):
                continue
            oi = translate_record(record, descriptor)
            dst.write(json.dumps(oi, ensure_ascii=False, default=str))
            dst.write("\n")
            n += 1
            if progress_every and n % progress_every == 0:
                print(f"translated {n:,} records…", flush=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="HF → OpenInference trace translator")
    ap.add_argument("--descriptor", required=True, type=Path,
                    help="Path to a catalog/<id>.py descriptor file.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output JSONL path for translated records.")
    args = ap.parse_args()

    descriptor = _load_descriptor(args.descriptor)
    n = translate_file(descriptor, args.output)
    print(f"wrote {n:,} records to {args.output}")


if __name__ == "__main__":
    main()
