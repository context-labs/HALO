from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from engine.traces.models.canonical_span import SpanRecord, SpanResource, SpanScope, SpanStatus

JsonObject = Mapping[str, Any]

_DEFAULT_TIMESTAMP = "1970-01-01T00:00:00.000000000Z"
_LANGFUSE_GENERATION_TYPES = {"GENERATION"}
_LANGFUSE_SPAN_TYPES = {"SPAN"}
_ERROR_LEVELS = {"ERROR"}
_ERROR_STATUSES = {"error", "errored", "failed", "failure", "cancelled", "timeout"}


def load_langfuse_export(path: Path) -> Any:
    """Load a Langfuse export from JSON or JSONL.

    The local `lf api ...` commands usually emit JSON, but ad-hoc exports are
    often JSONL. Accept both so the converter stays usable with shell pipelines.
    """
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"{path} is empty")
    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_halo_jsonl(spans: Iterable[SpanRecord], path: Path, *, append: bool = False) -> int:
    """Write canonical HALO span rows to `path` and return the row count."""
    count = 0
    with path.open("a" if append else "w") as fh:
        for span in spans:
            fh.write(span.model_dump_json())
            fh.write("\n")
            count += 1
    return count


def convert_langfuse_export_to_spans(
    payload: Any,
    *,
    project_id: str = "care-ai",
    service_name: str = "care-ai-agents",
) -> list[SpanRecord]:
    """Convert Care AI Langfuse trace exports into HALO's OTel-shaped JSONL spans.

    Supported input shapes:

    - `{"trace": {...}, "observations": [...]}`
    - `{"traces": [...], "observations": [...]}`
    - a plain list of observations from `lf api observations list`
    - a list of bundle objects containing `trace` + `observations`

    Care AI writes one Langfuse trace per customer message and groups the wider
    conversation via `sessionId`. HALO indexes by trace id, so this converter
    keeps each Langfuse trace as its own HALO trace while preserving session and
    UCID metadata in attributes.
    """
    spans: list[SpanRecord] = []
    for trace, observations in _iter_trace_bundles(payload):
        spans.extend(
            _convert_trace_bundle(
                trace=trace,
                observations=observations,
                project_id=project_id,
                service_name=service_name,
            )
        )
    return spans


def _iter_trace_bundles(payload: Any) -> Iterable[tuple[JsonObject, list[JsonObject]]]:
    if isinstance(payload, Mapping):
        if "body" in payload and isinstance(payload["body"], Mapping):
            yield from _iter_trace_bundles(payload["body"])
            return

        if _looks_like_bundle(payload):
            yield (
                _coerce_trace(payload.get("trace")),
                _coerce_observations(payload.get("observations")),
            )
            return

        if "items" in payload:
            yielded = False
            for item in _as_sequence(payload["items"]):
                if isinstance(item, Mapping) and _looks_like_bundle(item):
                    yield (
                        _coerce_trace(item.get("trace")),
                        _coerce_observations(item.get("observations")),
                    )
                    yielded = True
            if yielded:
                return

        traces = _coerce_traces(payload.get("traces"))
        observations = _coerce_observations(payload.get("observations") or payload.get("data"))
        if traces:
            by_trace = _group_observations_by_trace(observations)
            for trace in traces:
                trace_id = _string_value(trace, "id") or _string_value(trace, "traceId")
                yield trace, by_trace.pop(trace_id, []) if trace_id else []
            for trace_id, unclaimed in by_trace.items():
                yield {"id": trace_id}, unclaimed
            return
        if observations:
            yield from _bundles_from_observations(observations)
            return
        if _looks_like_trace(payload):
            yield payload, []
            return

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        bundle_items = [
            item for item in payload if isinstance(item, Mapping) and _looks_like_bundle(item)
        ]
        if bundle_items:
            for item in bundle_items:
                yield (
                    _coerce_trace(item.get("trace")),
                    _coerce_observations(item.get("observations")),
                )
            return
        observations = [item for item in payload if isinstance(item, Mapping)]
        yield from _bundles_from_observations(observations)
        return

    raise ValueError("Unsupported Langfuse export shape")


def _looks_like_bundle(value: Mapping[str, Any]) -> bool:
    return "trace" in value and "observations" in value


def _looks_like_trace(value: Mapping[str, Any]) -> bool:
    return "id" in value and ("sessionId" in value or "metadata" in value or "name" in value)


def _coerce_trace(value: Any) -> JsonObject:
    if isinstance(value, Mapping) and isinstance(value.get("body"), Mapping):
        return value["body"]
    if isinstance(value, Mapping):
        return value
    return {}


def _coerce_traces(value: Any) -> list[JsonObject]:
    return [item for item in _as_sequence(value) if isinstance(item, Mapping)]


def _coerce_observations(value: Any) -> list[JsonObject]:
    if isinstance(value, Mapping) and isinstance(value.get("body"), Mapping):
        body = value["body"]
        return _coerce_observations(body.get("data") or body.get("items") or body)
    if isinstance(value, Mapping) and ("data" in value or "items" in value):
        return _coerce_observations(value.get("data") or value.get("items"))
    return [item for item in _as_sequence(value) if isinstance(item, Mapping)]


def _as_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return [value]


def _group_observations_by_trace(
    observations: Iterable[JsonObject],
) -> dict[str | None, list[JsonObject]]:
    by_trace: dict[str | None, list[JsonObject]] = defaultdict(list)
    for observation in observations:
        by_trace[_string_value(observation, "traceId")].append(observation)
    return by_trace


def _bundles_from_observations(
    observations: list[JsonObject],
) -> Iterable[tuple[JsonObject, list[JsonObject]]]:
    for trace_id, trace_observations in _group_observations_by_trace(observations).items():
        yield (
            {"id": trace_id or _synthetic_id("trace", json.dumps(trace_observations))},
            trace_observations,
        )


def _convert_trace_bundle(
    *,
    trace: JsonObject,
    observations: list[JsonObject],
    project_id: str,
    service_name: str,
) -> list[SpanRecord]:
    raw_trace_id = _string_value(trace, "id") or _first_string(observations, "traceId")
    if not raw_trace_id:
        raw_trace_id = _synthetic_id("trace", json.dumps(trace, sort_keys=True, default=str))
    trace_id = _otel_trace_id(raw_trace_id)

    root_span_id = _otel_span_id(f"trace:{raw_trace_id}")
    trace_start = _timestamp(
        _string_value(trace, "timestamp")
        or _string_value(trace, "createdAt")
        or _first_string(observations, "startTime")
    )
    trace_end = _timestamp(
        _string_value(trace, "updatedAt")
        or _string_value(trace, "endTime")
        or _last_string(observations, "endTime")
        or trace_start
    )
    metadata = _metadata(trace)
    trace_name = _string_value(trace, "name") or metadata.get("agentName") or "care_ai_trace"

    spans = [
        SpanRecord(
            trace_id=trace_id,
            span_id=root_span_id,
            parent_span_id="",
            name=f"custom.{trace_name}",
            kind="SPAN_KIND_INTERNAL",
            start_time=trace_start,
            end_time=trace_end,
            status=SpanStatus(code="STATUS_CODE_OK", message=""),
            resource=SpanResource(attributes=_resource_attributes(trace, service_name)),
            scope=SpanScope(name="care-ai-langfuse", version="1"),
            attributes=_base_attributes(
                project_id=project_id,
                observation_kind="CHAIN",
                trace=trace,
                observation=None,
                metadata=metadata,
            ),
        )
    ]

    observation_id_to_span_id: dict[str, str] = {}
    for observation in observations:
        observation_id = _observation_id(observation)
        observation_id_to_span_id[observation_id] = _otel_span_id(f"observation:{observation_id}")

    for observation in observations:
        observation_id = _observation_id(observation)
        parent_observation_id = _string_value(observation, "parentObservationId")
        parent_span_id = (
            observation_id_to_span_id.get(parent_observation_id, root_span_id)
            if parent_observation_id
            else root_span_id
        )
        spans.append(
            _convert_observation(
                trace=trace,
                observation=observation,
                trace_id=trace_id,
                span_id=observation_id_to_span_id[observation_id],
                parent_span_id=parent_span_id,
                project_id=project_id,
                service_name=service_name,
            )
        )
    return spans


def _convert_observation(
    *,
    trace: JsonObject,
    observation: JsonObject,
    trace_id: str,
    span_id: str,
    parent_span_id: str,
    project_id: str,
    service_name: str,
) -> SpanRecord:
    metadata = _metadata(observation)
    kind = _observation_kind(observation, metadata)
    model = _model_name(observation, metadata)
    name = _span_name(observation, metadata, kind, model)
    input_tokens, output_tokens, total_tokens = _usage(observation)
    attributes = _base_attributes(
        project_id=project_id,
        observation_kind=kind,
        trace=trace,
        observation=observation,
        metadata=metadata,
    )

    if kind == "LLM":
        planned_tool_names = _function_call_names(observation.get("output"))
        attributes.update(
            {
                "llm.provider": "openai",
                "llm.model_name": model,
                "inference.llm.provider": "openai",
                "inference.llm.model_name": model,
                "inference.llm.input_tokens": input_tokens,
                "inference.llm.output_tokens": output_tokens,
                "llm.token_count.prompt": input_tokens,
                "llm.token_count.completion": output_tokens,
                "llm.token_count.total": total_tokens,
                "llm.response.id": _string_from(metadata, "responseId"),
            }
        )
        if planned_tool_names:
            attributes["care_ai.llm_planned_tool_names"] = planned_tool_names
            attributes["care_ai.llm_planned_tool_count"] = len(planned_tool_names)
        _maybe_set(attributes, "llm.input_messages", _json_value(observation.get("input")))
        _maybe_set(attributes, "llm.output_messages", _json_value(observation.get("output")))
    elif kind == "TOOL":
        tool_name = _tool_name(observation, metadata)
        attributes.update(
            {
                "tool.name": tool_name,
                "input.value": _json_value(observation.get("input")),
                "output.value": _json_value(observation.get("output")),
            }
        )
    elif kind == "AGENT":
        response_count, tool_call_count, final_output_length = _agent_execution_counts(
            observation.get("output")
        )
        attributes.update(
            {
                "input.value": _json_value(observation.get("input")),
                "output.value": _json_value(observation.get("output")),
            }
        )
        _maybe_set(attributes, "care_ai.agent_response_count", response_count)
        _maybe_set(attributes, "care_ai.agent_reported_tool_call_count", tool_call_count)
        _maybe_set(attributes, "care_ai.agent_final_output_length", final_output_length)
    else:
        attributes.update(
            {
                "input.value": _json_value(observation.get("input")),
                "output.value": _json_value(observation.get("output")),
            }
        )

    if _string_from(metadata, "agentName") or _string_from(metadata, "agentDisplayName"):
        agent_name = _string_from(metadata, "agentName") or _string_from(
            metadata, "agentDisplayName"
        )
        attributes["agent.name"] = agent_name
        attributes["inference.agent_name"] = agent_name

    return SpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=name,
        kind="SPAN_KIND_CLIENT" if kind == "LLM" else "SPAN_KIND_INTERNAL",
        start_time=_timestamp(
            _string_value(observation, "startTime") or _string_value(observation, "createdAt")
        ),
        end_time=_timestamp(
            _string_value(observation, "endTime")
            or _string_value(observation, "updatedAt")
            or _string_value(observation, "startTime")
            or _string_value(observation, "createdAt")
        ),
        status=_status(observation, metadata),
        resource=SpanResource(attributes=_resource_attributes(trace, service_name)),
        scope=SpanScope(name="care-ai-langfuse", version="1"),
        attributes=attributes,
    )


def _base_attributes(
    *,
    project_id: str,
    observation_kind: str,
    trace: JsonObject,
    observation: JsonObject | None,
    metadata: JsonObject,
) -> dict[str, Any]:
    trace_metadata = _metadata(trace)
    attrs = {
        "openinference.span.kind": observation_kind,
        "inference.export.schema_version": 1,
        "inference.project_id": project_id,
        "inference.observation_kind": observation_kind,
        "langfuse.trace_id": _string_value(trace, "id")
        or _string_value(observation or {}, "traceId"),
        "langfuse.session_id": _string_value(trace, "sessionId")
        or _string_from(trace_metadata, "conversationId"),
        "langfuse.observation_id": _string_value(observation or {}, "id"),
        "care_ai.trace_kind": _string_from(metadata, "traceKind"),
        "care_ai.conversation_id": _string_from(metadata, "conversationId")
        or _string_value(trace, "sessionId"),
        "care_ai.ucid": _string_from(metadata, "ucid") or _string_from(trace_metadata, "ucid"),
        "care_ai.session_executor_name": _string_from(metadata, "sessionExecutorName")
        or _string_from(trace_metadata, "sessionExecutorName"),
        "care_ai.request_type": _string_from(metadata, "requestType"),
        "care_ai.request_id": _string_from(metadata, "requestId"),
        "care_ai.user_type": _string_from(metadata, "userType"),
        "care_ai.path": _string_from(metadata, "path"),
        "care_ai.prompt_name": _string_from(metadata, "promptName"),
        "care_ai.prompt_version": metadata.get("promptVersion"),
        "care_ai.parent_agent": _string_from(metadata, "parentAgent"),
        "care_ai.target_agent": _string_from(metadata, "targetAgent"),
        "care_ai.target_agent_display_name": _string_from(metadata, "targetAgentDisplayName"),
        "care_ai.tool_call_id": _string_from(metadata, "toolCallId"),
        "care_ai.tags": _json_value(trace.get("tags")),
    }
    return {key: value for key, value in attrs.items() if value not in (None, "")}


def _resource_attributes(trace: JsonObject, service_name: str) -> dict[str, Any]:
    metadata = _metadata(trace)
    attrs = {
        "service.name": service_name,
        "deployment.environment": _string_value(trace, "environment")
        or _string_from(metadata, "environment"),
        "service.version": _string_value(trace, "version") or _string_from(metadata, "version"),
        "service.release": _string_value(trace, "release") or _string_from(metadata, "release"),
    }
    return {key: value for key, value in attrs.items() if value not in (None, "")}


def _observation_kind(observation: JsonObject, metadata: JsonObject) -> str:
    obs_type = (_string_value(observation, "type") or "").upper()
    trace_kind = (_string_from(metadata, "traceKind") or "").lower()
    observation_name = _string_value(observation, "name") or ""
    normalized_name = observation_name.lower()
    if obs_type in _LANGFUSE_GENERATION_TYPES or trace_kind == "llm_generation":
        return "LLM"
    if trace_kind in {"tool_call", "agent_tool_invocation"} or normalized_name.startswith(
        ("tool_call:", "agent_tool:", "transfer_to_")
    ):
        return "TOOL"
    if trace_kind == "agent_execution" or observation_name == "agent_execution":
        return "AGENT"
    if obs_type in _LANGFUSE_SPAN_TYPES:
        return "SPAN"
    return "SPAN"


def _span_name(observation: JsonObject, metadata: JsonObject, kind: str, model: str | None) -> str:
    name = _string_value(observation, "name")
    if kind == "LLM":
        return name or f"response.{model or 'unknown'}"
    if kind == "TOOL":
        return name or f"function.{_tool_name(observation, metadata)}"
    if kind == "AGENT":
        return name or f"agent.{_string_from(metadata, 'agentName') or 'unknown'}"
    return name or "custom.langfuse_observation"


def _tool_name(observation: JsonObject, metadata: JsonObject) -> str:
    if tool_name := _string_from(metadata, "toolName"):
        return tool_name
    name = _string_value(observation, "name") or ""
    for prefix in ("tool_call: ", "agent_tool: "):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    if name.lower().startswith("transfer_to_"):
        return name
    return name or "unknown"


def _model_name(observation: JsonObject, metadata: JsonObject) -> str | None:
    return (
        _string_value(observation, "model")
        or _string_value(observation, "providedModelName")
        or _string_value(observation, "internalModelId")
        or _string_from(metadata, "model")
        or _string_from(metadata, "conversationModel")
    )


def _status(observation: JsonObject, metadata: JsonObject) -> SpanStatus:
    level = (_string_value(observation, "level") or "").upper()
    status = (_string_from(metadata, "status") or "").lower()
    status_message = _string_value(observation, "statusMessage") or ""
    is_error = level in _ERROR_LEVELS or status in _ERROR_STATUSES
    return SpanStatus(
        code="STATUS_CODE_ERROR" if is_error else "STATUS_CODE_OK",
        message=status_message,
    )


def _usage(observation: JsonObject) -> tuple[int, int, int]:
    usage = observation.get("usage")
    usage_details = observation.get("usageDetails")
    input_tokens = _int_from(usage, "promptTokens") or _int_from(usage_details, "input") or 0
    output_tokens = _int_from(usage, "completionTokens") or _int_from(usage_details, "output") or 0
    total_tokens = (
        _int_from(usage, "totalTokens")
        or _int_from(usage_details, "total")
        or input_tokens + output_tokens
    )
    return input_tokens, output_tokens, total_tokens


def _metadata(value: JsonObject) -> JsonObject:
    metadata = value.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata
    return {}


def _observation_id(observation: JsonObject) -> str:
    return _string_value(observation, "id") or _synthetic_id(
        "observation", json.dumps(observation, sort_keys=True, default=str)
    )


def _string_value(value: JsonObject, key: str) -> str | None:
    entry = value.get(key)
    return entry if isinstance(entry, str) and entry else None


def _string_from(value: Any, key: str) -> str | None:
    if isinstance(value, Mapping):
        entry = value.get(key)
        return entry if isinstance(entry, str) and entry else None
    return None


def _first_string(values: Sequence[JsonObject], key: str) -> str | None:
    for value in values:
        if found := _string_value(value, key):
            return found
    return None


def _last_string(values: Sequence[JsonObject], key: str) -> str | None:
    for value in reversed(values):
        if found := _string_value(value, key):
            return found
    return None


def _int_from(value: Any, key: str) -> int | None:
    if not isinstance(value, Mapping):
        return None
    entry = value.get(key)
    if isinstance(entry, bool):
        return None
    if isinstance(entry, int):
        return entry
    if isinstance(entry, float) and entry.is_integer():
        return int(entry)
    return None


def _json_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _function_call_names(value: Any) -> list[str]:
    names: list[str] = []

    def visit(entry: Any) -> None:
        if isinstance(entry, str):
            stripped = entry.strip()
            if stripped.startswith(("{", "[")):
                try:
                    visit(json.loads(stripped))
                except json.JSONDecodeError:
                    return
            return
        if isinstance(entry, Mapping):
            entry_type = entry.get("type")
            name = entry.get("name")
            if entry_type == "function_call" and isinstance(name, str) and name:
                names.append(name)
            for child in entry.values():
                visit(child)
            return
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
            for child in entry:
                visit(child)

    visit(value)
    return names


def _agent_execution_counts(value: Any) -> tuple[int | None, int | None, int | None]:
    parsed = _parse_jsonish(value)
    if not isinstance(parsed, Mapping):
        return None, None, None
    return (
        _int_from(parsed, "responseCount"),
        _int_from(parsed, "toolCallCount"),
        _int_from(parsed, "finalOutputLength"),
    )


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _maybe_set(attrs: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        attrs[key] = value


def _timestamp(value: str | None) -> str:
    if not value:
        return _DEFAULT_TIMESTAMP
    raw = value.strip()
    if raw.endswith("Z") and "." in raw:
        prefix, fractional = raw[:-1].split(".", maxsplit=1)
        if len(fractional) == 9:
            return raw
        return f"{prefix}.{fractional[:9].ljust(9, '0')}Z"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return raw
    return f"{parsed.strftime('%Y-%m-%dT%H:%M:%S')}.{parsed.microsecond:06d}000Z"


def _synthetic_id(namespace: str, value: str) -> str:
    return sha256(f"{namespace}:{value}".encode()).hexdigest()


def _otel_trace_id(value: str) -> str:
    cleaned = value.replace("-", "").lower()
    if len(cleaned) == 32 and all(ch in "0123456789abcdef" for ch in cleaned):
        return cleaned
    return sha256(f"trace:{value}".encode()).hexdigest()[:32]


def _otel_span_id(value: str) -> str:
    cleaned = value.replace("-", "").lower()
    if len(cleaned) == 16 and all(ch in "0123456789abcdef" for ch in cleaned):
        return cleaned
    return sha256(f"span:{value}".encode()).hexdigest()[:16]
