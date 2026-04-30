from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from engine.traces.models.canonical_span import SpanRecord, SpanResource, SpanScope, SpanStatus
from engine.traces.models.trace_index_config import TraceIndexConfig

TraceSource = Literal["auto", "otel", "pi-session"]

_PI_ENTRY_TYPES = {
    "session",
    "message",
    "model_change",
    "thinking_level_change",
    "compaction",
    "branch_summary",
    "custom",
    "custom_message",
    "label",
    "session_info",
}
_GENERATED_JSONL_NAMES = {
    ".halo-pi-session-spans.jsonl",
}
_DEFAULT_EXCERPT_CHARS = 240


class TraceSourceError(ValueError):
    """Raised when a requested trace input source cannot be detected or adapted."""


@dataclass(frozen=True)
class PreparedTraceInput:
    """Resolved trace input after any source-specific adapter has run."""

    trace_path: Path
    config: TraceIndexConfig
    source: Literal["otel", "pi-session"]


def prepare_trace_input(
    trace_path: Path,
    config: TraceIndexConfig,
    *,
    source: TraceSource = "auto",
    pi_session_full_content: bool = False,
    pi_session_excerpt_chars: int = _DEFAULT_EXCERPT_CHARS,
) -> PreparedTraceInput:
    """Return a canonical span JSONL path ready for ``TraceIndexBuilder``.

    OTel/OpenInference JSONL files are already HALO's canonical substrate and are
    returned unchanged. Pi session JSONL files (or directories of session files)
    are converted to bounded synthetic ``SpanRecord`` rows in a sidecar file so
    the existing index, store, and trace tools can operate unchanged.
    """
    if pi_session_excerpt_chars < 0:
        raise TraceSourceError("pi_session_excerpt_chars must be >= 0")

    resolved_source = detect_trace_source(trace_path) if source == "auto" else source

    if resolved_source == "otel":
        if trace_path.is_dir():
            raise TraceSourceError("source='otel' expects a JSONL file, not a directory")
        return PreparedTraceInput(trace_path=trace_path, config=config, source="otel")

    if resolved_source == "pi-session":
        canonical_path = convert_pi_sessions_to_span_jsonl(
            trace_path,
            include_full_content=pi_session_full_content,
            excerpt_chars=pi_session_excerpt_chars,
        )
        # Any caller-provided index_path remains respected; otherwise the normal
        # sidecar convention attaches to the generated canonical JSONL.
        return PreparedTraceInput(trace_path=canonical_path, config=config, source="pi-session")

    raise TraceSourceError(f"unsupported trace source: {resolved_source}")


def detect_trace_source(trace_path: Path) -> Literal["otel", "pi-session"]:
    """Detect whether ``trace_path`` is OTel/OpenInference span JSONL or Pi session JSONL."""
    if trace_path.is_dir():
        files = list(_iter_pi_session_files(trace_path))
        if not files:
            raise TraceSourceError(f"no JSONL files found under Pi session directory: {trace_path}")
        first = _first_json_object(files[0])
        if _looks_like_pi_entry(first):
            return "pi-session"
        raise TraceSourceError(
            f"directory input is supported for Pi sessions only; first JSONL row in {files[0]} "
            "is not a Pi session entry"
        )

    first = _first_json_object(trace_path)
    if first is None:
        raise TraceSourceError(f"trace input has no JSON objects: {trace_path}")

    try:
        SpanRecord.model_validate(first)
        return "otel"
    except ValidationError:
        pass

    if _looks_like_pi_entry(first):
        return "pi-session"

    raise TraceSourceError(
        f"could not detect trace source for {trace_path}; expected OTel/OpenInference SpanRecord "
        "JSONL or Pi session JSONL. Pass --source explicitly if this is intentional."
    )


def convert_pi_sessions_to_span_jsonl(
    source_path: Path,
    *,
    include_full_content: bool = False,
    excerpt_chars: int = _DEFAULT_EXCERPT_CHARS,
) -> Path:
    """Convert a Pi session file or directory of session files into canonical span JSONL."""
    if excerpt_chars < 0:
        raise TraceSourceError("excerpt_chars must be >= 0")

    session_files = list(_iter_pi_session_files(source_path))
    if not session_files:
        raise TraceSourceError(f"no Pi session JSONL files found at {source_path}")

    output_path = _pi_session_sidecar_path(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as fh:
        for session_file in session_files:
            for span in _iter_pi_session_spans(
                session_file,
                include_full_content=include_full_content,
                excerpt_chars=excerpt_chars,
            ):
                fh.write(span.model_dump_json())
                fh.write("\n")

    if output_path.exists() and output_path.read_bytes() == tmp_path.read_bytes():
        tmp_path.unlink()
    else:
        tmp_path.replace(output_path)

    return output_path


def _iter_pi_session_files(source_path: Path) -> list[Path]:
    if source_path.is_file():
        return [source_path]
    if not source_path.exists():
        raise TraceSourceError(f"trace input does not exist: {source_path}")
    if not source_path.is_dir():
        raise TraceSourceError(f"trace input is neither a file nor directory: {source_path}")
    return sorted(
        p for p in source_path.rglob("*.jsonl") if p.is_file() and not _is_generated_trace_file(p)
    )


def _is_generated_trace_file(path: Path) -> bool:
    return (
        path.name in _GENERATED_JSONL_NAMES
        or path.name.endswith(".engine-index.jsonl")
        or path.name.endswith(".tmp")
    )


def _pi_session_sidecar_path(source_path: Path) -> Path:
    if source_path.is_dir():
        return source_path / ".halo-pi-session-spans.jsonl"
    return source_path.with_name(source_path.name + ".halo-pi-session-spans.jsonl")


def _first_json_object(path: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TraceSourceError(f"invalid JSON in {path}:{line_no}: {exc.msg}") from exc
            if isinstance(value, dict):
                return value
            raise TraceSourceError(f"expected JSON object in {path}:{line_no}")
    return None


def _looks_like_pi_entry(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict):
        return False
    entry_type = value.get("type")
    return isinstance(entry_type, str) and entry_type in _PI_ENTRY_TYPES


def _iter_pi_session_spans(
    session_file: Path,
    *,
    include_full_content: bool,
    excerpt_chars: int,
) -> list[SpanRecord]:
    spans: list[SpanRecord] = []
    file_id = _stable_file_id(session_file)
    session: dict[str, Any] = {
        "id": file_id,
        "cwd": None,
        "parentSession": None,
        "timestamp": None,
        "path": str(session_file),
    }

    with session_file.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TraceSourceError(
                    f"invalid JSON in {session_file}:{line_no}: {exc.msg}"
                ) from exc
            if not isinstance(entry, dict):
                raise TraceSourceError(f"expected JSON object in {session_file}:{line_no}")
            if not _looks_like_pi_entry(entry):
                raise TraceSourceError(
                    f"unsupported Pi session entry type in {session_file}:{line_no}: "
                    f"{entry.get('type')!r}"
                )

            if entry.get("type") == "session":
                session = {
                    "id": str(entry.get("id") or file_id),
                    "cwd": entry.get("cwd"),
                    "parentSession": entry.get("parentSession"),
                    "timestamp": entry.get("timestamp"),
                    "path": str(session_file),
                }

            spans.append(
                _entry_to_span(
                    entry=entry,
                    session=session,
                    session_file=session_file,
                    file_id=file_id,
                    include_full_content=include_full_content,
                    excerpt_chars=excerpt_chars,
                )
            )
    return spans


def _entry_to_span(
    *,
    entry: dict[str, Any],
    session: dict[str, Any],
    session_file: Path,
    file_id: str,
    include_full_content: bool,
    excerpt_chars: int,
) -> SpanRecord:
    entry_type = str(entry.get("type") or "unknown")
    raw_message = entry.get("message")
    message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
    role = message.get("role")
    timestamp = _entry_timestamp(entry, message)
    span_id = str(entry.get("id") or _stable_entry_id(session_file, entry))
    if entry_type == "session":
        span_id = f"session-{file_id[:16]}"
    parent_span_id = str(entry.get("parentId") or "")

    attributes = _base_attributes(entry=entry, session=session, session_file=session_file)
    attributes.update(
        _entry_attributes(
            entry=entry,
            message=message,
            include_full_content=include_full_content,
            excerpt_chars=excerpt_chars,
        )
    )
    _project_index_attributes(attributes, entry=entry, message=message)

    return SpanRecord(
        trace_id=str(session.get("id") or file_id),
        span_id=span_id,
        parent_span_id=parent_span_id,
        trace_state="",
        name=_span_name(entry_type=entry_type, role=role),
        kind="INTERNAL",
        start_time=timestamp,
        end_time=timestamp,
        status=_span_status(entry=entry, message=message),
        resource=SpanResource(
            attributes={
                "service.name": "pi-session",
                "pi.session.cwd": session.get("cwd") or "",
                "pi.session.source_file": str(session_file),
            }
        ),
        scope=SpanScope(name="halo.pi-session-adapter", version="1"),
        attributes=attributes,
    )


def _base_attributes(
    *,
    entry: dict[str, Any],
    session: dict[str, Any],
    session_file: Path,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "pi.source": "session_jsonl",
        "pi.session.id": session.get("id") or "",
        "pi.session.cwd": session.get("cwd") or "",
        "pi.session.source_file": str(session_file),
        "pi.entry.type": entry.get("type") or "",
    }
    if entry.get("id") is not None:
        attrs["pi.entry.id"] = str(entry.get("id"))
    if entry.get("parentId") is not None:
        attrs["pi.entry.parent_id"] = str(entry.get("parentId"))
    if session.get("parentSession"):
        attrs["pi.session.parent_session"] = str(session["parentSession"])
    return attrs


def _entry_attributes(
    *,
    entry: dict[str, Any],
    message: dict[str, Any],
    include_full_content: bool,
    excerpt_chars: int,
) -> dict[str, Any]:
    entry_type = str(entry.get("type") or "unknown")
    if entry_type == "message":
        return _message_attributes(
            message=message,
            include_full_content=include_full_content,
            excerpt_chars=excerpt_chars,
        )
    if entry_type == "session":
        return _copy_fields(entry, ["version", "cwd", "parentSession"], prefix="pi.session")
    if entry_type == "model_change":
        return _copy_fields(entry, ["provider", "modelId"], prefix="pi.model_change")
    if entry_type == "thinking_level_change":
        return _copy_fields(entry, ["thinkingLevel"], prefix="pi.thinking_level_change")
    if entry_type in {"compaction", "branch_summary"}:
        attrs = _copy_fields(
            entry,
            ["firstKeptEntryId", "tokensBefore", "fromId", "fromHook"],
            prefix=f"pi.{entry_type}",
        )
        _add_text_field(
            attrs,
            f"pi.{entry_type}.summary",
            entry.get("summary"),
            include_full_content,
            excerpt_chars,
        )
        if "details" in entry:
            _add_json_field(
                attrs,
                f"pi.{entry_type}.details",
                entry.get("details"),
                include_full_content,
                excerpt_chars,
            )
        return attrs
    if entry_type == "custom":
        attrs = _copy_fields(entry, ["customType"], prefix="pi.custom")
        if "data" in entry:
            _add_json_field(
                attrs, "pi.custom.data", entry.get("data"), include_full_content, excerpt_chars
            )
        return attrs
    if entry_type == "custom_message":
        attrs = _copy_fields(entry, ["customType", "display"], prefix="pi.custom_message")
        _add_content_attributes(
            attrs,
            entry.get("content"),
            text_key="pi.custom_message.content.text",
            include_full_content=include_full_content,
            excerpt_chars=excerpt_chars,
        )
        if "details" in entry:
            _add_json_field(
                attrs,
                "pi.custom_message.details",
                entry.get("details"),
                include_full_content,
                excerpt_chars,
            )
        return attrs
    if entry_type == "label":
        return _copy_fields(entry, ["targetId", "label"], prefix="pi.label")
    if entry_type == "session_info":
        return _copy_fields(entry, ["name"], prefix="pi.session_info")
    return {}


def _message_attributes(
    *,
    message: dict[str, Any],
    include_full_content: bool,
    excerpt_chars: int,
) -> dict[str, Any]:
    attrs = _copy_fields(
        message,
        [
            "role",
            "api",
            "provider",
            "model",
            "stopReason",
            "errorMessage",
            "toolCallId",
            "toolName",
            "isError",
            "exitCode",
            "cancelled",
            "truncated",
            "fullOutputPath",
            "excludeFromContext",
            "customType",
            "display",
            "fromId",
            "tokensBefore",
        ],
        prefix="pi.message",
    )
    role = message.get("role")
    if isinstance(role, str):
        attrs["inference.agent_name"] = f"pi.{role}"

    usage = message.get("usage")
    if isinstance(usage, dict):
        for key in ["input", "output", "cacheRead", "cacheWrite", "totalTokens"]:
            if isinstance(usage.get(key), int):
                attrs[f"pi.message.usage.{key}"] = usage[key]
        cost = usage.get("cost")
        if isinstance(cost, dict):
            for key in ["input", "output", "cacheRead", "cacheWrite", "total"]:
                if isinstance(cost.get(key), int | float):
                    attrs[f"pi.message.usage.cost.{key}"] = cost[key]

    _add_content_attributes(
        attrs,
        message.get("content"),
        text_key="pi.message.content.text",
        include_full_content=include_full_content,
        excerpt_chars=excerpt_chars,
    )

    if role == "bashExecution":
        _add_text_field(
            attrs,
            "pi.message.command",
            message.get("command"),
            include_full_content,
            excerpt_chars,
        )
        _add_text_field(
            attrs,
            "pi.message.output",
            message.get("output"),
            include_full_content,
            excerpt_chars,
        )
    elif role in {"branchSummary", "compactionSummary"}:
        _add_text_field(
            attrs,
            "pi.message.summary",
            message.get("summary"),
            include_full_content,
            excerpt_chars,
        )
    elif role == "custom":
        if "details" in message:
            _add_json_field(
                attrs,
                "pi.message.details",
                message.get("details"),
                include_full_content,
                excerpt_chars,
            )

    return attrs


def _add_content_attributes(
    attrs: dict[str, Any],
    content: Any,
    *,
    text_key: str,
    include_full_content: bool,
    excerpt_chars: int,
) -> None:
    if isinstance(content, str):
        _add_text_field(
            attrs,
            text_key,
            content,
            include_full_content,
            excerpt_chars,
        )
    elif isinstance(content, list):
        attrs.update(
            _content_block_attributes(
                content,
                include_full_content=include_full_content,
                excerpt_chars=excerpt_chars,
            )
        )


def _content_block_attributes(
    blocks: list[Any],
    *,
    include_full_content: bool,
    excerpt_chars: int,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    block_types: list[str] = []
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    image_count = 0

    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if isinstance(block_type, str):
            block_types.append(block_type)
        if block_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block_type == "thinking" and isinstance(block.get("thinking"), str):
            thinking_parts.append(block["thinking"])
        elif block_type == "toolCall":
            tool_call = {
                "id": block.get("id"),
                "name": block.get("name"),
                "arguments": block.get("arguments")
                if isinstance(block.get("arguments"), dict)
                else {},
            }
            tool_calls.append(tool_call)
        elif block_type == "image":
            image_count += 1

    attrs["pi.content.block_types"] = block_types
    attrs["pi.content.block_count"] = len(blocks)
    if image_count:
        attrs["pi.content.image_count"] = image_count

    if text_parts:
        _add_text_field(
            attrs,
            "pi.content.text",
            "\n".join(text_parts),
            include_full_content,
            excerpt_chars,
        )
    if thinking_parts:
        _add_text_field(
            attrs,
            "pi.content.thinking",
            "\n".join(thinking_parts),
            include_full_content,
            excerpt_chars,
        )
    if tool_calls:
        attrs["pi.tool_call.count"] = len(tool_calls)
        attrs["pi.tool_call.names"] = [tc.get("name") for tc in tool_calls if tc.get("name")]
        attrs["pi.tool_call.ids"] = [tc.get("id") for tc in tool_calls if tc.get("id")]
        attrs["pi.tool_call.argument_keys"] = {
            str(tc.get("name") or tc.get("id") or i): sorted(tc["arguments"].keys())
            for i, tc in enumerate(tool_calls)
        }
        _add_json_field(
            attrs,
            "pi.tool_call.arguments",
            [
                {"id": tc.get("id"), "name": tc.get("name"), "arguments": tc["arguments"]}
                for tc in tool_calls
            ],
            include_full_content,
            excerpt_chars,
        )
    return attrs


def _copy_fields(value: dict[str, Any], field_names: list[str], *, prefix: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for field_name in field_names:
        if field_name in value and _json_safe_scalar(value[field_name]):
            attrs[f"{prefix}.{field_name}"] = value[field_name]
    return attrs


def _json_safe_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _add_text_field(
    attrs: dict[str, Any],
    key: str,
    value: Any,
    include_full_content: bool,
    excerpt_chars: int,
) -> None:
    if not isinstance(value, str):
        return
    attrs[f"{key}.char_count"] = len(value)
    if include_full_content:
        attrs[key] = value
    else:
        attrs[f"{key}.excerpt"] = _excerpt(value, excerpt_chars)


def _add_json_field(
    attrs: dict[str, Any],
    key: str,
    value: Any,
    include_full_content: bool,
    excerpt_chars: int,
) -> None:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    attrs[f"{key}.json_char_count"] = len(serialized)
    if include_full_content:
        attrs[key] = value
    else:
        attrs[f"{key}.excerpt"] = _excerpt(serialized, excerpt_chars)


def _excerpt(value: str, excerpt_chars: int) -> str:
    if len(value) <= excerpt_chars:
        return value
    return f"{value[:excerpt_chars]}... [HALO pi-session excerpt: original {len(value)} chars]"


def _project_index_attributes(
    attrs: dict[str, Any], *, entry: dict[str, Any], message: dict[str, Any]
) -> None:
    attrs["inference.project_id"] = "pi-session"
    model = message.get("model") or entry.get("modelId")
    if isinstance(model, str) and model:
        attrs["inference.llm.model_name"] = model
        attrs["llm.model_name"] = model
    usage = message.get("usage")
    if isinstance(usage, dict):
        input_tokens = usage.get("input")
        output_tokens = usage.get("output")
        if isinstance(input_tokens, int):
            attrs["inference.llm.input_tokens"] = input_tokens
        if isinstance(output_tokens, int):
            attrs["inference.llm.output_tokens"] = output_tokens


def _span_status(*, entry: dict[str, Any], message: dict[str, Any]) -> SpanStatus:
    entry_type = entry.get("type")
    if entry_type != "message":
        return SpanStatus(code="STATUS_CODE_OK", message="")

    role = message.get("role")
    if role == "assistant" and (
        message.get("stopReason") == "error" or message.get("errorMessage")
    ):
        return SpanStatus(
            code="STATUS_CODE_ERROR",
            message=str(
                message.get("errorMessage") or message.get("stopReason") or "assistant error"
            ),
        )
    if role == "toolResult" and message.get("isError") is True:
        return SpanStatus(code="STATUS_CODE_ERROR", message="tool result marked as error")
    if role == "bashExecution":
        if message.get("cancelled") is True:
            return SpanStatus(code="STATUS_CODE_ERROR", message="bash execution cancelled")
        exit_code = message.get("exitCode")
        if isinstance(exit_code, int) and exit_code != 0:
            return SpanStatus(code="STATUS_CODE_ERROR", message=f"bash exited with {exit_code}")
    return SpanStatus(code="STATUS_CODE_OK", message="")


def _span_name(*, entry_type: str, role: Any) -> str:
    if entry_type == "message":
        role_name = role if isinstance(role, str) and role else "unknown"
        return f"message.{role_name}"
    return entry_type


def _entry_timestamp(entry: dict[str, Any], message: dict[str, Any]) -> str:
    for value in (entry.get("timestamp"), message.get("timestamp")):
        normalized = _normalize_timestamp(value)
        if normalized:
            return normalized
    return "1970-01-01T00:00:00.000Z"


def _normalize_timestamp(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int | float):
        dt = datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return None


def _stable_file_id(path: Path) -> str:
    return sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:32]


def _stable_entry_id(path: Path, entry: dict[str, Any]) -> str:
    blob = json.dumps(entry, ensure_ascii=False, sort_keys=True)
    return sha256(f"{path.resolve()}\n{blob}".encode("utf-8")).hexdigest()[:16]
