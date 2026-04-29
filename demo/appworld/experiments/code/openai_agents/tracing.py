"""Wire the openai-agents SDK to write inference.net-format JSONL traces.

This is the single integration file for HALO. Drop it into your project, call
``setup_tracing()`` once at startup before constructing any ``Agent``, and every
agent / LLM / tool span emitted by the SDK will be appended to a JSONL
file in the inference.net OTLP-shaped export format that the HALO Engine reads.

Usage:

    from agents import Agent, Runner
    from tracing import setup_tracing

    processor = setup_tracing(service_name="my-agent", project_id="my-project")
    agent = Agent(name="assistant", instructions="Be helpful.")
    Runner.run_sync(agent, "Hello")
    processor.shutdown()  # flush the file

The processor is *additive* — the default OpenAI-dashboard processor still runs
unless disabled. Set ``OPENAI_AGENTS_DISABLE_TRACING=1`` or call
``agents.set_trace_processors([])`` before ``setup_tracing()`` to suppress it.

Reference: inference.net export spec ``07-export.md`` and the conversion notes
in ``openai-agents-sdk-span-conversion.md``.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from agents import add_trace_processor
from agents.tracing import Span, Trace
from agents.tracing.processor_interface import TracingProcessor

# ---------------------------------------------------------------------------
# Constants — keep in sync with the spec in 07-export.md
# ---------------------------------------------------------------------------

EXPORT_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_PATH = "traces.jsonl"

# observation_kind vocabulary we emit. Pick from the same enum the router
# uses server-side so the projection is stable across sources.
OBSERVATION_KIND_BY_TYPE: dict[str, str] = {
    "agent":        "AGENT",
    "generation":   "LLM",
    "response":     "LLM",
    "function":     "TOOL",
    "mcp_tools":    "TOOL",
    "handoff":      "CHAIN",
    "guardrail":    "GUARDRAIL",
    "custom":       "SPAN",
    "task":         "SPAN",
    "turn":         "SPAN",
    "transcription": "SPAN",
    "speech":       "SPAN",
    "speech_group": "SPAN",
}


# ---------------------------------------------------------------------------
# Context you attach at export time (the SDK can't know any of this).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExportContext:
    """Per-process identity stamped onto every exported span.

    These are the fields the router normally fills server-side from the
    ingest edge. When you're exporting directly from the SDK (no edge
    proxy in the path) you have to stamp them yourself.
    """
    project_id: str
    service_name: str
    service_version: str | None = None
    deployment_environment: str | None = None
    # Optional OpenTelemetry-style resource attributes merged into
    # resource.attributes verbatim.
    extra_resource_attributes: Mapping[str, Any] | None = None


# ---------------------------------------------------------------------------
# Pure conversion: one SDK span -> one dict shaped like our JSONL line.
# ---------------------------------------------------------------------------

def span_to_otlp_line(
    span: Span[Any],
    *,
    ctx: ExportContext,
    workflow_name: str | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    """Convert an openai-agents Span into one JSONL line dict.

    `workflow_name` and `group_id` come from the parent Trace — the SDK
    gives you those on `on_trace_start`; stash them keyed by trace_id and
    look them up here. See `InferenceOtlpFileProcessor` below for the
    wiring.
    """
    raw = span.export() or {}
    span_data = raw.get("span_data") or {}
    span_type = str(span_data.get("type") or "custom")

    # --- identifiers ------------------------------------------------------
    trace_id = _strip_prefix(raw.get("trace_id"), "trace_")
    span_id = _strip_prefix(raw.get("id"), "span_")
    parent_span_id = _strip_prefix(raw.get("parent_id"), "span_") or ""

    # --- timestamps -------------------------------------------------------
    start_time = _to_otlp_timestamp(raw.get("started_at"))
    end_time = _to_otlp_timestamp(raw.get("ended_at"))

    # --- status -----------------------------------------------------------
    error = raw.get("error")
    if error:
        status = {
            "code": "STATUS_CODE_ERROR",
            "message": str(error.get("message") or ""),
        }
    else:
        status = {"code": "STATUS_CODE_OK", "message": ""}

    # --- resource ---------------------------------------------------------
    resource_attributes: dict[str, Any] = {
        "service.name": ctx.service_name,
    }
    if ctx.service_version:
        resource_attributes["service.version"] = ctx.service_version
    if ctx.deployment_environment:
        resource_attributes["deployment.environment"] = ctx.deployment_environment
    if ctx.extra_resource_attributes:
        resource_attributes.update(ctx.extra_resource_attributes)

    # --- scope ------------------------------------------------------------
    scope = {
        "name": "openai-agents-sdk",
        "version": _sdk_version(),
    }

    # --- raw upstream attributes + inference.* projection -----------------
    attributes, projection = _attributes_for_span_type(span_type, span_data)

    # Trace-level context is useful for grouping; put it on every span.
    if workflow_name:
        attributes.setdefault("agent.workflow.name", workflow_name)
    if group_id:
        attributes.setdefault("agent.workflow.group_id", group_id)

    # inference.* projections — ALWAYS present per 07-export.md.
    attributes.update({
        "inference.export.schema_version": EXPORT_SCHEMA_VERSION,
        "inference.project_id": ctx.project_id,
        "inference.observation_kind": OBSERVATION_KIND_BY_TYPE.get(span_type, "SPAN"),
        "inference.llm.provider": projection.get("llm_provider"),
        "inference.llm.model_name": projection.get("llm_model_name"),
        "inference.llm.input_tokens": projection.get("input_tokens"),
        "inference.llm.output_tokens": projection.get("output_tokens"),
        "inference.llm.cost.total": projection.get("cost_total"),   # we don't know cost client-side
        "inference.user_id": projection.get("user_id"),
        "inference.session_id": group_id,  # SDK's group_id is the closest analogue
        "inference.agent_name": projection.get("agent_name") or "",
    })

    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "trace_state": "",
        "name": _span_name(span_type, span_data),
        "kind": _span_kind(span_type),
        "start_time": start_time,
        "end_time": end_time,
        "status": status,
        "resource": {"attributes": resource_attributes},
        "scope": scope,
        "attributes": attributes,
    }


# ---------------------------------------------------------------------------
# Per-span-type mapping. Each branch returns:
#   (raw_attributes_dict, projection_dict_for_inference_star)
# ---------------------------------------------------------------------------

def _attributes_for_span_type(
    span_type: str, d: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if span_type == "agent":
        return _agent_attrs(d)
    if span_type == "generation":
        return _generation_attrs(d)
    if span_type == "response":
        return _response_attrs(d)
    if span_type == "function":
        return _function_attrs(d)
    if span_type == "mcp_tools":
        return _mcp_tools_attrs(d)
    if span_type == "handoff":
        return _handoff_attrs(d)
    if span_type == "guardrail":
        return _guardrail_attrs(d)
    # custom / task / turn / transcription / speech / speech_group /
    # anything new the SDK adds in the future:
    return _custom_attrs(span_type, d)


def _agent_attrs(d: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    name = d.get("name") or ""
    attrs = {
        "openinference.span.kind": "AGENT",
        "agent.name": name,
        "agent.handoffs": _json(d.get("handoffs")),
        "agent.tools": _json(d.get("tools")),
        "agent.output_type": d.get("output_type"),
    }
    return _drop_none(attrs), {"agent_name": name}


def _generation_attrs(d: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = d.get("model")
    usage = d.get("usage") or {}
    input_msgs = d.get("input") or []
    output_msgs = d.get("output") or []

    attrs: dict[str, Any] = {
        "openinference.span.kind": "LLM",
        "llm.provider": "openai",
        "llm.model_name": model,
        "llm.invocation_parameters": _json(d.get("model_config")),
        "llm.input_messages":  _json(list(input_msgs)),
        "llm.output_messages": _json(list(output_msgs)),
        "llm.token_count.prompt":     _int(usage.get("input_tokens") or usage.get("prompt_tokens")),
        "llm.token_count.completion": _int(usage.get("output_tokens") or usage.get("completion_tokens")),
        "llm.token_count.total":      _int(usage.get("total_tokens")),
    }

    # Expand input/output into the flat OpenInference .N.message.* keys so
    # Phoenix / Arize-style viewers get a native read.
    attrs.update(_expand_messages("llm.input_messages",  input_msgs))
    attrs.update(_expand_messages("llm.output_messages", output_msgs))

    projection = {
        "llm_provider": "openai",
        "llm_model_name": model,
        "input_tokens":  _int(usage.get("input_tokens") or usage.get("prompt_tokens")),
        "output_tokens": _int(usage.get("output_tokens") or usage.get("completion_tokens")),
    }
    return _drop_none(attrs), projection


def _response_attrs(d: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    # `response` spans export only response_id + usage by default; the
    # full Response object isn't in .export() output. If you need the
    # body, capture it from the OpenAI client side-channel.
    usage = d.get("usage") or {}
    attrs = {
        "openinference.span.kind": "LLM",
        "llm.provider": "openai",
        "llm.response.id": d.get("response_id"),
        "llm.token_count.prompt":     _int(usage.get("input_tokens") or usage.get("prompt_tokens")),
        "llm.token_count.completion": _int(usage.get("output_tokens") or usage.get("completion_tokens")),
        "llm.token_count.total":      _int(usage.get("total_tokens")),
    }
    projection = {
        "llm_provider": "openai",
        "input_tokens":  _int(usage.get("input_tokens") or usage.get("prompt_tokens")),
        "output_tokens": _int(usage.get("output_tokens") or usage.get("completion_tokens")),
    }
    return _drop_none(attrs), projection


def _function_attrs(d: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    attrs = {
        "openinference.span.kind": "TOOL",
        "tool.name": d.get("name"),
        "input.value":  d.get("input"),
        "output.value": d.get("output"),
        "mcp.data": _json(d.get("mcp_data")),
    }
    return _drop_none(attrs), {}


def _mcp_tools_attrs(d: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    attrs = {
        "openinference.span.kind": "TOOL",
        "mcp.server": d.get("server"),
        "mcp.tools.listed": _json(d.get("result")),
    }
    return _drop_none(attrs), {}


def _handoff_attrs(d: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    attrs = {
        "openinference.span.kind": "CHAIN",
        "agent.handoff.from": d.get("from_agent"),
        "agent.handoff.to":   d.get("to_agent"),
    }
    return _drop_none(attrs), {"agent_name": d.get("to_agent")}


def _guardrail_attrs(d: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    attrs = {
        "openinference.span.kind": "GUARDRAIL",
        "guardrail.name": d.get("name"),
        "guardrail.triggered": bool(d.get("triggered")),
    }
    return _drop_none(attrs), {}


def _custom_attrs(span_type: str, d: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    attrs: dict[str, Any] = {
        "openinference.span.kind": "CHAIN",
        "sdk.span.type": span_type,
    }
    if "name" in d:
        attrs["sdk.span.name"] = d.get("name")
    # Flatten `data` one level: `foo.bar` keys are friendlier than a
    # blob string for most consumers.
    data = d.get("data") or {}
    for k, v in data.items():
        attrs[f"sdk.data.{k}"] = v if _json_safe(v) else _json(v)
    if "usage" in d:
        attrs["llm.token_count.total"] = _int((d.get("usage") or {}).get("total_tokens"))
    return _drop_none(attrs), {}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _strip_prefix(value: Any, prefix: str) -> str | None:
    if not value:
        return None
    s = str(value)
    return s[len(prefix):] if s.startswith(prefix) else s


def _to_otlp_timestamp(iso_str: str | None) -> str:
    """SDK emits datetime.isoformat() with microseconds; our contract
    wants ISO-8601 with *nanosecond* precision and a trailing Z."""
    if not iso_str:
        return ""
    # SDK default is `datetime.now(timezone.utc).isoformat()` which ends
    # in "+00:00". datetime.fromisoformat handles it.
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    # microseconds -> nanoseconds by right-padding three zeros.
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond:06d}000Z"


def _span_kind(span_type: str) -> str:
    # openai-agents doesn't carry OTLP SpanKind. LLM-ish spans are the
    # most obvious CLIENT calls; everything else is INTERNAL.
    if span_type in ("generation", "response"):
        return "SPAN_KIND_CLIENT"
    return "SPAN_KIND_INTERNAL"


def _span_name(span_type: str, d: Mapping[str, Any]) -> str:
    name = d.get("name")
    if name:
        return f"{span_type}.{name}"
    model = d.get("model")
    if model:
        return f"{span_type}.{model}"
    return span_type


def _expand_messages(prefix: str, messages: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, msg in enumerate(messages or []):
        if not isinstance(msg, Mapping):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role is not None:
            out[f"{prefix}.{i}.message.role"] = role
        if isinstance(content, str):
            out[f"{prefix}.{i}.message.content"] = content
        elif content is not None:
            out[f"{prefix}.{i}.message.content"] = _json(content)
        # tool call fan-out
        for j, tc in enumerate(msg.get("tool_calls") or []):
            if not tc:
                continue
            fn = tc.get("function") or {}
            out[f"{prefix}.{i}.message.tool_calls.{j}.tool_call.id"] = tc.get("id")
            out[f"{prefix}.{i}.message.tool_calls.{j}.tool_call.function.name"] = fn.get("name")
            out[f"{prefix}.{i}.message.tool_calls.{j}.tool_call.function.arguments"] = fn.get("arguments")
        if msg.get("tool_call_id"):
            out[f"{prefix}.{i}.message.tool_call_id"] = msg["tool_call_id"]
        if msg.get("name"):
            out[f"{prefix}.{i}.message.name"] = msg["name"]
    return {k: v for k, v in out.items() if v is not None}


def _json(v: Any) -> str | None:
    if v is None:
        return None
    try:
        return json.dumps(v, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return json.dumps(str(v))


def _json_safe(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def _int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _drop_none(d: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _sdk_version() -> str:
    try:
        from importlib.metadata import version
        return version("openai-agents")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# TracingProcessor — plugs the converter into the SDK.
# ---------------------------------------------------------------------------

class InferenceOtlpFileProcessor(TracingProcessor):
    """Append-only JSONL writer, one line per span, spec-compliant with
    07-export.md. Safe to use in dev / evals / one-off exports. For
    production ingest you'd replace the file sink with an HTTP POST to
    your edge proxy — the `span_to_otlp_line` call is the same."""

    def __init__(self, path: str, *, ctx: ExportContext):
        self._path = path
        self._ctx = ctx
        self._lock = threading.Lock()
        self._fh = open(path, mode="a", encoding="utf-8")
        self._trace_meta: dict[str, tuple[str | None, str | None]] = {}

    # ----- Trace lifecycle ------------------------------------------------

    def on_trace_start(self, trace: Trace) -> None:
        # Export() is an option on Trace too; we only need the workflow
        # name + group id to stamp onto each span.
        data = trace.export() or {}
        tid = _strip_prefix(data.get("id"), "trace_") or ""
        self._trace_meta[tid] = (data.get("workflow_name"), data.get("group_id"))

    def on_trace_end(self, trace: Trace) -> None:
        data = trace.export() or {}
        tid = _strip_prefix(data.get("id"), "trace_") or ""
        self._trace_meta.pop(tid, None)

    # ----- Span lifecycle -------------------------------------------------

    def on_span_start(self, span: Span[Any]) -> None:
        # We only write on_span_end so the line carries ended_at + status.
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        exported = span.export() or {}
        tid = _strip_prefix(exported.get("trace_id"), "trace_") or ""
        workflow_name, group_id = self._trace_meta.get(tid, (None, None))
        line = span_to_otlp_line(
            span,
            ctx=self._ctx,
            workflow_name=workflow_name,
            group_id=group_id,
        )
        encoded = json.dumps(line, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            self._fh.write(encoded)
            self._fh.write("\n")

    # ----- Shutdown -------------------------------------------------------

    def shutdown(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass

    def force_flush(self) -> None:
        with self._lock:
            self._fh.flush()


# ---------------------------------------------------------------------------
# One-call wiring — what most users will import.
# ---------------------------------------------------------------------------

def setup_tracing(
    service_name: str = "my-agent",
    project_id: str = "my-project",
) -> InferenceOtlpFileProcessor:
    """Construct an `InferenceOtlpFileProcessor` and register it with the SDK.

    Output path defaults to ``./traces.jsonl``; override with the
    ``HALO_TRACES_PATH`` env var. Call ``processor.shutdown()`` before exit
    to flush the file.
    """
    path = os.getenv("HALO_TRACES_PATH", DEFAULT_OUTPUT_PATH)
    ctx = ExportContext(project_id=project_id, service_name=service_name)
    processor = InferenceOtlpFileProcessor(path, ctx=ctx)
    add_trace_processor(processor)
    return processor
