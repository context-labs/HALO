"""TraceView over Claude Code native OTel span trees.

Claude Code (the CLI subprocess the Claude Agent SDK spawns) emits traces
under its own ``claude_code.*`` span namespace. The shape is documented
at https://docs.claude.com/en/docs/claude-code/monitoring-usage ·
"Traces (beta)" section. Concretely:

::

    [openinference AGENT wrapper]             (optional, from the Python side)
    └── claude_code.interaction               user_prompt, interaction.sequence
        ├── claude_code.llm_request           model, *_tokens, request_id
        │     └─ event gen_ai.request.attempt
        ├── claude_code.tool                  tool_name, file_path, full_command
        │     ├─ claude_code.tool.blocked_on_user   decision, source
        │     ├─ claude_code.tool.execution         success, error
        │     └─ event tool.output                  content/output/bash_command/file_path
        └── ... more llm_request / tool cycles

Content-bearing attributes are gated by producer env vars:
``OTEL_LOG_USER_PROMPTS`` for ``user_prompt``, ``OTEL_LOG_TOOL_DETAILS``
for ``file_path`` / ``full_command`` etc., ``OTEL_LOG_TOOL_CONTENT``
for the ``tool.output`` event. When those gates are off the producer
writes ``<REDACTED>`` or omits the attribute; this view handles both
(redacted values are treated as missing).

Claude Code does **not** emit assistant text as span attributes. The only
way to recover per-turn assistant messages is ``OTEL_LOG_RAW_API_BODIES``,
which writes OTel *log events* (separate signal) — not in scope here.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from dataset.descriptor import ClaudeCodeMapping, DatasetDescriptor
from dataset.formats.openinference import _attrs_to_dict
from dataset.formats.trace_view import (
    DocumentView,
    MessageView,
    ToolCallView,
    TraceView,
)


# Producer sentinels that mean "content would be here if the gate were on".
_REDACTED_TOKENS = frozenset({"<REDACTED>", "REDACTED"})


def _clean_str(v: Any) -> str:
    """Return ``v`` as a string, dropping producer-side redaction markers."""
    if v is None:
        return ""
    if isinstance(v, str):
        return "" if v.strip() in _REDACTED_TOKENS else v
    try:
        return json.dumps(v)
    except (TypeError, ValueError):
        return str(v)


def _event_attrs(event: dict | None) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {}
    return _attrs_to_dict(event.get("attributes"))


# Keys we treat as *tool input* material when they ride on a ``tool.output``
# event or on the ``claude_code.tool`` span itself. Kept deliberately small
# and explicit — Claude Code's schema here is stable per the monitoring
# docs.
_TOOL_INPUT_KEYS = ("full_command", "bash_command", "file_path", "command")
# Keys that carry the tool's textual output body.
_TOOL_OUTPUT_KEYS = ("content", "output", "stdout", "stderr")


def _tool_arguments(span_attrs: dict[str, Any],
                    tool_output_event: dict[str, Any]) -> str:
    """JSON-encoded tool arguments derived from span attrs + tool.output event.

    We merge whatever input-y keys are present without inventing names; if
    nothing is available (gates off) the result is ``""``.
    """
    merged: dict[str, Any] = {}
    for key in _TOOL_INPUT_KEYS:
        for source in (tool_output_event, span_attrs):
            v = source.get(key)
            if v is None:
                continue
            sv = _clean_str(v)
            if sv and key not in merged:
                merged[key] = sv
    if not merged:
        return ""
    return json.dumps(merged, ensure_ascii=False)


def _tool_result(tool_output_event: dict[str, Any]) -> str | None:
    """Pick the most useful text body out of a ``tool.output`` event."""
    for key in _TOOL_OUTPUT_KEYS:
        v = tool_output_event.get(key)
        if v is None:
            continue
        sv = _clean_str(v)
        if sv:
            return sv
    if tool_output_event:
        # Fall back to a JSON dump of whatever did land in the event — still
        # more informative than nothing.
        try:
            return json.dumps(tool_output_event, ensure_ascii=False)
        except (TypeError, ValueError):
            return None
    return None


class ClaudeCodeTraceView(TraceView):
    """Walk a Claude Code native OTel span tree."""

    # --- construction -------------------------------------------------

    def __init__(self, record: dict, descriptor: DatasetDescriptor):
        assert isinstance(descriptor.mapping, ClaudeCodeMapping), (
            "ClaudeCodeTraceView requires a descriptor with ClaudeCodeMapping"
        )
        self._rec = record
        self._desc = descriptor
        self._m: ClaudeCodeMapping = descriptor.mapping

        spans: list[tuple[dict, dict[str, Any]]] = []
        for s in record.get("spans") or []:
            if not isinstance(s, dict):
                continue
            spans.append((s, _attrs_to_dict(s.get("attributes"))))
        self._spans = spans

        # Group spans by name — Claude Code's schema makes span name the
        # reliable classifier (``span.type`` is a duplicate of the name, per
        # the monitoring docs).
        by_name: dict[str, list[tuple[dict, dict[str, Any]]]] = {}
        for s, a in spans:
            by_name.setdefault(str(s.get("name") or ""), []).append((s, a))
        self._by_name = by_name

        # Root detection: first span whose parent is absent or external.
        own_ids = {str(s.get("spanId") or "") for s, _ in spans}
        self._root: tuple[dict, dict[str, Any]] | None = None
        for s, a in spans:
            p = s.get("parentSpanId")
            if not p or p not in own_ids:
                self._root = (s, a)
                break

        # Primary interaction span (first, by start time). When the
        # producer wraps the trace in an OI AGENT root, the interaction
        # is a child of that root; otherwise it *is* the root.
        interactions = self._by_name.get("claude_code.interaction", [])
        interactions.sort(
            key=lambda sa: int(sa[0].get("startTimeUnixNano") or 0)
        )
        self._interaction = interactions[0] if interactions else None

    # --- helpers ------------------------------------------------------

    def _lookup_attribute(self, key: str) -> Any | None:
        """Read a custom attribute — root first, then any span."""
        if not key:
            return None
        if self._root is not None:
            v = self._root[1].get(key)
            if v is not None:
                return v
        for _, a in self._spans:
            if key in a:
                return a[key]
        return None

    def _start_time(self, span: dict) -> int:
        return int(span.get("startTimeUnixNano") or 0)

    # --- identity / semantics ----------------------------------------

    @property
    def id(self) -> str:
        key = self._m.id_attribute
        if key:
            v = self._lookup_attribute(key)
            if v is not None:
                return str(v)
        return str(self._rec.get("traceId") or "?")

    @property
    def query(self) -> str:
        """``user_prompt`` on the interaction span, when the producer has
        ``OTEL_LOG_USER_PROMPTS=1`` set. Falls back to an OI-style
        ``input.value`` on the root if present, else empty."""
        if self._interaction is not None:
            v = _clean_str(self._interaction[1].get("user_prompt"))
            if v:
                return v
        if self._root is not None:
            v = _clean_str(self._root[1].get("input.value"))
            if v:
                return v
        return ""

    @property
    def final_answer(self) -> Any | None:
        """Claude Code's stable span schema carries no assistant text, so
        there's no in-trace final answer to surface. A downstream producer
        that merges ``api_response_body`` log events onto the trace could
        populate ``output.value`` on the root; we honor that if present."""
        if self._root is not None:
            ra = self._root[1]
            if "output.value" in ra:
                ov = _clean_str(ra["output.value"])
                return ov or None
        return None

    @property
    def ground_truth(self) -> Any | None:
        return self._lookup_attribute(self._desc.ground_truth_source or "")

    def metric_value(self, name: str) -> float | None:
        metric = self._desc.metric(name)
        if metric is None:
            return None
        v = self._lookup_attribute(metric.source)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @property
    def outcome(self) -> float | None:
        if self._desc.primary_metric is None:
            return None
        return self.metric_value(self._desc.primary_metric.name)

    @property
    def labels(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for lbl in self._desc.labels:
            v = self._lookup_attribute(lbl.source)
            if v is not None:
                out[lbl.name] = str(v)
        return out

    # --- trajectory --------------------------------------------------

    def _ordered_tool_spans(self) -> list[tuple[dict, dict[str, Any]]]:
        tools = list(self._by_name.get("claude_code.tool", []))
        tools.sort(key=lambda sa: self._start_time(sa[0]))
        return tools

    def _tool_output_event(self, span: dict) -> dict[str, Any]:
        for ev in span.get("events") or []:
            if isinstance(ev, dict) and ev.get("name") == "tool.output":
                return _event_attrs(ev)
        return {}

    def _tool_call_for_span(
        self,
        span: dict,
        span_attrs: dict[str, Any],
    ) -> ToolCallView:
        event_attrs = self._tool_output_event(span)
        return ToolCallView(
            id=str(span.get("spanId") or "") or None,
            name=str(span_attrs.get("tool_name") or "").strip(),
            arguments=_tool_arguments(span_attrs, event_attrs),
            result=_tool_result(event_attrs),
        )

    def messages(self) -> Iterator[MessageView]:
        """Synthesize an ordered user → (tool_call / tool_result)* trajectory.

        Claude Code doesn't emit assistant text in the stable span schema;
        we therefore don't fabricate assistant content. Consumers that only
        care about tool usage (the indexer's ``tools_used`` + tool count,
        the analyst's trace walk) still get a faithful sequence."""
        # 1. One user turn at the top, if a query is present.
        q = self.query
        if q:
            yield MessageView(role="user", content=q)

        # 2. One assistant turn per ``claude_code.tool`` span, each paired
        #    with a tool-result message. Tools are yielded in trace start
        #    order so the sequence mirrors the real interaction.
        for span, attrs in self._ordered_tool_spans():
            tc = self._tool_call_for_span(span, attrs)
            if not tc.name:
                continue
            yield MessageView(
                role="assistant",
                content="",
                tool_calls=[tc],
            )
            if tc.result is not None:
                yield MessageView(
                    role="tool",
                    content=tc.result,
                    tool_call_id=tc.id,
                )

    def tool_calls(self) -> Iterator[ToolCallView]:
        for span, attrs in self._ordered_tool_spans():
            tc = self._tool_call_for_span(span, attrs)
            if tc.name:
                yield tc

    # --- auxiliary ----------------------------------------------------

    def documents(self) -> list[DocumentView]:
        return []

    @property
    def turns_used(self) -> int | None:
        n = len(self._by_name.get("claude_code.llm_request", []))
        return n or None

    @property
    def tool_errors(self) -> int:
        n = 0
        for s, a in self._by_name.get("claude_code.tool.execution", []):
            status = (s.get("status") or {}).get("code")
            if status == "STATUS_CODE_ERROR":
                n += 1
                continue
            success = a.get("success")
            if success is False or (isinstance(success, str)
                                    and success.lower() == "false"):
                n += 1
        return n

    @property
    def tool_calls_total(self) -> int:
        return len(self._by_name.get("claude_code.tool", []))

    @property
    def usage(self) -> dict[str, int]:
        prompt = completion = cache_read = cache_create = 0
        for _, a in self._by_name.get("claude_code.llm_request", []):
            for key, bucket in (
                ("input_tokens", "prompt"),
                ("output_tokens", "completion"),
                ("cache_read_tokens", "cache_read"),
                ("cache_creation_tokens", "cache_create"),
            ):
                v = a.get(key)
                if v is None:
                    continue
                try:
                    n = int(v)
                except (TypeError, ValueError):
                    continue
                if bucket == "prompt":
                    prompt += n
                elif bucket == "completion":
                    completion += n
                elif bucket == "cache_read":
                    cache_read += n
                else:
                    cache_create += n
        out: dict[str, int] = {}
        if prompt:
            out["prompt_tokens"] = prompt
        if completion:
            out["completion_tokens"] = completion
        total = prompt + completion
        if total:
            out["total_tokens"] = total
        # Surface Anthropic-style cache counters too — consumers that don't
        # care just ignore the extra keys.
        if cache_read:
            out["cache_read_tokens"] = cache_read
        if cache_create:
            out["cache_creation_tokens"] = cache_create
        return out
