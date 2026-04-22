"""TraceView over OpenInference-shaped OTLP span trees.

Canonical attribute names come from Arize-ai/openinference's semantic
conventions. Each record is one trace (one span tree) in the shape the
``otel-interceptor`` ``compact:traces`` step emits.

For descriptor-driven fields (metric values, ground truth, labels), the
``source`` string on each :class:`Metric` / :class:`Label` is treated
as a span *attribute key* and looked up on the root span first, then on
any span in the trace (first match wins).
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from dataset.descriptor import (
    DatasetDescriptor,
    OpenInferenceMapping,
)
from dataset.formats.trace_view import (
    DocumentView,
    MessageView,
    ToolCallView,
    TraceView,
)


# ----------------------------------------------------------------------
# Attribute decoding
# ----------------------------------------------------------------------

def _attr_value(v: dict) -> Any:
    """Decode an OTLP ``AnyValue``."""
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        iv = v["intValue"]
        try:
            return int(iv)
        except (TypeError, ValueError):
            return iv
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "doubleValue" in v:
        try:
            return float(v["doubleValue"])
        except (TypeError, ValueError):
            return v["doubleValue"]
    if "arrayValue" in v:
        arr = (v["arrayValue"] or {}).get("values") or []
        return [_attr_value(x) for x in arr]
    if "kvlistValue" in v:
        kvs = (v["kvlistValue"] or {}).get("values") or []
        return {kv.get("key"): _attr_value(kv.get("value") or {}) for kv in kvs}
    return None


def _attrs_to_dict(attrs: list[dict] | None) -> dict[str, Any]:
    if not attrs:
        return {}
    out: dict[str, Any] = {}
    for kv in attrs:
        if not isinstance(kv, dict):
            continue
        k = kv.get("key")
        if not k:
            continue
        out[k] = _attr_value(kv.get("value") or {})
    return out


# ----------------------------------------------------------------------
# Message extraction
# ----------------------------------------------------------------------

def _indexed_groups(attrs: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    """Collect attributes of the form ``<prefix>.N.<...>`` into per-index dicts."""
    pfx = prefix + "."
    groups: dict[int, dict[str, Any]] = {}
    for k, v in attrs.items():
        if not k.startswith(pfx):
            continue
        rest = k[len(pfx):]
        idx_str, _, tail = rest.partition(".")
        if not idx_str.isdigit() or not tail:
            continue
        groups.setdefault(int(idx_str), {})[tail] = v
    return [groups[i] for i in sorted(groups)]


def _extract_messages(attrs: dict[str, Any], prefix: str) -> list[MessageView]:
    out: list[MessageView] = []
    for g in _indexed_groups(attrs, prefix):
        role = str(g.get("message.role") or "")
        content = g.get("message.content") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        tcs: list[ToolCallView] = []
        for tg in _indexed_groups(g, "message.tool_calls"):
            tcs.append(ToolCallView(
                id=tg.get("tool_call.id"),
                name=str(tg.get("tool_call.function.name") or ""),
                arguments=str(tg.get("tool_call.function.arguments") or ""),
            ))
        out.append(MessageView(
            role=role,
            content=content,
            tool_calls=tcs,
            tool_call_id=g.get("message.tool_call_id"),
        ))
    return out


# ----------------------------------------------------------------------
# The view itself
# ----------------------------------------------------------------------

class OpenInferenceTraceView(TraceView):
    """Walk an OTLP span tree to expose trace-level canonical accessors."""

    def __init__(self, record: dict, descriptor: DatasetDescriptor):
        assert isinstance(descriptor.mapping, OpenInferenceMapping), (
            "OpenInferenceTraceView requires a descriptor with OpenInferenceMapping"
        )
        self._rec = record
        self._desc = descriptor
        self._m: OpenInferenceMapping = descriptor.mapping
        spans = record.get("spans") or []
        self._spans: list[tuple[dict, dict[str, Any]]] = []
        for s in spans:
            if not isinstance(s, dict):
                continue
            self._spans.append((s, _attrs_to_dict(s.get("attributes"))))
        own_ids = {s.get("spanId") for s, _ in self._spans}
        self._root: tuple[dict, dict[str, Any]] | None = None
        for s, a in self._spans:
            p = s.get("parentSpanId")
            if p is None or p == "" or p not in own_ids:
                self._root = (s, a)
                break
        self._by_kind: dict[str, list[tuple[dict, dict[str, Any]]]] = {}
        for s, a in self._spans:
            kind = str(a.get("openinference.span.kind") or "").upper()
            self._by_kind.setdefault(kind, []).append((s, a))

    # --- helpers ---

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

    # --- identity / semantics ---
    @property
    def id(self) -> str:
        """Honor ``mapping.id_attribute`` if set; fall back to record.traceId."""
        key = self._m.id_attribute
        if key:
            v = self._lookup_attribute(key)
            if v is not None:
                return str(v)
        return str(self._rec.get("traceId") or "?")

    @property
    def query(self) -> str:
        if self._root is not None:
            _, ra = self._root
            iv = ra.get("input.value")
            if isinstance(iv, str) and iv:
                return iv
        for _, a in self._by_kind.get("LLM", []):
            for m in _extract_messages(a, "llm.input_messages"):
                if m.role == "user" and m.content:
                    return m.content
        return ""

    @property
    def final_answer(self) -> Any | None:
        """Root's ``output.value`` (if present as an attribute at all — even
        empty-string, which encodes "explicitly no final answer"). Falls
        back to the last LLM span's last output message when the root has
        no ``output.value`` attribute set."""
        if self._root is not None:
            _, ra = self._root
            if "output.value" in ra:
                ov = ra["output.value"]
                return ov if ov != "" else None
        llm_spans = self._by_kind.get("LLM", [])
        if not llm_spans:
            return None
        last = max(llm_spans, key=lambda sa: int(sa[0].get("startTimeUnixNano") or 0))
        msgs = _extract_messages(last[1], "llm.output_messages")
        return msgs[-1].content if msgs else None

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

    # --- trajectory ---
    def messages(self) -> Iterator[MessageView]:
        """First LLM span's input in full, then each subsequent span's tail
        of new inputs + all outputs. Dedup is positional, not content-based —
        identical consecutive replies ("ok", "ok") are preserved."""
        llm_spans = sorted(
            self._by_kind.get("LLM", []),
            key=lambda sa: int(sa[0].get("startTimeUnixNano") or 0),
        )
        prev_input_len = 0
        for _, a in llm_spans:
            in_msgs = _extract_messages(a, "llm.input_messages")
            out_msgs = _extract_messages(a, "llm.output_messages")
            for m in in_msgs[prev_input_len:]:
                yield m
            prev_input_len = len(in_msgs)
            for m in out_msgs:
                yield m
            prev_input_len += len(out_msgs)

    def tool_calls(self) -> Iterator[ToolCallView]:
        """TOOL-span data preferred when linkable to an assistant tool_call
        (carries the result); otherwise falls back to messages-side intents.

        Linkage precedence:

        1. ``tool_call.id`` / ``tool.call_id`` on the TOOL span matches
           the assistant tool_call.id.
        2. TOOL span's ``parentSpanId`` matches an LLM span we've already
           processed AND the message tool_call index matches the TOOL
           span's order among siblings of that LLM span.

        The second rule catches partially-instrumented OI traces where
        TOOL spans omit ``tool_call.id``. Without it those TOOL spans
        would double-emit (once via the message, once as "orphan").
        """
        tool_spans = self._by_kind.get("TOOL", [])
        tool_by_callid: dict[str, tuple[dict, dict[str, Any]]] = {}
        for s, a in tool_spans:
            cid = a.get("tool_call.id") or a.get("tool.call_id")
            if cid:
                tool_by_callid[str(cid)] = (s, a)

        # Group TOOL spans by parent LLM span, ordered by start time, for
        # the positional fallback.
        tool_children: dict[str, list[tuple[dict, dict[str, Any]]]] = {}
        for s, a in tool_spans:
            pid = s.get("parentSpanId")
            if not pid:
                continue
            tool_children.setdefault(str(pid), []).append((s, a))
        for pid, group in tool_children.items():
            group.sort(key=lambda sa: int(sa[0].get("startTimeUnixNano") or 0))

        # Which LLM span emitted each message.tool_call — we walk LLM spans
        # in the same order as ``messages()`` does so position-based
        # matching aligns.
        llm_spans_by_start = sorted(
            self._by_kind.get("LLM", []),
            key=lambda sa: int(sa[0].get("startTimeUnixNano") or 0),
        )

        emitted_tool_span_ids: set[str] = set()
        for llm_s, llm_a in llm_spans_by_start:
            parent_id = str(llm_s.get("spanId") or "")
            children = tool_children.get(parent_id, [])
            child_cursor = 0
            out_msgs = _extract_messages(llm_a, "llm.output_messages")
            for m in out_msgs:
                for tc in m.tool_calls:
                    linked: tuple[dict, dict[str, Any]] | None = None
                    if tc.id and tc.id in tool_by_callid:
                        linked = tool_by_callid[tc.id]
                    elif child_cursor < len(children):
                        # Positional fallback — consume the next un-linked
                        # child of this LLM span.
                        linked = children[child_cursor]
                        child_cursor += 1
                    if linked is not None:
                        s, a = linked
                        sid = str(s.get("spanId") or "")
                        if sid in emitted_tool_span_ids:
                            yield tc
                            continue
                        emitted_tool_span_ids.add(sid)
                        yield ToolCallView(
                            id=tc.id or str(a.get("tool_call.id")
                                            or a.get("tool.call_id") or "")
                                or None,
                            name=str(a.get("tool.name") or s.get("name") or tc.name),
                            arguments=str(a.get("input.value") or tc.arguments),
                            result=(str(a.get("output.value"))
                                    if "output.value" in a else None),
                        )
                    else:
                        yield tc

        # Any TOOL spans still un-emitted: they weren't linked via a
        # message-side tool_call (trace with TOOL spans but no
        # corresponding assistant tool_calls — uncommon but possible).
        for s, a in tool_spans:
            sid = str(s.get("spanId") or "")
            if sid in emitted_tool_span_ids:
                continue
            call_id = a.get("tool_call.id") or a.get("tool.call_id")
            yield ToolCallView(
                id=str(call_id) if call_id else None,
                name=str(a.get("tool.name") or s.get("name") or ""),
                arguments=str(a.get("input.value") or ""),
                result=(str(a.get("output.value"))
                        if "output.value" in a else None),
            )

    # --- auxiliary ---
    @property
    def turns_used(self) -> int | None:
        n = len(self._by_kind.get("LLM", []))
        return n or None

    @property
    def tool_errors(self) -> int:
        n = 0
        for s, a in self._by_kind.get("TOOL", []):
            status = (s.get("status") or {}).get("code")
            if status == "STATUS_CODE_ERROR" or a.get("exception.type"):
                n += 1
        return n

    @property
    def tool_calls_total(self) -> int:
        return len(self._by_kind.get("TOOL", []))

    @property
    def usage(self) -> dict[str, int]:
        out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for _, a in self._by_kind.get("LLM", []):
            p = a.get("llm.token_count.prompt")
            c = a.get("llm.token_count.completion")
            t = a.get("llm.token_count.total")
            try:
                out["prompt_tokens"] += int(p) if p is not None else 0
                out["completion_tokens"] += int(c) if c is not None else 0
                out["total_tokens"] += int(t) if t is not None else 0
            except (TypeError, ValueError):
                continue
        if out["total_tokens"] == 0 and (out["prompt_tokens"] or out["completion_tokens"]):
            out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
        return {k: v for k, v in out.items() if v}

    def documents(self) -> list[DocumentView]:
        """OpenInference doesn't have a canonical document attribute yet;
        teams carrying documents should expose them as RETRIEVER spans
        or custom attrs, which future adapters can surface."""
        return []
