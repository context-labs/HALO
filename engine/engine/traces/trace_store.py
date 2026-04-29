from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.traces.models.canonical_span import SpanRecord
from engine.traces.models.trace_index_models import TraceIndexRow

if TYPE_CHECKING:
    from engine.traces.models.trace_query_models import (
        DatasetOverview,
        TraceCountResult,
        TraceFilters,
        TraceQueryResult,
        TraceSearchResult,
        TraceView,
    )


_OVERVIEW_SAMPLE_TRACE_IDS = 20

# Cap per-attribute payload size when returning spans to the LLM. Large fields
# like input.value / output.value / llm.input_messages can be tens of KB each
# and easily blow the model's context window when many spans come back at once.
# 4 KB preserves enough head-of-payload for the model to see what was called and
# roughly what came back, without the long tail.
_ATTR_TRUNCATION_BYTES = 4096

# Per-call total size budget for ``view_trace``. When a trace's truncated serialized
# size exceeds this, ``view_trace`` returns metadata + statistics instead of the
# spans, and the agent is told to use ``search_trace`` / ``view_spans`` for surgical
# reads. 150_000 chars is a comfortable fraction of even modest context windows
# (~37K tokens) and leaves headroom for conversation history.
_VIEW_TRACE_CHAR_BUDGET = 150_000

# How many top-frequency span names to surface in the oversized summary.
_OVERSIZED_TOP_SPAN_NAMES = 10


def _truncate_attribute_value(value: Any) -> Any:
    """Cap a single attribute value at ``_ATTR_TRUNCATION_BYTES`` of its JSON form.

    Strings beyond the threshold get a head slice plus a marker. Non-string values
    only get truncated if their JSON serialization exceeds the threshold (in which
    case they're replaced by the truncated JSON string with a marker). Small values
    pass through untouched.
    """
    if isinstance(value, str):
        if len(value) <= _ATTR_TRUNCATION_BYTES:
            return value
        return (
            f"{value[:_ATTR_TRUNCATION_BYTES]}"
            f"... [HALO truncated: original {len(value)} chars]"
        )
    try:
        serialized = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return value
    if len(serialized) <= _ATTR_TRUNCATION_BYTES:
        return value
    return (
        f"{serialized[:_ATTR_TRUNCATION_BYTES]}"
        f"... [HALO truncated: original {len(serialized)} chars; non-string attribute serialized for truncation]"
    )


# OpenInference instrumentations emit per-message flat projections under keys
# like ``llm.input_messages.0.message.contents.0.message_content.text``. A single
# LLM span in a long agent trace can have 400+ such keys totaling 60+ KB even
# though most individual values are tiny (so the per-attribute truncation
# doesn't catch them). The JSON-blob equivalents — ``llm.input_messages`` and
# ``llm.output_messages`` — carry the same content and ARE caught by the per-
# attribute truncation, so we drop the flat projections to keep the per-span
# size bounded. The string ``__halo_dropped_flat_projections`` is added to
# preserve discoverability when the model needs to know what's missing.
_NOISY_FLAT_PROJECTION_RE = re.compile(
    r"^(?:llm\.(?:input|output)_messages|mcp\.tools)\.\d+\."
)


def _is_noisy_flat_projection(key: str) -> bool:
    """True for OpenInference flat-projection keys (per-message / per-tool fan-outs)."""
    return bool(_NOISY_FLAT_PROJECTION_RE.match(key))


def _truncate_span_attributes(span: SpanRecord) -> SpanRecord:
    """Return a copy of ``span`` whose oversized attribute values are head-capped
    and whose noisy OpenInference flat projections are dropped.

    The ``attributes`` dict on ``SpanRecord`` is ``dict[str, Any]`` and the model
    is ``extra="allow"``, so replacing dict/list values with truncated strings
    is schema-safe.
    """
    new_attrs: dict[str, Any] = {}
    dropped = 0
    for k, v in span.attributes.items():
        if _is_noisy_flat_projection(k):
            dropped += 1
            continue
        new_attrs[k] = _truncate_attribute_value(v)
    if dropped:
        new_attrs["__halo_dropped_flat_projections"] = (
            f"{dropped} llm.input_messages.<i>.* / llm.output_messages.<i>.* / "
            "mcp.tools.<i>.* projection keys dropped to keep span size bounded. "
            "The JSON-blob attributes llm.input_messages / llm.output_messages / "
            "mcp.tools.listed (head-capped at ~4KB) carry the same content."
        )
    return span.model_copy(update={"attributes": new_attrs})


class TraceStore:
    """Pure read/query/render API over a built index plus the canonical JSONL.

    Deliberately depends only on stdlib + Pydantic + ``engine.traces.models`` so the
    sandbox can import and instantiate it directly inside user code, with no agent
    SDK / async runtime / tool dependencies in the import graph.
    """

    def __init__(self, trace_path: Path, index_path: Path, rows: list[TraceIndexRow]) -> None:
        """Hold paths plus the in-memory index rows; prefer ``load`` for constructing from disk."""
        self._trace_path = trace_path
        self._index_path = index_path
        self._rows = rows
        self._rows_by_id: dict[str, TraceIndexRow] = {r.trace_id: r for r in rows}

    @classmethod
    def load(cls, trace_path: Path, index_path: Path) -> "TraceStore":
        """Read the sidecar index file line-by-line and construct a TraceStore."""
        raw = index_path.read_text().splitlines()
        rows = [TraceIndexRow.model_validate_json(line) for line in raw if line]
        return cls(trace_path=trace_path, index_path=index_path, rows=rows)

    @property
    def trace_count(self) -> int:
        """Total trace count in the loaded index (no filtering)."""
        return len(self._rows)

    @property
    def trace_path(self) -> Path:
        """The canonical JSONL path this store reads spans from."""
        return self._trace_path

    @property
    def index_path(self) -> Path:
        """The sidecar index path this store was loaded from."""
        return self._index_path

    def view_trace(self, trace_id: str) -> "TraceView":
        """Read all spans of one trace by seeking to each indexed byte offset and parsing as SpanRecord.

        Per-attribute payloads are head-capped at ``_ATTR_TRUNCATION_BYTES`` so a single
        big trace can't blow the model's context window. If the truncated serialized size
        still exceeds ``_VIEW_TRACE_CHAR_BUDGET``, the spans are dropped and an
        ``OversizedTraceSummary`` is returned in their place — the agent should switch to
        ``search_trace`` + ``view_spans`` for surgical reads.
        """
        from engine.traces.models.trace_query_models import (
            OversizedTraceSummary,
            TraceView,
        )

        if trace_id not in self._rows_by_id:
            raise KeyError(trace_id)
        row = self._rows_by_id[trace_id]

        with self._trace_path.open("rb") as fh:
            spans: list[SpanRecord] = []
            for offset, length in zip(row.byte_offsets, row.byte_lengths, strict=True):
                fh.seek(offset)
                blob = fh.read(length)
                spans.append(_truncate_span_attributes(SpanRecord.model_validate_json(blob)))

        # Total-size guard: if the truncated spans collectively exceed the budget,
        # don't return the spans — return a summary that lets the agent plan smaller
        # follow-up calls without blowing context.
        per_span_sizes = [len(s.model_dump_json()) for s in spans]
        total_chars = sum(per_span_sizes)
        if total_chars > _VIEW_TRACE_CHAR_BUDGET:
            sorted_sizes = sorted(per_span_sizes)
            mid = sorted_sizes[len(sorted_sizes) // 2] if sorted_sizes else 0
            from collections import Counter
            name_counts = Counter(s.name for s in spans)
            error_spans = sum(1 for s in spans if s.status.code == "STATUS_CODE_ERROR")
            recommendation = (
                f"This trace exceeds the per-call view budget "
                f"({total_chars:,} chars > {_VIEW_TRACE_CHAR_BUDGET:,}). "
                "Do not retry view_trace. Instead: "
                "(1) call search_trace(trace_id, pattern) with a specific substring "
                "(error string, tool name, attribute key) to surface the spans you "
                "actually need; or (2) call view_spans(trace_id, span_ids=[...]) with "
                "specific span ids you've already seen in search_trace results or "
                "in another tool's output. The top_span_names below give you a sense "
                "of what's in the trace."
            )
            summary = OversizedTraceSummary(
                trace_id=trace_id,
                span_count=len(spans),
                total_serialized_chars=total_chars,
                char_budget=_VIEW_TRACE_CHAR_BUDGET,
                span_size_min=sorted_sizes[0] if sorted_sizes else 0,
                span_size_median=mid,
                span_size_max=sorted_sizes[-1] if sorted_sizes else 0,
                top_span_names=name_counts.most_common(_OVERSIZED_TOP_SPAN_NAMES),
                error_span_count=error_spans,
                recommendation=recommendation,
            )
            return TraceView(trace_id=trace_id, spans=[], oversized=summary)

        return TraceView(trace_id=trace_id, spans=spans)

    def view_spans(self, trace_id: str, span_ids: list[str]) -> "TraceView":
        """Read only the named ``span_ids`` from ``trace_id`` (truncated, same as ``view_trace``).

        For surgical reads after ``search_trace`` has identified a few interesting spans
        in a long trace. Walks the trace's byte offsets and returns spans whose ``span_id``
        is in ``span_ids``; ids that don't match any span are silently skipped.
        """
        from engine.traces.models.trace_query_models import TraceView

        if trace_id not in self._rows_by_id:
            raise KeyError(trace_id)
        row = self._rows_by_id[trace_id]
        wanted = set(span_ids)
        if not wanted:
            return TraceView(trace_id=trace_id, spans=[])

        spans: list[SpanRecord] = []
        with self._trace_path.open("rb") as fh:
            for offset, length in zip(row.byte_offsets, row.byte_lengths, strict=True):
                fh.seek(offset)
                blob = fh.read(length)
                span = SpanRecord.model_validate_json(blob)
                if span.span_id in wanted:
                    spans.append(_truncate_span_attributes(span))
        return TraceView(trace_id=trace_id, spans=spans)

    def query_traces(
        self,
        filters: "TraceFilters",
        limit: int = 50,
        offset: int = 0,
    ) -> "TraceQueryResult":
        """Filter rows in memory, slice with ``offset:offset+limit``, and project each into a TraceSummary."""
        from engine.traces.models.trace_query_models import TraceQueryResult, TraceSummary

        filtered = [row for row in self._rows if _matches_filters(row, filters)]
        summaries = [
            TraceSummary(
                trace_id=row.trace_id,
                span_count=row.span_count,
                start_time=row.start_time,
                end_time=row.end_time,
                has_errors=row.has_errors,
                service_names=row.service_names,
                model_names=row.model_names,
                total_input_tokens=row.total_input_tokens,
                total_output_tokens=row.total_output_tokens,
                agent_names=row.agent_names,
            )
            for row in filtered[offset : offset + limit]
        ]
        return TraceQueryResult(traces=summaries, total=len(filtered))

    def count_traces(self, filters: "TraceFilters") -> "TraceCountResult":
        """Count matching rows without materializing summaries — cheaper than ``query_traces``."""
        from engine.traces.models.trace_query_models import TraceCountResult

        total = sum(1 for row in self._rows if _matches_filters(row, filters))
        return TraceCountResult(total=total)

    def get_overview(self, filters: "TraceFilters") -> "DatasetOverview":
        """Aggregate the filtered subset into a single DatasetOverview rollup row."""
        from engine.traces.models.trace_query_models import DatasetOverview

        rows = [r for r in self._rows if _matches_filters(r, filters)]
        if not rows:
            return DatasetOverview(
                total_traces=0,
                total_spans=0,
                earliest_start_time="",
                latest_end_time="",
                service_names=[],
                model_names=[],
                agent_names=[],
                error_trace_count=0,
                total_input_tokens=0,
                total_output_tokens=0,
            )

        services: set[str] = set()
        models: set[str] = set()
        agents: set[str] = set()
        for r in rows:
            services.update(r.service_names)
            models.update(r.model_names)
            agents.update(r.agent_names)

        return DatasetOverview(
            total_traces=len(rows),
            total_spans=sum(r.span_count for r in rows),
            earliest_start_time=min(r.start_time for r in rows),
            latest_end_time=max(r.end_time for r in rows),
            service_names=sorted(services),
            model_names=sorted(models),
            agent_names=sorted(agents),
            error_trace_count=sum(1 for r in rows if r.has_errors),
            total_input_tokens=sum(r.total_input_tokens for r in rows),
            total_output_tokens=sum(r.total_output_tokens for r in rows),
            sample_trace_ids=[r.trace_id for r in rows[:_OVERVIEW_SAMPLE_TRACE_IDS]],
        )

    def search_trace(self, trace_id: str, pattern: str) -> "TraceSearchResult":
        """Substring search on raw JSON span text within one trace.

        Pattern matching is done against the raw on-disk JSON (so the pattern can
        target keys inside large attribute values). Returned matches are re-serialized
        with attribute payloads head-capped at ``_ATTR_TRUNCATION_BYTES`` to keep the
        per-call response size bounded.
        """
        from engine.traces.models.trace_query_models import TraceSearchResult

        if trace_id not in self._rows_by_id:
            raise KeyError(trace_id)
        row = self._rows_by_id[trace_id]

        matches: list[str] = []
        with self._trace_path.open("rb") as fh:
            for offset, length in zip(row.byte_offsets, row.byte_lengths, strict=True):
                fh.seek(offset)
                blob = fh.read(length).decode("utf-8", errors="replace")
                if pattern in blob:
                    span = SpanRecord.model_validate_json(blob)
                    matches.append(_truncate_span_attributes(span).model_dump_json())
        return TraceSearchResult(trace_id=trace_id, match_count=len(matches), matches=matches)

    def render_trace(self, trace_id: str, budget: int) -> str:
        """Render a trace as plain text suitable for prompt/tool consumption, truncated to ``budget`` bytes."""
        view = self.view_trace(trace_id)
        lines: list[str] = [f"trace_id: {trace_id}", f"spans: {len(view.spans)}"]
        for s in view.spans:
            lines.append(
                f"- span_id={s.span_id} parent={s.parent_span_id or '∅'} "
                f"name={s.name} kind={s.kind} status={s.status.code}"
            )
            lines.append(f"  start={s.start_time} end={s.end_time}")
            model = s.attributes.get("inference.llm.model_name") or s.attributes.get(
                "llm.model_name"
            )
            if model:
                lines.append(f"  model={model}")
            in_tok = s.attributes.get("inference.llm.input_tokens")
            out_tok = s.attributes.get("inference.llm.output_tokens")
            if in_tok is not None or out_tok is not None:
                lines.append(f"  tokens: input={in_tok} output={out_tok}")

        rendered = "\n".join(lines)
        if len(rendered) > budget:
            return rendered[:budget] + "... [truncated]"
        return rendered


def _matches_filters(row: TraceIndexRow, filters: "TraceFilters") -> bool:
    """ANDed predicate: row passes only if every set filter matches."""
    if filters.has_errors is not None and row.has_errors != filters.has_errors:
        return False
    if filters.model_names is not None and not any(
        m in row.model_names for m in filters.model_names
    ):
        return False
    if filters.service_names is not None and not any(
        s in row.service_names for s in filters.service_names
    ):
        return False
    if filters.agent_names is not None and not any(
        a in row.agent_names for a in filters.agent_names
    ):
        return False
    if filters.project_id is not None and row.project_id != filters.project_id:
        return False
    if filters.start_time_gte is not None and row.start_time < filters.start_time_gte:
        return False
    if filters.end_time_lte is not None and row.end_time > filters.end_time_lte:
        return False
    return True
