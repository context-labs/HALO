from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.traces.models.canonical_span import SpanRecord
from engine.traces.models.trace_index_models import TraceIndexRow

if TYPE_CHECKING:
    from engine.traces.models.trace_query_models import (
        DatasetOverview,
        SpanMatchRecord,
        SpanSearchResult,
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
#
# Two caps, by tool. The cap is compared against ``len(...)`` of the JSON-serialized
# attribute value, so the unit is *characters* of the string form (which is also a
# reasonable proxy for bytes on mostly-ASCII OTel attributes).
#   - ``_DISCOVERY_ATTR_TRUNCATION_CHARS`` (4 KB) is the cheap-discovery cap used by
#     ``view_trace`` (which can pull every span of a trace) and ``search_trace``
#     (which can match many spans at once). It preserves enough head-of-payload
#     for the model to see what was called and roughly what came back, without
#     the long tail.
#   - ``_SURGICAL_ATTR_TRUNCATION_CHARS`` (16 KB) is the surgical-read cap used by
#     ``view_spans``. The agent has explicitly named the spans it wants, capped
#     to 200 ids, so a higher per-attribute budget is appropriate — that's what
#     makes ``view_spans`` genuinely complementary to ``search_trace`` rather
#     than a duplicate.
_DISCOVERY_ATTR_TRUNCATION_CHARS = 4096
_SURGICAL_ATTR_TRUNCATION_CHARS = 16384

# Per-call total size budget for ``view_trace`` / ``view_spans``. Computed as the
# UTF-8 byte length of the truncated, serialized response. When over budget, the
# spans are dropped and an ``OversizedTraceSummary`` is returned in their place
# so the agent can plan smaller follow-up calls instead of blowing context.
# 150_000 bytes is a comfortable fraction of even modest context windows
# (~37K tokens) and leaves headroom for conversation history.
_VIEW_TRACE_RESPONSE_BYTES_BUDGET = 150_000
_VIEW_SPANS_RESPONSE_BYTES_BUDGET = 150_000

# How many top-frequency span names to surface in the oversized summary.
_OVERSIZED_TOP_SPAN_NAMES = 10


def _truncate_attribute_value(value: Any, cap_chars: int) -> Any:
    """Cap a single attribute value at ``cap_chars`` characters of its JSON form.

    Strings beyond the threshold get a head slice plus a marker. Non-string values
    only get truncated if their JSON serialization exceeds the threshold (in which
    case they're replaced by the truncated JSON string with a marker). Small values
    pass through untouched.
    """
    if isinstance(value, str):
        if len(value) <= cap_chars:
            return value
        return f"{value[:cap_chars]}... [HALO truncated: original {len(value)} chars]"
    try:
        serialized = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return value
    if len(serialized) <= cap_chars:
        return value
    return (
        f"{serialized[:cap_chars]}"
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
_NOISY_FLAT_PROJECTION_RE = re.compile(r"^(?:llm\.(?:input|output)_messages|mcp\.tools)\.\d+\.")


def _is_noisy_flat_projection(key: str) -> bool:
    """True for OpenInference flat-projection keys (per-message / per-tool fan-outs)."""
    return bool(_NOISY_FLAT_PROJECTION_RE.match(key))


def _truncate_span_attributes(span: SpanRecord, cap_chars: int) -> SpanRecord:
    """Return a copy of ``span`` whose oversized attribute values are head-capped at ``cap_chars``
    characters and whose noisy OpenInference flat projections are dropped.

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
        new_attrs[k] = _truncate_attribute_value(v, cap_chars)
    if dropped:
        cap_kb = cap_chars // 1024
        new_attrs["__halo_dropped_flat_projections"] = (
            f"{dropped} llm.input_messages.<i>.* / llm.output_messages.<i>.* / "
            "mcp.tools.<i>.* projection keys dropped to keep span size bounded. "
            "The JSON-blob attributes llm.input_messages / llm.output_messages / "
            f"mcp.tools.listed (head-capped at ~{cap_kb}KB) carry the same content."
        )
    return span.model_copy(update={"attributes": new_attrs})


def _compile_regex_or_raise(regex_pattern: str) -> re.Pattern[str]:
    """Compile a Python regex string or raise ``ValueError`` with a clear message.

    Tools take regex inputs as ``str`` (not ``re.Pattern``) so they can be JSON-
    serialized; this helper keeps the failure mode consistent across all tools
    that accept a regex argument.
    """
    try:
        return re.compile(regex_pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern {regex_pattern!r}: {exc}") from exc


def _build_match_record(
    *,
    trace_id: str,
    span: SpanRecord,
    span_index: int,
    raw_jsonl_bytes: int,
    raw: str,
    match: re.Match[str],
    context_buffer_chars: int,
) -> "SpanMatchRecord":
    """Build a SpanMatchRecord from one regex match against ``raw`` (the decoded span JSON).

    Indices are character offsets into ``raw``. ``matched_context`` clips a window
    of ``context_buffer_chars`` characters around the match.
    """
    from engine.traces.models.trace_query_models import SpanMatchRecord

    start = match.start()
    end = match.end()
    ctx_start = max(0, start - context_buffer_chars)
    ctx_end = min(len(raw), end + context_buffer_chars)
    return SpanMatchRecord(
        trace_id=trace_id,
        span_id=span.span_id,
        span_index=span_index,
        span_name=span.name,
        kind=span.kind,
        status_code=span.status.code,
        parent_span_id=span.parent_span_id,
        raw_jsonl_bytes=raw_jsonl_bytes,
        match_text=raw[start:end],
        matched_context=raw[ctx_start:ctx_end],
        match_start_char=start,
        match_end_char=end,
    )


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

        Per-attribute payloads are head-capped at ``_DISCOVERY_ATTR_TRUNCATION_CHARS``
        (4 KB) so a single big trace can't blow the model's context window. If the
        truncated serialized UTF-8 byte size still exceeds
        ``_VIEW_TRACE_RESPONSE_BYTES_BUDGET``, the spans are dropped and an
        ``OversizedTraceSummary`` is returned in their place — the agent should
        switch to ``search_trace`` for discovery and ``view_spans`` for surgical
        reads at a higher per-attribute budget (16 KB), or ``search_span`` when an
        individual span itself is too large.
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
                spans.append(
                    _truncate_span_attributes(
                        SpanRecord.model_validate_json(blob),
                        _DISCOVERY_ATTR_TRUNCATION_CHARS,
                    )
                )

        per_span_bytes = [len(s.model_dump_json().encode("utf-8")) for s in spans]
        total_bytes = sum(per_span_bytes)
        if total_bytes > _VIEW_TRACE_RESPONSE_BYTES_BUDGET:
            sorted_sizes = sorted(per_span_bytes)
            mid = sorted_sizes[len(sorted_sizes) // 2] if sorted_sizes else 0
            name_counts = Counter(s.name for s in spans)
            error_spans = sum(1 for s in spans if s.status.code == "STATUS_CODE_ERROR")
            recommendation = (
                f"This trace exceeds the per-call view budget "
                f"({total_bytes:,} bytes > {_VIEW_TRACE_RESPONSE_BYTES_BUDGET:,}). "
                "Do not retry view_trace. Instead: "
                "(1) call search_trace(trace_id, regex_pattern) with a specific regex "
                "(error string, tool name, attribute key) to surface the spans you "
                "actually need; or (2) call view_spans(trace_id, span_ids=[...]) with "
                "specific span ids you've already seen in search_trace results. If a "
                "single span surfaces as too large, use "
                "search_span(trace_id, span_id, regex_pattern) to extract matches "
                "from inside that span. The top_span_names below give a sense of "
                "what's in the trace."
            )
            summary = OversizedTraceSummary(
                trace_id=trace_id,
                span_count=len(spans),
                truncated_response_bytes=total_bytes,
                response_bytes_budget=_VIEW_TRACE_RESPONSE_BYTES_BUDGET,
                span_response_bytes_min=sorted_sizes[0] if sorted_sizes else 0,
                span_response_bytes_median=mid,
                span_response_bytes_max=sorted_sizes[-1] if sorted_sizes else 0,
                top_span_names=name_counts.most_common(_OVERSIZED_TOP_SPAN_NAMES),
                error_span_count=error_spans,
                recommendation=recommendation,
            )
            return TraceView(trace_id=trace_id, spans=[], oversized=summary)

        return TraceView(trace_id=trace_id, spans=spans)

    def view_spans(self, trace_id: str, span_ids: list[str]) -> "TraceView":
        """Read only the named ``span_ids`` from ``trace_id`` at the surgical-read cap.

        Surgical follow-up to ``search_trace`` (or any other source of span ids the
        agent has on hand). Per-attribute payloads are head-capped at
        ``_SURGICAL_ATTR_TRUNCATION_CHARS`` (16 KB) — 4× higher than the discovery
        cap used by ``view_trace`` and ``search_trace`` — so re-fetching a span the
        agent already saw via ``search_trace`` actually returns more bytes for any
        attribute that was head-capped on the discovery path.

        Walks the trace's byte offsets and returns spans whose ``span_id`` is in
        ``span_ids``; ids that don't match any span are silently skipped. Enforces
        ``_VIEW_SPANS_RESPONSE_BYTES_BUDGET`` as a hard cap on the truncated
        serialized UTF-8 byte size: when the selected spans collectively exceed the
        budget, ``spans`` is returned empty and ``oversized`` carries summary
        statistics + a recommendation to use ``search_span`` (per-span regex
        extraction) or a smaller ``span_ids`` set.
        """
        from engine.traces.models.trace_query_models import (
            OversizedTraceSummary,
            TraceView,
        )

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
                    spans.append(_truncate_span_attributes(span, _SURGICAL_ATTR_TRUNCATION_CHARS))

        per_span_bytes = [len(s.model_dump_json().encode("utf-8")) for s in spans]
        total_bytes = sum(per_span_bytes)
        if total_bytes > _VIEW_SPANS_RESPONSE_BYTES_BUDGET:
            sorted_sizes = sorted(per_span_bytes)
            mid = sorted_sizes[len(sorted_sizes) // 2] if sorted_sizes else 0
            name_counts = Counter(s.name for s in spans)
            error_spans = sum(1 for s in spans if s.status.code == "STATUS_CODE_ERROR")
            recommendation = (
                f"The selected spans exceed the per-call view budget "
                f"({total_bytes:,} bytes > {_VIEW_SPANS_RESPONSE_BYTES_BUDGET:,}). "
                "Do not retry view_spans with the same set. Instead: "
                "(1) call search_span(trace_id, span_id, regex_pattern) on individual "
                "large spans (look at span_response_bytes_max below) to extract only "
                "the regex matches you need; or "
                "(2) call view_spans with a smaller subset of span_ids."
            )
            summary = OversizedTraceSummary(
                trace_id=trace_id,
                span_count=len(spans),
                truncated_response_bytes=total_bytes,
                response_bytes_budget=_VIEW_SPANS_RESPONSE_BYTES_BUDGET,
                span_response_bytes_min=sorted_sizes[0] if sorted_sizes else 0,
                span_response_bytes_median=mid,
                span_response_bytes_max=sorted_sizes[-1] if sorted_sizes else 0,
                top_span_names=name_counts.most_common(_OVERSIZED_TOP_SPAN_NAMES),
                error_span_count=error_spans,
                recommendation=recommendation,
            )
            return TraceView(trace_id=trace_id, spans=[], oversized=summary)

        return TraceView(trace_id=trace_id, spans=spans)

    def query_traces(
        self,
        filters: "TraceFilters",
        limit: int = 50,
        offset: int = 0,
    ) -> "TraceQueryResult":
        """Filter rows in memory and project each surviving row into a TraceSummary.

        Indexed predicates on ``filters`` are applied first (cheap, no JSONL reads).
        ``filters.regex_pattern`` is the one scan-heavy predicate: when set, the
        remaining candidates are scanned span-by-span and a trace is kept iff at
        least one of its spans matches.
        """
        from engine.traces.models.trace_query_models import TraceQueryResult, TraceSummary

        filtered = self._apply_filters(filters)
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
                raw_jsonl_bytes=sum(row.byte_lengths),
            )
            for row in filtered[offset : offset + limit]
        ]
        return TraceQueryResult(traces=summaries, total=len(filtered))

    def count_traces(self, filters: "TraceFilters") -> "TraceCountResult":
        """Count matching rows without materializing summaries.

        Same filter semantics as ``query_traces``: indexed predicates first, then
        the optional ``filters.regex_pattern`` raw-content scan.
        """
        from engine.traces.models.trace_query_models import TraceCountResult

        return TraceCountResult(total=len(self._apply_filters(filters)))

    def get_overview(self, filters: "TraceFilters") -> "DatasetOverview":
        """Aggregate the filtered subset into a single DatasetOverview rollup row.

        Same filter semantics as ``query_traces``/``count_traces``.
        """
        from engine.traces.models.trace_query_models import DatasetOverview

        rows = self._apply_filters(filters)
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
                raw_jsonl_bytes=0,
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
            raw_jsonl_bytes=sum(sum(r.byte_lengths) for r in rows),
            sample_trace_ids=[r.trace_id for r in rows[:_OVERVIEW_SAMPLE_TRACE_IDS]],
        )

    def search_trace(
        self,
        trace_id: str,
        regex_pattern: str,
        context_buffer_chars: int = 100,
        max_matches: int = 50,
    ) -> "TraceSearchResult":
        """Regex-search the raw on-disk JSON of every span in ``trace_id``.

        Returns up to ``max_matches`` ``SpanMatchRecord``s — one per regex match
        (so a single span can produce multiple records). ``match_count`` is the
        unbounded total across the trace; ``has_more`` is true when the result
        was capped. The full trace is scanned even after the cap is reached so
        the count stays exact.
        """
        from engine.traces.models.trace_query_models import TraceSearchResult

        if trace_id not in self._rows_by_id:
            raise KeyError(trace_id)
        row = self._rows_by_id[trace_id]
        pattern = _compile_regex_or_raise(regex_pattern)

        matches: list["SpanMatchRecord"] = []
        match_count = 0

        with self._trace_path.open("rb") as fh:
            for span_index, (offset, length) in enumerate(
                zip(row.byte_offsets, row.byte_lengths, strict=True)
            ):
                fh.seek(offset)
                blob = fh.read(length)
                raw = blob.decode("utf-8", errors="replace")
                iterator = pattern.finditer(raw)
                first = next(iterator, None)
                if first is None:
                    continue
                # Parse the span only when it has at least one match — avoids paying
                # the SpanRecord validate cost on spans we'll never report on.
                span = SpanRecord.model_validate_json(blob)
                # Lazy chain over the first match + remaining iterator rather than
                # ``(first, *iterator)`` which would eagerly materialize every match
                # into a tuple — for a broad regex on a large span that's millions of
                # ``re.Match`` objects held in memory at once.
                for m in itertools.chain([first], iterator):
                    match_count += 1
                    if len(matches) < max_matches:
                        matches.append(
                            _build_match_record(
                                trace_id=trace_id,
                                span=span,
                                span_index=span_index,
                                raw_jsonl_bytes=length,
                                raw=raw,
                                match=m,
                                context_buffer_chars=context_buffer_chars,
                            )
                        )

        return TraceSearchResult(
            trace_id=trace_id,
            match_count=match_count,
            returned_match_count=len(matches),
            has_more=match_count > len(matches),
            matches=matches,
        )

    def search_span(
        self,
        trace_id: str,
        span_id: str,
        regex_pattern: str,
        context_buffer_chars: int = 100,
        max_matches: int = 50,
    ) -> "SpanSearchResult":
        """Regex-search the raw on-disk JSON of a single span.

        For drilling into a single span when ``view_spans`` of that span would exceed
        the response budget. Returns up to ``max_matches`` records; ``match_count``
        is the unbounded total of regex matches inside the span. Raises ``KeyError``
        if ``trace_id`` or ``span_id`` is unknown.
        """
        from engine.traces.models.trace_query_models import SpanSearchResult

        if trace_id not in self._rows_by_id:
            raise KeyError(trace_id)
        row = self._rows_by_id[trace_id]
        pattern = _compile_regex_or_raise(regex_pattern)

        matches: list["SpanMatchRecord"] = []
        match_count = 0

        with self._trace_path.open("rb") as fh:
            for span_index, (offset, length) in enumerate(
                zip(row.byte_offsets, row.byte_lengths, strict=True)
            ):
                fh.seek(offset)
                blob = fh.read(length)
                # Cheap pre-check: stdlib JSON parse to read just ``span_id`` without
                # paying the SpanRecord field-by-field validation cost on every
                # non-matching span. Full Pydantic validation only runs once we know
                # this is the span we want.
                if json.loads(blob).get("span_id") != span_id:
                    continue
                span = SpanRecord.model_validate_json(blob)
                raw = blob.decode("utf-8", errors="replace")
                for m in pattern.finditer(raw):
                    match_count += 1
                    if len(matches) < max_matches:
                        matches.append(
                            _build_match_record(
                                trace_id=trace_id,
                                span=span,
                                span_index=span_index,
                                raw_jsonl_bytes=length,
                                raw=raw,
                                match=m,
                                context_buffer_chars=context_buffer_chars,
                            )
                        )
                return SpanSearchResult(
                    trace_id=trace_id,
                    span_id=span_id,
                    match_count=match_count,
                    returned_match_count=len(matches),
                    has_more=match_count > len(matches),
                    matches=matches,
                )

        raise KeyError(f"span_id={span_id!r} not found in trace_id={trace_id!r}")

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

    def _apply_filters(self, filters: "TraceFilters") -> list[TraceIndexRow]:
        """Apply ``filters`` to ``self._rows`` and return the surviving rows.

        Indexed predicates run first (cheap, in-memory). ``filters.regex_pattern``
        runs last because it requires reading the JSONL — early-exits per trace
        on the first matching span so a typical match stays cheap.
        """
        rows = [r for r in self._rows if _matches_indexed_filters(r, filters)]
        if filters.regex_pattern is None:
            return rows
        pattern = _compile_regex_or_raise(filters.regex_pattern)
        return [r for r in rows if self._row_has_content_match(r, pattern)]

    def _row_has_content_match(self, row: TraceIndexRow, pattern: re.Pattern[str]) -> bool:
        """True iff at least one span of ``row`` has a regex match in its raw JSON.

        Stops at the first hit per trace so a typical match is cheap; only forces a
        full read when the trace doesn't match.
        """
        with self._trace_path.open("rb") as fh:
            for offset, length in zip(row.byte_offsets, row.byte_lengths, strict=True):
                fh.seek(offset)
                blob = fh.read(length)
                if pattern.search(blob.decode("utf-8", errors="replace")):
                    return True
        return False


def _matches_indexed_filters(row: TraceIndexRow, filters: "TraceFilters") -> bool:
    """ANDed predicate over the index-only fields of ``filters``.

    ``filters.regex_pattern`` is intentionally NOT consulted here — that field
    requires reading the JSONL and is applied separately by ``_apply_filters``
    after this cheap pass narrows the candidate set.
    """
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
