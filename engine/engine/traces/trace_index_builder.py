from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from engine.traces.models.canonical_span import SpanRecord
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_index_models import TraceIndexMeta, TraceIndexRow

# TODO: Switch all dataclasses to pydantic
@dataclass
class _RowAccumulator:
    """Mutable per-trace_id rollup used during a single index-building pass; converts to TraceIndexRow at the end."""

    trace_id: str
    byte_offsets: list[int] = field(default_factory=list)
    byte_lengths: list[int] = field(default_factory=list)
    span_count: int = 0
    start_time: str = ""
    end_time: str = ""
    has_errors: bool = False
    service_names: set[str] = field(default_factory=set)
    model_names: set[str] = field(default_factory=set)
    agent_names: set[str] = field(default_factory=set)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    project_id: str | None = None

    # TODO: Use LLM for smart parsing
    def absorb(self, *, span: SpanRecord, byte_offset: int, byte_length: int) -> None:
        """Fold one span into the accumulator: record its byte slice and update rollup fields."""
        self.byte_offsets.append(byte_offset)
        self.byte_lengths.append(byte_length)
        self.span_count += 1

        if not self.start_time or span.start_time < self.start_time:
            self.start_time = span.start_time
        if not self.end_time or span.end_time > self.end_time:
            self.end_time = span.end_time

        if span.status.code == "STATUS_CODE_ERROR":
            self.has_errors = True

        svc = span.resource.attributes.get("service.name")
        if isinstance(svc, str):
            self.service_names.add(svc)

        model = span.attributes.get("inference.llm.model_name") or span.attributes.get("llm.model_name")
        if isinstance(model, str) and model:
            self.model_names.add(model)

        agent = span.attributes.get("inference.agent_name")
        if isinstance(agent, str) and agent:
            self.agent_names.add(agent)

        input_tokens = span.attributes.get("inference.llm.input_tokens")
        if isinstance(input_tokens, int):
            self.total_input_tokens += input_tokens
        output_tokens = span.attributes.get("inference.llm.output_tokens")
        if isinstance(output_tokens, int):
            self.total_output_tokens += output_tokens

        proj = span.attributes.get("inference.project_id")
        if isinstance(proj, str) and self.project_id is None:
            self.project_id = proj

    def finalize(self) -> TraceIndexRow:
        """Snapshot the accumulated state into the immutable TraceIndexRow that gets written to the sidecar."""
        return TraceIndexRow(
            trace_id=self.trace_id,
            byte_offsets=self.byte_offsets,
            byte_lengths=self.byte_lengths,
            span_count=self.span_count,
            start_time=self.start_time,
            end_time=self.end_time,
            has_errors=self.has_errors,
            service_names=sorted(self.service_names),
            model_names=sorted(self.model_names),
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            project_id=self.project_id,
            agent_names=sorted(self.agent_names),
        )


class TraceIndexBuilder:
    """Sidecar index creator for the flat OTel JSONL trace input.

    Index is sidecar-style next to the trace file. ``ensure_index_exists`` reuses
    an existing index when its stored stat fingerprint (size + mtime_ns) still
    matches the trace file, and rebuilds it otherwise. Schema-version mismatches
    still fail fast. ``build_index`` is the actual scan + write path.
    """

    @classmethod
    async def ensure_index_exists(
        cls,
        trace_path: Path,
        config: TraceIndexConfig,
    ) -> Path:
        """Return a usable index path, rebuilding when missing or stale.

        The sidecar is a derived cache: any mismatch — missing files, schema
        version drift, or a different ``source_size``/``source_mtime_ns`` — is
        treated as staleness and triggers a rebuild. ``build_index`` itself
        fails fast on requested versions it does not know how to write.
        """
        index_path = config.index_path or Path(str(trace_path) + ".engine-index.jsonl")
        meta_path = cls._meta_path_for(index_path)

        current_size, current_mtime_ns = cls._fingerprint_trace_file(trace_path)

        if index_path.exists() and meta_path.exists():
            existing = TraceIndexMeta.model_validate_json(meta_path.read_text())
            if (
                existing.schema_version == config.schema_version
                and existing.source_size == current_size
                and existing.source_mtime_ns == current_mtime_ns
            ):
                return index_path

        await cls.build_index(
            trace_path=trace_path,
            index_path=index_path,
            meta_path=meta_path,
            schema_version=config.schema_version,
            source_size=current_size,
            source_mtime_ns=current_mtime_ns,
        )
        return index_path

    @staticmethod
    def _fingerprint_trace_file(trace_path: Path) -> tuple[int, int]:
        """Return ``(size_bytes, mtime_ns)`` for the trace file via a single stat."""
        st = trace_path.stat()
        return st.st_size, st.st_mtime_ns

    @staticmethod
    def _meta_path_for(index_path: Path) -> Path:
        """Convention: ``<trace>.engine-index.jsonl`` ↔ ``<trace>.engine-index.meta.json``."""
        name = index_path.name
        if name.endswith(".engine-index.jsonl"):
            return index_path.with_name(name[: -len(".jsonl")] + ".meta.json")
        return index_path.with_name(name + ".meta.json")

    @classmethod
    async def build_index(
        cls,
        trace_path: Path,
        index_path: Path,
        meta_path: Path,
        schema_version: int,
        source_size: int | None = None,
        source_mtime_ns: int | None = None,
    ) -> None:
        """Single-pass binary scan over the JSONL, grouping by trace_id and writing the sidecars atomically.

        Reads in binary mode so byte_offset/byte_length are exact, accumulates per-trace
        rollups, then writes ``<file>.tmp`` then renames into place so a partially-written
        index is never observable. ``source_size`` and ``source_mtime_ns`` are the
        freshness fingerprint stored in meta; callers may pass precomputed values to
        avoid a redundant stat.
        """
        if schema_version != 1:
            raise ValueError(f"unsupported trace index schema_version={schema_version}")

        if source_size is None or source_mtime_ns is None:
            source_size, source_mtime_ns = cls._fingerprint_trace_file(trace_path)

        rows_by_trace: dict[str, _RowAccumulator] = {}

        # TODO: Stream file to avoid loading all into memory
        with trace_path.open("rb") as fh:
            offset = 0

            # TODO: Parallelize / async, can be slow
            for raw_line in fh:
                byte_length = len(raw_line)
                stripped = raw_line.rstrip(b"\n")
                if stripped:
                    span = SpanRecord.model_validate_json(stripped)
                    acc = rows_by_trace.setdefault(span.trace_id, _RowAccumulator(trace_id=span.trace_id))
                    acc.absorb(span=span, byte_offset=offset, byte_length=len(stripped))
                offset += byte_length

        rows = [acc.finalize() for acc in rows_by_trace.values()]

        tmp_index = index_path.with_suffix(index_path.suffix + ".tmp")
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")

        tmp_index.write_text(
            "\n".join(row.model_dump_json() for row in rows) + ("\n" if rows else "")
        )
        tmp_meta.write_text(
            TraceIndexMeta(
                schema_version=schema_version,
                trace_count=len(rows),
                source_size=source_size,
                source_mtime_ns=source_mtime_ns,
            ).model_dump_json()
        )

        tmp_index.replace(index_path)
        tmp_meta.replace(meta_path)
