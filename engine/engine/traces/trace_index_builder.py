from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from engine.traces.models.canonical_span import SpanRecord
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_index_models import TraceIndexMeta, TraceIndexRow


@dataclass
class _RowAccumulator:
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

    def absorb(self, *, span: SpanRecord, byte_offset: int, byte_length: int) -> None:
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
    @classmethod
    async def ensure_index_exists(
        cls,
        trace_path: Path,
        config: TraceIndexConfig,
    ) -> Path:
        index_path = config.index_path or Path(str(trace_path) + ".engine-index.jsonl")
        meta_path = cls._meta_path_for(index_path)

        if index_path.exists() and meta_path.exists():
            existing = TraceIndexMeta.model_validate_json(meta_path.read_text())
            if existing.schema_version != config.schema_version:
                raise ValueError(
                    f"existing index schema_version={existing.schema_version} "
                    f"does not match requested {config.schema_version}"
                )
            return index_path

        await cls.build_index(
            trace_path=trace_path,
            index_path=index_path,
            meta_path=meta_path,
            schema_version=config.schema_version,
        )
        return index_path

    @staticmethod
    def _meta_path_for(index_path: Path) -> Path:
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
    ) -> None:
        if schema_version != 1:
            raise ValueError(f"unsupported trace index schema_version={schema_version}")

        rows_by_trace: dict[str, _RowAccumulator] = {}

        with trace_path.open("rb") as fh:
            offset = 0
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
            TraceIndexMeta(schema_version=schema_version, trace_count=len(rows)).model_dump_json()
        )

        tmp_index.replace(index_path)
        tmp_meta.replace(meta_path)
