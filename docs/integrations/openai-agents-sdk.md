# HALO — OpenAI Agents SDK integration

Wire your existing OpenAI Agents SDK app into HALO's trace pipeline. Drop in one file (`tracing.py`), call `setup_tracing()` once at startup, and every agent / LLM / tool call becomes a JSONL span line on disk in the inference.net OTLP-shaped export format that the HALO Engine consumes.

> **Hosted sink not yet launched.** This guide covers the local file sink, which works today. See [Hosted (HALO Cloud)](#hosted-halo-cloud) for the placeholder.

## Prereqs

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- An OpenAI API key (`OPENAI_API_KEY`)

## Install

Add these dependencies to your project:

```toml
[project]
dependencies = [
    "openai-agents",
    "python-dotenv",
]
```

Or with uv:

```bash
uv add openai-agents python-dotenv
```

No OpenTelemetry packages required. `tracing.py` is a single self-contained module — stdlib + `openai-agents` only — so there's no OTLP exporter, no collector, and no instrumentor in the dependency graph.

## Add `tracing.py`

Copy [`demo/openai-agents-sdk-demo/tracing.py`](../../demo/openai-agents-sdk-demo/tracing.py) into your project verbatim. It's one ~450-line module that bundles three things:

- **`ExportContext`** — frozen dataclass for per-process identity (`project_id`, `service_name`, optional `service_version`, `deployment_environment`). `project_id` becomes `inference.project_id` (the Engine filters on this); `service_name` becomes `resource.attributes."service.name"`.
- **`InferenceOtlpFileProcessor`** — an `agents.tracing.processor_interface.TracingProcessor` subclass that converts each `Span` to a JSON line via `span_to_otlp_line()` and appends it to a gzipped JSONL file. Thread-safe; writes on `on_span_end`. Stamps every span with the `inference.*` projection keys (`inference.project_id`, `inference.observation_kind`, `inference.llm.model_name`, `inference.llm.input_tokens`, etc.) per the inference.net `07-export.md` spec — these are what the HALO Engine indexes on.
- **`setup_tracing()`** — the one function you call from your app. It builds an `ExportContext`, instantiates the processor at `$HALO_TRACES_PATH` (default `./traces.jsonl.gz`), registers it with `add_trace_processor(...)`, and returns the processor so you can call `.shutdown()` before exit.

The module is vendored, not packaged — copy it as-is and don't edit it. Future spec changes will land in the demo copy first.

## Wire it into your app

Call `setup_tracing()` once at startup, before constructing any `Agent`, and call `processor.shutdown()` before exit so the gzip stream flushes:

```python
from dotenv import load_dotenv

from tracing import setup_tracing
from my_app import build_agent  # your existing factory


def main():
    load_dotenv()
    processor = setup_tracing(service_name="my-agent", project_id="my-project")
    try:
        agent = build_agent()
        # ... Runner.run_sync(agent, question) etc.
    finally:
        processor.shutdown()
```

Order matters: `setup_tracing()` must run before the first `Agent(...)` so the processor is in place when the SDK starts emitting trace lifecycle events.

## Run your app and unzip the traces

```bash
uv run main.py "your question"
```

Traces land at `./traces.jsonl.gz` (or wherever `HALO_TRACES_PATH` points).

**The HALO Engine reads plain JSONL, not gzip.** `engine/traces/trace_index_builder.py` opens the trace file with `Path.open("rb")` and parses one span per line — it does not transparently decompress. Before pointing the Engine at a traces file, decompress it:

```bash
gunzip traces.jsonl.gz          # produces traces.jsonl, removes the .gz
# or, keep the gzipped copy:
gunzip -k traces.jsonl.gz       # produces traces.jsonl, keeps traces.jsonl.gz
```

Then pass `traces.jsonl` (not `.gz`) to the Engine. The index builder writes a sidecar `traces.jsonl.engine-index.jsonl` next to it on first read.

## Hosted (HALO Cloud)

> **Not yet launched.** When the hosted ingest endpoint ships, this section will describe how to swap `InferenceOtlpFileProcessor` for an HTTP processor that POSTs the same JSONL line shape to HALO Cloud — same `ExportContext`, same `span_to_otlp_line` projection, different sink. Until then, the local file path above is the only supported sink.

## Trace shape

A single agent run produces a tree like this (one `Trace` → many `Span`s):

```
agent.MyAgent              (AGENT)
├── response.gpt-4o-mini   (LLM)   turn 1
├── function.grep          (TOOL)
├── response.gpt-4o-mini   (LLM)   turn 2
├── function.read_file     (TOOL)
└── response.gpt-4o-mini   (LLM)   turn 3, final
```

Each line in `traces.jsonl(.gz)` is one span. Every line carries OTLP-compatible identity (`trace_id`, `span_id`, `parent_span_id`, `name`, `kind`, `start_time`, `end_time`, `status`), a `resource.attributes` block, a `scope` block (`openai-agents-sdk` + version), and an `attributes` map containing both raw upstream keys and the normalised `inference.*` projection.

Selected attributes:

| Attribute | Example | Which span |
|---|---|---|
| `openinference.span.kind` | `AGENT`, `LLM`, `TOOL`, `CHAIN`, `GUARDRAIL` | all |
| `inference.observation_kind` | `AGENT`, `LLM`, `TOOL`, `CHAIN`, `GUARDRAIL`, `SPAN` | all |
| `inference.project_id` | whatever you passed to `setup_tracing()` | all |
| `inference.export.schema_version` | `1` | all |
| `llm.model_name` / `inference.llm.model_name` | `gpt-4o-mini` | LLM |
| `llm.input_messages`, `llm.output_messages` | JSON-encoded message arrays | LLM |
| `llm.input_messages.{i}.message.role` etc. | flat OpenInference projection | LLM |
| `llm.token_count.prompt` / `inference.llm.input_tokens` | `1234` | LLM |
| `llm.token_count.completion` / `inference.llm.output_tokens` | `56` | LLM |
| `tool.name`, `input.value`, `output.value` | `grep`, JSON args, JSON result | TOOL |
| `agent.name`, `agent.tools`, `agent.handoffs` | `MyAgent`, JSON arrays | AGENT |
| `service.name` (under `resource.attributes`) | from `ExportContext.service_name` | all |

The full vocabulary lives in the per-span-type converters in `tracing.py` — `_generation_attrs`, `_response_attrs`, `_function_attrs`, etc.

## Verify it's working

The demo ships [`verify_traces.py`](../../demo/openai-agents-sdk-demo/verify_traces.py), a stdlib-only assertion script. Copy it (or run it from the demo directory) against your output:

```bash
uv run python verify_traces.py traces.jsonl.gz
# OK: 23 spans passed all spec assertions
```

It checks: top-level keys present, `kind` is `SPAN_KIND_*`, `status.code` is `STATUS_CODE_*`, timestamps are ISO-8601 with nanosecond precision and trailing `Z`, and the four required `inference.*` keys (`schema_version`, `project_id`, `observation_kind`) are populated.

For an ad-hoc look:

```bash
zcat traces.jsonl.gz | head -1 | jq '{trace_id, span_id, name, kind, observation_kind: .attributes."inference.observation_kind"}'
zcat traces.jsonl.gz | jq -r '[.trace_id, .name, .attributes."inference.observation_kind"] | @tsv'
```

## Troubleshooting

**`traces.jsonl.gz` is empty or missing.**
You probably exited before `processor.shutdown()` ran, so the gzip stream never flushed its final block. Always call `processor.shutdown()` (or `processor.force_flush()`) before process exit. The demo's `main.py` is the canonical pattern.

**The HALO Engine errors out reading `traces.jsonl.gz` directly.**
Decompress first — see [Run your app and unzip the traces](#run-your-app-and-unzip-the-traces). The Engine reads the file as plain UTF-8 JSONL with no gzip layer.

**Duplicate span exports, or errors about uploading to `platform.openai.com`.**
The OpenAI Agents SDK ships a default `TracingProcessor` that uploads to OpenAI's trace dashboard. `add_trace_processor(...)` is *additive* — both run by default. If you only want the inference.net file:

```python
import os
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
# or, programmatically before setup_tracing():
#   import agents
#   agents.set_trace_processors([])
```

**Spans appear but `inference.llm.input_tokens` / `output_tokens` are `None`.**
The SDK only populates `usage` on `response` / `generation` spans for models where it was returned by the API. Check that your model returns usage in the OpenAI response payload — older / streaming-only configurations sometimes omit it.

**`agents.add_trace_processor` import fails.**
You're on an older `openai-agents` version. The trace processor API landed in 0.0.4+; bump with `uv add openai-agents@latest`.

**Lines in `traces.jsonl.gz` look fine but the Engine reports `0 traces`.**
After decompression, check that `inference.project_id` is set on every line — the index builder filters on it. Pass `project_id=` to `setup_tracing()` (the default `"my-project"` works but is intentionally generic).

## See a working example

[`demo/openai-agents-sdk-demo/`](../../demo/openai-agents-sdk-demo/) is a runnable agent that uses exactly this `tracing.py`. It answers questions about a local codebase using three file tools (`list_files`, `grep`, `read_file`) and produces multi-turn traces suitable as Engine fixtures. Sample output is checked in at [`demo/openai-agents-sdk-demo/sample-traces/traces.jsonl.gz`](../../demo/openai-agents-sdk-demo/sample-traces/).
