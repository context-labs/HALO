# HALO — OpenAI Agents SDK integration

Wire your existing OpenAI Agents SDK app into HALO's OpenInference-shaped OTEL pipeline. Both local (self-hosted `otel-interceptor` + compactor) and hosted (HALO Cloud) sinks are covered. After this guide, every LLM call and tool call in your agent becomes an OTEL span with OpenInference attributes, captured on disk locally or forwarded to HALO Cloud with no code change between the two.

> **Hosted sink not yet launched.** The endpoint and API key format are TODO. Fill in `<TODO-halo-ingest-endpoint>` and `<TODO-halo-api-key>` once we publish them. Local sink works today — scroll to [Pick a sink → Local](#local-otel-interceptor).

## Prereqs

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- An OpenAI API key (`OPENAI_API_KEY`)
- A clone of [`github.com/context-labs/otel-interceptor`](https://github.com/context-labs/otel-interceptor) for the local sink path

## Install

Add these dependencies to your project:

```toml
[project]
dependencies = [
    "openai-agents",
    "openinference-instrumentation-openai-agents",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp-proto-http",
    "python-dotenv",
]
```

Or in one line with uv:

```bash
uv add openai-agents openinference-instrumentation-openai-agents \
       opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
       python-dotenv
```

## The tracing snippet

Drop this `tracing.py` into your project. This is the whole integration.

```python
"""Minimal OpenInference + OTEL wiring. Copy-paste into any OpenAI Agents SDK app."""
import os
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

DEFAULT_OTLP_ENDPOINT = "http://localhost:4318"


def setup_tracing(service_name: str = "my-agent") -> TracerProvider:
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT).rstrip("/")
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces")))
    OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)
    return provider
```

What each block does:

- `Resource.create({"service.name": ...})` tags every span so you can group them in any backend. Without this, spans appear as `unknown_service`.
- `SimpleSpanProcessor` over `BatchSpanProcessor` exports one span at a time. For short-lived agent runs, batching adds flush-at-exit friction without meaningful throughput gain.
- `OTLPSpanExporter` from the HTTP/protobuf package avoids pulling in `grpcio` (~40 MB of native code).
- `OTEL_EXPORTER_OTLP_ENDPOINT` is treated as a base URL per OTEL spec; `/v1/traces` is appended in code. `OTEL_EXPORTER_OTLP_HEADERS` is read natively by the OTEL SDK, so hosted-sink auth works without code changes.
- `OpenAIAgentsInstrumentor().instrument(...)` hooks the OpenAI Agents SDK's `TracingProcessor` ABC so every `Agent` / `Runner` call, LLM call, and tool call becomes an OpenInference span automatically.

## Wire it into your app

Call `setup_tracing()` once at startup, before building any `Agent`:

```python
from dotenv import load_dotenv

from tracing import setup_tracing
from my_app import build_agent  # your existing factory

def main():
    load_dotenv()
    setup_tracing(service_name="my-agent")
    agent = build_agent()
    # ... Runner.run_sync(agent, question) or Runner.run(...) as you already do
```

Order matters: `setup_tracing()` must run before the first `Agent(...)` construction — the instrumentor hooks methods at install time.

## Pick a sink

Same `tracing.py`. Different destination.

### Local (otel-interceptor)

Three terminals run in parallel. All three are required — the interceptor receives OTLP export requests, the compactor turns batch-fragmented NDJSON into the canonical one-line-per-trace shape, and your agent emits the traces.

```bash
# Terminal 1 — OTLP receiver. Listens on :4318 (HTTP) + :4317 (gRPC).
cd ~/dev/otel-interceptor
task install    # one-time
task start
```

```bash
# Terminal 2 — compactor. Groups spans by traceId and writes JSONL.
cd ~/dev/otel-interceptor
task compact:watch:all
```

```bash
# Terminal 3 — your agent.
uv run main.py
```

The default endpoint in `tracing.py` is `http://localhost:4318`, matching the interceptor's HTTP port. No env vars needed.

**Why both interceptor and compactor?** The interceptor appends one line to `data/traces.ndjson` per OTLP export *request* — spans from one trace can split across many lines as batches flush. Downstream HALO tooling (including the RLM harness in this repo) expects one-line-per-*trace*, which is what `task compact:watch:all` produces at `data/traces-by-trace.jsonl`. Running only the interceptor gets you raw telemetry; running both gets you HALO-compatible traces.

See the [otel-interceptor README](https://github.com/context-labs/otel-interceptor) for install details, alternative transports (gRPC, HTTP/JSON), and the full list of tasks.

### Hosted (HALO Cloud)

> **Not yet launched.** The two values below are placeholders. Once HALO Cloud publishes an ingest endpoint and API key format, replace them.

Set two env vars in your shell or `.env`:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=<TODO-halo-ingest-endpoint>
export OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer <TODO-halo-api-key>"
```

Run your agent as normal. **No compactor step** — HALO Cloud compacts server-side.

The OTEL OTLP HTTP exporter reads both env vars natively, so `tracing.py` needs no changes to switch sinks.

## Trace shape

A single agent run produces an OpenInference tree, all in-process:

```
AgentWorkflow
└── Agent
    ├── LLM   (turn 1)
    ├── Tool  (e.g. grep)
    ├── LLM   (turn 2)
    ├── Tool  (e.g. read_file)
    └── LLM   (turn 3, final)
```

Every span carries these attributes in the OpenInference shape:

| Attribute | Example | Which span |
|---|---|---|
| `openinference.span.kind` | `AGENT`, `LLM`, `TOOL` | all |
| `llm.model_name` | `gpt-4o-mini` | LLM |
| `llm.input_messages` | JSON array of `{role, content}` objects | LLM |
| `llm.output_messages` | JSON array of `{role, content}` objects | LLM |
| `llm.token_count.prompt` | `1234` | LLM |
| `llm.token_count.completion` | `56` | LLM |
| `tool.name` | `grep` | TOOL |
| `tool.parameters` | JSON string of the tool arguments | TOOL |
| `service.name` | whatever you passed to `setup_tracing()` | all |

All spans share one `trace_id`. Child spans carry their parent's `span_id` in `parent_span_id`.

## Verify it's working

With the interceptor and compactor running, and your agent emitting traces:

```bash
# Confirm export requests are arriving.
tail -n 1 ~/dev/otel-interceptor/data/traces.ndjson | jq '.payload.resourceSpans[0].resource'

# Confirm the compacted file has one line per trace.
tail -n 1 ~/dev/otel-interceptor/data/traces-by-trace.jsonl | jq '{traceId, spanCount, services, spanNames: [.spans[].name]}'
```

A healthy compacted line looks like:

```json
{
  "traceId": "c39c72dabeb2c3fe8aab4fdacb5805d8",
  "spanCount": 7,
  "services": ["my-agent"],
  "spanNames": ["AgentWorkflow", "Agent", "LLM", "Tool", "LLM", "Tool", "LLM"]
}
```

Confirm the `openinference.span.kind` attribute appears on each span:

```bash
tail -n 1 ~/dev/otel-interceptor/data/traces-by-trace.jsonl \
  | jq '[.spans[] | {name, kind: (.attributes[] | select(.key == "openinference.span.kind") | .value.stringValue)}]'
```

If every span has a kind, instrumentation is working end-to-end.

## Troubleshooting

**Nothing appears in `traces.ndjson`.**
Your process may be exiting before the first export. Add `OTEL_TRACES_EXPORT_INTERVAL=1000` to your `.env` while debugging. `SimpleSpanProcessor` already exports synchronously, so an empty file usually means the exporter never reached the receiver (see next entry).

**`Connection refused` on port 4318.**
The interceptor isn't running. Start it with `task start` in the interceptor repo. Confirm nothing else is bound: `lsof -iTCP:4318 -sTCP:LISTEN`.

**Duplicate span exports, or errors about uploading to `platform.openai.com`.**
The OpenAI Agents SDK ships a default `TracingProcessor` that uploads to OpenAI's trace dashboard. If you see both destinations exporting, disable the built-in one:

```python
import os
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
# or, programmatically before setup_tracing():
#   import agents
#   agents.set_trace_processors([])
```

**Spans arrive but OpenInference attributes are empty.**
`OpenAIAgentsInstrumentor().instrument(...)` must run before the first `Agent(...)` construction. Move `setup_tracing()` to the top of `main()` and re-run.

**`OTEL_EXPORTER_OTLP_HEADERS` format errors.**
The value is comma-separated `key=value` pairs, *not* a JSON object. Use `authorization=Bearer <TODO-halo-api-key>`, not `{"authorization": "Bearer <TODO-halo-api-key>"}`.

**404 on `/v1/traces/v1/traces`.**
Your `OTEL_EXPORTER_OTLP_ENDPOINT` includes the path suffix. `tracing.py` appends `/v1/traces` itself. Set the env var to a base URL with no path.

## See a working example

[`demo/openai-agents-sdk-demo/`](../../demo/openai-agents-sdk-demo/) in this repo is a runnable agent that uses exactly this `tracing.py`. It answers questions about a local codebase using three file tools (`list_files`, `grep`, `read_file`) and produces multi-turn traces suitable as fixtures. See also the [OpenInference instrumentor docs](https://github.com/Arize-ai/openinference/tree/main/python/instrumentation/openinference-instrumentation-openai-agents).
