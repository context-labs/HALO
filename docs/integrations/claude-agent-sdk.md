# HALO — Claude Agent SDK integration

Wire your existing Claude Agent SDK app into HALO's OTEL trace pipeline. Both local (self-hosted `otel-interceptor` + compactor) and hosted (HALO Cloud) sinks are covered. After this guide, every Claude LLM call and every tool call the Claude Code binary makes becomes an OTEL span, captured on disk locally or forwarded to HALO Cloud with no code change between the two.

> **Hosted sink not yet launched.** The endpoint and API key format are TODO. Fill in `<TODO-halo-ingest-endpoint>` and `<TODO-halo-api-key>` once we publish them. Local sink works today — scroll to [Pick a sink → Local](#local-otel-interceptor).

## How it works

Unlike a pure-Python SDK, the Claude Agent SDK spawns a `claude` CLI subprocess that emits its own OTEL spans. The integration has two halves:

1. The Python side emits one OpenInference `AGENT` wrapper span via a context manager.
2. The `claude` subprocess emits `claude_code.*` spans (one per LLM turn, one per tool invocation) as children of that AGENT span. W3C trace context propagates through environment variables injected by the SDK's `[otel]` extra.

Attribute namespaces differ by span (OpenInference on the root, `claude_code.*` on children), but semantic content — model, tokens, messages, tool args, tool output — is captured end-to-end.

## Prereqs

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Node 18+ and the Claude Code CLI: `npm i -g @anthropic-ai/claude-code` (the SDK spawns the `claude` binary; the CLI must be on `$PATH`)
- An Anthropic API key in `ANTHROPIC_API_KEY`, or run `claude login` once
- A clone of [`github.com/context-labs/otel-interceptor`](https://github.com/context-labs/otel-interceptor) for the local sink path

## Install

```toml
[project]
dependencies = [
    "claude-agent-sdk[otel]",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp-proto-http",
    "python-dotenv",
]
```

The `[otel]` extra is what propagates W3C trace context into the subprocess — without it, only the Python-side AGENT span will reach the receiver.

Or in one line with uv:

```bash
uv add 'claude-agent-sdk[otel]' opentelemetry-sdk opentelemetry-exporter-otlp-proto-http python-dotenv
```

## The tracing snippet

Drop this `tracing.py` into your project. This is the whole integration.

```python
"""Minimal Claude Agent SDK + OTEL wiring. Copy-paste into any Claude Agent SDK app."""
import os
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

DEFAULT_OTLP_ENDPOINT = "http://localhost:4318"

# Env vars the Claude Code CLI subprocess reads. setdefault so users can override
# any of them in their shell or .env file.
_CLAUDE_ENV = {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",   # distributed trace spans
    "OTEL_TRACES_EXPORTER": "otlp",
    "OTEL_METRICS_EXPORTER": "otlp",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    "OTEL_TRACES_EXPORT_INTERVAL": "1000",         # faster flush for short runs
    # Content capture — on by default so traces are useful as fixtures.
    "OTEL_LOG_USER_PROMPTS": "1",
    "OTEL_LOG_TOOL_DETAILS": "1",
    "OTEL_LOG_TOOL_CONTENT": "1",
    "OTEL_LOG_RAW_API_BODIES": "1",
}


def setup_tracing(service_name: str = "my-claude-agent") -> TracerProvider:
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT)
    for k, v in _CLAUDE_ENV.items():
        os.environ.setdefault(k, v)

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    endpoint = f"{os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'].rstrip('/')}/v1/traces"
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return provider


@contextmanager
def agent_span(name: str, model: str):
    tracer = trace.get_tracer("my-claude-agent")
    with tracer.start_as_current_span(
        name,
        attributes={
            "openinference.span.kind": "AGENT",
            "llm.model_name": model,
        },
    ) as span:
        yield span
```

What each block does:

- **`_CLAUDE_ENV` dict** — the `claude` subprocess reads these env vars at startup. `os.environ.setdefault` means values you set in your shell or `.env` take precedence. The content-capture flags default to ON because unredacted traces are more useful as fixtures; see [Content-capture flags](#content-capture-flags) to selectively disable.
- **Python-side `TracerProvider` + `OTLPSpanExporter`** — so the wrapper AGENT span ships to the same destination as the subprocess spans.
- **`trace.set_tracer_provider(provider)`** — registers globally so any tracer name resolves against this provider.
- **`agent_span()` context manager** — opens a single OpenInference `AGENT` span. The SDK's `[otel]` extra picks up the active W3C trace context from the Python tracer and injects it into the subprocess, so every `claude_code.*` span becomes a child of this one.

## Wire it into your app

Call `setup_tracing()` at startup, and wrap each `query(...)` call in `agent_span()`:

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions, query
from dotenv import load_dotenv

from tracing import agent_span, setup_tracing

async def run(question: str, model: str = "claude-sonnet-4-6") -> str:
    options = ClaudeAgentOptions(model=model)  # your existing options
    with agent_span("my-agent.run", model=model):
        async for message in query(prompt=question, options=options):
            ...  # process messages as you already do
    return "..."

def main():
    load_dotenv()
    setup_tracing(service_name="my-claude-agent")
    asyncio.run(run("what's up"))
```

`setup_tracing()` must run before the first `query(...)` call — the subprocess reads the env vars at spawn time.

## Pick a sink

Same `tracing.py`. Different destination.

### Local (otel-interceptor)

Three terminals in parallel.

```bash
# Terminal 1 — OTLP receiver.
cd ~/dev/otel-interceptor
task install    # one-time
task start
```

```bash
# Terminal 2 — compactor.
cd ~/dev/otel-interceptor
task compact:watch:all
```

```bash
# Terminal 3 — your agent.
uv run main.py
```

Default endpoint in `tracing.py` is `http://localhost:4318`, matching the interceptor's HTTP port. `setup_tracing()` appends `/v1/traces` internally. No env vars needed for the local sink.

**Why both interceptor and compactor?** The interceptor appends one line to `data/traces.ndjson` per OTLP export *request* — spans from one trace can split across many lines as batches flush. Downstream HALO tooling expects one-line-per-*trace*, which is what `task compact:watch:all` produces at `data/traces-by-trace.jsonl`. Running only the interceptor gets raw telemetry; running both gets HALO-compatible traces.

See the [otel-interceptor README](https://github.com/context-labs/otel-interceptor) for install details and alternative transports.

### Hosted (HALO Cloud)

> **Not yet launched.** The two values below are placeholders.

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=<TODO-halo-ingest-endpoint>
export OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer <TODO-halo-api-key>"
```

Run your agent. **No compactor step** — HALO Cloud compacts server-side.

Both env vars are read by the OTEL exporter on both sides of the split (Python-side wrapper span and subprocess-side `claude_code.*` spans land at the same endpoint with the same auth header).

## Trace shape

One agent run produces one trace with a cross-process span tree:

```
my-claude-agent  AGENT  (Python — opened by agent_span())
└── claude-code  claude_code.interaction
    ├── claude-code  claude_code.llm_request
    ├── claude-code  claude_code.tool.Grep
    ├── claude-code  claude_code.llm_request
    ├── claude-code  claude_code.tool.Read
    └── claude-code  claude_code.llm_request
```

Attribute namespaces differ by origin:

| Span | Attribute namespace | Example attributes |
|---|---|---|
| Python `AGENT` root | OpenInference | `openinference.span.kind=AGENT`, `llm.model_name` |
| `claude_code.interaction` | `claude_code.*` | session ID, duration |
| `claude_code.llm_request` | `gen_ai.*` / `claude_code.*` | `gen_ai.request.model`, token usage, stop reason, full request/response bodies if `OTEL_LOG_RAW_API_BODIES=1` |
| `claude_code.tool.<Name>` | `claude_code.*` | tool args, tool output (if `OTEL_LOG_TOOL_CONTENT=1`) |

All spans share one `trace_id`. `service.name` differs by emitter: the Python span carries the value from `setup_tracing(service_name=...)`; the subprocess spans carry `claude-code`.

## Content-capture flags

Claude Code's OTEL instrumentation redacts prompts and tool I/O by default. The shipped `tracing.py` flips all four flags ON via `os.environ.setdefault` so local traces are useful as fixtures. Set any of them to `0` in your shell or `.env` to opt out.

| Env var | What it captures |
|---|---|
| `OTEL_LOG_USER_PROMPTS=1` | User prompt text in `claude_code.user_prompt` events. |
| `OTEL_LOG_TOOL_DETAILS=1` | Bash commands, file paths, tool args. |
| `OTEL_LOG_TOOL_CONTENT=1` | Raw tool I/O in span events (truncated at 60 KB). |
| `OTEL_LOG_RAW_API_BODIES=1` | Full Anthropic API request/response bodies. |

**Production note.** If the traces you emit from production shouldn't contain raw prompts or API bodies, set any of these to `0` in the production environment *before* `setup_tracing()` runs — `os.environ.setdefault` will respect the user-set value.

## Verify it's working

With the interceptor and compactor running, and your agent emitting traces:

```bash
# Confirm export requests arrive.
tail -n 1 ~/dev/otel-interceptor/data/traces.ndjson | jq '.payload.resourceSpans[0].resource'

# Confirm the compacted file has one line per trace and contains both the
# Python AGENT span and Claude Code subprocess spans.
tail -n 1 ~/dev/otel-interceptor/data/traces-by-trace.jsonl \
  | jq '{traceId, spanCount, services, spanNames: [.spans[].name]}'
```

A healthy compacted line shows `services` containing both your service name and `claude-code`, and `spanNames` includes at least one `claude_code.interaction` plus one or more `claude_code.llm_request` and `claude_code.tool.<Name>`.

## Troubleshooting

**Nothing appears in `traces.ndjson`.**
`OTEL_TRACES_EXPORT_INTERVAL=1000` is set by `tracing.py`, but if your process exits very quickly the subprocess's first batch may not flush. Let the agent finish and wait a second or two before killing the process.

**`Connection refused` on port 4318.**
Interceptor isn't running. `task start` in the interceptor repo.

**Only the AGENT span arrives — no `claude_code.*` children.**
Check `claude --version` on `$PATH`; the SDK can't spawn the subprocess if the binary is missing. Also confirm the SDK was installed with the `[otel]` extra (`claude-agent-sdk[otel]`) — without it, W3C trace context doesn't propagate.

**`claude_code.*` spans arrive but without prompts/tool content.**
The content-capture flags default to ON in `tracing.py`, but if your environment pre-sets any of them to `0` that wins. Explicitly set them in your shell or `.env` if you need capture.

**Two sessions of spans appear in one agent run.**
Calling `setup_tracing()` more than once registers duplicate processors. Call it exactly once at startup.

**404 on `/v1/traces/v1/traces`.**
Your `OTEL_EXPORTER_OTLP_ENDPOINT` includes the path suffix. `tracing.py` appends `/v1/traces` itself, and the subprocess does the same. Set the env var to a base URL with no path.

## See a working example

[`demo/claude-agent-sdk-demo/`](../../demo/claude-agent-sdk-demo/) in this repo is a runnable agent that uses exactly this `tracing.py`. It answers questions about a local codebase using Claude Code's built-in `Read`, `Grep`, and `Glob` tools and produces multi-turn traces suitable as fixtures.
