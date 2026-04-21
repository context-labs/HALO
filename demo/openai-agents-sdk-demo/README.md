# HALO demo: OpenAI Agents SDK with OTEL tracing

A minimal OpenAI Agents SDK agent that answers questions about a local codebase using three read-only tools (`list_files`, `grep`, `read_file`). Every LLM call, tool call, and agent turn is exported as an OTEL trace with [OpenInference](https://github.com/Arize-ai/openinference) attributes.

## Run it locally (self-hosted sink)

You need the [otel-interceptor](https://github.com/context-labs/otel-interceptor) running in another terminal.

```bash
# Terminal 1 — interceptor
cd ~/dev/otel-interceptor
task install
task start

# Terminal 2 — demo
cd ~/dev/HALO/demo/openai-agents-sdk-demo
uv sync
cp .env.example .env      # fill in OPENAI_API_KEY
uv run main.py "Where is tracing configured in this repo?" --root ../..

# Terminal 3 (optional) — watch captured traces
cd ~/dev/otel-interceptor
task tail:traces
```

Captured spans land in `~/dev/otel-interceptor/data/traces.ndjson`. Run `task compact:traces` in that repo to get one-line-per-trace JSONL.

## Run it against the HALO hosted ingest (when available)

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://ingest.halo.inference.net/v1/traces
export OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer sk_live_..."
uv run main.py "..." --root ../..
```

Or put those two variables in `.env` instead of exporting.

## Smoke test without any receiver

```bash
OTEL_TRACES_EXPORTER=console uv run main.py "list the python files" --root ../..
```
Spans print to stdout as JSON-ish blocks. Useful to sanity-check that tracing is wired up without running the interceptor.

## Requirements

- Python 3.12+ (managed by uv)
- `rg` (ripgrep) on `$PATH`
- An OpenAI API key in `.env`

## How it works

- `tracing.py` — ~25 lines wiring OpenInference's OpenAI Agents instrumentor onto an OTEL `TracerProvider` with either an OTLP HTTP exporter or a console exporter.
- `agent.py` — three `@function_tool`s scoped to a `--root` directory; one `Agent` with instructions that push toward a list/grep → read → answer loop.
- `main.py` — typer CLI that loads `.env`, calls `setup_tracing()`, and runs the agent via `Runner.run_sync`.

The entirety of the "how do I add OTEL to my own OpenAI Agents app" integration lives in `tracing.py`. Copy that file into your project, import `setup_tracing()`, call it once at startup, done.
