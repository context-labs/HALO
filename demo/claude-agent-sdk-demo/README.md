# HALO demo: Claude Agent SDK with OTEL tracing

A minimal Claude Agent SDK agent that answers questions about a local codebase using Claude Code's built-in `Read`, `Grep`, and `Glob` tools. Every LLM call, tool call, and interaction is exported as an OTEL trace under a single OpenInference `AGENT` root span.

## Architecture

This demo uses **Claude Code's native OTEL telemetry** (spans emitted by the CLI subprocess) plus **one OpenInference `AGENT` root span** emitted from Python. The Claude Agent SDK's `[otel]` extra propagates the W3C trace context into the subprocess, so every trace is one tree:

```
halo-claude-agents-demo  AGENT  (our root)
└── claude-code  claude_code.interaction
    ├── claude-code  claude_code.llm_request
    ├── claude-code  claude_code.tool.Grep
    ├── claude-code  claude_code.llm_request
    └── claude-code  claude_code.tool.Read
```

Attribute namespaces differ by span (OpenInference on the root, `claude_code.*` on children), but the semantic content (model, tokens, messages, tool args, tool output) is all captured.

## Prereqs

One-time:
```bash
npm i -g @anthropic-ai/claude-code   # provides the `claude` CLI on $PATH
```

Per machine:
- Python 3.12+ (managed by uv)
- An Anthropic API key, either in `.env` as `ANTHROPIC_API_KEY=...` or via `claude login`

## Run it locally (self-hosted sink)

You need the [otel-interceptor](https://github.com/context-labs/otel-interceptor) running in another terminal.

```bash
# Terminal 1 — interceptor
cd ~/dev/otel-interceptor
task install
task start

# Terminal 2 — demo
cd ~/dev/HALO/demo/claude-agent-sdk-demo
uv sync
cp .env.example .env      # fill in ANTHROPIC_API_KEY
uv run main.py "Where is tracing configured in this repo?" --root ../..

# Terminal 3 (optional) — watch captured traces
cd ~/dev/otel-interceptor
task tail:traces
```

Captured spans land in `~/dev/otel-interceptor/data/traces.ndjson`. Run `task compact:traces` there to get one-line-per-trace JSONL.

## Run it against the HALO hosted ingest (when available)

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://ingest.halo.inference.net
export OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer sk_live_..."
uv run main.py "..." --root ../..
```

Or put those two variables in `.env` instead of exporting.

## Content-capture flags

The demo defaults all Claude Code content-capture flags to ON so traces are useful as fixtures:

| Env var | What it captures |
|---|---|
| `OTEL_LOG_USER_PROMPTS=1` | User prompt text in `claude_code.user_prompt` events. |
| `OTEL_LOG_TOOL_DETAILS=1` | Bash commands, file paths, tool args. |
| `OTEL_LOG_TOOL_CONTENT=1` | Raw tool I/O in span events (truncated at 60 KB). |
| `OTEL_LOG_RAW_API_BODIES=1` | Full Anthropic API request/response bodies. |

If you don't want any of these in your traces, set that flag to `0` in your `.env`.

## How it works

- `tracing.py` — OTEL `TracerProvider` + OTLP HTTP exporter + env-var setup for Claude Code's native telemetry. Exposes `setup_tracing()` (call once at startup) and `agent_span(name, model)` (context manager opening an OpenInference `AGENT` root span).
- `main.py` — typer CLI. Loads `.env`, calls `setup_tracing()`, then runs `asyncio.run(_run(...))` which opens the agent span and iterates the async `query()` generator, collecting the last assistant text.

The entirety of the "how do I add OTEL to my own Claude Agent SDK app" integration lives in `tracing.py`. Copy that file into your project, call `setup_tracing()` at startup, wrap your `query()` call in `agent_span()`, done.
