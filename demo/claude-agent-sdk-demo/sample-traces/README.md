# Sample traces

Captured from running this demo against the HALO repo itself. Useful as reference fixtures for trace-shape inspection, downstream tooling work (e.g. the HALO harness in INF-2847), and documentation examples.

## Files

- **`raw-traces.ndjson`** — what the otel-interceptor captured on disk, one line per OTLP export batch. Each line is a JSON object with `received_at`, `transport`, `signal`, `source`, and `payload` (the decoded OTLP `ExportTraceServiceRequest`). This is the format HALO's RLM will consume.
- **`traces-by-trace.jsonl`** — post-processed via the interceptor's `task compact:traces`. One line per `traceId`, with all spans for that trace inlined and sorted by start time. Easier to eyeball.

## What's in these traces

Three runs of the demo, each a multi-turn agent loop. The agent used `Grep`, `Glob`, and `Read` to answer questions about HALO's own source. Every trace has:

- A **root span** named `claude-agent.run` with `openinference.span.kind = "AGENT"` and `llm.model_name = "claude-sonnet-4-6"`, under `service.name = "halo-claude-agents-demo"`.
- **Child spans emitted by the Claude Code CLI subprocess** (`claude_code.interaction`, `claude_code.llm_request`, `claude_code.tool.Grep/Glob/Read`) under `service.name = "claude-code"`, carrying model name, token counts, full input/output messages (since `OTEL_LOG_RAW_API_BODIES=1`), and tool args/results (since `OTEL_LOG_TOOL_DETAILS=1` + `OTEL_LOG_TOOL_CONTENT=1`).

All spans share a single `traceId` thanks to the `claude-agent-sdk[otel]` extra propagating `TRACEPARENT` into the subprocess.

## Attribute shape note

Unlike the OpenAI demo (which emits OpenInference attributes throughout), this demo emits **one OpenInference-shaped root span plus Claude Code-native children**. Downstream consumers (e.g. HALO's ingest layer) normalize across both shapes — see the `concepts/vendor-attribute-routing` wiki page for the pattern.

## Reproducing

```bash
# Terminal 1
cd ~/dev/otel-interceptor && task clean && task start

# Terminal 2
cd ~/dev/HALO/demo/claude-agent-sdk-demo
uv run main.py "Where is tracing configured in this repo?" --root ../..
uv run main.py "What does the agent_span context manager do, and where is it defined?" --root ../..
uv run main.py "Which file documents the sink story, and what are the three sinks?" --root ../..

# Back in otel-interceptor
task compact:traces
# Output lands in data/traces.ndjson and data/traces-by-trace.jsonl
```

No secrets are embedded — scanned for `sk-*` and `authorization`/`bearer` before committing. The spans contain prompts and completions about HALO's own source code (public).
