# Sample traces

Captured from running this demo against the HALO repo itself. Useful as reference fixtures for trace-shape inspection, downstream tooling work (e.g. the HALO harness in INF-2847), and documentation examples.

## Files

- **`raw-traces.ndjson`** — what the otel-interceptor captured on disk, one line per OTLP export batch. Each line is a JSON object with `received_at`, `transport`, `signal`, `source`, and `payload` (the decoded OTLP `ExportTraceServiceRequest`). This is the format HALO's RLM will consume.
- **`traces-by-trace.jsonl`** — post-processed via the interceptor's `task compact:traces`. One line per `traceId`, with all spans for that trace inlined and sorted by start time. Easier to eyeball.

## What's in these traces

Three runs of the demo, each a multi-turn agent loop. The agent used `grep`, `list_files`, and `read_file` to answer questions about HALO's own source. Every trace has:

- A top-level `AGENT`-kind span (`openinference.span.kind = "AGENT"`) under `service.name = halo-openai-agents-demo`.
- One or more `LLM`-kind spans per turn, carrying `llm.model_name`, `llm.token_count.prompt`, `llm.token_count.completion`, `llm.input_messages`, and `llm.output_messages`.
- `TOOL`-kind spans with `tool.name` and `tool.parameters`.
- `CHAIN`-kind wrappers for the per-turn orchestration.

All spans share a single `traceId`.

## Reproducing

```bash
# Terminal 1
cd ~/dev/otel-interceptor && task start

# Terminal 2
cd ~/dev/HALO/demo/openai-agents-sdk-demo
uv run main.py "Where is tracing configured in this repo?" --root ../..

# Back in otel-interceptor
task compact:traces
# Output lands in data/traces.ndjson and data/traces-by-trace.jsonl
```

No secrets are embedded — scanned for `sk-*` and `authorization`/`bearer` before committing. The spans contain prompts and completions about HALO's own source code (public).
