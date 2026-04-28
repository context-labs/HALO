# Sample traces

Captured from running this demo against its own directory. Useful as reference fixtures for trace-shape inspection and downstream tooling work.

## File

**`traces.jsonl.gz`** — one JSON object per line, one line per span. Shape is the OTLP-compatible JSONL format defined by the inference.net export spec (`07-export.md` in the monorepo), produced directly by the `InferenceOtlpFileProcessor` in [`../inference_otlp_exporter.py`](../inference_otlp_exporter.py). This is the format the RLM consumes.

Each line carries:
- OTLP span identity (`trace_id`, `span_id`, `parent_span_id`, `name`, `kind`, `start_time`, `end_time`, `status`).
- `resource.attributes` (service identity) and `scope` (`openai-agents-sdk` + version).
- `attributes` — raw upstream attributes (`openinference.span.kind`, `llm.*`, `tool.*`, `agent.*`, `sdk.data.*`) plus normalised `inference.*` projections (`inference.observation_kind`, `inference.llm.*`, `inference.project_id`, etc.).

## What's in these traces

Three runs of the demo against this directory as root. 3 traces / 58 spans total:

| Trace | Question | Mix |
|---|---|---|
| `301d…` | "Which Python file configures tracing? Answer with the filename only." | 1 AGENT · 9 LLM · 8 TOOL · 10 SPAN |
| `8455…` | "Which tools does the agent expose?" | 1 AGENT · 3 LLM · 2 TOOL · 4 SPAN |
| `a974…` | "What Python dependencies does this demo declare?" | 1 AGENT · 6 LLM · 6 TOOL · 7 SPAN |

Every trace has a single top-level `AGENT` span (`agent.HaloCodeHelper`), interleaved `response` (`LLM`) spans for each model turn, `function.*` (`TOOL`) spans for `list_files` / `grep` / `read_file` calls, and `custom.turn` / `custom.task` wrappers.

## Reproducing

```bash
cd ~/dev/HALO/demo/openai-agents-sdk-demo
HALO_TRACES_PATH=sample-traces/traces.jsonl.gz uv run main.py "Which Python file configures tracing? Answer with the filename only." --root .
HALO_TRACES_PATH=sample-traces/traces.jsonl.gz uv run main.py "Which tools does the agent expose?" --root .
HALO_TRACES_PATH=sample-traces/traces.jsonl.gz uv run main.py "What Python dependencies does this demo declare?" --root .
```

Each run appends to the gzipped JSONL file. Verify with `uv run python verify_traces.py sample-traces/traces.jsonl.gz`.

No secrets are embedded. The spans contain prompts and completions about this demo directory's own source code (public).
