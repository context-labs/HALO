# HALO demo: OpenAI Agents SDK → inference.net JSONL

Runnable example for [`docs/integrations/openai-agents-sdk.md`](../../docs/integrations/openai-agents-sdk.md). A toy file-aware code helper: three tools (`list_files`, `grep`, `read_file`) scoped to `--root`; multi-turn loop that answers questions with file:line citations.

The demo registers an `InferenceOtlpFileProcessor` (vendored from the inference.net monorepo) as an openai-agents `TracingProcessor` and writes one JSONL span per `.jsonl.gz` line in the format defined by `07-export.md`. No OTEL collector, no OTLP exporter, no external sink.

## Run it

Prereqs: ripgrep on `$PATH` and an OpenAI API key.

```bash
cd ~/dev/HALO/demo/openai-agents-sdk-demo
uv sync
cp .env.example .env                # fill in OPENAI_API_KEY
uv run main.py "Where is tracing configured in this repo?" --root ../..
```

Traces land in `./traces.jsonl.gz`. For integration into your own app, trace shape, and the RLM handoff, see [`docs/integrations/openai-agents-sdk.md`](../../docs/integrations/openai-agents-sdk.md).
