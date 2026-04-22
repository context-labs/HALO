# HALO demo: OpenAI Agents SDK with OTEL tracing

Runnable example for [`docs/integrations/openai-agents-sdk.md`](../../docs/integrations/openai-agents-sdk.md). A toy file-aware code helper: three tools (`list_files`, `grep`, `read_file`) scoped to `--root`; multi-turn loop that answers questions with file:line citations.

## Run it

Prereqs: ripgrep on `$PATH`, an OpenAI API key, and a clone of [github.com/context-labs/otel-interceptor](https://github.com/context-labs/otel-interceptor).

```bash
# Terminal 1 — interceptor
cd ~/dev/otel-interceptor
task install && task start

# Terminal 2 — compactor (one-line-per-trace JSONL)
cd ~/dev/otel-interceptor
task compact:watch:all

# Terminal 3 — demo
cd ~/dev/HALO/demo/openai-agents-sdk-demo
uv sync
cp .env.example .env                # fill in OPENAI_API_KEY
uv run main.py "Where is tracing configured in this repo?" --root ../..
```

Compacted traces land in `~/dev/otel-interceptor/data/traces-by-trace.jsonl`. For integration into your own app, hosted sink setup, and trace shape, see [`docs/integrations/openai-agents-sdk.md`](../../docs/integrations/openai-agents-sdk.md).
