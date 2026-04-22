# HALO demo: Claude Agent SDK with OTEL tracing

Runnable example for [`docs/integrations/claude-agent-sdk.md`](../../docs/integrations/claude-agent-sdk.md). Same toy as the OpenAI demo but using Claude Code's built-in `Read`, `Grep`, and `Glob` tools. Content-capture flags are ON by default so traces are useful as fixtures — see the integration guide for the flag table.

## Run it

Prereqs: the `claude` CLI on `$PATH` (`npm i -g @anthropic-ai/claude-code`), an Anthropic API key, and a clone of [github.com/context-labs/otel-interceptor](https://github.com/context-labs/otel-interceptor).

```bash
# Terminal 1 — interceptor
cd ~/dev/otel-interceptor
task install && task start

# Terminal 2 — compactor (one-line-per-trace JSONL)
cd ~/dev/otel-interceptor
task compact:watch:all

# Terminal 3 — demo
cd ~/dev/HALO/demo/claude-agent-sdk-demo
uv sync
cp .env.example .env                # fill in ANTHROPIC_API_KEY
uv run main.py "Where is tracing configured in this repo?" --root ../..
```

Compacted traces land in `~/dev/otel-interceptor/data/traces-by-trace.jsonl`. For integration into your own app, hosted sink setup, content-capture flag reference, and trace shape, see [`docs/integrations/claude-agent-sdk.md`](../../docs/integrations/claude-agent-sdk.md).
