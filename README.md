# HALO

**H**ierarchical **A**gent **L**oop **O**ptimizer — open-source tooling for capturing, reviewing, and improving agent traces.

HALO ships minimal demos that show how to wire any agent framework to OpenTelemetry. Traces can be captured locally with the [otel-interceptor](https://github.com/context-labs/otel-interceptor) or sent to the HALO hosted ingest service.

## Demos

- [`demo/openai-agents-sdk-demo/`](demo/openai-agents-sdk-demo) — OpenAI Agents SDK with OpenInference tracing. Multi-turn tool calls over a local codebase; custom `@function_tool`s for `list_files`/`grep`/`read_file`.
- [`demo/claude-agent-sdk-demo/`](demo/claude-agent-sdk-demo) — Claude Agent SDK with Claude Code native telemetry plus an OpenInference `AGENT` wrapper span. Multi-turn tool calls over a local codebase using Claude Code's built-in `Read`/`Grep`/`Glob`.

Integration guides (one per framework) land in `docs/` in the next phase.

## RLM harness

- [`rlm/`](rlm) — the HALO Recursive Language Model harness. An LLM-driven agent + CLI + web UI for exploring OpenInference trace datasets. Run it locally with `uv run halo ingest/ask/serve`, or call the hosted Modal endpoint with any OpenAI-format client. See [`rlm/README.md`](rlm/README.md) for the full setup and API.

## License

MIT. See `LICENSE` when added.
