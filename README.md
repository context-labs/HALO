# HALO

**H**ierarchical **A**gent **L**oop **O**ptimizer — open-source tooling for capturing, reviewing, and improving agent traces.

HALO ships minimal demos that show how to wire any agent framework to OpenTelemetry using [OpenInference](https://github.com/Arize-ai/openinference) attribute conventions. Traces can be captured locally with the [otel-interceptor](https://github.com/context-labs/otel-interceptor) or sent to the HALO hosted ingest service.

## Demos

- [`demo/openai-agents-sdk-demo/`](demo/openai-agents-sdk-demo) — OpenAI Agents SDK with OTEL tracing. Multi-turn tool calls over a local codebase.

More demos (Claude Agent SDK, integration guides) coming soon.

## RLM harness

- [`rlm/`](rlm) — the HALO Recursive Language Model harness. An LLM-driven agent + CLI + web UI for exploring OpenInference trace datasets. Run it locally with `uv run halo ingest/ask/serve`, or call the hosted Modal endpoint with any OpenAI-format client. See [`rlm/README.md`](rlm/README.md) for the full setup and API.

## License

MIT. See `LICENSE` when added.
