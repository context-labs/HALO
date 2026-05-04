# HALO CLI

Thin Typer wrapper around the HALO engine that streams the engine over a JSONL trace file.

## Install

```bash
pip install halo-engine
```

This installs the `halo` script onto your `PATH`. No extra configuration — the script is registered as a console entry point in the `halo-engine` wheel.

Verify:

```bash
halo --help
```

### Setup

The engine needs real LLM access:

```bash
export OPENAI_API_KEY=sk-...
```

## Usage

```bash
halo TRACE_PATH --prompt "your question"
```

### Required

| Arg              | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `TRACE_PATH`     | JSONL trace file (e.g. `tests/fixtures/realistic_traces.jsonl`) |
| `--prompt`, `-p` | User prompt sent to the root agent                              |

### Options

| Flag                 | Default            | Description                                                                                                                                                                                                                      |
| -------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model`, `-m`      | `gpt-5.4-mini`     | Model name for root, sub, synthesis, and compaction                                                                                                                                                                              |
| `--max-depth`        | `1`                | Max subagent recursion depth                                                                                                                                                                                                     |
| `--max-turns`        | `8`                | Max turns per agent                                                                                                                                                                                                              |
| `--max-parallel`     | `2`                | Max concurrent subagents                                                                                                                                                                                                         |
| `--instructions`     | _(engine default)_ | Override default trace-tool agent instructions                                                                                                                                                                                   |
| `--reasoning-effort` | _(model default)_  | Reasoning effort for root, subagent, and synthesis calls. One of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`. Compaction never uses reasoning. Omit to use the model family's documented max for known reasoning models. |

## Example

```bash
halo tests/fixtures/realistic_traces.jsonl \
  -p "What are the most common failure modes?" \
  --max-depth 2 \
  --max-turns 12 \
  --reasoning-effort high
```

Output streams to stdout: text deltas inline, then a rule-separated panel for each agent output item.

## Telemetry (optional)

HALO can emit OpenInference-shaped traces of its **own** LLM, tool, and agent activity — useful when you're tuning HALO and want to inspect what it actually did. Off by default; nothing is emitted unless you pass `--telemetry`.

### Enable on a run

```bash
halo TRACE_PATH --prompt "..." --telemetry
```

### Routing

The destination is decided by env vars:

- `CATALYST_OTLP_TOKEN` set → spans are uploaded to **inference.net Catalyst** over OTLP. Requires the optional `telemetry` extra: `pip install 'halo-engine[telemetry]'` (Python ≥ 3.11).
- `CATALYST_OTLP_TOKEN` unset → spans are written to a **local JSONL file** at `./halo-telemetry-{run_id}.jsonl` in the current working directory.

Either way, every span carries a `halo.run_id` resource attribute so you can filter to a single run.

### Environment variables

| Var | Default | Purpose |
|---|---|---|
| `CATALYST_OTLP_TOKEN` | *(unset)* | If set, uploads to Catalyst over OTLP. If unset, writes JSONL locally. |
| `CATALYST_OTLP_ENDPOINT` | catalyst-tracing default | OTLP endpoint **base URL** (e.g. `https://telemetry.inference.net`). catalyst-tracing appends `/v1/traces` automatically — do **not** include the path, or you'll get a `.../v1/traces/v1/traces` 404 and silently no traces. |
| `CATALYST_SERVICE_NAME` | `halo-engine` | Service identifier on traces. |
| `CATALYST_DEBUG` | *(unset)* | Set to `1` to surface OTLP export errors at WARNING level. Useful for troubleshooting "no errors, no traces" — the default `BatchSpanProcessor` swallows export failures. |
| `HALO_TELEMETRY_PATH` | `./halo-telemetry-{run_id}.jsonl` | Local fallback file path. Only consulted when `CATALYST_OTLP_TOKEN` is unset. |

### Local file format

The local JSONL is the inference.net OTLP-shaped form that HALO itself ingests, so traces produced by running HALO can be loaded back into HALO for analysis.

### Notes

- Enabling `--telemetry` clears the openai-agents SDK's default trace processor (which would otherwise upload to OpenAI's dashboard). HALO's own LLM traffic stays out of OpenAI's dashboard while telemetry is on.
- When telemetry is off (the default), no env vars are read and no files are written.

## Developing locally

If you want to hack on the CLI or the engine itself, install from a checkout of this repo with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/context-labs/HALO
cd HALO
uv sync
```

`uv sync` creates `.venv/` and installs `halo-engine` in editable mode. Use `uv run halo ...` (or activate the venv) to invoke the CLI against your local checkout.
