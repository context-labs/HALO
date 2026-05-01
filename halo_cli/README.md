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

| Arg | Description |
|---|---|
| `TRACE_PATH` | JSONL trace file (e.g. `tests/fixtures/realistic_traces.jsonl`) |
| `--prompt`, `-p` | User prompt sent to the root agent |

### Options

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `gpt-5.4-mini` | Model name for root, sub, synthesis, and compaction |
| `--max-depth` | `1` | Max subagent recursion depth |
| `--max-turns` | `8` | Max turns per agent |
| `--max-parallel` | `2` | Max concurrent subagents |
| `--instructions` | *(engine default)* | Override default trace-tool agent instructions |

## Example

```bash
halo tests/fixtures/realistic_traces.jsonl \
  -p "What are the most common failure modes?" \
  --max-depth 2 \
  --max-turns 12
```

Output streams to stdout: text deltas inline, then a rule-separated panel for each agent output item.

## Telemetry (optional)

HALO can emit OpenInference traces of its **own** LLM, tool, and agent activity — useful when you're tuning HALO and want to see what it actually did. This is opt-in and off by default; nothing is written or sent unless you pass `--telemetry`.

### Install the extra

The telemetry path requires Python ≥3.11 and a separate optional dependency:

```bash
pip install 'halo-engine[telemetry]'
```

### Enable on a run

```bash
halo TRACE_PATH --prompt "..." --telemetry
```

### Where the spans go

| Condition | Destination |
|---|---|
| `CATALYST_OTLP_TOKEN` set | OTLP upload to inference.net Catalyst |
| `CATALYST_OTLP_TOKEN` unset | Local JSONL file |

Local file path defaults to `./halo-telemetry-{run_id}.jsonl`. Override with `HALO_TELEMETRY_PATH=/some/path.jsonl`.

### Environment variables

| Var | Default | Purpose |
|---|---|---|
| `CATALYST_OTLP_TOKEN` | *(unset)* | If set, uploads to Catalyst over OTLP. If unset, writes JSONL locally. |
| `CATALYST_OTLP_ENDPOINT` | catalyst-tracing default | OTLP endpoint URL. |
| `CATALYST_SERVICE_NAME` | `halo-engine` | Service identifier on traces. |
| `HALO_TELEMETRY_PATH` | `./halo-telemetry-{run_id}.jsonl` | Local fallback file path. |

Every span carries a `halo.run_id` resource attribute so a single run is filterable in Catalyst.

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
