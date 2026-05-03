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

HALO can emit OpenInference-shaped traces of its **own** LLM, tool, and agent activity to a local JSONL file — useful when you're tuning HALO and want to inspect what it actually did. Off by default; nothing is written unless you pass `--telemetry`.

### Enable on a run

```bash
halo TRACE_PATH --prompt "..." --telemetry
```

### Where the spans go

Spans are written to `./halo-telemetry-{run_id}.jsonl` in the current working directory. Override with:

```bash
export HALO_TELEMETRY_PATH=/some/path.jsonl
```

The file format is the inference.net OTLP-shaped JSONL that HALO itself ingests, so traces produced by running HALO can be loaded back into HALO for analysis.

Every span carries a `halo.run_id` resource attribute matching the file suffix, useful for filtering when multiple runs share a path.

### Notes

- Enabling `--telemetry` clears the openai-agents SDK's default trace processor (which would otherwise upload to OpenAI's dashboard). HALO's own LLM traffic stays out of OpenAI's dashboard while telemetry is on.
- When telemetry is off (the default), no env vars are read and no files are written.
- A direct upload path to inference.net Catalyst is planned; it is currently blocked on an upstream incompatibility between catalyst-tracing's OpenAI instrumentation and openai-agents 0.14+ streaming responses.

## Developing locally

If you want to hack on the CLI or the engine itself, install from a checkout of this repo with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/context-labs/HALO
cd HALO
uv sync
```

`uv sync` creates `.venv/` and installs `halo-engine` in editable mode. Use `uv run halo ...` (or activate the venv) to invoke the CLI against your local checkout.
