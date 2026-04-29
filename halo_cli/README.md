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

## Developing locally

If you want to hack on the CLI or the engine itself, install from a checkout of this repo with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/context-labs/HALO
cd HALO
uv sync
```

`uv sync` creates `.venv/` and installs `halo-engine` in editable mode. Use `uv run halo ...` (or activate the venv) to invoke the CLI against your local checkout.
