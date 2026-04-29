# HALO CLI

Thin Typer wrapper around `halo-engine` that streams the engine over a JSONL trace file.

## Install

```bash
pip install halo-engine
```

Or from a checkout of this repo:

```bash
uv sync
```

Both install the `halo-engine` script onto your `PATH`.

## Setup

The engine needs real LLM access:

```bash
export OPENAI_API_KEY=sk-...
```

## Usage

```bash
halo-engine TRACE_PATH --prompt "your question"
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
halo-engine tests/fixtures/realistic_traces.jsonl \
  -p "What are the most common failure modes?" \
  --max-depth 2 \
  --max-turns 12
```

Output streams to stdout: text deltas inline, then a rule-separated panel for each agent output item.
