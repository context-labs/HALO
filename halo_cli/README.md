# HALO CLI

Thin Typer wrapper around the HALO engine that streams the engine over an OTel/OpenInference JSONL trace file or Pi session JSONL input.

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

`TRACE_PATH` can be an existing OTel/OpenInference span JSONL file, a Pi session JSONL file, or a Pi session directory such as `~/.pi/agent/sessions/--home-user-project--/`.

### Required

| Arg | Description |
|---|---|
| `TRACE_PATH` | OTel/OpenInference JSONL trace file, Pi session JSONL file, or Pi session directory |
| `--prompt`, `-p` | User prompt sent to the root agent |

### Options

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `gpt-5.4-mini` | Model name for root, sub, synthesis, and compaction |
| `--max-depth` | `1` | Max subagent recursion depth |
| `--max-turns` | `8` | Max turns per agent |
| `--max-parallel` | `2` | Max concurrent subagents |
| `--instructions` | *(engine default)* | Override default trace-tool agent instructions |
| `--source` | `auto` | Input source type: `auto`, `otel`, or `pi-session` |
| `--pi-session-full-content` / `--pi-session-redacted-content` | redacted | Opt into full Pi session text/tool payload indexing; redacted mode stores metadata and bounded excerpts |
| `--pi-session-excerpt-chars` | `240` | Maximum characters per Pi session excerpt in redacted mode |

## Examples

OTel/OpenInference trace JSONL:

```bash
halo tests/fixtures/realistic_traces.jsonl \
  -p "What are the most common failure modes?" \
  --max-depth 2 \
  --max-turns 12
```

Pi session directory, with explicit source mode and default redacted excerpts:

```bash
halo ~/.pi/agent/sessions/--home-user-project-- \
  --source pi-session \
  -p "Cluster the coding-agent failure modes across these sessions"
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
