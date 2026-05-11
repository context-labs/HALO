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

- `CATALYST_OTLP_TOKEN` set → spans are uploaded to **inference.net Catalyst** over OTLP.
- `CATALYST_OTLP_TOKEN` unset → spans are written to a **local JSONL file** at `./halo-telemetry-{run_id}.jsonl` in the current working directory.

Either way, every span carries a `halo.run.id` resource attribute so you can filter to a single run. On the Catalyst path, spans also carry `halo.engine.version` (the installed `halo-engine` package version) so you can correlate trace shape changes with engine releases.

### Environment variables

| Var | Default | Purpose |
|---|---|---|
| `CATALYST_OTLP_TOKEN` | *(unset)* | If set, uploads to Catalyst over OTLP. If unset, writes JSONL locally. |
| `CATALYST_OTLP_ENDPOINT` | catalyst-tracing default | OTLP endpoint **base URL** (e.g. `https://telemetry.inference.net`). catalyst-tracing appends `/v1/traces` automatically — do **not** include the path, or you'll get a `.../v1/traces/v1/traces` 404 and silently no traces. |
| `CATALYST_SERVICE_NAME` | `halo-engine` | Top-level service identifier on traces. Kept constant so dashboards have a stable HALO grouping; per-run identity (team, project, user, …) flows through the `halo.*` attributes below, not through `service.name`. |
| `CATALYST_SERVICE_VERSION` | installed `halo-engine` version | Service version stamped on traces. Defaults to the package version reported by `importlib.metadata` (`"unknown"` on un-installed source checkouts). |
| `CATALYST_DEBUG` | *(unset)* | Set to `1` to surface OTLP export errors at WARNING level. Useful for troubleshooting "no errors, no traces" — the default `BatchSpanProcessor` swallows export failures. |
| `CATALYST_TRACING_RUN_ID` | *(unset)* | When set, becomes the HALO run id (and the `halo.run.id` resource attribute) instead of a generated uuid. Lets a launching system (typically Catalyst) keep its own bookkeeping in sync with HALO's traces. |
| `CATALYST_TRACING_*` | *(unset)* | Generic passthrough — see below. |
| `HALO_TELEMETRY_PATH` | `./halo-telemetry-{run_id}.jsonl` | Local fallback file path. Only consulted when `CATALYST_OTLP_TOKEN` is unset. |

### Generic `CATALYST_TRACING_*` passthrough (Catalyst path only)

Any environment variable named `CATALYST_TRACING_<NAME>=<value>` is translated into a `halo.<name>=<value>` resource attribute on every span, with `<name>` lowercased and underscores converted to dots:

| Env var | Resource attribute |
|---|---|
| `CATALYST_TRACING_TEAM_ID=team-7` | `halo.team.id=team-7` |
| `CATALYST_TRACING_PROJECT_ID=proj-9` | `halo.project.id=proj-9` |
| `CATALYST_TRACING_USER_ID=user-42` | `halo.user.id=user-42` |
| `CATALYST_TRACING_DEPLOYMENT_ENV=staging` | `halo.deployment.env=staging` |

Notes:

- The dotted form matches the convention the Catalyst-side HALO runtime emits for its known fields, so dashboard filters work uniformly across both emitters.
- Empty / whitespace-only values are skipped (treated as unset).
- `CATALYST_TRACING_RUN_ID` is **not** included in the generic passthrough — it flows through the canonical `halo.run.id` attribute exactly once (see the table above).
- This mechanism lets Catalyst inject new identity / metadata fields onto HALO traces without requiring a HALO release.

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
