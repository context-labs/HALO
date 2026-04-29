# benchmarks/appworld — AppWorld via OpenAI Agents SDK + HALO tracing

> **Status: WIP. Do not merge.** First HALO benchmark adapter; still under
> active iteration. The trace pipeline works end-to-end (100/100 tasks
> produced verified JSONL traces) but the harness is intentionally
> minimal — no API predictor, no ReAct exemplars, no leaderboard
> scaffolding. Optimizing the harness is HALO's job.

A worked example of running the [AppWorld benchmark](https://github.com/StonyBrookNLP/appworld)
through an OpenAI Agents SDK harness, writing inference.net-format JSONL
traces via HALO's `tracing.py`. Single-task and batch runners ship; a
HuggingFace upload script is planned.

## What's in here

| File | Purpose |
|---|---|
| `agent.py` | Thin Agents SDK adapter — one `execute_code` `@function_tool` that wraps `world.execute(code)` (paper's ReAct baseline shape). |
| `tracing.py` | Vendored verbatim from `demo/openai-agents-sdk-demo/`. The single-file `InferenceOtlpFileProcessor` integration; writes plain JSONL. |
| `run_one_task.py` | Typer CLI: load one AppWorld task, run the agent, flush trace, print evaluation. Entry point for ad-hoc runs. |
| `run_batch.py` | Run a list of tasks in parallel (default: train + dev = 100 tasks, 8 workers). Per-task subprocess; resumable via `--skip-existing`. Writes `traces/index.jsonl` summary alongside the per-task traces. |
| `verify_traces.py` | Stdlib spec-conformance check on the trace file (vendored from the demo). |
| `pyproject.toml` | Pinned to AppWorld git `main` (PyPI 0.1.3.post1 still has `pydantic<2` — incompatible with the SDK). |

## Why git source for AppWorld?

PyPI `appworld 0.1.3.post1` pins `pydantic<2`. OpenAI Agents SDK requires
`pydantic>=2.10`. The conflict is unsatisfiable. `main` is `0.2.0.dev0`
and is on `pydantic>=2.12`, so we pull from git directly. The README of
the AppWorld repo explicitly invites this:

> "the main branch is **significantly ahead** of this release. We'll do
> a larger release with the latest code soon"

When AppWorld cuts a 0.2.x PyPI release, drop the `[tool.uv.sources]`
git pin and we'll be back on stable distribution.

## One quirk we work around

The OpenAI Agents SDK dispatches sync `@function_tool` callbacks through
`asyncio.to_thread` — the tool body lands on a worker thread. AppWorld's
default `_shell_run_cell` uses `signal.SIGALRM` to enforce a 100s
per-execution timeout, which Python only supports on the main thread,
so every tool call would otherwise raise:

```
signal only works in main thread of the main interpreter
```

Pass `timeout_seconds=None` to `AppWorld(...)` (a public escape hatch
in upstream — `environment.py:949` skips the timeout wrapper) and
`world.execute` runs straight through. We do this in `run_one_task.py`.

This is the only "patch" — and it's a public-API call, not a monkey
patch. We do not modify any AppWorld source.

## Quickstart

```bash
# 1. Install + materialize AppWorld
cp .env.example .env                        # paste your OPENAI_API_KEY
uv sync                                     # pulls AppWorld from git main
uv run appworld install                     # decrypts the bundled apps
uv run appworld download data --root .data  # ~few MB of task DBs + API docs

# 2. Run one task
uv run python run_one_task.py fac291d_1 --max-steps 30

# 3. Run a batch (default: 100 tasks, 8 workers, ~30 minutes, ~$5 on gpt-4.1-mini)
uv run python run_batch.py

# 4. Verify the traces
uv run python verify_traces.py traces/fac291d_1.jsonl
# OK: 69 spans passed all spec assertions
```

## Trace shape

One JSONL line per span, inference.net OTLP-shaped (`07-export.md`):

- One trace per task. `trace_id` constant for all spans of a task.
- `attributes."inference.observation_kind"` ∈ {`AGENT`, `LLM`, `TOOL`,
  `CHAIN`, `SPAN`}.
- TOOL spans (`function.execute_code`) carry `input.value` (the Python
  snippet sent to AppWorld) and `output.value` (stdout or stack trace).
- LLM spans (`response.<model>`) carry `llm.token_count.{prompt,completion,total}`
  but **not** the full message body — Responses API doesn't surface it
  via `Span.export()`.

See `tracing.py` for the per-span-type projection.

## Status of the 100-trace fixture

After running `run_batch.py` defaults:

- 100 trace files in `traces/<task_id>.jsonl` (gitignored, ~7.7 MB).
- `traces/index.jsonl` with per-task metadata.
- `verify_traces.py` passes for every file.

Planned next: upload the fixture to `inference-net/AppWorld-samples-100`
on HuggingFace so consumers can grab the traces without re-running the
harness. WIP.

## Known limitations / TODO

- **`shutil.rmtree` cleanup error** at `__exit__` — AppWorld's own
  safety guard blocks it during context-manager teardown. Cosmetic
  (trace already flushed before the error), but adds noise to the log.
  Likely an upstream bug to file.
- **`final_message` not extracted** in batch index — the Responses API
  doesn't put assistant text into the exported span. Workaround would
  be to capture `run_one_task.py` stdout per task; not done yet.
- **No HF upload script yet** — manual upload pending org access
  confirmation.
- **No tests** — once the harness shape is settled, add a
  `tests/test_smoke.py` that runs one task with `--max-steps 1` and
  asserts `verify_traces.py` accepts the output.
