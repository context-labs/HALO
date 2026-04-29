# HALO demo: AppWorld → HALO trace JSONL

Vendored fork of [StonyBrookNLP/appworld](https://github.com/StonyBrookNLP/appworld) wired to emit HALO-shaped trace JSONL from its OpenAI Agents SDK harness. AppWorld is an ACL'24 benchmark of 9 simulated apps (Spotify, Splitwise, Gmail, Venmo, etc.) operable via 457 APIs, with a public leaderboard at [appworld.dev/leaderboard](https://appworld.dev/leaderboard).

What this demo gives you:

- **End-to-end runs** on any of the four splits (`train`, `dev`, `test_normal`, `test_challenge`)
- **HALO traces** at `experiments/outputs/<experiment>/traces.jsonl` after every run, ready to feed to the HALO Engine
- **A Taskfile-driven workflow** so you don't have to memorize the install dance

For the full list of changes from upstream, see `[HALO_PATCH.md](HALO_PATCH.md)`.

## Scope

HALO tracing is wired into the `**openai_agents_mcp_agent`** harness only, and use default `AGENT=openai_agents_mcp_agent` for the HALO loop.

## Prereqs

- macOS or Linux (Windows untested)
- `[uv](https://docs.astral.sh/uv/)` for Python env management
- [Task](https://taskfile.dev) (`brew install go-task`) recommended; everything is reachable manually too
- An OpenAI API key (or another provider's; see [Other models](#other-models))

## Setup

```bash
cp .env.example .env   # then fill in OPENAI_API_KEY
task setup
```

`task setup` is idempotent. It runs four steps; each only does work the first time:


| Step            | What it does                                                      | First-run cost |
| --------------- | ----------------------------------------------------------------- | -------------- |
| `setup:venv`    | Creates `.venv` with Python 3.12                                  | ~5s            |
| `setup:install` | Installs `appworld` and `appworld-agents[openai_agents]` editable | ~30s, ~250MB   |
| `setup:bundles` | `appworld install --repo` unpacks the four encrypted bundles    | ~2s            |
| `setup:data`    | `appworld download data` fetches the 728-task dataset from S3   | ~10s, ~190MB   |


After setup, `task --list` shows everything available.

## Run a smoke test

```bash
task run:smoke
```

Runs one task (`50e1ac9_1` from `dev`) with `gpt-4o-mini-2024-07-18`. Takes ~20 seconds, costs about $0.005. On success you'll see:

```
1 of 1 tasks completed.
Json evaluation report saved in: ./experiments/outputs/.../evaluations/on_only_50e1ac9_1.json
Text Evaluation Report:
    type     | task_goal_completion | scenario_goal_completion
    ...
```

The trace lands at `experiments/outputs/openai_agents_mcp_agent/openai/gpt-4o-mini-2024-07-18/dev/traces.jsonl`. To verify it conforms to the HALO spec:

```bash
task verify:traces
# expected: OK: 17 spans passed all spec assertions
```

## Run on a full split

```bash
task run:dev               # 57 tasks
task run:test-normal       # 168 tasks
task run:test-challenge    # 417 tasks
task run:train             # 90 tasks
```

Default is sequential (`PARALLEL=1`). Override with `PARALLEL=N` to run N tasks in parallel via process-level fan-out:

```bash
task run:dev PARALLEL=8                 # ~4 min instead of ~19 min
task run:test-challenge PARALLEL=16
task run:train PARALLEL=-1              # all CPUs - 1, clamped to task count
```

Approximate wallclock with `gpt-4o-mini-2024-07-18` (~20s/task sequential):


| Split          | Tasks | `PARALLEL=1` | `PARALLEL=8` | `PARALLEL=16` |
| -------------- | ----- | ------------ | ------------ | ------------- |
| dev            | 57    | ~19 min      | ~3-5 min     | ~3 min        |
| train          | 90    | ~30 min      | ~5-7 min     | ~4 min        |
| test_normal    | 168   | ~56 min      | ~9-12 min    | ~6-8 min      |
| test_challenge | 417   | ~2.3 h       | ~22-30 min   | ~15-20 min    |


Override the model with the `MODEL=` task variable:

```bash
task run:dev MODEL=gpt-4.1-2025-04-14
task run:test-normal MODEL=gpt-4.1-mini-2025-04-14 PARALLEL=12
```

`AGENT=` exists too but only `openai_agents_mcp_agent` is HALO-traced, see [Scope](#scope) above. `task agents:list` lists every paradigm upstream ships, not the subset HALO patched.

For the full menu of supported model names:

```bash
task models:list
```

### How parallelism works under the hood

`--num-processes N` causes AppWorld's CLI to re-shell itself N times via `subprocess.Popen`. Each subprocess gets a balanced chunk of `task_ids` via `chunk_and_return(...)`, spins up its own AppWorld FastAPI server + MCP server on auto-selected ports, and runs its tasks sequentially. The parent waits on all subprocesses, then runs eval once over the union of outputs.

Resource implications:

- **Memory:** ~300-500 MB per subprocess (AppWorld engine + 9 simulated apps + MCP). 8 processes ≈ 3-4 GB.
- **OpenAI rate limits:** scale linearly with parallelism. `gpt-4o-mini` won't bottleneck below ~50 processes; higher-tier models hit limits sooner.
- **Eval determinism:** task order is shuffled before chunking (with a fixed `random_seed=100` from the config), so results are reproducible at a given `PARALLEL` value but task scheduling differs across `PARALLEL` settings.

### Trace files when running in parallel

Each subprocess writes its own trace file to avoid gzip-stream contention:

```
experiments/outputs/.../dev/
├── traces-p0.jsonl
├── traces-p1.jsonl
├── traces-p2.jsonl
└── traces-p3.jsonl     (one per process)
```

The Taskfile auto-merges these into a single `traces.jsonl` after every `run:*` task by calling `traces:merge`. The per-process files are deleted after merge. If a run was killed before merge ran (e.g. SIGKILL), call `task traces:merge` manually to consolidate. Order doesn't matter — HALO indexes by `trace_id`, not by line position.

## Where outputs land

For an experiment run with the default config and `dev` split:

```
experiments/outputs/openai_agents_mcp_agent/openai/gpt-4o-mini-2024-07-18/dev/
├── traces.jsonl                       # HALO traces (one line per span)
├── configs/{dev.json, dev.jsonnet}    # the config that was actually run
├── evaluations/                       # JSON + text reports per task / per dataset
├── logs/server.log                    # AppWorld background-server log
└── tasks/<task_id>/
    ├── dbs/                           # per-task simulated app state snapshots
    └── logs/lm_calls.jsonl            # per-task LLM-call log (separate from HALO traces)
```

The HALO trace shape per span:


| Field                                                                | Notes                                                                          |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `trace_id`, `span_id`, `parent_span_id`                              | OTLP IDs; one trace per task                                                   |
| `name`                                                               | `agent.Assistant`, `generation.<model>`, `function.<app>__<tool>`, `mcp_tools` |
| `attributes."inference.observation_kind"`                            | `AGENT` / `LLM` / `TOOL`                                                       |
| `attributes."inference.project_id"`                                  | `appworld-<experiment_name_with_slashes_underscored>`                          |
| `attributes."inference.llm.{model_name,input_tokens,output_tokens}"` | LLM spans only                                                                 |
| `attributes."tool.name"`, `input.value`, `output.value`              | TOOL spans only                                                                |


A typical 20-step task trace contains ~50 spans: 1 root AGENT, ~10 LLM, ~10 TOOL function calls, ~10 MCP tool-listing spans.

## Feeding traces to the HALO Engine

```bash
task analyze
```

Streams the most recent `traces.jsonl` through the HALO Engine with a default question about harness improvements. To customize:

```bash
task analyze \
  TRACE_PATH=experiments/outputs/.../traces.jsonl \
  PROMPT="Which tasks failed because the API predictor under-fetched? Group by app." \
  MODEL=gpt-5.4-mini
```

This task assumes the demo lives at `HALO/demo/appworld/` so it can find the HALO CLI at `../../cli`. If you've moved things, edit `HALO_CLI_DIR` in `Taskfile.yml`.

## Iterating on the harness

The whole point of this demo is HALO-driven harness improvement. The loop:

1. `task run:dev PARALLEL=8` — produce traces + eval results in ~3-5 min
2. `task analyze` — ask HALO Engine what to fix
3. Edit the harness (most likely places below)
4. `task clean:run-outputs && task run:dev PARALLEL=8` — re-run on the same split
5. Diff the eval reports

`PARALLEL=8` is the sweet spot for iteration: fast enough to keep the loop tight, low enough to leave headroom for other work. Drop to `PARALLEL=1` for debugging (deterministic stdout, no interleaved logs from sibling subprocesses).

Most-improvable surfaces:


| Path                                                                          | What's there                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiments/code/openai_agents/run.py`                                       | The agent loop, including the API predictor → main agent handoff and the `ModelBehaviorError` recovery gap that upstream itself flagged with `# no easy way to give it feedback about the error in this framework, so leave it.` |
| `experiments/code/openai_agents/api_predictor.py`                             | The first-pass model that whitelists ≤20 APIs per task; under-fetching here cascades into main-agent failures                                                                                                                    |
| `experiments/prompts/function_calling_agent/instructions.txt`                 | The agent's system prompt template                                                                                                                                                                                               |
| `experiments/prompts/function_calling_agent/demos.json`                       | Few-shot demonstrations                                                                                                                                                                                                          |
| `experiments/prompts/api_predictor.txt`                                       | The API predictor's prompt                                                                                                                                                                                                       |
| `experiments/configs/_generator/templates/openai_agents_mcp_agent.jsonnet.j2` | The template that produces per-model run configs (`max_steps`, tool_choice, parallelism, etc.)                                                                                                                                   |


## Other models

The default is `gpt-4o-mini-2024-07-18`, routed through the `"openai"` adapter (`AsyncOpenAI` → `api.openai.com`). The harness can also route to non-OpenAI providers via the `"litellm"` adapter. The model registry at `experiments/configs/_generator/models/<provider>.py` declares which adapter each model uses; entries for `anthropic/`, `google/`, `meta/`, `deepseek/` and others are tagged `"client_name": "litellm"`.

Practical caveats:

- **Only OpenAI configs are checked in for `openai_agents_mcp_agent`.** `experiments/configs/openai_agents_mcp_agent/` ships only `openai/gpt-4o-2024-05-13` and `openai/gpt-4o-mini-2024-07-18`. Running a non-OpenAI model under HALO tracing means generating a fresh config from the registry; `task models:list` prints names the generator accepts.
- **Set the relevant API key in `.env`** before invoking with a non-OpenAI model.
- **The 90s `AsyncOpenAI` request timeout is OpenAI-path only** — `LitellmModel` has its own httpx client and is unaffected (see [HALO_PATCH.md](HALO_PATCH.md)).
- **Claude Haiku 4.5 won't route via litellm in this fork** — the `litellm>=1.78.2` floor was dropped to make the `[openai_agents]` extra installable, and Haiku 4.5 needs that floor.

## Cleaning up

```bash
task clean:run-outputs   # delete just experiments/outputs/
task clean               # delete .venv, data, run outputs, and unpacked bundles (prompts for confirmation)
```

After `task clean`, a `task setup` from scratch takes ~1–2 minutes.

## Troubleshooting

`**uv pip install -e 'experiments[openai_agents]'` installs the wrong package.**
There is a generic PyPI package named `experiments`. Use `'./experiments[openai_agents]'` with the leading `./`. The Taskfile already does this; this only bites if you run `uv pip install` by hand.

`**appworld download data` fails with "package not fully installed".**
You ran `appworld install` (package mode) instead of `appworld install --repo`. The verify-installation heuristic checks paths that only the `--repo` mode populates when AppWorld is installed editable from outside `site-packages`. The Taskfile uses `--repo`; if running by hand, do the same.

**Traces are empty or missing.**
The HALO file processor flushes on `processor.shutdown()`. The patched `run.py` calls this in a `finally` block. If you killed the run with `SIGKILL` rather than `SIGTERM` / Ctrl-C, the gzip stream may have been truncated. Re-run the task.

`**appworld run` errors with "Could not find dataset".**
You skipped `task setup:data`. Run `task setup` (or just `task setup:data`) first.

**Some non-OpenAI model fails to load a prompt at `experiments/prompts/function_calling_v2_zero_shot.txt`.**
That prompt isn't shipped in this fork. Models with `function_calling_demos: True` in `experiments/configs/_generator/models/<creator>.py` are unaffected. For a model that needs the zero-shot prompt, either flip the flag or supply the file.

**Subprocess hangs on a single OpenAI request for minutes.**
This fork sets a 90-second per-request timeout on every `AsyncOpenAI` client (defined as `_OPENAI_REQUEST_TIMEOUT_SECONDS` in `experiments/code/openai_agents/run.py` and `language_model.py`). Without it, a half-closed TCP connection can stall a subprocess for ~10 minutes — and because the parent collects subprocesses in launch order, one stuck process blocks the whole run from finishing. If you're running a slow model like `o3` or `gpt-5-...-high-reasoning` and seeing legitimate timeouts on long completions, raise the constant (e.g. to 300s) in both files.

## Related docs

- [HALO_PATCH.md](HALO_PATCH.md) — what this fork changes vs upstream and how to resync
- Upstream AppWorld: [stonybrooknlp/appworld](https://github.com/StonyBrookNLP/appworld), [appworld.dev](https://appworld.dev), [paper](https://arxiv.org/abs/2407.18901)
- HALO OpenAI Agents SDK integration: `[docs/integrations/openai-agents-sdk.md](../../docs/integrations/openai-agents-sdk.md)`

## License

Apache 2.0, inherited from upstream — see `[LICENSE](LICENSE)`.