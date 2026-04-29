# HALO patch notes

This directory is a vendored fork of [StonyBrookNLP/appworld](https://github.com/StonyBrookNLP/appworld) wired to emit HALO trace JSONL. There is no live link to upstream — to pull updates, manually diff against a newer upstream commit and re-apply the changes recorded here.

## Snapshot

| Field | Value |
|---|---|
| Upstream repo | https://github.com/StonyBrookNLP/appworld |
| Upstream commit | `a072b7a86e7c1d5b1d7175659d750ebb9b79f10a` |
| Upstream subject | "Fix bundle files (#210)" |
| Upstream date | 2026-02-17 |
| AppWorld DATA_VERSION | `0.2.0` |

To resync: clone upstream at a newer sha, diff against this directory's tree (excluding the HALO additions below), cherry-pick what you want, and re-apply the patches.

## Removed from upstream

The following upstream files/directories were removed because they are not needed to run the HALO loop. Removed paths produce nothing at runtime and are not referenced by anything in `src/appworld/` or `experiments/code/openai_agents/`.

- `.github/` — GitHub Actions workflows; inert at this depth (Actions only reads root-level `.github/workflows/`)
- `.private.gitignore` — upstream's internal release-tooling artifact
- `.pre-commit-config.yaml` — HALO has its own dev hooks
- `.dockerignore`, `dockerfile` — HALO does not use upstream's container setup
- `RELEASE_PROCESS.md`, `DEVELOPMENT.md` — about developing AppWorld upstream, not consuming it
- `pytest.ini` — AppWorld's own test harness
- `images/` — banner SVGs and screenshots, only referenced by upstream's README
- `notebooks/` — exploration notebooks, not on the runtime path
- `guides/` — documentation about extending AppWorld with new apps/tasks
- `scripts/` — 35 maintainer/release/leaderboard scripts; none used at runtime
- `README.md` (the 81K root one) — replaced with a HALO-targeted [`README.md`](README.md)
- `tests/experiments/` — experiment-related test fixtures
- `tests/package/test_appworld.py`, `test_factories.py`, `test_load_task_ids.py`, `test_model_collection.py`, `test_model_relationships.py`, `test_prepare_api_docs.py`, `test_responses.py`, `test_safety_guard.py`, `test_sqlite_fts.py`, `test_sqlmodel.py` — standalone test files (not the per-app eval modules; those live at `tests/package/apps/` and are runtime, unpacked from `tests.bundle` during install)
- `tests/{LICENSE, README_BEFORE_SHARING.md}` and `tests/package/{LICENSE, README_BEFORE_SHARING.md}` — internal release artifacts
- `generate/code/`, `generate/images/`, `generate/tasks/` — task-generation infrastructure (used to extend AppWorld with new tasks; not used to run the existing benchmark). The `generate/.source/{tasks,data}.bundle` files are kept because `appworld install --repo` requires them.

`tests/package/test_*.py` is gitignored after removal because `appworld install --repo` re-creates these files from `tests.bundle` and we do not want them to drift between vendored state and post-install state.

## Patched files

### `experiments/pyproject.toml` — litellm pin

The upstream `appworld-agents` package has a real dependency conflict between its core dependencies and the `[openai_agents]` extra:

- core: `litellm>=1.78.2` (bottom-pinned for `claude-haiku-4-5-20251001`)
- `[openai_agents]` extra: `openai<=1.96.1` (pinned to avoid [openai-python#2489](https://github.com/openai/openai-python/issues/2489))
- but every `litellm>=1.78.2` requires `openai>=1.99.5`

Result: the `[openai_agents]` extra is uninstallable as-published. We removed the `litellm>=1.78.2` floor:

```diff
- "litellm>=1.78.2",        # bottom-pin needed for claude-haiku-4-5-20251001.
+ "litellm",                # HALO patch: floor removed (upstream `>=1.78.2` for haiku-4-5 conflicts with openai_agents extra).
```

Side effect: Claude Haiku 4.5 will not work via litellm with this fork. Acceptable for a v1 OpenAI-focused HALO demo. If/when upstream resolves the conflict, drop this patch.

### `experiments/code/openai_agents/run.py` — HALO trace processor

Upstream explicitly disables tracing globally to avoid the SDK's default upload to OpenAI's dashboard. We replace that with the HALO `InferenceOtlpFileProcessor`. Diff:

```diff
- from agents import Agent, set_default_openai_api, set_tracing_disabled
+ from agents import Agent, set_default_openai_api, set_trace_processors
  ...
+ from appworld_agents.code.openai_agents.tracing import setup_tracing

  set_default_openai_api("chat_completions")
- set_tracing_disabled(True)
```

Inside `run_agent_on_tasks`, around the existing `async with AgentsMCP(...)` block:

```diff
+ # HALO patch: replace SDK default OpenAI uploader with HALO file processor.
+ traces_filename = "traces.jsonl"
+ if process_index is not None and num_processes > 1:
+     traces_filename = f"traces-p{process_index}.jsonl"
+ traces_path = os.path.join(
+     os.environ.get("APPWORLD_ROOT", os.getcwd()),
+     "experiments", "outputs", experiment_name, traces_filename,
+ )
+ os.makedirs(os.path.dirname(traces_path), exist_ok=True)
+ os.environ["HALO_TRACES_PATH"] = traces_path
+ set_trace_processors([])
+ halo_processor = setup_tracing(
+     service_name="appworld",
+     project_id=f"appworld-{experiment_name}".replace("/", "_"),
+ )
+ try:
      async with AgentsMCP(...) as mcp:
          ...
-         run_config = RunConfig(tracing_disabled=True)
+         run_config = RunConfig()
          for task_id in task_ids:
              await run_agent_on_task(...)
+ finally:
+     halo_processor.shutdown()
```

`set_trace_processors([])` clears the SDK's auto-registered default processor (which uploads to OpenAI's trace dashboard). `setup_tracing(...)` then adds the HALO file processor on top. Per the SDK's own docs, this is the canonical pattern for replacing the default.

### `.gitignore` — HALO additions

Appended to upstream's gitignore to cover artifacts that upstream's gitignore does not:

```gitignore
# HALO additions: runtime artifacts not in upstream's gitignore
.tmp/
generate/data/
tests/package/test_*.py
experiments/configs/openai_agents_mcp_agent/openai/*
!experiments/configs/openai_agents_mcp_agent/openai/gpt-4o-2024-05-13
!experiments/configs/openai_agents_mcp_agent/openai/gpt-4o-2024-05-13/**
```

## Added files

| Path | Purpose |
|---|---|
| `experiments/code/openai_agents/tracing.py` | The HALO `InferenceOtlpFileProcessor`. Vendored verbatim from `HALO/demo/openai-agents-sdk-demo/tracing.py`. Do not edit in place — sync with the upstream demo copy when the spec changes. |
| `Taskfile.yml` | One-command setup + run + analyze for this demo. See `task --list`. |
| `README.md` | HALO-targeted readme (replaces upstream's 81K one) |
| `HALO_PATCH.md` | This file |

`README.pypi.md` is untouched — it is the body of the upstream PyPI listing.

## Verifying the patch is intact

After re-running setup, a fresh smoke run should produce 17 spans that all pass `verify_traces.py`:

```bash
task setup
task run:smoke
task verify:traces
# expected: "OK: 17 spans passed all spec assertions"
```

A clean trace will contain at minimum:
- 1 `AGENT` span (`agent.Assistant`)
- N `LLM` spans (`generation.<model>`), one per agent turn
- N `TOOL` spans of two flavors:
  - `mcp_tools` — SDK's per-server tool-listing span
  - `function.<app>__<tool>` — actual MCP tool invocations (e.g. `function.spotify__login`)

If `verify_traces.py` fails or no `function.*` spans appear, the SDK's MCP tool wrapping has changed shape — see `agents/mcp/util.py:to_function_tool` and `agents/_run_impl.py:execute_function_tool_calls` in the installed SDK to confirm `function_span(...)` is still wrapping MCP tool calls.
