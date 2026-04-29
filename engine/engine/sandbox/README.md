# `engine.sandbox` — Deno + Pyodide WASM sandbox

Runs partially-trusted user Python (the `run_code` tool) under a locked-down
[Deno](https://deno.com/) subprocess that hosts [Pyodide](https://pyodide.org/)
for the actual Python interpreter. Adapted from DSPy's
[`primitives/python_interpreter.py`](https://github.com/stanfordnlp/dspy/blob/main/dspy/primitives/python_interpreter.py)
+ `primitives/runner.js`, but trimmed and re-shaped — see *Differences from
DSPy* below.

## Why this design

The agent surface includes a `run_code` tool that lets the model write and
execute arbitrary Python against the trace dataset. We need the model's code to
be able to:

- read the loaded `trace_store` and use `numpy` / `pandas`
- raise exceptions, print to stdout, return text — the normal feedback loop

…and absolutely not be able to:

- read host files outside the trace + index
- write anywhere the host can see
- open network sockets
- spawn host subprocesses
- read the host's environment

Earlier iterations used Linux `bubblewrap` and macOS `sandbox-exec`. Those
hit a wall on Ubuntu 24.04+ where AppArmor's `unprivileged_userns` profile
denies the capability operations bwrap needs (CAP_NET_ADMIN for loopback,
CAP_SETUID for uid_map, CAP_SYS_ADMIN for `mount(MS_SLAVE)`) when invoked from
unconfined processes. Working around it required `sudo`. Pyodide-in-Deno
sidesteps the kernel-namespace mess entirely — isolation lives at the Deno
permission layer, which is just process-level argv flags.

## Architecture

```
┌─────────────────── host (engine package) ──────────────────┐
│                                                            │
│  Sandbox.run_python(code, trace_path, index_path)          │
│       │                                                    │
│       ▼                                                    │
│  _run_session ── spawn ──► deno run --allow-read=...       │
│       │                          │                         │
│       │                          ▼                         │
│       │      ┌──── runner.js (in Deno) ─────┐              │
│       │      │   • boots Pyodide            │              │
│       │      │   • stages engine.traces     │              │
│       │      │     into /halo/ in WASM FS   │              │
│       │      │   • runs pyodide_runtime.py  │              │
│       │      │     (defines halo_bootstrap, │              │
│       │      │      halo_execute)           │              │
│       │      └──────────────┬───────────────┘              │
│       │       JSON-RPC 2.0  │ (one msg / line)             │
│       │  ◄──────stdin/stdout──────►                        │
│       │                                                    │
│       │ 1. mount_file (trace + index → /input/...)         │
│       │ 2. bootstrap (paths) → halo_bootstrap loads        │
│       │    TraceStore, populates user_globals              │
│       │ 3. execute (code) → halo_execute runs user code    │
│       │ 4. shutdown                                        │
│       ▼                                                    │
│  CodeExecutionResult(exit_code, stdout, stderr, timed_out) │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

Each `run_python` call spawns a **fresh** Deno subprocess and tears it down
after one bootstrap+execute cycle. WASM filesystem state cannot leak between
calls — different from DSPy's persistent-process model (see below).

## Files

| File | Role |
|------|------|
| `sandbox.py` | Whole host-side surface: discovery, argv build, JSON-RPC driver, subprocess lifecycle. Public: `Sandbox`, `Sandbox.resolve()`. |
| `runner.js` | The Deno-side entry point. Boots Pyodide, stages the engine package + runtime into the WASM FS, runs the JSON-RPC main loop. |
| `pyodide_runtime.py` | Loaded **inside** Pyodide. Defines `halo_bootstrap(trace_path, index_path)` and `halo_execute(code)`. |
| `models.py` | Public IO models: `CodeExecutionResult`, `RunCodeArguments`. |
| `__init__.py` | Re-exports `Sandbox`, `CodeExecutionResult`, `RunCodeArguments`. |

`pyodide_trace_compat.py` no longer exists — the in-Pyodide bootstrap imports
the **real** `engine.traces.trace_store` after the runner stages the host's
`engine/traces/` source tree into `/halo/engine/traces/` inside Pyodide. Pydantic
v2 is loaded as a Pyodide package so the same models work in both contexts; we
do not maintain a parallel stdlib-only TraceStore.

## Security model

The Deno subprocess is launched with exactly:

```
deno run --no-prompt --allow-read=<runner.js>,<pyodide_runtime.py>,<engine/__init__.py>,<engine/traces>,<deno cache>,<trace>,<index>
```

…and **nothing else**. In particular:

- No `--allow-net` → user code cannot open sockets, can't fetch URLs, can't even
  hit `127.0.0.1`. The Pyodide wheels are pre-cached so the runner never needs
  network at execute time (`Sandbox._ensure_pyodide_wheels` does a one-time
  download via host-side `urllib`).
- No `--allow-write` → user code's writes hit Pyodide's in-memory virtual FS,
  never the host. `open("/etc/anything", "w")` writes to `/etc/anything` *in
  WASM*, vanishes when the subprocess exits.
- No `--allow-env` → host env is invisible. `os.environ` shows Pyodide's canned
  defaults (`HOME=/home/pyodide`, `USER=web_user`).
- No `--allow-run` → no subprocess spawn. Pyodide also has no working
  `fork`/`execve`, so this is belt-and-suspenders.
- `--allow-read` is **enumerated**, never wildcard. Each path is an explicit
  file or one specific directory subtree.

There's a unit test (`test_run_python_does_not_pass_unsafe_flags`) that asserts
none of the unsafe flags ever appear in the argv. If you find yourself wanting
to add one, you almost certainly want a JSON-RPC method on the runner instead.

The integration suite (`tests/integration/test_sandbox_policy_denials.py`)
exercises each of the five denials end-to-end against a real Deno+Pyodide.

## Wire protocol

JSON-RPC 2.0, one message per line on stdin/stdout. The methods are:

| Method | Direction | Payload | Purpose |
|--------|-----------|---------|---------|
| (boot) | runner → host | `{"id": 0, "result": {"ready": true}}` | Sentinel after Pyodide finishes booting |
| `mount_file` | host → runner | `{host_path, virtual_path}` | Read host file, copy bytes into Pyodide FS |
| `bootstrap` | host → runner | `{trace_path, index_path}` | Populate `user_globals` (trace_store, np, pd, …) |
| `execute` | host → runner | `{code}` | Run user code with captured stdout/stderr |
| `shutdown` | host → runner | (notification) | Trigger `Deno.exit(0)` |

`bootstrap` and `execute` both return `{exit_code, stdout, stderr}`. Splitting
them lets the host distinguish setup failures (malformed index, missing
package) from user-code failures (assertion errors, the agent's bugs) — both
travel on the same response shape but the host knows which phase produced
each.

The runner encodes outgoing payloads with `ensure_ascii=False` so non-ASCII
content rides as raw UTF-8 (smaller wire, and the path our `TextDecoder`
actually has to handle across stdin chunk boundaries).

## Subprocess lifecycle

`_run_session` owns it; `_drive` is pure protocol.

```
_run_session
├── spawn deno  +  start stderr_task (drains capped)
├── try
│   ├── result = wait_for(_drive(proc), TIMEOUT)
│   │    │
│   │    └── _drive: read ready ▸ mount* ▸ bootstrap ▸ execute
│   │              returns CodeExecutionResult on any exit path
│   │              (no shutdown / wait responsibility)
│   │
│   ├── on TimeoutError: kill pgid, await wait, drain, return timed_out
│   │
│   └── _drive returned: send shutdown (best-effort), wait, drain, return
└── except BaseException: kill pgid, await wait, drain, raise
```

Cleanup ownership is critical. Earlier shapes had `_drive` send shutdown only
on the execute happy path, which left the runner orphaned on every early-
return path (mount-error, bootstrap-error, bootstrap-Python-error) — `proc.wait`
and `stderr_task` waited forever for an exit that never happened. The current
shape runs the post-drive cleanup block exactly once per session no matter
which `_drive` exit path produced the result.

## Differences from DSPy

DSPy's interpreter is the design ancestor. The trim and re-shape are
deliberate — see the original brief. Concretely:

**Same** (the core mechanics we copied):
- JSON-RPC 2.0 framing over stdin/stdout
- Error code conventions (`-32700`/`-32600`/`-32601` for protocol,
  `-32007`/`-32008`/`-32099` for app)
- `unhandledrejection` → JSON-RPC error pattern
- `pyodide.loadPackage([...])` preloading
- Per-execute `StringIO` capture wrapper
- "Skip non-JSON status lines until matching id" reader

**Removed** (DSPy's general-purpose machinery, dropped per the trim plan):
- Host-side tool bridging (`toolCallBridge`, `_js_tool_call`, makeToolWrapper)
- `SUBMIT` / `FinalOutput` control-flow exceptions
- Env var forwarding
- `sync_file` (write-back from sandbox to host)
- `inject_var` / `inject_text` + the 100 MB filesystem path for large vars
- Tool/output schema registration (`register` method)
- Variable injection (`_inject_variables`, `_serialize_value`)
- Health check
- Thread-ownership tracking

**Inverted:**
- DSPy embeds Python in JS template literals (`PYTHON_SETUP_CODE`,
  `makeToolWrapper` returning Python source). HALO does the opposite —
  `pyodide_runtime.py` is real Python that ruff and pyright cover. Templates
  are small and don't need codegen.

**Architectural divergence:**
- DSPy's `PythonInterpreter` is **long-lived**: one Pyodide subprocess
  persists across many `execute()` calls; loaded packages, mounted files, and
  registered tools carry over between calls. State leakage is fine because
  DSPy's threat model assumes trusted user code.
- HALO's `Sandbox` is **per-call**: every `run_python` spawns a fresh
  subprocess and tears it down. WASM filesystem and globals can't leak
  between calls. Important when the agent is partially-trusted — this is the
  single largest design difference.

**Added** (production hardening with no DSPy equivalent):
- The locked-down permission model — `--allow-read` enumerated, all others
  forbidden
- Pyodide wheel pre-cache (`_ensure_pyodide_wheels`) via host-side `urllib`,
  so the locked-down `deno run` never needs network
- `Sandbox.resolve()` with module-level memoization (avoids the `deno info`
  subprocess + file-existence checks on every engine call)
- Wall-clock budget (`_TIMEOUT_SECONDS`) with kill-process-group on overrun
- UTF-8 chunk-boundary handling (`{stream: true}` decoder)
- Engine package staging into Pyodide FS (`copyDirToPyodide` + sys.path
  insert) so the real `engine.traces.trace_store` runs in WASM — no parallel
  stdlib-only shim
- Split `halo_bootstrap` / `halo_execute` so setup vs. user-code failures
  are distinguishable
- Byte-aware (not character-aware) output truncation matching the cap field
  names

## Adding capabilities

Resist the urge to lift a Deno permission. The right move is almost always a
new JSON-RPC method:

- New mount type → new method on the runner side, new caller in `_drive`.
- Need to read a different host file? Add it to the per-call `extra_read_paths`
  argument of `_build_argv`, mount it into the WASM FS via `mount_file`.
- Need to run more setup before user code? Extend `halo_bootstrap` in
  `pyodide_runtime.py`. The shared globals dict is the bridge.

If you find yourself wanting `--allow-net` or `--allow-write`, the threat
model is wrong — back off and rethink. The locked-down policy is the whole
point.

## Test surface

| File | What it covers |
|------|------|
| `tests/unit/sandbox/test_sandbox.py` | Resolve, argv shape, byte-aware truncation, RPC error surfacing, lifecycle (mount/bootstrap/execute/shutdown), no-hang regression, `_read_ready` JSON tolerance |
| `tests/unit/sandbox/test_models.py` | `CodeExecutionResult` / `RunCodeArguments` shape |
| `tests/integration/test_sandbox_availability.py` | Real `Sandbox.resolve()` against installed Deno + cached Pyodide |
| `tests/integration/test_sandbox_runner.py` | Real Pyodide running real `engine.traces.TraceStore`; numpy/pandas aliases; tracebacks; UTF-8 across chunk boundaries; timeout |
| `tests/integration/test_sandbox_policy_denials.py` | All five denial axes (host write, host read, network, subprocess, env) end-to-end |

The unit tests use `monkeypatch.setattr(sandbox_module, ...)` to swap in stubs
— pytest's standard mocking, no mocking framework. The integration tests
require Deno on PATH (which the `deno` PyPI dep provides; CI doesn't need a
separate setup-deno step).
