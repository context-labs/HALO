"""HALO Deno+Pyodide WASM sandbox.

One ``Sandbox`` class owns the host-side surface: discovery, asset prep,
the Deno argv, and ``run_python``. A private ``_RunnerSession`` owns one
subprocess + JSON-RPC roundtrip. Pyodide-specific knobs (version pin,
required packages, lockfile-driven wheel resolution) live as module
constants and helpers in this same file.

The single per-process knob (``_TIMEOUT_SECONDS``) is a module constant
rather than a config field. Production runs all want the same wall-clock
budget; the one test that needs a different value monkeypatches the
constant directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypedDict

from engine.sandbox.models import CodeExecutionResult

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RUNNER_FILENAME = "runner.js"
_RUNTIME_FILENAME = "pyodide_runtime.py"

# ``_PYODIDE_VERSION`` is the npm version this sandbox expects; the matching
# pin lives in ``runner.js`` (``npm:pyodide@<version>/pyodide.js``). Both
# must move together — Deno caches the npm package by version under
# ``<deno_dir>/npm/registry.npmjs.org/pyodide/<version>/`` and the wheel
# filenames are ABI-tied to that release.
_PYODIDE_VERSION = "0.29.3"

# Top-level Pyodide packages we need at boot. The transitive closure (numpy
# pulls nothing extra; pandas pulls dateutil/pytz/six; pydantic pulls
# pydantic_core/typing_extensions/annotated_types/typing_inspection) is
# resolved from ``pyodide-lock.json`` at wheel-cache time, so a Pyodide
# bump moves the resolved filenames automatically without a code change here.
_REQUIRED_PACKAGES: tuple[str, ...] = ("numpy", "pandas", "pydantic")

_WHEEL_BASE_URL = f"https://cdn.jsdelivr.net/pyodide/v{_PYODIDE_VERSION}/full/"

# Where the trace and index files live inside the Pyodide FS. Hardcoded so
# the sandbox and the in-Pyodide ``halo_bootstrap`` stay aligned without
# leaking host paths through the WASM filesystem.
_TRACE_VIRTUAL_PATH = "/input/traces.jsonl"
_INDEX_VIRTUAL_PATH = "/input/index.jsonl"

# Wall-clock budget for one ``run_python`` call. Generous default — cold
# Pyodide boot can take 5-10s, so anything below ~30s would mask real bugs
# as timeouts. Tests that exercise the timeout path (``test_sandbox_timeout``)
# monkeypatch this to a shorter value.
_TIMEOUT_SECONDS = 60.0

# Defensive caps on captured output. Constants rather than config: the agent
# should not be able to provoke arbitrarily large prompt growth by emitting
# a multi-megabyte stdout from inside the sandbox, and there's no realistic
# use case for raising the cap (any analysis that needs more than this
# should be summarizing in code, not in stdout).
_MAX_STDOUT_BYTES = 64_000
_MAX_STDERR_BYTES = 64_000

_TRUNCATION_MARKER = "\n[... output truncated ...]\n"

# How long to wait for the ``{"ready": true}`` sentinel from runner.js.
# Cold-boot of npm:pyodide on a fresh Deno cache can take ~10s.
_READY_TIMEOUT_SECONDS = 30.0

# How long the graceful-shutdown send is given to flush before we escalate
# to SIGKILL. The runner exits immediately on the shutdown frame; this
# only matters if the runner is already wedged.
_SHUTDOWN_GRACE_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SandboxError(RuntimeError):
    """Raised when the Deno/Pyodide subprocess returns a JSON-RPC error or dies."""


class _ResolutionError(Exception):
    """Internal: any failure during Deno + wheel pre-cache resolution."""


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Sandbox:
    """Resolved Deno+Pyodide WASM sandbox: paths to read + binary to spawn.

    Construct via :meth:`get` — that's the only path that does Deno
    discovery and Pyodide pre-cache work, and it caches the result on
    the class so repeated calls don't reshell out to ``deno info``.

    Each :meth:`run_python` call spawns a fresh ``deno run`` subprocess
    so the WASM filesystem cannot leak between runs. The subprocess is
    launched with the locked-down permission set (``--allow-read`` only,
    scoped to runner + sibling .py files + Deno cache + the per-run
    trace and index files; never ``--allow-net`` / ``--allow-write`` /
    ``--allow-env`` / ``--allow-run``).
    """

    # Successful resolves are memoized so a long-lived process (e.g., a
    # server handling many engine runs) doesn't pay the ``deno info``
    # subprocess + file-existence checks on every request. Failed resolves
    # are deliberately not cached: a transient failure (e.g., wheel download
    # blip) shouldn't poison the rest of the process.
    _cached: ClassVar["Sandbox | None"] = None

    deno_executable: Path
    runner_path: Path
    runtime_path: Path
    # The host's ``engine`` package root and ``engine/traces`` subtree.
    # Both are added to ``--allow-read`` so the runner can stage them
    # into Pyodide's WASM filesystem, where the real
    # ``engine.traces.trace_store`` becomes importable. This is how we
    # avoid maintaining a parallel stdlib-only TraceStore.
    engine_init_path: Path
    traces_pkg_dir: Path
    deno_dir: Path

    @classmethod
    def get(cls) -> "Sandbox | None":
        """Find ``deno``, verify sibling files, pre-cache Pyodide wheels.

        Resolution order for ``deno``:
          1. ``deno.find_deno_bin()`` from the ``deno`` PyPI dependency —
             the normal path, the binary ships with ``pip install``.
          2. ``shutil.which("deno")`` — fallback for unsupported platforms
             (musl Linux, FreeBSD) or system-managed Deno installs.

        On any failure (binary missing, ``deno info`` failing, sibling
        files missing, wheel download failing) a remediation warning is
        emitted via :func:`_log_unavailable` and ``None`` is returned so
        ``run_code`` is silently dropped from the agent's tool surface
        rather than registered with broken plumbing.
        """
        if cls._cached is not None:
            return cls._cached

        deno = _locate_deno()
        if deno is None:
            _log_unavailable(
                diagnostic="deno binary not found (expected from `deno` PyPI dep or PATH)",
            )
            return None
        try:
            here = Path(__file__).parent
            runner_path = (here / _RUNNER_FILENAME).resolve()
            runtime_path = (here / _RUNTIME_FILENAME).resolve()
            engine_pkg_root = here.parent  # engine/sandbox/.. == engine/
            engine_init_path = (engine_pkg_root / "__init__.py").resolve()
            traces_pkg_dir = (engine_pkg_root / "traces").resolve()
            for required_file in (runner_path, runtime_path, engine_init_path):
                if not required_file.is_file():
                    raise _ResolutionError(f"required sandbox file missing at {required_file}")
            if not traces_pkg_dir.is_dir():
                raise _ResolutionError(f"engine.traces package missing at {traces_pkg_dir}")
            deno_dir = _query_deno_dir(deno)
            pyodide_dir = _ensure_npm_cache(deno, deno_dir)
            _ensure_wheels(pyodide_dir)
        except _ResolutionError as exc:
            _log_unavailable(diagnostic=str(exc))
            return None

        sandbox = cls(
            deno_executable=deno,
            runner_path=runner_path,
            runtime_path=runtime_path,
            engine_init_path=engine_init_path,
            traces_pkg_dir=traces_pkg_dir,
            deno_dir=deno_dir,
        )
        cls._cached = sandbox
        return sandbox

    async def run_python(
        self,
        *,
        code: str,
        trace_path: Path,
        index_path: Path,
    ) -> CodeExecutionResult:
        """Run ``code`` in the WASM sandbox; returns a typed result regardless of pass/fail/timeout.

        Mounting:
          - The runner.js + sibling .py files + Deno cache are read-only.
          - The host trace + index are added to ``--allow-read`` for this
            invocation only, so Deno can read them once to copy bytes into
            Pyodide's virtual FS.
          - Inside Pyodide, files appear at fixed virtual paths. The
            runner stages the trace compat shim itself; the host only
            tells the bootstrap which mount points to load.
        """
        trace = trace_path.resolve()
        index = index_path.resolve()
        session = _RunnerSession(argv=self._build_argv(extra_read_paths=[trace, index]))
        try:
            await session.start()
            try:
                result = await asyncio.wait_for(
                    _run_protocol(session, trace=trace, index=index, code=code),
                    timeout=_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                stderr = await session.stop(hard=True)
                return CodeExecutionResult(
                    exit_code=session.returncode if session.returncode is not None else -1,
                    stdout="",
                    stderr=stderr.decode("utf-8", errors="replace"),
                    timed_out=True,
                )
            stderr = await session.stop(hard=False)
            return _attach_deno_stderr(result, stderr)
        except SandboxError as exc:
            # Most ``SandboxError`` causes (RPC error responses) leave the
            # runner alive and responsive, so graceful shutdown is correct.
            # On the rarer flavors (premature stdout EOF, ready-timeout)
            # the runner is already dead — graceful stop degrades to a
            # best-effort write that ``BrokenPipeError``s, then ``proc.wait``
            # resolves immediately. Either way, no orphan.
            #
            # The agent sees ``stderr`` on its end; operators don't, so log
            # at warning so a tail of the host process surfaces the failure
            # too. Same shape as the broader ``Exception`` handler below.
            _logger.warning("sandbox.run_python: runner failure: %s", exc)
            await session.stop(hard=False)
            return CodeExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"sandbox runner failure: {exc}",
                timed_out=False,
            )
        except Exception as exc:
            # Anything else — ``OSError`` from a spawn failure,
            # ``json`` errors from a corrupt cache, surprising bugs in the
            # driver itself — must not crash the agent loop. ``run_code``
            # is a tool: the right behavior is to log the failure for
            # operators and hand the agent a sad result it can recover
            # from. ``BaseException`` (cancel/interrupt) is intentionally
            # NOT caught here so user-initiated termination still works.
            _logger.warning("sandbox.run_python failed unexpectedly: %r", exc)
            await session.stop(hard=True)
            return CodeExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"sandbox unexpected failure: {exc!r}",
                timed_out=False,
            )
        except BaseException:
            await session.stop(hard=True)
            raise

    def _build_argv(self, *, extra_read_paths: list[Path]) -> list[str]:
        """Build the ``deno run`` argv with HALO's locked-down permission set.

        ``--allow-read`` is enumerated explicitly (no wildcards) and covers:
          - the runner script + its sibling runtime .py
          - the host's ``engine/__init__.py`` and the ``engine/traces/``
            subtree, which the runner stages into Pyodide's FS so the
            real ``engine.traces.trace_store`` is importable
          - the Deno cache directory (where ``npm:pyodide`` resolves and
            the pre-cached ``*.whl`` wheels live next to ``pyodide.asm.wasm``)
          - any additional per-run paths the caller mounts (trace, index)

        Everything else — ``--allow-net``, ``--allow-write``, ``--allow-env``,
        ``--allow-run`` — is intentionally absent so the sandboxed process
        has no host network, no host writes, no host env vars, no subprocess
        spawn. ``--no-prompt`` is passed so a missing permission errors
        instead of pausing on a TTY prompt.
        """
        read_paths = [
            self.runner_path,
            self.runtime_path,
            self.engine_init_path,
            self.traces_pkg_dir,
            self.deno_dir,
            *extra_read_paths,
        ]
        allow_read = ",".join(str(p) for p in read_paths)
        return [
            str(self.deno_executable),
            "run",
            "--no-prompt",
            f"--allow-read={allow_read}",
            str(self.runner_path),
        ]


async def _run_protocol(
    session: "_RunnerSession",
    *,
    trace: Path,
    index: Path,
    code: str,
) -> CodeExecutionResult:
    """Mount → bootstrap → execute against a started session.

    Pure protocol: takes a session that's already passed its ready
    sentinel and runs the deterministic JSON-RPC sequence. Lifecycle
    (spawn, shutdown, kill, drain) belongs to ``Sandbox.run_python``
    and the session itself.
    """
    await session.mount(trace, _TRACE_VIRTUAL_PATH)
    await session.mount(index, _INDEX_VIRTUAL_PATH)
    boot = await session.bootstrap(_TRACE_VIRTUAL_PATH, _INDEX_VIRTUAL_PATH)
    if boot.exit_code != 0:
        return boot
    return await session.execute(code)


# ---------------------------------------------------------------------------
# _RunnerSession: subprocess + JSON-RPC for one run_python call
# ---------------------------------------------------------------------------


class _RpcErrorRequired(TypedDict):
    code: int
    message: str


class _RpcError(_RpcErrorRequired, total=False):
    """JSON-RPC 2.0 error object. ``data`` is optional in the spec."""

    data: object


class _ExecutionPayload(TypedDict):
    """Shape of the result dict returned by ``halo_bootstrap`` / ``halo_execute``."""

    exit_code: int
    stdout: str
    stderr: str


@dataclass
class _RpcResult:
    """One JSON-RPC response line, parsed for the caller.

    ``result`` is the runner's payload — typed loosely as ``dict``
    because the *caller* knows whether it expects an
    ``_ExecutionPayload`` (bootstrap/execute) or a small ack
    (mount_file). Caller-side validation in ``_result_from_rpc``
    coerces to ``CodeExecutionResult`` with defaults so a malformed
    runner reply can't crash the host.
    """

    result: dict | None
    error: _RpcError | None


class _RunnerSession:
    """One Deno subprocess + JSON-RPC roundtrip for a single ``run_python`` call.

    Lifecycle:
      ``start()``  → spawn, begin draining stderr, wait for ready sentinel
      ``mount()`` / ``bootstrap()`` / ``execute()`` → JSON-RPC roundtrips
      ``stop(hard=False)`` → graceful shutdown frame, wait, drain (returns stderr)
      ``stop(hard=True)``  → SIGKILL the pgroup, wait, drain (returns stderr)

    Fresh per call. No state carried between calls; that's the whole
    point of the per-call subprocess design (vs DSPy's long-lived one).
    """

    def __init__(self, *, argv: list[str]) -> None:
        self._argv = argv
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[bytes] | None = None
        self._next_id = 0
        self._stopped = False

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode if self._proc is not None else None

    async def start(self) -> None:
        """Spawn the deno subprocess and block until it signals ready."""
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._stderr_task = asyncio.create_task(_drain_capped(self._proc.stderr, _MAX_STDERR_BYTES))
        await self._await_ready()

    async def stop(self, *, hard: bool) -> bytes:
        """Tear down the subprocess and return whatever stderr was captured.

        Idempotent — calling twice (e.g., once on the happy path, once
        from an outer exception handler) is harmless. The first call
        does the work; subsequent calls return the same captured stderr.
        """
        if self._stopped:
            # Already stopped: stderr_task has resolved, just return its
            # value. ``await``ing a completed task returns its result.
            if self._stderr_task is not None:
                try:
                    return await self._stderr_task
                except Exception as exc:
                    # ``_drain_capped`` blew up reading the pipe — rare,
                    # but worth a breadcrumb. Without this, an empty
                    # stderr in the result silently hides a real I/O
                    # failure.
                    _logger.debug("stderr_task failed on re-entry stop: %r", exc)
                    return b""
            return b""
        self._stopped = True

        if self._proc is None:
            return b""

        if hard:
            _kill_process_group(self._proc.pid)
            await self._proc.wait()
        else:
            try:
                await self._write_message({"jsonrpc": "2.0", "method": "shutdown"})
            except (BrokenPipeError, ConnectionError):
                # Expected race: the runner can exit between us building
                # the shutdown frame and writing it (e.g., it just finished
                # responding to ``execute`` and tore down). The OS error
                # IS the "runner already left" signal — silent pass is
                # correct, logging would spam every happy-path session.
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=_SHUTDOWN_GRACE_SECONDS)
            except asyncio.TimeoutError:
                # Runner accepted shutdown but didn't exit in 5s — that's
                # a wedge (deadlock in pyodide_runtime, an unkillable
                # async task in the runner, ...). Operators want to know
                # this happened before we SIGKILL.
                _logger.warning(
                    "runner did not exit within %.1fs of shutdown; escalating to SIGKILL",
                    _SHUTDOWN_GRACE_SECONDS,
                )
                _kill_process_group(self._proc.pid)
                await self._proc.wait()

        if self._stderr_task is not None:
            try:
                return await self._stderr_task
            except Exception as exc:
                # Same rationale as the re-entry case above.
                _logger.debug("stderr_task failed during stop: %r", exc)
                return b""
        return b""

    # -- public RPC methods ---------------------------------------------------

    async def mount(self, host_path: Path, virtual_path: str) -> None:
        """Copy a host file into Pyodide's virtual FS at ``virtual_path``.

        Raises ``SandboxError`` on RPC failure (host file missing, Deno
        --allow-read denial). Without this surfacing, a denied mount
        would let bootstrap run anyway and crash with a confusing
        ``FileNotFoundError`` deep inside Pyodide.
        """
        rpc = await self._request(
            "mount_file",
            {"host_path": str(host_path), "virtual_path": virtual_path},
        )
        if rpc.error is not None:
            raise SandboxError(_format_rpc_error(f"mount_file({virtual_path})", rpc.error))

    async def bootstrap(
        self, trace_virtual_path: str, index_virtual_path: str
    ) -> CodeExecutionResult:
        """Load the trace + index inside Pyodide and prepare ``user_globals``.

        Returns a ``CodeExecutionResult``: the runner's ``halo_bootstrap``
        wraps the load in stdout/stderr capture, so a Python-level
        failure (malformed index, missing wheel) returns a result with
        ``exit_code != 0`` and the traceback in stderr. RPC-level
        failures (runner rejected the request) raise ``SandboxError``.
        """
        rpc = await self._request(
            "bootstrap",
            {"trace_path": trace_virtual_path, "index_path": index_virtual_path},
        )
        if rpc.error is not None:
            raise SandboxError(_format_rpc_error("bootstrap", rpc.error))
        return _result_from_rpc(rpc.result)

    async def execute(self, code: str) -> CodeExecutionResult:
        """Run ``code`` in Pyodide and return its captured stdout/stderr/exit_code."""
        rpc = await self._request("execute", {"code": code})
        if rpc.error is not None:
            raise SandboxError(_format_rpc_error("execute", rpc.error))
        return _result_from_rpc(rpc.result)

    # -- private wire helpers -------------------------------------------------

    async def _request(self, method: str, params: dict) -> _RpcResult:
        """Send one JSON-RPC request, await the matching response."""
        self._next_id += 1
        request_id = self._next_id
        await self._write_message(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        return await self._read_until_id(request_id, method)

    async def _write_message(self, payload: dict) -> None:
        """Serialize ``payload`` as one JSON-RPC line and write it to the runner's stdin.

        ``ensure_ascii=False`` keeps non-ASCII content as raw UTF-8 rather
        than ``\\uXXXX`` escapes — smaller wire, and (more importantly)
        the path the runner's ``TextDecoder`` actually has to handle
        across chunk boundaries. With ASCII escaping every byte on stdin
        is single-byte, so the multi-byte decode path would never be
        exercised in production.
        """
        assert self._proc is not None and self._proc.stdin is not None
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self._proc.stdin.write(data)
        await self._proc.stdin.drain()

    async def _read_until_id(self, expected_id: int, method: str) -> _RpcResult:
        """Read JSON-RPC lines from stdout until the matching id arrives.

        Pyodide's package loader emits status lines ("Loading numpy, ...")
        before/between JSON responses. Skip those; only treat lines starting
        with '{' as JSON-RPC.
        """
        assert self._proc is not None and self._proc.stdout is not None
        max_skip = 200
        for _ in range(max_skip):
            line = await self._proc.stdout.readline()
            if not line:
                raise SandboxError(f"runner closed stdout before responding to {method!r}")
            text = line.decode("utf-8", errors="replace").strip()
            if not text or not text.startswith("{"):
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                # ``{`` start-byte heuristic isn't enough — a status line
                # could legitimately begin with one. Skip-and-continue is
                # the loader-noise control flow; logging would fire on
                # every cold boot.
                continue
            if msg.get("id") != expected_id:
                continue
            return _RpcResult(result=msg.get("result"), error=msg.get("error"))
        raise SandboxError(f"too many non-JSON lines while waiting for {method!r} response")

    async def _await_ready(self) -> None:
        """Wait for the ``{"result": {"ready": true}}`` sentinel from runner.js.

        Pyodide's package loader prints non-JSON status lines (``Loading
        numpy, pandas, ...``) to stdout before the runner emits its first
        JSON message, so we skip non-JSON / non-id-zero lines until the
        sentinel arrives. A blank stdout (process exited) is fatal.
        """
        assert self._proc is not None and self._proc.stdout is not None
        deadline = asyncio.get_event_loop().time() + _READY_TIMEOUT_SECONDS
        max_lines = 200
        for _ in range(max_lines):
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise SandboxError("Pyodide runner did not become ready in time")
            try:
                line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise SandboxError("Pyodide runner did not become ready in time") from exc
            if not line:
                raise SandboxError("Pyodide runner exited before signalling ready")
            text = line.decode("utf-8", errors="replace").strip()
            if not text or not text.startswith("{"):
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                # Same rationale as ``_read_until_id``: status-line noise
                # is the rule during boot, not the exception.
                continue
            if msg.get("error") is not None:
                raise SandboxError(f"runner failed at startup: {msg['error']}")
            if msg.get("id") == 0 and msg.get("result", {}).get("ready") is True:
                return
            # Future Pyodide / Deno releases could emit JSON diagnostics on
            # stdout during boot. Skip-and-keep-looking matches what
            # ``_read_until_id`` does for non-matching ids; raising here
            # would kill the session for nothing.
            continue
        raise SandboxError("too many non-JSON lines while waiting for ready sentinel")


def _result_from_rpc(rpc_result: dict | None) -> CodeExecutionResult:
    """Coerce a ``halo_bootstrap`` / ``halo_execute`` dict into a result model.

    Both phases route through here, so byte-cap truncation lives here
    too — a recursive-import traceback from bootstrap can be just as
    big as a chatty execute, and either one bloats the agent's prompt
    if it flows back uncapped. ``_truncate_to_bytes`` slices safely on
    UTF-8 boundaries.
    """
    rpc_result = rpc_result or {}
    return CodeExecutionResult(
        exit_code=int(rpc_result.get("exit_code", 1)),
        stdout=_truncate_to_bytes(str(rpc_result.get("stdout", "")), _MAX_STDOUT_BYTES),
        stderr=_truncate_to_bytes(str(rpc_result.get("stderr", "")), _MAX_STDERR_BYTES),
        timed_out=False,
    )


# ---------------------------------------------------------------------------
# Discovery + Pyodide asset preparation
# ---------------------------------------------------------------------------


def _locate_deno() -> Path | None:
    """Resolve the Deno binary, preferring the bundled PyPI wheel over PATH.

    The ``deno`` PyPI package ships ``deno`` as a per-platform binary in
    its wheel and exposes ``deno.find_deno_bin()`` to locate it; that's
    the out-of-the-box path. ``shutil.which`` is the system fallback for
    rare platforms with no wheel (musl Linux, FreeBSD) or for power users
    with a system-managed Deno.
    """
    try:
        import deno as deno_module  # type: ignore[import-not-found]
    except ImportError:
        # Expected when the user is running with a system-managed Deno
        # and didn't install the ``deno`` PyPI extra. Fall through to
        # ``shutil.which``; silent is correct because PATH discovery is
        # the documented alternative path.
        bundled_path: str | None = None
    else:
        try:
            bundled_path = deno_module.find_deno_bin()
        except Exception as exc:
            # The PyPI wheel is broken in some way (corrupt install,
            # missing per-platform binary). We fall back to PATH, but
            # leave a breadcrumb so an operator who later finds nothing
            # on PATH either can connect the two.
            _logger.debug("deno.find_deno_bin() raised, falling back to PATH: %r", exc)
            bundled_path = None

    if bundled_path:
        candidate = Path(bundled_path)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    system = shutil.which("deno")
    if system is not None:
        return Path(system)
    return None


def _query_deno_dir(deno_path: Path) -> Path:
    """Read ``deno info --json`` for ``denoDir``; the cache root we whitelist."""
    try:
        result = subprocess.run(
            [str(deno_path), "info", "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _ResolutionError(f"failed to invoke `deno info`: {exc}") from exc

    if result.returncode != 0:
        raise _ResolutionError(f"`deno info` exited {result.returncode}: {result.stderr.strip()}")

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise _ResolutionError(f"`deno info` did not return valid JSON: {exc}") from exc

    deno_dir = info.get("denoDir")
    if not deno_dir:
        raise _ResolutionError("`deno info` did not report a denoDir")
    return Path(deno_dir)


def _ensure_npm_cache(deno_path: Path, deno_dir: Path) -> Path:
    """Return the Deno-cached ``pyodide@<version>`` directory; warm it via ``deno cache``."""
    target = deno_dir / "npm" / "registry.npmjs.org" / "pyodide" / _PYODIDE_VERSION
    if (target / "pyodide.asm.wasm").is_file():
        return target

    runner_path = (Path(__file__).parent / _RUNNER_FILENAME).resolve()
    try:
        result = subprocess.run(
            [str(deno_path), "cache", str(runner_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _ResolutionError(f"failed to invoke `deno cache`: {exc}") from exc

    if result.returncode != 0:
        raise _ResolutionError(
            "`deno cache` exited "
            f"{result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
        )

    if not (target / "pyodide.asm.wasm").is_file():
        raise _ResolutionError(f"deno cache did not populate Pyodide assets at {target}")
    return target


def _normalize_pkg_name(name: str) -> str:
    """PEP 503 distribution-name normalization: lowercase, ``-``/``_``/``.`` → ``-``.

    Pyodide's lockfile keys ``packages`` by the normalized form
    (``"pydantic-core"``, ``"python-dateutil"``) but its ``depends``
    lists sometimes leak the importable underscore form
    (``"pydantic_core"``). Normalizing both sides lets the graph walk
    follow either spelling without a hand-maintained alias table.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _resolve_required_wheels(pyodide_dir: Path) -> list[str]:
    """Walk ``pyodide-lock.json`` from ``_REQUIRED_PACKAGES`` to wheel filenames.

    The lockfile keys ``packages`` by PEP 503-normalized distribution
    name (``"numpy"``, ``"python-dateutil"``, ``"pydantic-core"``);
    ``depends`` lists occasionally use the importable underscore form
    (``"pydantic_core"``). Normalize once, then recursively collect
    every ``file_name`` reachable from ``_REQUIRED_PACKAGES``. On a
    Pyodide bump the resolved filenames update automatically — no
    human bookkeeping per release.
    """
    lockfile = pyodide_dir / "pyodide-lock.json"
    try:
        packages_raw = json.loads(lockfile.read_text())["packages"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise _ResolutionError(f"failed to read packages from {lockfile}: {exc}") from exc

    packages = {_normalize_pkg_name(name): entry for name, entry in packages_raw.items()}
    wheels: set[str] = set()

    def collect(pkg_name: str) -> None:
        entry = packages.get(_normalize_pkg_name(pkg_name))
        if entry is None:
            raise _ResolutionError(f"{lockfile} has no entry for required package {pkg_name!r}")
        if entry["file_name"] in wheels:
            return  # dedup the diamond case (e.g., ``six`` reached twice)
        wheels.add(entry["file_name"])
        for dep in entry.get("depends", []):
            collect(dep)

    for pkg in _REQUIRED_PACKAGES:
        collect(pkg)
    return sorted(wheels)


def _ensure_wheels(pyodide_dir: Path) -> None:
    """Backfill the wheels Pyodide needs at boot from the public Pyodide CDN.

    The Pyodide loader looks for these next to ``pyodide.asm.wasm``. When a
    wheel is missing it falls back to ``cdn.jsdelivr.net``, which fails
    under our locked-down ``deno run`` because ``--allow-net`` is not
    granted. Downloading them here (Python-side, no Deno permission scope
    involved) is a one-time setup cost on a fresh machine.
    """
    wheels = _resolve_required_wheels(pyodide_dir)
    missing = [w for w in wheels if not (pyodide_dir / w).is_file()]
    if not missing:
        return
    for wheel in missing:
        url = _WHEEL_BASE_URL + wheel
        target = pyodide_dir / wheel
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60.0) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out)
            os.replace(tmp, target)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise _ResolutionError(
                f"failed to download Pyodide wheel {wheel} from {url}: {exc}"
            ) from exc


def _log_unavailable(diagnostic: str) -> None:
    """Emit the sandbox-unavailable warning to logging and stderr.

    Called when the host can't produce a working sandbox (Deno binary
    missing, Pyodide assets missing, wheel pre-cache failed). The
    warning is intentionally visible in every common deployment surface
    — CLI, library import, container logs — so operators see why
    ``run_code`` is missing from the agent's tool list. The remediation
    is hardcoded here rather than passed in: the failure modes all have
    the same fix path (the ``deno`` PyPI dep ships a binary; if it
    failed, reinstall or fall back to a system Deno).
    """
    warning = (
        "HALO run_code disabled: sandbox unavailable.\n\n"
        f"Reason:\n  {diagnostic}\n\n"
        "How to fix:\n"
        "  The ``deno`` PyPI dependency normally ships a per-platform binary\n"
        "  alongside the engine. If it didn't (uncommon platforms like musl\n"
        "  Linux or FreeBSD, or a broken install), reinstall the engine\n"
        "  package or drop a Deno >=2.7 binary on PATH:\n"
        "    curl -fsSL https://deno.land/install.sh | sh\n\n"
        "The engine will continue without exposing run_code to the agent."
    )
    _logger.warning(warning)
    print(warning, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Pure utilities
# ---------------------------------------------------------------------------


async def _drain_capped(stream: asyncio.StreamReader | None, cap: int) -> bytes:
    """Read ``stream`` into a buffer with the same cap+truncation marker as stdout."""
    if stream is None:
        return b""
    buf = bytearray()
    reached_eof = False
    while len(buf) < cap:
        chunk = await stream.read(min(4096, cap - len(buf)))
        if not chunk:
            reached_eof = True
            break
        buf.extend(chunk)
    truncated = False
    if not reached_eof:
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            truncated = True
    if truncated:
        marker = _TRUNCATION_MARKER.encode("utf-8")
        marker_len = min(len(marker), cap)
        del buf[cap - marker_len :]
        buf.extend(marker[:marker_len])
    return bytes(buf)


def _truncate_to_bytes(text: str, cap_bytes: int) -> str:
    """Truncate ``text`` so its UTF-8 encoding is at most ``cap_bytes`` bytes.

    The cap is named in bytes (``_MAX_STDOUT_BYTES`` etc.) so we honor
    that contract: encode, slice on bytes, decode with ``errors="ignore"``
    to drop any trailing partial UTF-8 sequence (no U+FFFD smearing
    when the cut lands mid-character). Multi-byte content like emoji
    or CJK shrinks the visible character count but never lets the byte
    output exceed the cap.

    The truncation marker is pure ASCII so its byte length equals its
    character length; we reserve those bytes at the tail. With the
    realistic 64 KB caps the engine ships, the marker (~30 bytes) is
    always tiny relative to the budget.
    """
    if cap_bytes <= 0:
        return text
    encoded = text.encode("utf-8")
    if len(encoded) <= cap_bytes:
        return text
    head_budget = max(0, cap_bytes - len(_TRUNCATION_MARKER))
    head = encoded[:head_budget].decode("utf-8", errors="ignore")
    return head + _TRUNCATION_MARKER


def _attach_deno_stderr(result: CodeExecutionResult, stderr_extra: bytes) -> CodeExecutionResult:
    """Append any deno-side stderr noise to the result's ``stderr`` with a marker."""
    if not stderr_extra:
        return result
    extra = stderr_extra.decode("utf-8", errors="replace").strip()
    if not extra:
        return result
    sep = "\n" if result.stderr else ""
    return result.model_copy(update={"stderr": result.stderr + sep + f"[deno stderr] {extra}"})


def _format_rpc_error(context: str, error: _RpcError) -> str:
    return f"[{context}] runner error code={error['code']}: {error['message']}"


def _kill_process_group(pid: int) -> None:
    """Send SIGKILL to ``pid``'s process group so any orphan Deno workers die with it.

    Refuses ``pid <= 0``: ``os.getpgid(0)`` returns the *caller's* group,
    so passing a falsy stub pid would have us sending SIGKILL to our own
    process group. Production never produces such a pid (asyncio
    subprocesses always have valid > 0 pids); the guard exists to keep
    test seams that fabricate a process object safe by default.
    """
    if pid <= 0:
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        # The proc died between us deciding to kill and getpgid/killpg
        # actually running — common race, semantically "already done".
        # Logging would fire on every clean-shutdown timeout escalation.
        pass


__all__ = [
    "Sandbox",
    "SandboxError",
]
