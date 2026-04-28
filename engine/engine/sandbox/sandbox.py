"""HALO Deno+Pyodide WASM sandbox: discovery, argv build, subprocess driver.

One module owns the whole sandbox lifecycle so callers see exactly two
public names: the :class:`Sandbox` value object that holds the resolved
paths, and its :meth:`Sandbox.resolve` factory. Everything else —
``deno`` discovery, Pyodide wheel pre-cache, the JSON-RPC subprocess
driver, the per-run ``--allow-read`` set — is private machinery in this
file.

The single per-process knob (``_TIMEOUT_SECONDS``) is a module constant
rather than a config field. Production runs all want the same wall-clock
budget; the one test that needs a different value monkeypatches the
constant directly.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from engine.sandbox.log import log_unavailable
from engine.sandbox.models import CodeExecutionResult

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
_REQUIRED_WHEELS = (
    # numpy + pandas + their transitive deps — preloaded for user analysis code.
    "numpy-2.2.5-cp313-cp313-pyodide_2025_0_wasm32.whl",
    "pandas-2.3.3-cp313-cp313-pyodide_2025_0_wasm32.whl",
    "python_dateutil-2.9.0.post0-py2.py3-none-any.whl",
    "pytz-2025.2-py2.py3-none-any.whl",
    "six-1.17.0-py2.py3-none-any.whl",
    # pydantic + transitive deps. Required so the in-Pyodide bootstrap can
    # import the real ``engine.traces.trace_store`` (and its models) rather
    # than maintain a duplicate stdlib-only shim. Versions are whatever
    # Pyodide 0.29.3's lockfile ships; the host pins a slightly newer
    # pydantic but the API surface ``TraceStore`` uses (``BaseModel``,
    # ``ConfigDict``, ``Field``, ``model_validate_json``) is stable.
    "pydantic-2.12.5-py3-none-any.whl",
    "pydantic_core-2.41.5-cp313-cp313-pyodide_2025_0_wasm32.whl",
    "typing_extensions-4.15.0-py3-none-any.whl",
    "annotated_types-0.7.0-py3-none-any.whl",
    "typing_inspection-0.4.2-py3-none-any.whl",
)
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

_DENO_MISSING_REMEDIATION = (
    "The ``deno`` PyPI dependency normally ships a per-platform binary.\n"
    "If it failed to provide one (uncommon platform like musl Linux, or a\n"
    "broken install), reinstall the engine env (``task env:setup``) or\n"
    "drop a Deno >=2.7 binary on PATH:\n"
    "  curl -fsSL https://deno.land/install.sh | sh"
)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class SandboxError(RuntimeError):
    """Raised when the Deno/Pyodide subprocess returns a JSON-RPC error or dies."""


# Successful resolves are memoized at module level. Resolution work is
# deterministic in a given process — same Deno binary, same sibling files,
# same Deno cache — and shells out to ``deno info`` plus does file-existence
# checks on every call, which adds noticeable per-call latency on a server
# handling many requests. Failed resolves are *not* cached: a transient
# failure (e.g., wheel download blip) shouldn't poison the rest of the
# process. Tests that need a fresh resolve clear via
# ``monkeypatch.setattr(sandbox_module, "_RESOLVED_SANDBOX", None)``.
_RESOLVED_SANDBOX: "Sandbox | None" = None


@dataclass(frozen=True, kw_only=True)
class Sandbox:
    """Resolved Deno+Pyodide WASM sandbox: paths to read + binary to spawn.

    Construct via :meth:`resolve` — that's the only path that does Deno
    discovery and Pyodide pre-cache work. Frozen dataclass because every
    field is a value computed at resolve time and never mutated.

    Each :meth:`run_python` call spawns a fresh ``deno run`` subprocess so
    the WASM filesystem cannot leak between runs. The subprocess is
    launched with the locked-down permission set (``--allow-read`` only,
    scoped to runner + sibling .py files + Deno cache + the per-run trace
    and index files; never ``--allow-net`` / ``--allow-write`` /
    ``--allow-env`` / ``--allow-run``).
    """

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
    def resolve(cls) -> "Sandbox | None":
        """Find ``deno``, verify sibling files, pre-cache Pyodide wheels.

        Resolution order for ``deno``:
          1. ``deno.find_deno_bin()`` from the ``deno`` PyPI dependency —
             the normal path, the binary ships with ``pip install`` /
             ``uv sync``.
          2. ``shutil.which("deno")`` — fallback for unsupported platforms
             (musl Linux, FreeBSD) or system-managed Deno installs.

        On any failure (binary missing, ``deno info`` failing, sibling
        files missing, wheel download failing) a remediation warning is
        emitted via :func:`log_unavailable` and ``None`` is returned so
        ``run_code`` is silently dropped from the agent's tool surface
        rather than registered with broken plumbing.

        Successful results are memoized in ``_RESOLVED_SANDBOX`` so a
        long-lived process (e.g., a server handling many engine runs)
        doesn't pay the ``deno info`` subprocess + file-existence checks
        on every request. Failures are deliberately not cached.
        """
        global _RESOLVED_SANDBOX
        if _RESOLVED_SANDBOX is not None:
            return _RESOLVED_SANDBOX

        deno = _locate_deno_executable()
        if deno is None:
            log_unavailable(
                diagnostic="deno binary not found (expected from `deno` PyPI dep or PATH)",
                remediation=_DENO_MISSING_REMEDIATION,
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
            pyodide_dir = _ensure_pyodide_npm_cache(deno, deno_dir)
            _ensure_pyodide_wheels(pyodide_dir)
        except _ResolutionError as exc:
            log_unavailable(diagnostic=str(exc), remediation=_DENO_MISSING_REMEDIATION)
            return None
        sandbox = cls(
            deno_executable=deno,
            runner_path=runner_path,
            runtime_path=runtime_path,
            engine_init_path=engine_init_path,
            traces_pkg_dir=traces_pkg_dir,
            deno_dir=deno_dir,
        )
        _RESOLVED_SANDBOX = sandbox
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
        try:
            return await _run_session(
                argv=self._build_argv(
                    extra_read_paths=[trace_path.resolve(), index_path.resolve()],
                ),
                mounts=[
                    (trace_path.resolve(), _TRACE_VIRTUAL_PATH),
                    (index_path.resolve(), _INDEX_VIRTUAL_PATH),
                ],
                user_code=code,
            )
        except SandboxError as exc:
            return CodeExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"sandbox runner failure: {exc}",
                timed_out=False,
            )

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


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


class _ResolutionError(Exception):
    """Internal: any failure during Deno + wheel pre-cache resolution."""


def _locate_deno_executable() -> Path | None:
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
        bundled_path: str | None = None
    else:
        try:
            bundled_path = deno_module.find_deno_bin()
        except Exception:
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


def _ensure_pyodide_npm_cache(deno_path: Path, deno_dir: Path) -> Path:
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


def _ensure_pyodide_wheels(pyodide_dir: Path) -> None:
    """Backfill the wheels Pyodide needs at boot from the public Pyodide CDN.

    The Pyodide loader looks for these next to ``pyodide.asm.wasm``. When a
    wheel is missing it falls back to ``cdn.jsdelivr.net``, which fails
    under our locked-down ``deno run`` because ``--allow-net`` is not
    granted. Downloading them here (Python-side, no Deno permission scope
    involved) is a one-time setup cost on a fresh machine.
    """
    missing = [w for w in _REQUIRED_WHEELS if not (pyodide_dir / w).is_file()]
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


# ---------------------------------------------------------------------------
# Subprocess + JSON-RPC driver
# ---------------------------------------------------------------------------


@dataclass
class _RpcResult:
    """One JSON-RPC response line, parsed for the caller."""

    result: dict | None
    error: dict | None


async def _run_session(
    *,
    argv: list[str],
    mounts: list[tuple[Path, str]],
    user_code: str,
) -> CodeExecutionResult:
    """Spawn one Deno subprocess and drive the full mount/bootstrap/execute cycle.

    A single function rather than a class because the subprocess is
    one-shot per ``run_python`` call: every call opens a fresh process,
    sends a deterministic sequence, collects the result, sends shutdown,
    waits. No state worth carrying between calls.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    stderr_task = asyncio.create_task(_drain_capped(proc.stderr, _MAX_STDERR_BYTES))
    try:
        try:
            result = await asyncio.wait_for(
                _drive(proc, mounts, user_code),
                timeout=_TIMEOUT_SECONDS,
            )
            stderr_extra = await stderr_task
            return _attach_deno_stderr(result, stderr_extra)
        except asyncio.TimeoutError:
            _kill_process_group(proc.pid)
            await proc.wait()
            stderr_bytes = await stderr_task
            return CodeExecutionResult(
                exit_code=proc.returncode if proc.returncode is not None else -1,
                stdout="",
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                timed_out=True,
            )
    except BaseException:
        _kill_process_group(proc.pid)
        await proc.wait()
        try:
            await stderr_task
        except Exception:
            pass
        raise


async def _drive(
    proc: asyncio.subprocess.Process,
    mounts: list[tuple[Path, str]],
    user_code: str,
) -> CodeExecutionResult:
    """Send the deterministic mount → bootstrap → execute sequence over JSON-RPC."""
    assert proc.stdin is not None and proc.stdout is not None

    await _read_ready(proc.stdout)

    request_id = 0
    for host_path, virtual_path in mounts:
        request_id += 1
        mount = await _call(
            proc,
            "mount_file",
            {"host_path": str(host_path), "virtual_path": virtual_path},
            request_id,
        )
        # Surface mount failures up front. Without this, a Deno permission
        # denial or missing host file makes the next bootstrap call fall
        # over with a confusing ``FileNotFoundError`` deep inside Pyodide
        # — the user sees a Python traceback when the real cause is the
        # file never landing in the WASM FS.
        if mount.error is not None:
            return CodeExecutionResult(
                exit_code=1,
                stdout="",
                stderr=_format_rpc_error(f"mount_file({virtual_path})", mount.error),
                timed_out=False,
            )

    request_id += 1
    boot = await _call(
        proc,
        "bootstrap",
        {"trace_path": _TRACE_VIRTUAL_PATH, "index_path": _INDEX_VIRTUAL_PATH},
        request_id,
    )
    if boot.error is not None:
        return CodeExecutionResult(
            exit_code=1,
            stdout="",
            stderr=_format_rpc_error("bootstrap", boot.error),
            timed_out=False,
        )
    boot_result = boot.result or {}
    if int(boot_result.get("exit_code", 1)) != 0:
        return CodeExecutionResult(
            exit_code=int(boot_result.get("exit_code", 1)),
            stdout=str(boot_result.get("stdout", "")),
            stderr=str(boot_result.get("stderr", "")),
            timed_out=False,
        )

    request_id += 1
    run = await _call(proc, "execute", {"code": user_code}, request_id)
    # Shutdown is best-effort cleanup — the execute result is already in
    # hand. A ``BrokenPipeError`` here means the runner exited between
    # responding and reading our shutdown frame; the process is already
    # gone, there's nothing to shut down, and propagating the OSError
    # would discard a perfectly good ``CodeExecutionResult``.
    try:
        await _send(proc, {"jsonrpc": "2.0", "method": "shutdown"})
    except (BrokenPipeError, ConnectionError):
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        _kill_process_group(proc.pid)
        await proc.wait()

    if run.error is not None:
        return CodeExecutionResult(
            exit_code=1,
            stdout="",
            stderr=_format_rpc_error("execute", run.error),
            timed_out=False,
        )
    run_result = run.result or {}
    return CodeExecutionResult(
        exit_code=int(run_result.get("exit_code", 1)),
        stdout=_truncate(str(run_result.get("stdout", "")), _MAX_STDOUT_BYTES),
        stderr=_truncate(str(run_result.get("stderr", "")), _MAX_STDERR_BYTES),
        timed_out=False,
    )


async def _read_ready(stdout: asyncio.StreamReader) -> None:
    """Wait for the ``{"result": {"ready": true}}`` sentinel from runner.js.

    Pyodide's package loader prints non-JSON status lines (``Loading
    numpy, pandas, ...``) to stdout before the runner emits its first
    JSON message, so we skip non-JSON / non-id-zero lines until the
    sentinel arrives. A blank stdout (process exited) is fatal.
    """
    deadline = asyncio.get_event_loop().time() + _READY_TIMEOUT_SECONDS
    max_lines = 200
    for _ in range(max_lines):
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise SandboxError("Pyodide runner did not become ready in time")
        try:
            line = await asyncio.wait_for(stdout.readline(), timeout=remaining)
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
            continue
        if msg.get("error") is not None:
            raise SandboxError(f"runner failed at startup: {msg['error']}")
        if msg.get("id") == 0 and msg.get("result", {}).get("ready") is True:
            return
        # Future Pyodide / Deno releases could emit JSON diagnostics on
        # stdout during boot. Skip-and-keep-looking matches what
        # ``_read_response`` already does for non-matching ids; raising
        # here would kill the session for nothing.
        continue
    raise SandboxError("too many non-JSON lines while waiting for ready sentinel")


async def _call(
    proc: asyncio.subprocess.Process,
    method: str,
    params: dict,
    request_id: int,
) -> _RpcResult:
    """Send one JSON-RPC request, await the matching response, return result/error."""
    await _send(
        proc,
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
    )
    return await _read_response(proc, request_id, method)


async def _send(proc: asyncio.subprocess.Process, payload: dict) -> None:
    """Serialize ``payload`` as one JSON-RPC line and write it to the runner's stdin.

    ``ensure_ascii=False`` keeps non-ASCII content as raw UTF-8 rather
    than ``\\uXXXX`` escapes — smaller wire, and (more importantly)
    the path the runner's ``TextDecoder`` actually has to handle
    across chunk boundaries. With ASCII escaping every byte on stdin
    is single-byte, so the multi-byte decode path would never be
    exercised in production.
    """
    assert proc.stdin is not None
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    proc.stdin.write(data)
    await proc.stdin.drain()


async def _read_response(
    proc: asyncio.subprocess.Process,
    expected_id: int,
    method: str,
) -> _RpcResult:
    """Read JSON-RPC lines from ``proc.stdout`` until the matching id arrives."""
    assert proc.stdout is not None
    # Pyodide's package loader emits status lines ("Loading numpy, ...")
    # before/between JSON responses. Skip those; only treat lines starting
    # with '{' as JSON-RPC.
    max_skip = 200
    for _ in range(max_skip):
        line = await proc.stdout.readline()
        if not line:
            raise SandboxError(f"runner closed stdout before responding to {method!r}")
        text = line.decode("utf-8", errors="replace").strip()
        if not text or not text.startswith("{"):
            continue
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            continue
        if msg.get("id") != expected_id:
            continue
        return _RpcResult(result=msg.get("result"), error=msg.get("error"))
    raise SandboxError(f"too many non-JSON lines while waiting for {method!r} response")


# ---------------------------------------------------------------------------
# Output handling
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


def _truncate(text: str, cap_bytes: int) -> str:
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


def _format_rpc_error(context: str, error: dict) -> str:
    code = error.get("code", "?")
    message = error.get("message", "unknown")
    return f"[{context}] runner error code={code}: {message}"


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
        pass


__all__ = [
    "Sandbox",
    "SandboxError",
]
