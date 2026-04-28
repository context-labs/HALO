from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from engine.sandbox.log import log_unavailable

_RUNNER_FILENAME = "runner.js"
_TRACE_COMPAT_FILENAME = "pyodide_trace_compat.py"

# Wheels Pyodide must load to satisfy ``import numpy``/``import pandas``.
# Pinned to the set the runner declares in ``pyodide.loadPackage``; the
# version string comes from the ``pyodide`` npm package and must stay aligned
# with whatever npm resolves at boot.
_PYODIDE_VERSION = "0.29.3"
_REQUIRED_WHEELS = (
    "numpy-2.2.5-cp313-cp313-pyodide_2025_0_wasm32.whl",
    "pandas-2.3.3-cp313-cp313-pyodide_2025_0_wasm32.whl",
    "python_dateutil-2.9.0.post0-py2.py3-none-any.whl",
    "pytz-2025.2-py2.py3-none-any.whl",
    "six-1.17.0-py2.py3-none-any.whl",
)
_WHEEL_BASE_URL = f"https://cdn.jsdelivr.net/pyodide/v{_PYODIDE_VERSION}/full/"

_DENO_MISSING_REMEDIATION = (
    "The ``deno`` PyPI dependency normally ships a per-platform binary.\n"
    "If it failed to provide one (uncommon platform like musl Linux, or a\n"
    "broken install), reinstall the engine env (``task env:setup``) or\n"
    "drop a Deno >=2.7 binary on PATH:\n"
    "  curl -fsSL https://deno.land/install.sh | sh"
)


@dataclass(frozen=True)
class PyodideAssets:
    """Resolved on-disk locations the runner needs read access to."""

    runner_path: Path
    deno_dir: Path
    pyodide_npm_dir: Path


class PyodideError(RuntimeError):
    """Raised when the Deno/Pyodide subprocess returns a JSON-RPC error or dies."""


class PyodideClient:
    """Sync subprocess client for the Deno+Pyodide WASM sandbox.

    Spawns a single long-lived ``deno run`` per :meth:`run_python` call.
    Permissions are hardcoded to the narrow HALO policy: ``--allow-read``
    only on the runner script, the Deno cache (so npm:pyodide resolves),
    and the per-run trace + index paths. We never request ``--allow-net``,
    ``--allow-write``, ``--allow-env``, or ``--allow-run``, so the
    sandboxed Python has no path to host filesystem writes, host network,
    host environment, or subprocess execution.
    """

    def __init__(self, *, deno_executable: Path, assets: PyodideAssets) -> None:
        self._deno = deno_executable
        self._assets = assets

    @property
    def deno_executable(self) -> Path:
        """Resolved ``deno`` binary; surfaced for tests."""
        return self._deno

    @property
    def assets(self) -> PyodideAssets:
        """Assets bundle (runner path + deno cache); surfaced for tests."""
        return self._assets

    @staticmethod
    def resolve() -> PyodideClient | None:
        """Find ``deno`` and pre-cache the runner's Pyodide wheels.

        Resolution order:
          1. ``deno.find_deno_bin()`` from the ``deno`` PyPI dependency —
             this is the normal path: a per-platform binary ships in the
             wheel and ``pip install``/``uv sync`` puts it next to the
             other entry points in ``.venv/bin``. No user action needed.
          2. ``shutil.which("deno")`` — fallback for platforms where the
             ``deno`` wheel is unavailable (e.g. musl Linux) or for users
             who prefer a system-managed Deno.

        Failure modes (no Deno found, ``deno info`` failing, wheel
        backfill failing) log a remediation warning and return ``None``
        so ``resolve_sandbox`` can degrade gracefully.
        """
        deno_path = _locate_deno_executable()
        if deno_path is None:
            log_unavailable(
                diagnostic="deno binary not found (expected from `deno` PyPI dep or PATH)",
                remediation=_DENO_MISSING_REMEDIATION,
            )
            return None

        try:
            assets = _resolve_assets(deno_path)
        except _ResolutionError as exc:
            log_unavailable(diagnostic=str(exc), remediation=_DENO_MISSING_REMEDIATION)
            return None

        return PyodideClient(deno_executable=deno_path, assets=assets)

    def build_argv(self, *, extra_read_paths: list[Path]) -> list[str]:
        """Build the ``deno run`` argv with HALO's locked-down permission set.

        ``--allow-read`` is enumerated explicitly (no wildcards) and covers:
          - the runner script
          - the Deno cache directory (where ``npm:pyodide`` resolves and
            the pre-cached ``*.whl`` wheels live next to ``pyodide.asm.wasm``)
          - any additional per-run paths the caller mounts (trace, index)

        Everything else — ``--allow-net``, ``--allow-write``, ``--allow-env``,
        ``--allow-run`` — is intentionally absent so the sandboxed process
        has no host network, no host writes, no host env vars, no subprocess
        spawn. ``--no-prompt`` is passed so a missing permission errors
        instead of pausing on a TTY prompt.
        """
        read_paths = [self._assets.runner_path, self._assets.deno_dir]
        read_paths.extend(extra_read_paths)
        allow_read = ",".join(str(p) for p in read_paths)
        return [
            str(self._deno),
            "run",
            "--no-prompt",
            f"--allow-read={allow_read}",
            str(self._assets.runner_path),
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _ResolutionError(Exception):
    """Internal: any failure during Deno + wheel pre-cache resolution."""


def _locate_deno_executable() -> Path | None:
    """Resolve the Deno binary, preferring the bundled PyPI wheel over PATH.

    The ``deno`` PyPI package ships ``deno`` as a per-platform binary in
    its wheel and exposes ``deno.find_deno_bin()`` to locate it; that is
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


def _resolve_assets(deno_path: Path) -> PyodideAssets:
    """Locate the Deno cache directory and ensure the Pyodide wheels are pre-cached.

    Wheels live next to ``pyodide.asm.wasm`` inside the Deno npm cache so
    Pyodide's package loader finds them without needing ``--allow-net``.
    Pre-caching is opportunistic: a working install will already have them
    after the first time Deno fetches the npm package, but we backfill any
    missing files here so the sandbox is usable on a fresh checkout.
    """
    runner_path = (Path(__file__).parent / _RUNNER_FILENAME).resolve()
    if not runner_path.is_file():
        raise _ResolutionError(f"runner script missing at {runner_path}")

    deno_dir = _query_deno_dir(deno_path)
    pyodide_dir = _ensure_pyodide_npm_cache(deno_path, deno_dir)
    _ensure_pyodide_wheels(pyodide_dir)

    return PyodideAssets(
        runner_path=runner_path,
        deno_dir=deno_dir,
        pyodide_npm_dir=pyodide_dir,
    )


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
# Subprocess + JSON-RPC plumbing
# ---------------------------------------------------------------------------


@dataclass
class _RpcResult:
    """One JSON-RPC response line, parsed for the caller."""

    result: dict | None
    error: dict | None
    raw: dict


class _RunnerSession:
    """One-shot session: spawn deno, send mount/bootstrap/execute, capture exec result.

    Designed for a single ``run_python`` call: each public Sandbox call
    spawns a fresh Deno process so the WASM filesystem state never leaks
    between runs (same isolation guarantee as the bwrap-based backend).
    """

    _READY_TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        *,
        argv: list[str],
        timeout_seconds: float,
        max_stdout: int,
        max_stderr: int,
    ) -> None:
        self._argv = argv
        self._timeout = timeout_seconds
        self._max_stdout = max_stdout
        self._max_stderr = max_stderr

    async def run(
        self,
        *,
        mounts: list[tuple[Path, str]],
        injects: list[tuple[str, str]],
        bootstrap_code: str,
        user_code: str,
    ) -> "_ExecutionOutcome":
        """Drive the runner: ready → mount → inject → bootstrap → execute → shutdown."""
        proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stderr_task = asyncio.create_task(_drain_capped(proc.stderr, self._max_stderr))
        try:
            try:
                outcome = await asyncio.wait_for(
                    self._drive(proc, mounts, injects, bootstrap_code, user_code),
                    timeout=self._timeout,
                )
                stderr_bytes = await stderr_task
                outcome.stderr_extra = stderr_bytes
                return outcome
            except asyncio.TimeoutError:
                _kill_process_group(proc.pid)
                await proc.wait()
                stderr_bytes = await stderr_task
                return _ExecutionOutcome(
                    exit_code=proc.returncode if proc.returncode is not None else -1,
                    stdout="",
                    stderr=_decode(stderr_bytes),
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
        self,
        proc: asyncio.subprocess.Process,
        mounts: list[tuple[Path, str]],
        injects: list[tuple[str, str]],
        bootstrap_code: str,
        user_code: str,
    ) -> "_ExecutionOutcome":
        assert proc.stdin is not None and proc.stdout is not None

        await self._read_ready(proc.stdout)

        request_id = 0
        for host_path, virtual_path in mounts:
            request_id += 1
            await self._call(
                proc,
                "mount_file",
                {"host_path": str(host_path), "virtual_path": virtual_path},
                request_id,
            )

        for virtual_path, text in injects:
            request_id += 1
            await self._call(
                proc, "inject_text", {"virtual_path": virtual_path, "text": text}, request_id
            )

        request_id += 1
        boot = await self._call(proc, "bootstrap", {"code": bootstrap_code}, request_id)
        if boot.error is not None:
            return _ExecutionOutcome(
                exit_code=1,
                stdout="",
                stderr=_format_rpc_error("bootstrap", boot.error),
                timed_out=False,
            )
        boot_result = boot.result or {}
        if int(boot_result.get("exit_code", 1)) != 0:
            return _ExecutionOutcome(
                exit_code=int(boot_result.get("exit_code", 1)),
                stdout=str(boot_result.get("stdout", "")),
                stderr=str(boot_result.get("stderr", "")),
                timed_out=False,
            )

        request_id += 1
        run = await self._call(proc, "execute", {"code": user_code}, request_id)
        await self._send(proc, {"jsonrpc": "2.0", "method": "shutdown"})
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            _kill_process_group(proc.pid)
            await proc.wait()

        if run.error is not None:
            return _ExecutionOutcome(
                exit_code=1,
                stdout="",
                stderr=_format_rpc_error("execute", run.error),
                timed_out=False,
            )
        run_result = run.result or {}
        return _ExecutionOutcome(
            exit_code=int(run_result.get("exit_code", 1)),
            stdout=_truncate_text(str(run_result.get("stdout", "")), self._max_stdout),
            stderr=_truncate_text(str(run_result.get("stderr", "")), self._max_stderr),
            timed_out=False,
        )

    async def _read_ready(self, stdout: asyncio.StreamReader) -> None:
        """Wait for the ``{"result": {"ready": true}}`` sentinel line from runner.js.

        Pyodide's package loader prints non-JSON status lines (``Loading
        numpy, pandas, ...``) to stdout before the runner emits its first
        JSON message, so we skip non-JSON / non-id-zero lines until the
        sentinel arrives. A blank stdout (process exited) is fatal.
        """
        deadline = asyncio.get_event_loop().time() + self._READY_TIMEOUT_SECONDS
        max_lines = 200
        for _ in range(max_lines):
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise PyodideError("Pyodide runner did not become ready in time")
            try:
                line = await asyncio.wait_for(stdout.readline(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise PyodideError("Pyodide runner did not become ready in time") from exc
            if not line:
                raise PyodideError("Pyodide runner exited before signalling ready")
            text = line.decode("utf-8", errors="replace").strip()
            if not text or not text.startswith("{"):
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            if msg.get("error") is not None:
                raise PyodideError(f"runner failed at startup: {msg['error']}")
            if msg.get("id") == 0 and msg.get("result", {}).get("ready") is True:
                return
            # Some other JSON line we didn't expect; surface clearly.
            raise PyodideError(f"runner emitted unexpected JSON before ready: {msg}")
        raise PyodideError("too many non-JSON lines while waiting for ready sentinel")

    async def _call(
        self,
        proc: asyncio.subprocess.Process,
        method: str,
        params: dict,
        request_id: int,
    ) -> _RpcResult:
        await self._send(
            proc,
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
        )
        return await self._read_response(proc, request_id, method)

    async def _send(self, proc: asyncio.subprocess.Process, payload: dict) -> None:
        assert proc.stdin is not None
        data = (json.dumps(payload) + "\n").encode("utf-8")
        proc.stdin.write(data)
        await proc.stdin.drain()

    async def _read_response(
        self,
        proc: asyncio.subprocess.Process,
        expected_id: int,
        method: str,
    ) -> _RpcResult:
        assert proc.stdout is not None
        # Pyodide's package loader emits status lines ("Loading numpy, ...")
        # before/between JSON responses. Skip those; only treat lines
        # starting with '{' as JSON-RPC.
        max_skip = 200
        for _ in range(max_skip):
            line = await proc.stdout.readline()
            if not line:
                raise PyodideError(f"runner closed stdout before responding to {method!r}")
            text = line.decode("utf-8", errors="replace").strip()
            if not text or not text.startswith("{"):
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            if msg.get("id") != expected_id:
                continue
            return _RpcResult(
                result=msg.get("result"),
                error=msg.get("error"),
                raw=msg,
            )
        raise PyodideError(f"too many non-JSON lines while waiting for {method!r} response")


@dataclass
class _ExecutionOutcome:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    stderr_extra: bytes = b""


_TRUNCATION_MARKER = "\n[... output truncated ...]\n"


def _truncate_text(text: str, cap: int) -> str:
    if cap <= 0 or len(text) <= cap:
        return text
    head_len = max(0, cap - len(_TRUNCATION_MARKER))
    return text[:head_len] + _TRUNCATION_MARKER[: cap - head_len]


async def _drain_capped(stream: asyncio.StreamReader | None, cap: int) -> bytes:
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


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _format_rpc_error(context: str, error: dict) -> str:
    code = error.get("code", "?")
    message = error.get("message", "unknown")
    return f"[{context}] runner error code={code}: {message}"


def _kill_process_group(pid: int) -> None:
    """Send SIGKILL to ``pid``'s process group so any orphan Deno workers die with it."""
    import signal

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


__all__ = [
    "PyodideAssets",
    "PyodideClient",
    "PyodideError",
    "_ExecutionOutcome",  # surfaced for tests/sandbox glue, not public API
    "_RunnerSession",
]
