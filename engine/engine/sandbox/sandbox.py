from __future__ import annotations

import asyncio
import os
import platform
import signal
import tempfile
from pathlib import Path

from engine.sandbox.bootstrap import render_bootstrap_script
from engine.sandbox.linux_client import LinuxClient
from engine.sandbox.log import log_unavailable
from engine.sandbox.macos_client import MacosClient
from engine.sandbox.models import (
    CodeExecutionResult,
    PythonRuntimeMounts,
    SandboxConfig,
)
from engine.sandbox.runtime_mounts import discover_python_runtime_mounts

_TRUNCATION_MARKER = b"\n[... output truncated ...]\n"


class Sandbox:
    """Single front door for running Python under a per-platform sandbox client.

    A ``Sandbox`` always represents a working backend: ``resolve_sandbox()``
    is the only blessed factory and returns ``None`` when the host cannot
    provide one. Holds the resolved client (Linux bubblewrap or macOS
    sandbox-exec), the Python runtime mount manifest, and the run config.

    Per call to :meth:`run_python`: creates a temp work dir, renders the
    bootstrap script that preloads ``TraceStore``, builds platform argv via
    the client, and spawns the subprocess with capped output and a
    wall-clock timeout.
    """

    def __init__(
        self,
        *,
        client: LinuxClient | MacosClient,
        runtime_mounts: PythonRuntimeMounts,
        config: SandboxConfig,
    ) -> None:
        self._client = client
        self._runtime_mounts = runtime_mounts
        self._config = config

    @property
    def client(self) -> LinuxClient | MacosClient:
        """Return the underlying platform client (read-only test seam)."""
        return self._client

    async def run_python(
        self,
        *,
        code: str,
        trace_path: Path,
        index_path: Path,
    ) -> CodeExecutionResult:
        """Run user-supplied Python in the sandbox; returns a typed result regardless of pass/fail/timeout."""
        with tempfile.TemporaryDirectory(prefix="halo-sbx-") as tmp:
            work_dir = Path(tmp)
            (work_dir / "tmp").mkdir()
            script = work_dir / "bootstrap.py"
            script.write_text(
                render_bootstrap_script(
                    user_code=code,
                    trace_path=str(trace_path),
                    index_path=str(index_path),
                )
            )

            readonly_paths: list[Path] = [
                trace_path,
                index_path,
                *self._runtime_mounts.runtime_paths,
            ]

            if isinstance(self._client, LinuxClient):
                argv = self._client.build_argv(
                    python_executable=self._runtime_mounts.python_executable,
                    script_path=script,
                    work_dir=work_dir,
                    readonly_paths=readonly_paths,
                    library_paths=list(self._runtime_mounts.library_paths),
                )
            else:
                profile_path = work_dir / "profile.sb"
                profile_path.write_text(
                    self._client.render_profile(
                        readonly_paths=readonly_paths,
                        writable_paths=[work_dir],
                    )
                )
                argv = self._client.build_argv(
                    python_executable=self._runtime_mounts.python_executable,
                    script_path=script,
                    profile_path=profile_path,
                    work_dir=work_dir,
                )

            return await self._run_capped(argv=argv)

    async def _run_capped(self, *, argv: list[str]) -> CodeExecutionResult:
        """Spawn ``argv`` in a new session, cap stdout/stderr, enforce timeout, kill the whole process group on timeout."""
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        async def _read_capped(stream: asyncio.StreamReader | None, cap: int) -> bytes:
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
                marker_len = min(len(_TRUNCATION_MARKER), cap)
                del buf[cap - marker_len :]
                buf.extend(_TRUNCATION_MARKER[:marker_len])
            return bytes(buf)

        try:
            stdout_task = asyncio.create_task(
                _read_capped(proc.stdout, self._config.maximum_stdout_bytes)
            )
            stderr_task = asyncio.create_task(
                _read_capped(proc.stderr, self._config.maximum_stderr_bytes)
            )

            try:
                exit_code = await asyncio.wait_for(
                    proc.wait(), timeout=self._config.timeout_seconds
                )
                timed_out = False
            except asyncio.TimeoutError:
                _kill_process_group(proc.pid)
                await proc.wait()
                exit_code = proc.returncode if proc.returncode is not None else -1
                timed_out = True

            stdout = await stdout_task
            stderr = await stderr_task
        except BaseException:
            _kill_process_group(proc.pid)
            raise

        return CodeExecutionResult(
            exit_code=exit_code,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            timed_out=timed_out,
        )


def resolve_sandbox(*, config: SandboxConfig) -> Sandbox | None:
    """Probe the host once; return a ready ``Sandbox`` or ``None``.

    The platform clients log their own unavailability warning before
    returning ``None``; this function just propagates that ``None`` back to
    the caller without inspecting why.
    """
    client = _resolve_client()
    if client is None:
        return None
    runtime_mounts = discover_python_runtime_mounts(python_executable=config.python_executable)
    return Sandbox(client=client, runtime_mounts=runtime_mounts, config=config)


def _resolve_client() -> LinuxClient | MacosClient | None:
    """Pick the platform client. Logs + returns ``None`` on unsupported platforms."""
    system = platform.system()
    if system == "Linux":
        return LinuxClient.resolve()
    if system == "Darwin":
        return MacosClient.resolve()
    log_unavailable(
        diagnostic=f"unsupported platform: {system}",
        remediation="run_code requires Linux (bubblewrap) or macOS (sandbox-exec).",
    )
    return None


def _kill_process_group(pid: int) -> None:
    """Signal SIGKILL to the whole process group of ``pid``; tolerate already-exited processes."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


# Re-exports kept here so callers can grab everything from one module.
__all__ = [
    "Sandbox",
    "SandboxConfig",
    "resolve_sandbox",
]
