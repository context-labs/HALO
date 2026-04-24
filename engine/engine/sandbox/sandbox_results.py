from __future__ import annotations

import asyncio
import os
import signal

from engine.sandbox.sandbox_config import CodeExecutionResult, SandboxConfig


async def run_process_capped(
    *,
    argv: list[str],
    config: SandboxConfig,
) -> CodeExecutionResult:
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
        while len(buf) < cap:
            chunk = await stream.read(min(4096, cap - len(buf)))
            if not chunk:
                break
            buf.extend(chunk)
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
        return bytes(buf)

    try:
        stdout_task = asyncio.create_task(_read_capped(proc.stdout, config.maximum_stdout_bytes))
        stderr_task = asyncio.create_task(_read_capped(proc.stderr, config.maximum_stderr_bytes))

        try:
            exit_code = await asyncio.wait_for(proc.wait(), timeout=config.timeout_seconds)
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


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
