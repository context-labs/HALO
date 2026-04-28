from __future__ import annotations

import pytest

from engine.sandbox.sandbox_config import SandboxConfig
from engine.sandbox.sandbox_results import run_process_capped


@pytest.mark.asyncio
async def test_simple_stdout_capture() -> None:
    result = await run_process_capped(
        argv=["/bin/sh", "-c", "echo hello; echo bye"],
        config=SandboxConfig(timeout_seconds=5.0),
    )
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_stdout_cap_truncates() -> None:
    result = await run_process_capped(
        argv=["/bin/sh", "-c", "yes A | head -c 100000"],
        config=SandboxConfig(timeout_seconds=5.0, maximum_stdout_bytes=1000),
    )
    assert len(result.stdout.encode()) <= 1000
    assert result.stdout.endswith("[... output truncated ...]\n")


@pytest.mark.asyncio
async def test_stdout_under_cap_has_no_truncation_marker() -> None:
    result = await run_process_capped(
        argv=["/bin/sh", "-c", "printf hello"],
        config=SandboxConfig(timeout_seconds=5.0, maximum_stdout_bytes=1000),
    )
    assert result.stdout == "hello"


@pytest.mark.asyncio
async def test_timeout_kills_and_reports() -> None:
    result = await run_process_capped(
        argv=["/bin/sh", "-c", "sleep 10"],
        config=SandboxConfig(timeout_seconds=0.5),
    )
    assert result.timed_out is True
    assert result.exit_code != 0
