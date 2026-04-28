from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox.models import SandboxConfig
from engine.sandbox.pyodide_client import PyodideClient
from engine.sandbox.sandbox import Sandbox, resolve_sandbox
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder


async def _ready(tmp_path: Path, fixtures_dir: Path) -> tuple[Sandbox, Path, Path]:
    sandbox = resolve_sandbox(config=SandboxConfig(timeout_seconds=60.0))
    if sandbox is None:
        pytest.fail("Pyodide sandbox unavailable in CI; this must work for release.")

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = tmp_path / "traces.jsonl.engine-index.jsonl"
    await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig(index_path=index_path)
    )
    return sandbox, trace_path, index_path


@pytest.mark.asyncio
async def test_sandbox_runs_real_python_against_trace_store(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """The Pyodide-backed sandbox executes user code with a working ``trace_store``."""
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    assert isinstance(sandbox.client, PyodideClient)

    result = await sandbox.run_python(
        code="print('count=', trace_store.trace_count)",
        trace_path=trace_path,
        index_path=index_path,
    )

    assert result.exit_code == 0, (
        f"sandboxed run failed (exit={result.exit_code}, timed_out={result.timed_out}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "count= 3" in result.stdout


@pytest.mark.asyncio
async def test_sandbox_exposes_numpy_and_pandas_aliases(tmp_path: Path, fixtures_dir: Path) -> None:
    """``numpy``, ``pandas``, ``np``, ``pd`` must all be available without explicit imports.

    The bootstrap script preloads them into the user globals dict so user
    code can use the canonical short aliases (``np``, ``pd``) the rest of
    the engine documentation assumes.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)

    result = await sandbox.run_python(
        code=(
            "print('np=', np.__name__)\n"
            "print('pd=', pd.__name__)\n"
            "print('arr=', np.array([1, 2, 3]).sum())\n"
            "print('df=', pd.DataFrame({'a': [1, 2]}).shape)\n"
        ),
        trace_path=trace_path,
        index_path=index_path,
    )

    assert result.exit_code == 0, (
        f"sandboxed run failed (exit={result.exit_code}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "np= numpy" in result.stdout
    assert "pd= pandas" in result.stdout
    assert "arr= 6" in result.stdout
    assert "df= (2, 1)" in result.stdout


@pytest.mark.asyncio
async def test_sandbox_uncaught_exception_surfaces_traceback(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """User exceptions must produce ``exit_code != 0`` plus a traceback in stderr."""
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)

    result = await sandbox.run_python(
        code="raise ValueError('boom')",
        trace_path=trace_path,
        index_path=index_path,
    )

    assert result.exit_code != 0
    assert "ValueError: boom" in result.stderr


@pytest.mark.asyncio
async def test_sandbox_timeout_kills_process(tmp_path: Path, fixtures_dir: Path) -> None:
    """Long-running user code must trip the timeout and report ``timed_out=True``."""
    sandbox = resolve_sandbox(config=SandboxConfig(timeout_seconds=1.0))
    if sandbox is None:
        pytest.fail("Pyodide sandbox unavailable in CI; this must work for release.")

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())

    index_path = tmp_path / "traces.jsonl.engine-index.jsonl"
    await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig(index_path=index_path)
    )

    result = await sandbox.run_python(
        code="while True: pass",
        trace_path=trace_path,
        index_path=index_path,
    )

    assert result.timed_out is True
    assert result.exit_code != 0
