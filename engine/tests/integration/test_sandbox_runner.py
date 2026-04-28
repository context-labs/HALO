from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox.pyodide_client import PyodideClient
from engine.sandbox.sandbox import Sandbox, resolve_sandbox
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder


async def _ready(tmp_path: Path, fixtures_dir: Path) -> tuple[Sandbox, Path, Path]:
    sandbox = resolve_sandbox(timeout_seconds=60.0)
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
async def test_sandbox_handles_multibyte_utf8_across_chunk_boundary(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """User code with non-ASCII content must round-trip without UTF-8 corruption.

    Regression: the runner's ``TextDecoder`` previously decoded each stdin
    chunk as a complete UTF-8 sequence. When Deno's stdin reader split a
    multi-byte character across two chunks (entirely possible for any
    payload above the kernel pipe buffer, ~64KB on Linux), the trailing
    partial bytes turned into U+FFFD and JSON.parse failed silently or
    mangled the message.

    This test pushes a large payload (well above ``PIPE_BUF`` so the OS
    fragments the write) packed with multi-byte characters of varying
    widths (2-, 3-, and 4-byte UTF-8) so at least one boundary is likely
    to land mid-character. With ``stream: true`` on the decoder, the
    runner buffers partial sequences across reads and the unicode
    round-trips intact.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)

    # Mix widths so at least one chunk boundary is likely to split a
    # multi-byte sequence: é (2-byte), 日 (3-byte), 🔥 (4-byte).
    needle = "é日🔥"
    repetitions = 30_000
    # ``needle`` is 9 UTF-8 bytes, so the inlined string literal alone is
    # ~270KB — well past anything Deno's stdin reader could buffer in a
    # single chunk, guaranteeing the kernel delivers it in fragments. With
    # the bytes per character mixed (2, 3, 4) the chance that no split
    # ever lands mid-character is vanishingly small.
    padding = needle * repetitions
    code = (
        f"_payload = {padding!r}\n"
        f"assert _payload.count({needle!r}) == {repetitions}, "
        f"'unicode payload corrupted in transit'\n"
        f"print('len=', len(_payload))\n"
        f"print('head=', _payload[:6])\n"
        f"print('tail=', _payload[-6:])\n"
    )

    result = await sandbox.run_python(code=code, trace_path=trace_path, index_path=index_path)

    assert result.exit_code == 0, (
        f"sandboxed run failed (exit={result.exit_code}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # Python ``len`` counts code points: 3 chars × ``repetitions``.
    expected_len = 3 * repetitions
    assert f"len= {expected_len}" in result.stdout, (
        f"expected len(é日🔥 * {repetitions}) = {expected_len}; got: {result.stdout!r}"
    )
    assert "head= é日🔥é日🔥" in result.stdout
    assert "tail= é日🔥é日🔥" in result.stdout


@pytest.mark.asyncio
async def test_sandbox_timeout_kills_process(tmp_path: Path, fixtures_dir: Path) -> None:
    """Long-running user code must trip the timeout and report ``timed_out=True``."""
    sandbox = resolve_sandbox(timeout_seconds=1.0)
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
