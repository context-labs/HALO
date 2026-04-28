from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox.models import SandboxConfig
from engine.sandbox.sandbox import Sandbox, resolve_sandbox
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder


async def _ready(tmp_path: Path, fixtures_dir: Path) -> tuple[Sandbox, Path, Path]:
    sandbox = resolve_sandbox(config=SandboxConfig(timeout_seconds=10.0))
    if sandbox is None:
        pytest.fail("macOS sandbox unavailable in CI; this must work for release.")

    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    return sandbox, trace_path, index_path


@pytest.mark.asyncio
async def test_cannot_write_outside_workspace(tmp_path: Path, fixtures_dir: Path) -> None:
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code="open('/etc/attack', 'w').write('no')",
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_cannot_read_outside_allowed(tmp_path: Path, fixtures_dir: Path) -> None:
    """The strict allowlist must deny reads of paths that aren't in any allow.

    We seed a real file in ``$HOME`` (where user secrets like ``.ssh/`` and
    ``.aws/`` actually live) and confirm the sandbox cannot read it. The
    profile's broad ``file-read-metadata`` allow lets the kernel traverse
    ``/Users/<user>`` for path lookups, but ``file-read-data`` is granted
    only on explicit allow entries; nothing in ``$HOME`` is allowed unless
    the engine put it there.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)

    secret = Path.home() / "halo-sandbox-test-secret.tmp"
    secret.write_text("nope")
    try:
        result = await sandbox.run_python(
            code=f"print(open({str(secret)!r}).read())",
            trace_path=trace_path,
            index_path=index_path,
        )
    finally:
        secret.unlink()

    assert result.exit_code != 0
    assert "nope" not in result.stdout


@pytest.mark.asyncio
async def test_no_network(tmp_path: Path, fixtures_dir: Path) -> None:
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code="import socket; s = socket.socket(); s.connect(('1.1.1.1', 80))",
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0
