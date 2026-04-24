from __future__ import annotations

import platform
import shutil
from pathlib import Path

import pytest

from engine.sandbox.sandbox_config import SandboxConfig
from engine.sandbox.sandbox_runner import SandboxRunner


@pytest.mark.asyncio
async def test_sandboxed_hello_world(tmp_path: Path, fixtures_dir: Path) -> None:
    system = platform.system()
    if system == "Linux" and shutil.which("bwrap") is None:
        pytest.skip("bubblewrap not installed")
    if system == "Darwin" and shutil.which("sandbox-exec") is None:
        pytest.skip("sandbox-exec unavailable")
    if system not in ("Linux", "Darwin"):
        pytest.skip(f"unsupported platform {system}")

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    from engine.traces.models.trace_index_config import TraceIndexConfig
    from engine.traces.trace_index_builder import TraceIndexBuilder
    index_path = tmp_path / "traces.jsonl.engine-index.jsonl"
    await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig(index_path=index_path)
    )

    sandbox_venv = Path(__file__).resolve().parents[3] / ".sandbox-venv"
    if not (sandbox_venv / "bin" / "python").exists():
        pytest.skip("sandbox venv not built — run engine/scripts/build_sandbox_venv.sh")

    runner = SandboxRunner(sandbox_venv=sandbox_venv)
    result = await runner.run_python(
        code="print('count=', trace_store.trace_count)",
        trace_path=trace_path,
        index_path=index_path,
        config=SandboxConfig(timeout_seconds=15.0),
    )
    assert result.exit_code == 0
    assert "count= 3" in result.stdout
