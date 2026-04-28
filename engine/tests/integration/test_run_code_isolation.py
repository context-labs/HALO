"""Isolated integration test for ``run_code`` (live sandbox).

Resolves whatever sandbox backend the host provides (bubblewrap on Linux,
``sandbox-exec`` on macOS) and invokes the registered SDK ``FunctionTool``
with a tiny script that confirms the sandbox bootstrap preloaded
``trace_store`` correctly. Skips when no sandbox backend is available — the
platform-specific sandbox suites under ``tests/integration-{linux,macos}/``
are responsible for deeper coverage of policy denials and output capping.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from agents.tool_context import ToolContext as SdkToolContext

from engine.sandbox.models import SandboxConfig
from engine.sandbox.sandbox import resolve_sandbox
from tests.integration.tool_isolation_kit import (
    engine_config,
    load_store,
    new_agent_context,
    root_execution,
    wired_tools,
)


@pytest.mark.asyncio
async def test_run_code_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    sandbox = resolve_sandbox(config=SandboxConfig(timeout_seconds=20.0))
    if sandbox is None:
        pytest.skip("no sandbox backend available on this host")

    cfg = engine_config()
    store = await load_store(tmp_path, fixtures_dir)
    tools = wired_tools(
        cfg=cfg,
        store=store,
        agent_context=new_agent_context(cfg),
        parent_execution=root_execution(cfg),
        sandbox=sandbox,
    )

    raw = await tools["run_code"].on_invoke_tool(
        MagicMock(spec=SdkToolContext),
        json.dumps({"code": "print('count=', trace_store.trace_count)"}),
    )
    result = json.loads(raw)

    assert result["exit_code"] == 0, (
        f"run_code failed (exit={result['exit_code']}, timed_out={result['timed_out']}):\n"
        f"stdout:\n{result['stdout']}\nstderr:\n{result['stderr']}"
    )
    assert result["timed_out"] is False
    assert "count= 3" in result["stdout"]
