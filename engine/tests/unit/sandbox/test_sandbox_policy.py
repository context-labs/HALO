from __future__ import annotations

from pathlib import Path

from engine.sandbox.sandbox_config import SandboxConfig
from engine.sandbox.sandbox_policy import compose_policy


def test_compose_policy_sets_paths(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    index = tmp_path / "t.idx.jsonl"
    venv = tmp_path / ".sandbox-venv"
    work = tmp_path / "work"

    for p in (trace, index):
        p.write_text("")
    venv.mkdir()
    work.mkdir()

    policy = compose_policy(
        trace_path=trace,
        index_path=index,
        sandbox_venv=venv,
        work_dir=work,
        sandbox_config=SandboxConfig(timeout_seconds=7.0),
    )
    assert trace in policy.readonly_paths
    assert index in policy.readonly_paths
    assert venv in policy.readonly_paths
    assert work in policy.writable_paths
    assert policy.timeout_seconds == 7.0
    assert policy.network_enabled is False
