from __future__ import annotations

from pathlib import Path

from engine.sandbox.runtime_mounts import PythonRuntimeMounts
from engine.sandbox.sandbox_config import SandboxConfig
from engine.sandbox.sandbox_policy import compose_policy


def test_compose_policy_layers_paths_in_positional_order(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    index = tmp_path / "t.idx.jsonl"
    runtime_a = tmp_path / "runtime-a"
    runtime_b = tmp_path / "runtime-b"
    work = tmp_path / "work"
    libc = tmp_path / "libc.so.6"
    python = tmp_path / "bin" / "python"

    for p in (trace, index, libc):
        p.write_text("")
    for d in (runtime_a, runtime_b, work, python.parent):
        d.mkdir()
    python.write_text("")

    runtime = PythonRuntimeMounts(
        python_executable=python,
        runtime_paths=(runtime_a, runtime_b),
        library_paths=(libc,),
    )

    policy = compose_policy(
        trace_path=trace,
        index_path=index,
        runtime_mounts=runtime,
        work_dir=work,
        sandbox_config=SandboxConfig(timeout_seconds=7.0),
    )

    assert policy.python_executable == python
    assert policy.readonly_paths == [trace, index, runtime_a, runtime_b]
    assert policy.library_paths == [libc]
    assert policy.writable_paths == [work]
    assert policy.timeout_seconds == 7.0
    assert policy.network_enabled is False
