from __future__ import annotations

from pathlib import Path

from engine.sandbox.platform_commands import (
    build_linux_bubblewrap_command,
    build_macos_sandbox_exec_command,
    render_macos_profile,
)
from engine.sandbox.sandbox_config import SandboxPolicy


def _policy(tmp_path: Path) -> SandboxPolicy:
    trace = tmp_path / "t.jsonl"
    index = tmp_path / "t.idx.jsonl"
    venv = tmp_path / "venv"
    work = tmp_path / "work"
    for p in (trace, index):
        p.write_text("")
    for d in (venv, work):
        d.mkdir()
    return SandboxPolicy(
        readonly_paths=[trace, index, venv],
        writable_paths=[work],
        timeout_seconds=10.0,
    )


def test_linux_command_contains_core_flags(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    script = tmp_path / "work" / "bootstrap.py"
    script.write_text("print(1)")
    argv = build_linux_bubblewrap_command(policy=policy, script_path=script)

    joined = " ".join(argv)
    assert argv[0] == "bwrap"
    assert "--unshare-all" in joined
    assert "--unshare-net" in joined
    assert "--clearenv" in joined
    assert "--die-with-parent" in joined
    assert "--ro-bind" in joined
    assert str(script) in joined


def test_macos_profile_denies_by_default(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    profile = render_macos_profile(policy=policy)
    assert "(deny default)" in profile
    assert "(deny network*)" in profile
    assert "(allow file-write*" in profile


def test_macos_command_shape(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    script = tmp_path / "work" / "bootstrap.py"
    script.write_text("")
    profile_path = tmp_path / "profile.sb"
    argv = build_macos_sandbox_exec_command(
        policy=policy, script_path=script, profile_path=profile_path
    )
    assert argv[0] == "sandbox-exec"
    assert "-f" in argv
    assert str(profile_path) in argv
    assert "env" in argv
