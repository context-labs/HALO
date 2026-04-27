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
    runtime = tmp_path / "runtime"
    work = tmp_path / "work"
    python = tmp_path / "runtime" / "bin" / "python"
    libc = tmp_path / "lib" / "libc.so.6"

    for p in (trace, index):
        p.write_text("")
    runtime.mkdir()
    work.mkdir()
    (tmp_path / "runtime" / "bin").mkdir()
    python.write_text("")
    (tmp_path / "lib").mkdir()
    libc.write_text("")

    return SandboxPolicy(
        python_executable=python,
        readonly_paths=[trace, index, runtime],
        library_paths=[libc],
        writable_paths=[work],
        timeout_seconds=10.0,
    )


def test_linux_command_uses_explicit_bwrap_path(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    script = tmp_path / "work" / "bootstrap.py"
    script.write_text("print(1)")

    bwrap = tmp_path / "bin" / "bwrap"
    bwrap.parent.mkdir()
    bwrap.write_text("")

    argv = build_linux_bubblewrap_command(bwrap_executable=bwrap, policy=policy, script_path=script)

    assert argv[0] == str(bwrap)


def test_linux_command_carries_hardening_flags(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    script = tmp_path / "work" / "bootstrap.py"
    script.write_text("")

    bwrap = tmp_path / "bin" / "bwrap"
    bwrap.parent.mkdir()
    bwrap.write_text("")

    argv = build_linux_bubblewrap_command(bwrap_executable=bwrap, policy=policy, script_path=script)

    assert "--unshare-all" in argv
    assert "--unshare-net" in argv
    assert "--clearenv" in argv
    assert "--die-with-parent" in argv
    assert "--new-session" in argv


def test_linux_command_does_not_mount_proc(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    script = tmp_path / "work" / "bootstrap.py"
    script.write_text("")
    bwrap = tmp_path / "bin" / "bwrap"
    bwrap.parent.mkdir()
    bwrap.write_text("")

    argv = build_linux_bubblewrap_command(bwrap_executable=bwrap, policy=policy, script_path=script)

    assert "--proc" not in argv


def test_linux_command_binds_runtime_and_libraries_at_host_paths(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    script = tmp_path / "work" / "bootstrap.py"
    script.write_text("")
    bwrap = tmp_path / "bin" / "bwrap"
    bwrap.parent.mkdir()
    bwrap.write_text("")

    argv = build_linux_bubblewrap_command(bwrap_executable=bwrap, policy=policy, script_path=script)

    runtime_str = str(tmp_path / "runtime")
    libc_str = str(tmp_path / "lib" / "libc.so.6")

    assert _has_ro_bind(argv, runtime_str, runtime_str)
    assert _has_ro_bind(argv, libc_str, libc_str)


def test_linux_command_invokes_resolved_python(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    script = tmp_path / "work" / "bootstrap.py"
    script.write_text("")
    bwrap = tmp_path / "bin" / "bwrap"
    bwrap.parent.mkdir()
    bwrap.write_text("")

    argv = build_linux_bubblewrap_command(bwrap_executable=bwrap, policy=policy, script_path=script)

    assert argv[-2] == str(policy.python_executable)
    assert argv[-1] == str(script)


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
    sandbox_exec = tmp_path / "bin" / "sandbox-exec"
    sandbox_exec.parent.mkdir(exist_ok=True)
    sandbox_exec.write_text("")

    argv = build_macos_sandbox_exec_command(
        sandbox_exec_executable=sandbox_exec,
        policy=policy,
        script_path=script,
        profile_path=profile_path,
    )

    assert argv[0] == str(sandbox_exec)
    assert "-f" in argv
    assert str(profile_path) in argv
    assert "env" in argv
    assert str(policy.python_executable) in argv


def _has_ro_bind(argv: list[str], src: str, dst: str) -> bool:
    """Return True iff ``--ro-bind src dst`` appears as three consecutive tokens in ``argv``."""
    for i in range(len(argv) - 2):
        if argv[i] == "--ro-bind" and argv[i + 1] == src and argv[i + 2] == dst:
            return True
    return False
