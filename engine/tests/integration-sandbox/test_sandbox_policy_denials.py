from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox.models import SandboxConfig
from engine.sandbox.sandbox import Sandbox, resolve_sandbox
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder


async def _ready(tmp_path: Path, fixtures_dir: Path) -> tuple[Sandbox, Path, Path]:
    sandbox = resolve_sandbox(config=SandboxConfig(timeout_seconds=60.0))
    if sandbox is None:
        pytest.fail("Pyodide sandbox unavailable in CI; this must work for release.")

    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    return sandbox, trace_path, index_path


@pytest.mark.asyncio
async def test_cannot_write_to_host_filesystem(tmp_path: Path, fixtures_dir: Path) -> None:
    """User code must not be able to create or write a file outside the WASM FS.

    The Pyodide WASM filesystem is in-memory and isolated from the host;
    the only way to reach the host would be via Deno APIs, which Python
    cannot import. Any ``open(.., 'w')`` lands on the WASM FS, never on
    the host. We assert by checking the host path was not created.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    target = tmp_path / "must-not-exist.txt"
    await sandbox.run_python(
        code=f"open({str(target)!r}, 'w').write('no')",
        trace_path=trace_path,
        index_path=index_path,
    )
    # The write may succeed inside Pyodide's in-memory FS, but it must
    # never produce a file on the host filesystem. (We don't assert on
    # exit_code because Pyodide's FS layer happily creates parents.)
    assert not target.exists(), f"sandbox leaked write to host path {target}"


@pytest.mark.asyncio
async def test_cannot_read_outside_allowed_paths(tmp_path: Path, fixtures_dir: Path) -> None:
    """User code must not be able to read host files we did not mount.

    ``--allow-read`` is scoped to the runner, the Deno cache, and the
    trace + index files. Anything else (``/etc/passwd`` here) is invisible
    to the WASM FS and must surface as ``FileNotFoundError``.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code="print(open('/etc/passwd').read()[:10])",
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0
    assert (
        "FileNotFoundError" in result.stderr
        or "No such file" in result.stderr
        or "PermissionError" in result.stderr
    )


@pytest.mark.asyncio
async def test_no_network(tmp_path: Path, fixtures_dir: Path) -> None:
    """User code must not be able to open a network socket.

    Deno is launched without ``--allow-net``, so any TCP connect attempt
    from Pyodide must fail. We try a real socket connect to a public IP;
    Pyodide's emscripten layer surfaces this as ``OSError``.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code=("import socket\ns = socket.socket()\ns.connect(('1.1.1.1', 80))\n"),
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_no_subprocess_spawn(tmp_path: Path, fixtures_dir: Path) -> None:
    """User code must not be able to spawn host processes.

    Deno is launched without ``--allow-run`` and the WASM Python doesn't
    have a working ``fork``/``execve`` anyway, but we assert here so that
    a future regression that loosens permissions can't silently grant
    subprocess access.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code="import subprocess; subprocess.run(['/bin/echo', 'leak'], check=True)",
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_no_host_env_visible(tmp_path: Path, fixtures_dir: Path) -> None:
    """The sandboxed Python must not see host environment variables.

    Deno is launched without ``--allow-env``. Pyodide's Python populates
    ``os.environ`` with its own canned defaults (``HOME=/home/pyodide``,
    ``USER=web_user``); the test asserts that those defaults are what we
    see, not the host's ``$HOME`` / ``$USER``.
    """
    import getpass
    import os

    host_user = getpass.getuser()
    host_home = os.environ.get("HOME", "")

    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code=(
            "import os\n"
            "print('HALO_SANDBOX_PROBE_USER=' + os.environ.get('USER', '<unset>'))\n"
            "print('HALO_SANDBOX_PROBE_HOME=' + os.environ.get('HOME', '<unset>'))\n"
        ),
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code == 0, result.stderr
    assert f"HALO_SANDBOX_PROBE_USER={host_user}" not in result.stdout, (
        "host USER leaked into sandbox env"
    )
    if host_home:
        assert f"HALO_SANDBOX_PROBE_HOME={host_home}" not in result.stdout, (
            "host HOME leaked into sandbox env"
        )
