from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox.sandbox import Sandbox
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder


async def _ready(tmp_path: Path, fixtures_dir: Path) -> tuple[Sandbox, Path, Path]:
    sandbox = Sandbox.get()
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


@pytest.mark.asyncio
async def test_secret_env_var_does_not_leak(
    tmp_path: Path, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host-process env var with a sentinel value must not appear in sandbox stdout.

    Stronger than ``test_no_host_env_visible``: that test relies on the
    host's USER/HOME being non-empty. Here we plant a known sentinel
    on the parent process before spawning the sandbox and assert it
    never appears anywhere in the sandbox's view of the world.
    """
    sentinel = "HALO_SECRET_SENTINEL_2c6c4f1a"
    monkeypatch.setenv("HALO_SANDBOX_LEAK_PROBE", sentinel)

    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code=("import os\nfor k, v in os.environ.items():\n    print(f'{k}={v}')\n"),
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code == 0, result.stderr
    assert sentinel not in result.stdout, (
        f"sandbox env leaked sentinel {sentinel!r} from parent process"
    )
    assert sentinel not in result.stderr, (
        f"sandbox stderr leaked sentinel {sentinel!r} from parent process"
    )


@pytest.mark.asyncio
async def test_proc_self_environ_does_not_leak(tmp_path: Path, fixtures_dir: Path) -> None:
    """``/proc/self/environ`` is the standard side-channel for env exfiltration on Linux.

    Even with ``--allow-env`` denied, code that can read ``/proc`` could
    in principle walk to the host process's environ blob. The locked-down
    ``--allow-read`` doesn't include ``/proc``, so the read should fail
    with ``FileNotFoundError`` (Pyodide's WASM FS has no ``/proc`` either).
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code="print(open('/proc/self/environ', 'rb').read()[:64])",
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
async def test_no_http_via_urllib(tmp_path: Path, fixtures_dir: Path) -> None:
    """``urllib.request.urlopen`` is the most natural network call for an agent — must fail.

    Pyodide ships ``urllib``; without ``--allow-net`` the underlying
    socket / fetch attempt has nowhere to go. We pick a domain that
    resolves to a stable IP so DNS isn't the only thing being denied.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code=(
            "import urllib.request\n"
            "urllib.request.urlopen('http://example.com', timeout=5).read()\n"
        ),
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_dns_does_not_leak_to_host_resolver(tmp_path: Path, fixtures_dir: Path) -> None:
    """``socket.getaddrinfo`` returns synthetic addresses, not real DNS lookups.

    Some sandboxes accidentally permit DNS while denying TCP, which
    would let an agent exfiltrate via DNS query payloads
    (``${secret}.attacker.com``). Pyodide's ``getaddrinfo`` is a stub
    that returns deterministic placeholder addresses (e.g.,
    ``172.29.1.0`` for any hostname) without ever calling out to a
    real resolver — so even though it ``returns`` a result, the
    hostname never reaches a DNS server.

    We assert that property by checking that ``example.com`` does NOT
    resolve to its real public IPs. If Pyodide ever switched to a real
    resolver, this test would start returning the real address and
    fail loudly — at which point we'd know to lock it down.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code=(
            "import socket\n"
            "addrs = socket.getaddrinfo('example.com', 80, socket.AF_INET)\n"
            "ips = sorted({a[4][0] for a in addrs})\n"
            "print('IPS=' + ','.join(ips))\n"
        ),
        trace_path=trace_path,
        index_path=index_path,
    )
    # Real example.com IPv4 addresses (currently 23.215.0.{132..138} /
    # 23.220.75.{232..245} / 96.7.128.{175..200} per IANA's reserved
    # block). The synthetic Pyodide answer (172.29.x.x range) won't
    # collide with any of these.
    assert "IPS=23." not in result.stdout, (
        "getaddrinfo returned a real example.com address — DNS is leaking to host resolver"
    )
    assert "IPS=96.7." not in result.stdout, (
        "getaddrinfo returned a real example.com address — DNS is leaking to host resolver"
    )


@pytest.mark.asyncio
async def test_no_loopback_connection(tmp_path: Path, fixtures_dir: Path) -> None:
    """Loopback / 127.0.0.1 must be denied identically to public IPs.

    Some sandbox configs allow loopback by default. That's enough for
    a sandboxed agent to reach localhost-bound services on the host
    (Redis, Postgres, internal admin endpoints). Our policy is: no net
    means no net, including 127.0.0.1.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code=(
            "import socket\ns = socket.socket()\ns.settimeout(2.0)\ns.connect(('127.0.0.1', 22))\n"
        ),
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_no_os_system(tmp_path: Path, fixtures_dir: Path) -> None:
    """``os.system`` is a separate code path from ``subprocess`` — must also fail.

    ``subprocess.run`` going through ``fork+exec`` is one denial path;
    ``os.system`` historically calls ``system(3)`` directly. Both must
    end up unable to spawn a host process under ``--allow-run`` denied
    (and under Pyodide's WASM, both should error before they even
    reach Deno).
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code="import os; rc = os.system('/bin/echo leak'); raise SystemExit(0 if rc != 0 else 1)",
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_no_os_fork(tmp_path: Path, fixtures_dir: Path) -> None:
    """``os.fork`` must fail in the WASM Python regardless of Deno perms.

    Belt-and-suspenders alongside ``--allow-run`` denial: Pyodide has no
    working ``fork``, so a future regression that loosened ``--allow-run``
    would still hit this wall. Asserting it explicitly catches the case
    where someone wires up an actual fork shim.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code="import os; os.fork()",
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_cannot_write_to_mounted_trace_file(tmp_path: Path, fixtures_dir: Path) -> None:
    """Mounting copies bytes into the WASM FS — host trace must remain unmodified.

    The runner reads the host trace once at ``mount_file`` time and
    writes the bytes into Pyodide's in-memory FS. Writes by user code
    to the virtual path target the WASM FS, never the host. Regression
    target: a future change that 'optimized' mounting via a real bind
    or symlink would silently let user code modify the host trace
    file, corrupting the dataset for subsequent runs.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    original_bytes = trace_path.read_bytes()
    result = await sandbox.run_python(
        code=("with open('/input/traces.jsonl', 'wb') as f:\n    f.write(b'POISONED')\n"),
        trace_path=trace_path,
        index_path=index_path,
    )
    # The write itself may succeed in WASM — what matters is the host file.
    assert trace_path.read_bytes() == original_bytes, (
        f"sandbox write to virtual /input/traces.jsonl leaked to host {trace_path}"
        f" — got {result.stdout!r} / {result.stderr!r}"
    )


@pytest.mark.asyncio
async def test_cannot_write_to_runner_via_allowed_read_path(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """``--allow-read`` must not imply ``--allow-write`` for the runner script itself.

    Regression target: a Deno permission grant that overlaps read+write
    would let the agent corrupt ``runner.js`` for the next run. We try
    to write to it via the ``js`` bridge (Deno API directly) since
    Python in WASM can't reach the host writer at all.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    runner_path = sandbox.runner_path
    result = await sandbox.run_python(
        code=("import js\njs.Deno.writeTextFileSync({path!r}, 'POISONED')\n").format(
            path=str(runner_path)
        ),
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0
    # And host file must still be untouched.
    assert "import pyodideModule" in runner_path.read_text(), (
        "runner.js modified through JS bridge despite --allow-write being denied"
    )


@pytest.mark.asyncio
async def test_no_filesystem_via_js_bridge(tmp_path: Path, fixtures_dir: Path) -> None:
    """``import js`` exposes Deno's API; reading via ``Deno.readTextFileSync`` must respect --allow-read.

    Pyodide's ``js`` module is the Python<->JS bridge. Code that reaches
    for it can call any JS-side global, including Deno's namespace.
    Without this guarded, a clever agent could bypass Pyodide's stdlib
    sandboxing and use Deno APIs directly — meaning the only thing
    standing between user code and the host is the Deno permission set.
    Our test confirms that set holds: ``/etc/passwd`` is not in
    ``--allow-read``, so even the JS-side reader is denied.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code="import js; print(js.Deno.readTextFileSync('/etc/passwd')[:32])",
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0
    assert (
        "PermissionDenied" in result.stderr
        or "NotCapable" in result.stderr
        or ("denied" in result.stderr.lower())
    )


@pytest.mark.asyncio
async def test_no_network_via_js_bridge(tmp_path: Path, fixtures_dir: Path) -> None:
    """``js.fetch`` must fail because Deno was launched without ``--allow-net``.

    Same rationale as the FS bridge test: the JS surface is reachable
    from Python via ``import js``, so all Deno-side denials need to
    apply to that path too. ``fetch`` is the canonical check for net
    access on Deno.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code=(
            "import js, asyncio\n"
            "async def go():\n"
            "    return await js.fetch('http://example.com')\n"
            "asyncio.get_event_loop().run_until_complete(go())\n"
        ),
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_no_subprocess_via_js_bridge(tmp_path: Path, fixtures_dir: Path) -> None:
    """``js.Deno.Command`` must fail because Deno was launched without ``--allow-run``.

    Deno's ``Deno.Command`` is the subprocess primitive on the JS side.
    Reachable from Pyodide via ``import js``, denied at the Deno
    permission boundary.
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code=(
            "import js\n"
            "cmd = js.Deno.Command.new('/bin/echo', {'args': ['leak']})\n"
            "cmd.outputSync()\n"
        ),
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_no_ctypes_load(tmp_path: Path, fixtures_dir: Path) -> None:
    """``ctypes.CDLL`` must not be able to load a host shared library.

    ``ctypes`` is in Pyodide's stdlib, but loading a real shared library
    requires reading and mapping a host ``.so``. Under ``--allow-read``
    enumerated to the runner + Deno cache + trace/index, ``/lib/x86_64-
    linux-gnu/libc.so.6`` is invisible. Defensive: if a future Deno
    grant accidentally widened read access, ``ctypes.CDLL`` would
    suddenly become a workable escape (call ``system`` from libc).
    """
    sandbox, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await sandbox.run_python(
        code=("import ctypes\nlibc = ctypes.CDLL('libc.so.6')\nlibc.system(b'/bin/echo leak')\n"),
        trace_path=trace_path,
        index_path=index_path,
    )
    assert result.exit_code != 0
