from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from engine.sandbox import sandbox as sandbox_module
from engine.sandbox.models import CodeExecutionResult
from engine.sandbox.sandbox import Sandbox


def _fake_sandbox(tmp_path: Path) -> Sandbox:
    """Build a ``Sandbox`` with stub paths for argv-shape tests.

    Bypasses the discovery path: tests at this level care about
    ``run_python`` plumbing, not Deno detection. The actual subprocess
    is replaced by stubbing ``_RunnerSession`` methods in the test bodies.
    """
    deno = tmp_path / "deno"
    deno.write_text("")
    runner = tmp_path / "runner.js"
    runner.write_text("// stub")
    runtime = tmp_path / "pyodide_runtime.py"
    runtime.write_text("# stub")
    engine_init = tmp_path / "engine_init.py"
    engine_init.write_text("# stub")
    traces_pkg = tmp_path / "traces"
    traces_pkg.mkdir()
    deno_dir = tmp_path / "deno-cache"
    deno_dir.mkdir()
    return Sandbox(
        deno_executable=deno,
        runner_path=runner,
        runtime_path=runtime,
        engine_init_path=engine_init,
        traces_pkg_dir=traces_pkg,
        deno_dir=deno_dir,
    )


# -- Sandbox.get ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_sandbox_cache() -> Iterator[None]:
    """``Sandbox.get`` memoizes successful resolves on ``Sandbox._cached``.

    A test that runs after an integration suite would otherwise hit the
    cached real Sandbox and bypass the monkeypatched discovery helpers.
    Resetting before every unit test keeps each test hermetic without
    needing every test to do the reset itself.
    """
    Sandbox._cached = None
    yield
    Sandbox._cached = None


def test_get_returns_none_when_deno_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Deno on PATH or via the PyPI dep → ``Sandbox.get`` returns ``None``.

    Discovery emits its own remediation warning before bailing out;
    ``Sandbox.get`` just propagates the ``None`` so callers can
    silently drop ``run_code`` from the agent surface.
    """
    monkeypatch.setattr(sandbox_module, "_locate_deno", lambda: None)
    assert Sandbox.get() is None


def test_get_returns_none_when_required_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If runner.js or its sibling .py files vanish, get must refuse to fabricate a Sandbox."""
    deno = tmp_path / "deno"
    deno.write_text("")
    monkeypatch.setattr(sandbox_module, "_locate_deno", lambda: deno)
    # Point ``__file__``-derived parent at an empty dir so the required
    # sibling files don't exist relative to it.
    monkeypatch.setattr(sandbox_module, "__file__", str(tmp_path / "sandbox.py"))
    assert Sandbox.get() is None


def test_get_caches_successful_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful resolves must be memoized — discovery is subprocess-heavy.

    Regression: a long-lived process (e.g., a server handling many
    ``stream_engine_async`` requests) would otherwise pay the
    ``deno info`` subprocess + file-existence checks on every call.
    The cache should also produce **the same** instance — Sandbox is a
    frozen value object, so identity is meaningful.
    """
    deno = tmp_path / "deno"
    deno.write_text("")
    deno_dir = tmp_path / "deno-cache"
    deno_dir.mkdir()
    pyodide_dir = tmp_path / "pyodide"
    pyodide_dir.mkdir()

    counts = {"locate": 0, "deno_info": 0, "npm_cache": 0, "wheels": 0}

    def _locate():
        counts["locate"] += 1
        return deno

    def _query(_deno):
        counts["deno_info"] += 1
        return deno_dir

    def _npm_cache(_deno, _dir):
        counts["npm_cache"] += 1
        return pyodide_dir

    def _wheels(_dir):
        counts["wheels"] += 1

    monkeypatch.setattr(sandbox_module, "_locate_deno", _locate)
    monkeypatch.setattr(sandbox_module, "_query_deno_dir", _query)
    monkeypatch.setattr(sandbox_module, "_ensure_npm_cache", _npm_cache)
    monkeypatch.setattr(sandbox_module, "_ensure_wheels", _wheels)

    first = Sandbox.get()
    second = Sandbox.get()
    third = Sandbox.get()

    assert first is not None
    assert second is first, "second get must return the cached instance"
    assert third is first, "third get must return the cached instance"
    assert counts == {"locate": 1, "deno_info": 1, "npm_cache": 1, "wheels": 1}, (
        f"discovery helpers must run exactly once across cached resolves; got {counts}"
    )


def test_get_does_not_cache_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed get must not poison the cache — a transient failure
    should not silently disable ``run_code`` for the rest of the process.
    """
    monkeypatch.setattr(sandbox_module, "_locate_deno", lambda: None)
    assert Sandbox.get() is None
    # Cache stays empty so the next attempt re-runs discovery.
    assert Sandbox._cached is None


# -- Sandbox.run_python: argv shape -------------------------------------------


def _install_session_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    captured_argv: dict[str, list[str]] | None = None,
    result: CodeExecutionResult | None = None,
) -> None:
    """Replace ``_RunnerSession`` with a no-op stub that records argv and returns ``result``.

    The stub bypasses the entire subprocess / JSON-RPC plumbing: ``start``
    no-ops, ``mount`` / ``bootstrap`` succeed, ``execute`` returns the
    canned ``result`` (or a default success). ``stop`` returns empty
    stderr.
    """
    canned = result or CodeExecutionResult(exit_code=0, stdout="ok", stderr="", timed_out=False)

    class _StubSession:
        def __init__(self, *, argv: list[str]) -> None:
            if captured_argv is not None:
                captured_argv["argv"] = list(argv)
            self._returncode: int | None = None

        @property
        def returncode(self) -> int | None:
            return self._returncode

        async def start(self) -> None: ...
        async def stop(self, *, hard: bool) -> bytes:
            return b""

        async def mount(self, _host: Path, _virtual: str) -> None: ...
        async def bootstrap(self, _t: str, _i: str) -> CodeExecutionResult:
            return CodeExecutionResult(exit_code=0, stdout="", stderr="", timed_out=False)

        async def execute(self, _code: str) -> CodeExecutionResult:
            return canned

    monkeypatch.setattr(sandbox_module, "_RunnerSession", _StubSession)


@pytest.mark.asyncio
async def test_run_python_includes_trace_and_index_in_allow_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_python`` adds the trace + index files to ``--allow-read``.

    Permissions are scoped per-call: the runner script and Deno cache
    are constants, but the trace/index files are caller-supplied. Both
    must appear in the resolved Deno argv so the runner can read them
    once at mount time.
    """
    sandbox = _fake_sandbox(tmp_path)
    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    captured: dict[str, list[str]] = {}
    _install_session_stub(monkeypatch, captured_argv=captured)

    result = await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    assert result.exit_code == 0
    assert result.stdout == "ok"

    allow_read_arg = next(a for a in captured["argv"] if a.startswith("--allow-read="))
    allow = allow_read_arg.split("=", 1)[1].split(",")
    assert str(trace.resolve()) in allow
    assert str(index.resolve()) in allow


@pytest.mark.asyncio
async def test_run_python_does_not_pass_unsafe_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Deno argv must never carry ``--allow-net``, ``--allow-write``, ``--allow-env``, or ``--allow-run``.

    These flags would lift exactly the constraints HALO is enforcing
    (network, host writes, host env vars, subprocess spawn). A regression
    that adds any of them silently weakens the sandbox.
    """
    sandbox = _fake_sandbox(tmp_path)
    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    captured: dict[str, list[str]] = {}
    _install_session_stub(monkeypatch, captured_argv=captured)

    await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    forbidden = ("--allow-net", "--allow-write", "--allow-env", "--allow-run", "--allow-all")
    for flag in forbidden:
        assert not any(arg.startswith(flag) for arg in captured["argv"]), (
            f"sandbox argv must not contain {flag}: {captured['argv']}"
        )


@pytest.mark.asyncio
async def test_run_python_returns_sad_result_on_unexpected_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any non-``SandboxError``, non-``BaseException`` failure must degrade gracefully.

    ``run_code`` is a tool the agent calls — a hard crash from
    ``run_python`` (e.g., ``OSError`` on subprocess spawn, a JSON
    decode error in the driver, a surprise ``ValueError`` from a
    refactor bug) would propagate up through the SDK's tool dispatch
    and crash the agent loop mid-conversation. The right behavior is:
    log a warning so operators can investigate, and hand the agent a
    failed ``CodeExecutionResult`` it can recover from.

    ``BaseException`` (CancelledError / KeyboardInterrupt) must still
    propagate — that's user-initiated termination and shouldn't be
    swallowed.
    """
    sandbox = _fake_sandbox(tmp_path)
    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    class _ExplodingSession:
        def __init__(self, *, argv: list[str]) -> None: ...
        @property
        def returncode(self) -> int | None:
            return None

        async def start(self) -> None:
            raise OSError("simulated spawn failure")

        async def stop(self, *, hard: bool) -> bytes:
            return b""

        async def mount(self, _h: Path, _v: str) -> None: ...
        async def bootstrap(self, _t: str, _i: str) -> CodeExecutionResult:
            return CodeExecutionResult(exit_code=0, stdout="", stderr="", timed_out=False)

        async def execute(self, _c: str) -> CodeExecutionResult:
            return CodeExecutionResult(exit_code=0, stdout="", stderr="", timed_out=False)

    monkeypatch.setattr(sandbox_module, "_RunnerSession", _ExplodingSession)

    result = await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    assert result.exit_code == 1
    assert result.timed_out is False
    assert "OSError" in result.stderr or "simulated spawn failure" in result.stderr


@pytest.mark.asyncio
async def test_run_python_does_not_swallow_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``asyncio.CancelledError`` must propagate, not become a sad result.

    Cancellation is BaseException, not Exception — the broad ``except
    Exception`` graceful path doesn't catch it, and the
    ``except BaseException: raise`` guard explicitly preserves it. A
    regression that swapped these would mean asyncio cancel signals
    silently turn into a fake sandbox failure, which would deadlock
    callers waiting on a cancellation that never propagates.
    """
    sandbox = _fake_sandbox(tmp_path)
    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    class _CancellingSession:
        def __init__(self, *, argv: list[str]) -> None: ...
        @property
        def returncode(self) -> int | None:
            return None

        async def start(self) -> None:
            raise asyncio.CancelledError()

        async def stop(self, *, hard: bool) -> bytes:
            return b""

        async def mount(self, _h: Path, _v: str) -> None: ...
        async def bootstrap(self, _t: str, _i: str) -> CodeExecutionResult:
            return CodeExecutionResult(exit_code=0, stdout="", stderr="", timed_out=False)

        async def execute(self, _c: str) -> CodeExecutionResult:
            return CodeExecutionResult(exit_code=0, stdout="", stderr="", timed_out=False)

    monkeypatch.setattr(sandbox_module, "_RunnerSession", _CancellingSession)

    with pytest.raises(asyncio.CancelledError):
        await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)


@pytest.mark.asyncio
async def test_run_python_returns_runner_result_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runner result with non-zero exit must round-trip unchanged.

    ``timed_out=True`` is reserved for the wall-clock timeout path; the
    happy path (execute returns) always sets ``timed_out=False``.
    """
    sandbox = _fake_sandbox(tmp_path)
    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    expected = CodeExecutionResult(
        exit_code=137,
        stdout="partial",
        stderr="boom",
        timed_out=False,
    )
    _install_session_stub(monkeypatch, result=expected)

    result = await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    assert result == expected


# -- _truncate_to_bytes: byte-aware caps --------------------------------------


def test_truncate_below_cap_passthrough() -> None:
    """A string under the cap must round-trip unchanged."""
    assert sandbox_module._truncate_to_bytes("hello", 100) == "hello"


def test_truncate_ascii_above_cap_emits_marker() -> None:
    """ASCII content above the cap is replaced past the head with the truncation marker."""
    text = "x" * 200
    result = sandbox_module._truncate_to_bytes(text, 100)
    assert len(result.encode("utf-8")) <= 100
    assert sandbox_module._TRUNCATION_MARKER in result
    assert result.startswith("x")


def test_truncate_multibyte_respects_byte_cap_not_char_cap() -> None:
    """Cap is named ``_MAX_STDOUT_BYTES`` so it must be enforced in bytes.

    Regression: the old ``_truncate`` sliced on ``len(text)`` (character
    count). 200 emoji are 200 chars but 800 UTF-8 bytes; under the old
    code a 100-byte cap would let through ~800 bytes — silently breaking
    the byte-named contract. The fix encodes first, slices on bytes,
    then decodes with ``errors="ignore"`` so a partial UTF-8 sequence at
    the cut never becomes a U+FFFD replacement character.
    """
    # 4-byte emoji × 200 = 800 UTF-8 bytes.
    text = "🔥" * 200
    cap_bytes = 100
    result = sandbox_module._truncate_to_bytes(text, cap_bytes)
    assert len(result.encode("utf-8")) <= cap_bytes, (
        f"truncated output is {len(result.encode('utf-8'))} bytes, exceeds cap {cap_bytes}"
    )
    assert sandbox_module._TRUNCATION_MARKER in result
    # No replacement character: the partial 🔥 bytes at the cut were
    # dropped cleanly by ``errors='ignore'``.
    assert "�" not in result


def test_truncate_drops_partial_utf8_at_cut() -> None:
    """When the byte cap lands mid-character, the partial sequence is dropped, not replaced."""
    # ``é`` is 2 UTF-8 bytes. With a head budget of 5 bytes (= 2 full
    # ``é`` + 1 partial byte), the third ``é`` is half-cut and must be
    # dropped via ``errors="ignore"``. Input length must exceed the cap
    # to actually trigger truncation, hence ``é * 100`` not ``ééé``.
    text = "é" * 100
    cap_bytes = len(sandbox_module._TRUNCATION_MARKER) + 5
    result = sandbox_module._truncate_to_bytes(text, cap_bytes)
    assert len(result.encode("utf-8")) <= cap_bytes
    assert "�" not in result
    assert result.endswith(sandbox_module._TRUNCATION_MARKER)
    # Head holds exactly the two full ``é`` that fit in 5 bytes.
    assert result.startswith("éé")
    assert not result.startswith("ééé")


# -- _RunnerSession lifecycle: every protocol return path must shut down the runner ---


# Sentinel ``_RpcResult`` factories — short builders for the test responses below.
def _rpc_ok(result: dict) -> "sandbox_module._RpcResult":
    return sandbox_module._RpcResult(result=result, error=None)


def _rpc_error(message: str, code: int = -32008) -> "sandbox_module._RpcResult":
    return sandbox_module._RpcResult(result=None, error={"code": code, "message": message})


class _SessionCapture:
    """Records every ``_request`` method and every ``_write_message`` payload."""

    def __init__(self) -> None:
        self.phases: list[str] = []
        self.sent: list[dict] = []

    def shutdown_was_sent(self) -> bool:
        return any(p.get("method") == "shutdown" for p in self.sent)


def _install_session_internals_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rpc_responder,
    send_failure_for: str | None = None,
) -> _SessionCapture:
    """Wire stubs around ``_RunnerSession``'s internals.

    Patches the real ``_RunnerSession`` methods so the test exercises
    the actual ``Sandbox.run_python`` → ``_run_protocol`` flow but
    without spawning a real subprocess.

    ``rpc_responder(method)`` returns the ``_RpcResult`` to use for that
    JSON-RPC method call. Reaching an un-handled method is a test bug
    (the responder should ``raise AssertionError`` to surface that).

    ``send_failure_for`` (optional): the ``method`` whose ``_write_message``
    call should raise ``BrokenPipeError`` — used to simulate a runner that
    exits between responding and reading the next frame.
    """
    capture = _SessionCapture()

    async def _stub_request(self, method, _params):
        capture.phases.append(method)
        return rpc_responder(method)

    async def _stub_write_message(self, payload):
        capture.sent.append(payload)
        if send_failure_for is not None and payload.get("method") == send_failure_for:
            raise BrokenPipeError(f"runner already exited (simulated for {send_failure_for})")

    async def _stub_await_ready(self):
        return None

    async def _stub_drain_capped(_stream, _cap):
        return b""

    class _StubProc:
        stdin = object()
        stdout = object()
        stderr = None
        # Non-zero, non-existent pid so the cleanup path's
        # ``_kill_process_group`` raises ``ProcessLookupError`` (which it
        # silently ignores) instead of nuking the test runner's own pgid
        # — ``os.getpgid(0)`` returns the *caller's* group, so a stub of
        # 0 would have ``killpg`` sending SIGKILL to pytest itself.
        pid = 2**31 - 1
        returncode = 0

        async def wait(self):
            return 0

    async def _stub_create_subprocess(*_args, **_kwargs):
        return _StubProc()

    monkeypatch.setattr(sandbox_module._RunnerSession, "_request", _stub_request)
    monkeypatch.setattr(sandbox_module._RunnerSession, "_write_message", _stub_write_message)
    monkeypatch.setattr(sandbox_module._RunnerSession, "_await_ready", _stub_await_ready)
    monkeypatch.setattr(sandbox_module, "_drain_capped", _stub_drain_capped)
    monkeypatch.setattr(sandbox_module.asyncio, "create_subprocess_exec", _stub_create_subprocess)
    return capture


def _trace_and_index(tmp_path: Path) -> tuple[Path, Path]:
    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")
    return trace, index


@pytest.mark.asyncio
async def test_run_python_sends_shutdown_after_mount_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``mount_file`` JSON-RPC error must surface AND the runner must be shut down.

    Regression for two bugs:

    1. The driver previously discarded the result of every ``mount_file``
       call. A failed mount (host file missing, Deno --allow-read denial)
       let bootstrap run anyway and crash with a confusing
       ``FileNotFoundError`` deep inside Pyodide. The fix raises a
       ``SandboxError`` from ``_RunnerSession.mount`` and ``run_python``
       converts it into a clear stderr message.

    2. The early-return path skipped the shutdown send, leaving the
       Deno subprocess alive and blocking ``stderr_task``'s drain (which
       waits for the subprocess to close stderr on exit). The fix
       puts shutdown in ``_RunnerSession.stop`` so every error path
       runs cleanup uniformly via the outer try/except in ``run_python``.
    """
    sandbox = _fake_sandbox(tmp_path)
    trace, index = _trace_and_index(tmp_path)

    def _responder(method: str):
        if method == "mount_file":
            return _rpc_error("Failed to mount file: missing")
        raise AssertionError(f"protocol must return after mount_file error; reached {method!r}")

    capture = _install_session_internals_stubs(monkeypatch, rpc_responder=_responder)

    result = await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    assert result.exit_code == 1
    assert result.timed_out is False
    assert "mount_file" in result.stderr
    assert "Failed to mount file" in result.stderr
    assert capture.phases == ["mount_file"], (
        f"only mount_file should have been attempted; got {capture.phases}"
    )
    assert capture.shutdown_was_sent(), (
        "early return after mount_file error must still send shutdown — "
        "without it the subprocess orphans and stderr_task hangs"
    )


@pytest.mark.asyncio
async def test_run_python_sends_shutdown_after_bootstrap_rpc_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bootstrap JSON-RPC error must surface AND the runner must be shut down.

    Bootstrap RPC errors come from runner.js itself (e.g., the runner
    rejected the request before halo_bootstrap ran). Same lifecycle
    requirement as mount errors: the runner is alive and waiting; we
    must shut it down.
    """
    sandbox = _fake_sandbox(tmp_path)
    trace, index = _trace_and_index(tmp_path)

    def _responder(method: str):
        if method == "mount_file":
            return _rpc_ok({"mounted": "/x"})
        if method == "bootstrap":
            return _rpc_error("runner rejected bootstrap")
        raise AssertionError(f"protocol must return after bootstrap RPC error; reached {method!r}")

    capture = _install_session_internals_stubs(monkeypatch, rpc_responder=_responder)

    result = await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    assert result.exit_code == 1
    assert "bootstrap" in result.stderr
    assert "runner rejected bootstrap" in result.stderr
    # Two mounts then bootstrap; execute never reached.
    assert capture.phases == ["mount_file", "mount_file", "bootstrap"]
    assert capture.shutdown_was_sent(), (
        "early return after bootstrap RPC error must still send shutdown"
    )


@pytest.mark.asyncio
async def test_run_python_sends_shutdown_after_bootstrap_python_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bootstrap that returned ``exit_code != 0`` must shut the runner down.

    This is the in-Pyodide path: ``halo_bootstrap`` caught a Python
    exception (e.g., a malformed index, missing wheel), captured the
    traceback, and returned a non-zero exit_code. The runner is healthy
    — Python caught the error — so shutdown is the correct cleanup.
    """
    sandbox = _fake_sandbox(tmp_path)
    trace, index = _trace_and_index(tmp_path)

    def _responder(method: str):
        if method == "mount_file":
            return _rpc_ok({"mounted": "/x"})
        if method == "bootstrap":
            return _rpc_ok(
                {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "Traceback (most recent call last):\n  ...\nValueError: bad index\n",
                }
            )
        raise AssertionError(
            f"protocol must return after bootstrap python error; reached {method!r}"
        )

    capture = _install_session_internals_stubs(monkeypatch, rpc_responder=_responder)

    result = await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    assert result.exit_code == 1
    assert "ValueError: bad index" in result.stderr
    assert capture.phases == ["mount_file", "mount_file", "bootstrap"]
    assert capture.shutdown_was_sent(), (
        "early return after bootstrap exit_code != 0 must still send shutdown"
    )


@pytest.mark.asyncio
async def test_run_python_preserves_result_when_shutdown_send_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``BrokenPipeError`` on the shutdown send must not discard the result.

    Regression: the post-execute shutdown send is best-effort cleanup —
    the result is already in hand. If the runner exits between
    responding and reading our shutdown frame, ``_write_message`` raises
    ``BrokenPipeError`` (subclass of ``OSError``). Previously that
    propagated through ``run_python``'s outer ``except BaseException`` and
    out, which only catches ``SandboxError`` — so the caller saw the
    ``OSError`` instead of the valid result. The fix swallows
    ``BrokenPipeError``/``ConnectionError`` inside ``_RunnerSession.stop``.
    """
    sandbox = _fake_sandbox(tmp_path)
    trace, index = _trace_and_index(tmp_path)

    def _responder(method: str):
        if method == "mount_file":
            return _rpc_ok({"mounted": "/x"})
        if method == "bootstrap":
            return _rpc_ok({"exit_code": 0, "stdout": "", "stderr": ""})
        if method == "execute":
            return _rpc_ok({"exit_code": 0, "stdout": "real result\n", "stderr": ""})
        raise AssertionError(f"unexpected method {method!r}")

    capture = _install_session_internals_stubs(
        monkeypatch, rpc_responder=_responder, send_failure_for="shutdown"
    )

    result = await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    assert result.exit_code == 0
    assert result.stdout == "real result\n"
    assert result.timed_out is False
    # The driver attempted shutdown — we just swallowed its failure.
    assert capture.shutdown_was_sent()


@pytest.mark.asyncio
async def test_run_python_does_not_hang_when_protocol_returns_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end behavior check: no early-return path may leave the session hanging.

    The other tests in this section verify the structural contract
    (shutdown was sent). This test exercises the *symptom* the contract
    prevents: under the bug, ``stderr_task`` (which drains until the
    subprocess closes stderr on exit) and ``proc.wait`` would block
    forever because the runner was alive waiting for stdin. Stub
    semantics:

      * ``_drain_capped`` only resolves once the runner is "dead".
      * ``proc.wait`` only resolves once the runner is "dead".
      * ``_write_message(method=shutdown)`` flips the runner to "dead".
      * ``_kill_process_group`` also flips it to "dead" (the bug-free
        timeout/exception paths still cover us — but here we assert the
        *normal* early-return path does it on its own).

    Wrapped in a 2-second deadline; the bug presents as ``TimeoutError``.
    """
    sandbox = _fake_sandbox(tmp_path)
    trace, index = _trace_and_index(tmp_path)

    proc_dead = asyncio.Event()

    def _responder(method: str):
        if method == "mount_file":
            return _rpc_error("Failed to mount file: missing")
        raise AssertionError(f"reached {method!r}")

    async def _stub_request(self, method, _params):
        return _responder(method)

    async def _stub_write_message(self, payload):
        if payload.get("method") == "shutdown":
            proc_dead.set()

    async def _stub_await_ready(self):
        return None

    async def _stub_drain_capped(_stream, _cap):
        await proc_dead.wait()
        return b""

    def _stub_kill(_pid):
        # If the bug-free shutdown path didn't fire, the
        # outer-exception / timeout path would still set this — but for
        # the normal early-return path we never want to rely on it.
        proc_dead.set()

    class _StubProc:
        stdin = object()
        stdout = object()
        stderr = None
        pid = 2**31 - 1
        returncode = 0

        async def wait(self):
            await proc_dead.wait()
            return 0

    async def _stub_create_subprocess(*_args, **_kwargs):
        return _StubProc()

    monkeypatch.setattr(sandbox_module._RunnerSession, "_request", _stub_request)
    monkeypatch.setattr(sandbox_module._RunnerSession, "_write_message", _stub_write_message)
    monkeypatch.setattr(sandbox_module._RunnerSession, "_await_ready", _stub_await_ready)
    monkeypatch.setattr(sandbox_module, "_drain_capped", _stub_drain_capped)
    monkeypatch.setattr(sandbox_module, "_kill_process_group", _stub_kill)
    monkeypatch.setattr(sandbox_module.asyncio, "create_subprocess_exec", _stub_create_subprocess)

    result = await asyncio.wait_for(
        sandbox.run_python(code="x=1", trace_path=trace, index_path=index),
        timeout=2.0,
    )
    assert result.exit_code == 1
    assert "mount_file" in result.stderr
    assert proc_dead.is_set()


# -- bootstrap + execute share output truncation ------------------------------


@pytest.mark.asyncio
async def test_bootstrap_traceback_is_byte_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bootstrap failure with a huge traceback must not bloat the agent context.

    Regression: bootstrap previously returned ``_result_from_rpc(...)``
    raw, while execute applied ``_truncate_to_bytes``. A recursive
    import error in ``halo_bootstrap`` (or any pyodide-side import
    blowup) could push tens of megabytes of traceback back through the
    JSON-RPC frame, which then flowed verbatim into the agent's prompt.
    Both phases share ``_result_from_rpc`` now, so the byte cap applies
    to either kind of failure.
    """
    sandbox = _fake_sandbox(tmp_path)
    trace, index = _trace_and_index(tmp_path)

    huge_traceback = "Traceback (most recent call last):\n" + ("  " + "x" * 200 + "\n") * 5000

    def _responder(method: str):
        if method == "mount_file":
            return _rpc_ok({"mounted": "/x"})
        if method == "bootstrap":
            return _rpc_ok({"exit_code": 1, "stdout": "", "stderr": huge_traceback})
        raise AssertionError(f"unexpected method {method!r}")

    _install_session_internals_stubs(monkeypatch, rpc_responder=_responder)

    result = await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    assert result.exit_code == 1
    assert len(result.stderr.encode("utf-8")) <= sandbox_module._MAX_STDERR_BYTES, (
        f"bootstrap stderr is {len(result.stderr.encode('utf-8'))} bytes, "
        f"exceeds cap {sandbox_module._MAX_STDERR_BYTES}"
    )
    assert sandbox_module._TRUNCATION_MARKER in result.stderr


# -- _RunnerSession._await_ready: tolerate unexpected JSON before sentinel ----


class _FakeStdout:
    """Async-readline shim returning pre-scripted byte lines, then EOF."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)
        self._idx = 0

    async def readline(self) -> bytes:
        if self._idx >= len(self._lines):
            return b""
        line = self._lines[self._idx]
        self._idx += 1
        return line


def _session_with_fake_stdout(lines: list[bytes]) -> sandbox_module._RunnerSession:
    """Build a ``_RunnerSession`` whose ``_proc.stdout`` is the fake reader.

    Bypasses ``start()`` (no subprocess) — we wire the proc attribute
    directly so we can drive ``_await_ready`` against scripted input.
    """
    session = sandbox_module._RunnerSession(argv=["fake"])

    class _StubProc:
        stdout = _FakeStdout(lines)
        stdin = None
        stderr = None
        pid = 2**31 - 1
        returncode = 0

        async def wait(self):
            return 0

    session._proc = _StubProc()  # type: ignore[assignment]
    return session


@pytest.mark.asyncio
async def test_await_ready_skips_unexpected_json_before_sentinel() -> None:
    """Unexpected JSON before the ready sentinel must be skipped, not fatal.

    Regression: ``_await_ready`` previously raised on any valid JSON
    line that wasn't an error and wasn't the ``id=0, result.ready=True``
    sentinel. Future Pyodide / Deno releases could legitimately log JSON
    diagnostics on stdout during boot; that should not kill the session.
    Behavior must mirror ``_read_until_id``, which already
    skip-and-retries on non-matching ids.
    """
    session = _session_with_fake_stdout(
        [
            b'{"some": "diagnostic"}\n',
            b'{"jsonrpc":"2.0","method":"unknown_event"}\n',
            b'{"jsonrpc":"2.0","id":0,"result":{"ready":true}}\n',
        ]
    )
    # Should return cleanly (no exception).
    await session._await_ready()


@pytest.mark.asyncio
async def test_await_ready_still_raises_on_explicit_startup_error() -> None:
    """An ``error`` field in JSON before the sentinel is still a fatal startup signal.

    Skipping unexpected JSON must not also swallow real error
    notifications — the runner emits these from its
    ``unhandledrejection`` handler, and they encode genuine failures.
    """
    session = _session_with_fake_stdout(
        [
            b'{"jsonrpc":"2.0","error":{"code":-32007,"message":"boom"}}\n',
        ]
    )
    with pytest.raises(sandbox_module.SandboxError, match="runner failed at startup"):
        await session._await_ready()


@pytest.mark.asyncio
async def test_await_ready_raises_on_premature_eof() -> None:
    """Empty stdout (process exited before signalling) is still fatal."""
    session = _session_with_fake_stdout([])
    with pytest.raises(sandbox_module.SandboxError, match="exited before signalling ready"):
        await session._await_ready()


# -- _resolve_required_wheels: lockfile-driven wheel discovery -----------------


def _write_minimal_lockfile(pyodide_dir: Path, packages: dict) -> None:
    pyodide_dir.mkdir(parents=True, exist_ok=True)
    (pyodide_dir / "pyodide-lock.json").write_text(json.dumps({"packages": packages}))


def test_resolve_required_wheels_walks_transitive_dependencies(tmp_path: Path) -> None:
    """The resolver must transitively follow ``depends`` and dedupe shared deps.

    Regression target: a hand-maintained list misses a transitive dep
    on Pyodide bumps. The lockfile walk should expand to the full
    closure even for diamond-shaped dep graphs (two packages share a
    child).
    """
    pyodide_dir = tmp_path / "pyodide"
    _write_minimal_lockfile(
        pyodide_dir,
        {
            "numpy": {"file_name": "numpy.whl", "depends": []},
            "pandas": {
                "file_name": "pandas.whl",
                "depends": ["numpy", "python-dateutil"],
            },
            "python-dateutil": {"file_name": "python_dateutil.whl", "depends": ["six"]},
            "six": {"file_name": "six.whl", "depends": []},
            "pydantic": {"file_name": "pydantic.whl", "depends": ["six"]},  # diamond on six
            # Unrelated package the resolver must NOT pull in:
            "scipy": {"file_name": "scipy.whl", "depends": ["numpy"]},
        },
    )
    wheels = set(sandbox_module._resolve_required_wheels(pyodide_dir))
    assert wheels == {
        "numpy.whl",
        "pandas.whl",
        "python_dateutil.whl",
        "six.whl",
        "pydantic.whl",
    }


def test_resolve_required_wheels_raises_when_lockfile_missing(tmp_path: Path) -> None:
    """A missing ``pyodide-lock.json`` must raise ``_ResolutionError``."""
    with pytest.raises(sandbox_module._ResolutionError, match="failed to read packages"):
        sandbox_module._resolve_required_wheels(tmp_path / "nonexistent-pyodide")


def test_resolve_required_wheels_normalizes_underscore_in_depends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``depends`` entry in underscore form must resolve to its dash-keyed entry.

    Regression: the real Pyodide 0.29.3 lockfile keys ``packages`` by
    the PEP 503-normalized form (``pydantic-core``) but pydantic's
    ``depends`` list contains the importable underscore form
    (``pydantic_core``). Without normalization the resolver fails with
    "no entry for required package 'pydantic_core'" even though the
    package is right there under a different spelling. The resolver
    normalizes both keys and lookups via PEP 503, so the dependency
    walk follows either spelling.
    """
    monkeypatch.setattr(sandbox_module, "_REQUIRED_PACKAGES", ("pydantic",))
    pyodide_dir = tmp_path / "pyodide"
    _write_minimal_lockfile(
        pyodide_dir,
        {
            "pydantic": {"file_name": "pydantic.whl", "depends": ["pydantic_core"]},
            "pydantic-core": {"file_name": "pydantic_core.whl", "depends": []},
        },
    )
    wheels = set(sandbox_module._resolve_required_wheels(pyodide_dir))
    assert wheels == {"pydantic.whl", "pydantic_core.whl"}


def test_resolve_required_wheels_raises_when_required_package_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A required-package name absent from the lockfile must raise.

    This is the ``Pyodide bumped, dropped a package we depend on`` case.
    Better to fail loudly at resolve time than to silently skip it and
    crash inside Pyodide on ``loadPackage``.
    """
    monkeypatch.setattr(sandbox_module, "_REQUIRED_PACKAGES", ("numpy", "missingpkg"))
    pyodide_dir = tmp_path / "pyodide"
    _write_minimal_lockfile(
        pyodide_dir,
        {"numpy": {"file_name": "numpy.whl", "depends": []}},
    )
    with pytest.raises(
        sandbox_module._ResolutionError, match="no entry for required package 'missingpkg'"
    ):
        sandbox_module._resolve_required_wheels(pyodide_dir)
