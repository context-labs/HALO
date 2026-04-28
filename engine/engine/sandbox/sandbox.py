from __future__ import annotations

from pathlib import Path

from engine.sandbox.bootstrap import render_bootstrap_script
from engine.sandbox.log import log_unavailable
from engine.sandbox.models import CodeExecutionResult, SandboxConfig
from engine.sandbox.pyodide_client import (
    PyodideClient,
    PyodideError,
    _ExecutionOutcome,
    _RunnerSession,
)

# Where the trace, index, and trace compat shim live inside the Pyodide FS.
# Hardcoded so the bootstrap script and Sandbox stay in lock-step without
# leaking host paths through the WASM filesystem.
_TRACE_VIRTUAL_PATH = "/input/traces.jsonl"
_INDEX_VIRTUAL_PATH = "/input/index.jsonl"
_TRACE_COMPAT_VIRTUAL_PATH = "/halo/pyodide_trace_compat.py"

_TRACE_COMPAT_HOST_PATH = (Path(__file__).parent / "pyodide_trace_compat.py").resolve()


class Sandbox:
    """Single front door for running Python under the Deno+Pyodide WASM sandbox.

    A ``Sandbox`` always represents a working backend: ``resolve_sandbox()``
    is the only blessed factory and returns ``None`` when the host cannot
    provide one (e.g., Deno not installed or wheels can't be pre-cached).
    Holds the resolved client and the run config; each ``run_python`` call
    spawns a fresh subprocess so WASM filesystem state does not leak between
    runs.
    """

    def __init__(self, *, client: PyodideClient, config: SandboxConfig) -> None:
        self._client = client
        self._config = config

    @property
    def client(self) -> PyodideClient:
        """Return the underlying Pyodide client (read-only test seam)."""
        return self._client

    async def run_python(
        self,
        *,
        code: str,
        trace_path: Path,
        index_path: Path,
    ) -> CodeExecutionResult:
        """Run ``code`` in the WASM sandbox; returns a typed result regardless of pass/fail/timeout.

        Mounting policy:
          - The runner's runner.js + the Deno cache are read-only (covered
            by the client's ``--allow-read`` set).
          - The host trace + index are added to ``--allow-read`` for this
            invocation only, so Deno can read them once to copy bytes into
            Pyodide's virtual FS.
          - Inside Pyodide, files are visible at fixed virtual paths so the
            bootstrap script can find them without leaking host paths.
        """
        argv = self._client.build_argv(
            extra_read_paths=[trace_path.resolve(), index_path.resolve()],
        )

        bootstrap_code = render_bootstrap_script(
            trace_virtual_path=_TRACE_VIRTUAL_PATH,
            index_virtual_path=_INDEX_VIRTUAL_PATH,
        )

        compat_text = _TRACE_COMPAT_HOST_PATH.read_text()

        session = _RunnerSession(
            argv=argv,
            timeout_seconds=self._config.timeout_seconds,
            max_stdout=self._config.maximum_stdout_bytes,
            max_stderr=self._config.maximum_stderr_bytes,
        )
        try:
            outcome = await session.run(
                mounts=[
                    (trace_path.resolve(), _TRACE_VIRTUAL_PATH),
                    (index_path.resolve(), _INDEX_VIRTUAL_PATH),
                ],
                injects=[(_TRACE_COMPAT_VIRTUAL_PATH, compat_text)],
                bootstrap_code=bootstrap_code,
                user_code=code,
            )
        except PyodideError as exc:
            return CodeExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"sandbox runner failure: {exc}",
                timed_out=False,
            )

        return _outcome_to_result(outcome)


def resolve_sandbox(*, config: SandboxConfig) -> Sandbox | None:
    """Probe the host once; return a ready ``Sandbox`` or ``None``.

    The Pyodide client logs its own unavailability warning before returning
    ``None``; this function just propagates that ``None`` back to the caller
    without inspecting why.
    """
    client = PyodideClient.resolve()
    if client is None:
        return None
    if not _TRACE_COMPAT_HOST_PATH.is_file():
        # Should never happen: the file ships with the package. Log and bail
        # so we don't silently produce a Sandbox that crashes on first use.
        log_unavailable(
            diagnostic=f"trace compat shim missing at {_TRACE_COMPAT_HOST_PATH}",
            remediation="Reinstall halo-engine; the package is missing required data files.",
        )
        return None
    return Sandbox(client=client, config=config)


def _outcome_to_result(outcome: _ExecutionOutcome) -> CodeExecutionResult:
    """Translate the runner's outcome (incl. captured stderr noise) into the public result model."""
    stderr = outcome.stderr
    if outcome.stderr_extra:
        extra = outcome.stderr_extra.decode("utf-8", errors="replace").strip()
        if extra:
            sep = "\n" if stderr else ""
            stderr = stderr + sep + f"[deno stderr] {extra}"
    return CodeExecutionResult(
        exit_code=outcome.exit_code,
        stdout=outcome.stdout,
        stderr=stderr,
        timed_out=outcome.timed_out,
    )


__all__ = [
    "Sandbox",
    "SandboxConfig",
    "resolve_sandbox",
]
