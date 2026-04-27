from __future__ import annotations

import tempfile
from pathlib import Path

from engine.sandbox.platform_commands import (
    build_linux_bubblewrap_command,
    build_macos_sandbox_exec_command,
    render_macos_profile,
)
from engine.sandbox.runtime_mounts import PythonRuntimeMounts
from engine.sandbox.sandbox_availability import SandboxBackend, SandboxStatus
from engine.sandbox.sandbox_bootstrap import render_bootstrap_script
from engine.sandbox.sandbox_config import CodeExecutionResult, SandboxConfig
from engine.sandbox.sandbox_policy import compose_policy
from engine.sandbox.sandbox_results import run_process_capped


class SandboxRunner:
    """Top-level sandbox entrypoint: picks the platform backend and orchestrates one run.

    Per call to ``run_python``: creates a temp work dir, renders the bootstrap script
    that preloads TraceStore, composes the policy from the runtime mount manifest,
    hands argv to bubblewrap (Linux) or sandbox-exec (macOS), and captures capped
    output via ``run_process_capped``.
    """

    def __init__(
        self,
        *,
        sandbox_status: SandboxStatus,
        runtime_mounts: PythonRuntimeMounts,
    ) -> None:
        if not sandbox_status.available or sandbox_status.executable is None:
            raise RuntimeError(
                "SandboxRunner constructed with unavailable SandboxStatus; "
                "callers must gate on status.available."
            )
        self._sandbox_status = sandbox_status
        self._runtime_mounts = runtime_mounts

    async def run_python(
        self,
        *,
        code: str,
        trace_path: Path,
        index_path: Path,
        config: SandboxConfig,
    ) -> CodeExecutionResult:
        """Run user-supplied Python in the sandbox; returns a typed result regardless of pass/fail/timeout."""
        backend = self._sandbox_status.backend
        executable = self._sandbox_status.executable
        if backend is None or executable is None:
            raise RuntimeError("SandboxRunner has no resolved backend; this is a bug.")

        with tempfile.TemporaryDirectory(prefix="halo-sbx-") as tmp:
            work_dir = Path(tmp)
            (work_dir / "tmp").mkdir()
            script = work_dir / "bootstrap.py"

            script.write_text(
                render_bootstrap_script(
                    user_code=code,
                    trace_path=str(trace_path),
                    index_path=str(index_path),
                )
            )

            policy = compose_policy(
                trace_path=trace_path,
                index_path=index_path,
                runtime_mounts=self._runtime_mounts,
                work_dir=work_dir,
                sandbox_config=config,
            )

            if backend in (SandboxBackend.LINUX_BWRAP_SYSTEM, SandboxBackend.LINUX_BWRAP_PACKAGED):
                argv = build_linux_bubblewrap_command(
                    bwrap_executable=executable,
                    policy=policy,
                    script_path=script,
                )
            elif backend == SandboxBackend.MACOS_SANDBOX_EXEC:
                profile_path = work_dir / "profile.sb"
                profile_path.write_text(render_macos_profile(policy=policy))
                argv = build_macos_sandbox_exec_command(
                    sandbox_exec_executable=executable,
                    policy=policy,
                    script_path=script,
                    profile_path=profile_path,
                )
            else:
                raise RuntimeError(f"unhandled sandbox backend: {backend}")

            return await run_process_capped(argv=argv, config=config)
