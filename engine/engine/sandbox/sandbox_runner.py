from __future__ import annotations

import tempfile
from pathlib import Path

from engine.sandbox.platform_commands import (
    build_linux_bubblewrap_command,
    build_macos_sandbox_exec_command,
    render_macos_profile,
)
from engine.sandbox.sandbox_availability import SandboxBackend, SandboxRuntime
from engine.sandbox.sandbox_bootstrap import render_bootstrap_script
from engine.sandbox.sandbox_config import CodeExecutionResult, SandboxConfig
from engine.sandbox.sandbox_paths import (
    SANDBOX_BOOTSTRAP_FILENAME,
    SANDBOX_PROFILE_FILENAME,
    SANDBOX_TMP_DIRNAME,
)
from engine.sandbox.sandbox_policy import compose_policy
from engine.sandbox.sandbox_results import run_process_capped


class SandboxRunner:
    """Top-level sandbox entrypoint: picks the platform backend and orchestrates one run.

    Per call to ``run_python``: creates a temp work dir, renders the bootstrap script
    that preloads TraceStore, composes the policy from the runtime mount manifest,
    hands argv to bubblewrap (Linux) or sandbox-exec (macOS), and captures capped
    output via ``run_process_capped``.
    """

    def __init__(self, *, sandbox: SandboxRuntime) -> None:
        self._sandbox = sandbox

    async def run_python(
        self,
        *,
        code: str,
        trace_path: Path,
        index_path: Path,
        config: SandboxConfig,
    ) -> CodeExecutionResult:
        """Run user-supplied Python in the sandbox; returns a typed result regardless of pass/fail/timeout."""
        backend = self._sandbox.backend
        executable = self._sandbox.executable

        with tempfile.TemporaryDirectory(prefix="halo-sbx-") as tmp:
            work_dir = Path(tmp)
            (work_dir / SANDBOX_TMP_DIRNAME).mkdir()
            script = work_dir / SANDBOX_BOOTSTRAP_FILENAME

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
                runtime_mounts=self._sandbox.runtime_mounts,
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
                profile_path = work_dir / SANDBOX_PROFILE_FILENAME
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
