from __future__ import annotations

import platform
import tempfile
from pathlib import Path

from engine.sandbox.platform_commands import (
    build_linux_bubblewrap_command,
    build_macos_sandbox_exec_command,
    render_macos_profile,
)
from engine.sandbox.sandbox_bootstrap import render_bootstrap_script
from engine.sandbox.sandbox_config import CodeExecutionResult, SandboxConfig
from engine.sandbox.sandbox_policy import compose_policy
from engine.sandbox.sandbox_results import run_process_capped


class SandboxRunner:
    """Top-level sandbox entrypoint: picks the platform backend and orchestrates one run.

    Per call to ``run_python``: creates a temp work dir, renders the bootstrap script
    that preloads TraceStore, composes the policy, hands argv to bubblewrap (Linux)
    or sandbox-exec (macOS), and captures capped output via ``run_process_capped``.
    """

    def __init__(self, sandbox_venv: Path) -> None:
        self._sandbox_venv = sandbox_venv

    async def run_python(
        self,
        *,
        code: str,
        trace_path: Path,
        index_path: Path,
        config: SandboxConfig,
    ) -> CodeExecutionResult:
        """Run user-supplied Python in the sandbox; returns a typed result regardless of pass/fail/timeout."""
        system = platform.system()

        with tempfile.TemporaryDirectory(prefix="halo-sbx-") as tmp:
            work_dir = Path(tmp)
            (work_dir / "tmp").mkdir()
            script = work_dir / "bootstrap.py"

            script_body = render_bootstrap_script(
                user_code=code,
                trace_mount_path="/mnt/trace/traces.jsonl"
                if system == "Linux"
                else str(trace_path),
                index_mount_path="/mnt/trace/traces.jsonl.engine-index.jsonl"
                if system == "Linux"
                else str(index_path),
            )
            script.write_text(script_body)

            policy = compose_policy(
                trace_path=trace_path,
                index_path=index_path,
                sandbox_venv=self._sandbox_venv,
                work_dir=work_dir,
                sandbox_config=config,
            )

            if system == "Linux":
                in_sandbox_script = Path("/workspace/bootstrap.py")
                argv = build_linux_bubblewrap_command(policy=policy, script_path=in_sandbox_script)
            elif system == "Darwin":
                profile_path = work_dir / "profile.sb"
                profile_path.write_text(render_macos_profile(policy=policy))
                argv = build_macos_sandbox_exec_command(
                    policy=policy, script_path=script, profile_path=profile_path
                )
            else:
                raise RuntimeError(f"unsupported platform {system}")

            return await run_process_capped(argv=argv, config=config)
