import os
import shutil
import subprocess
import sys

from appworld.common.constants import DEFAULT_REMOTE_APIS_URL, DEFAULT_REMOTE_UIS_PORT
from appworld.common.path_store import path_store
from appworld.common.system import maybe_kill_processes


def _run_single_worker(env: dict[str, str]) -> None:
    frontend_directory = os.path.join(
        path_store.root, "appworld-frontend"
    )  # TODO: move this properly + into path_store.
    subprocess.Popen(
        ["node", "build/index.js"],
        cwd=frontend_directory,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env or os.environ,
    )


def _run_multiple_workers(env: dict[str, str], pm2_binary: str, num_workers: int = 1) -> None:
    frontend_directory = os.path.join(
        path_store.root, "appworld-frontend"
    )  # TODO: move this properly + into path_store.
    subprocess.Popen(
        [
            pm2_binary,
            "start",
            "build/index.js",
            "--name",
            "appworld",
            "-i",
            str(num_workers),
        ],
        cwd=frontend_directory,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env or os.environ,
    )


def run(
    remote_apis_url: str = DEFAULT_REMOTE_APIS_URL,
    port: int = DEFAULT_REMOTE_UIS_PORT,
    num_workers: int = 1,
) -> None:
    """Start AppWorld UI Server."""
    env = {
        **os.environ,
        "API_URL": remote_apis_url,
        "ORIGIN": f"http://127.0.0.1:{port}",
        "HOST": "0.0.0.0",
        "PORT": str(port),
    }
    pm2_binary = shutil.which("pm2") or shutil.which("pm2-runtime")
    if not pm2_binary:
        raise RuntimeError("Neither pm2 nor pm2-runtime was found in PATH.")
    subprocess.run([pm2_binary, "delete", "appworld"], stdout=sys.stdout, stderr=sys.stderr)
    maybe_kill_processes("port", [port])
    if num_workers > 1:
        _run_multiple_workers(env, pm2_binary, num_workers)
    else:
        _run_single_worker(env)
