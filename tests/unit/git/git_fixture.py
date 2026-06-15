"""Shared deterministic git-repo builder for the git-tool tests.

Builds a tiny repo with three fixed commits over two files so log ordering,
date windowing, pickaxe, blame attribution, and read-at-ref all have stable,
hardcoded expectations. ``GIT_CONFIG_GLOBAL``/``GIT_CONFIG_SYSTEM`` are pinned
to ``os.devnull`` and author/committer identity + dates are fixed so the build
is independent of the host's git config.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from engine.git.git_repo import _REPO_REDIRECT_GIT_ENV

# Three commits, oldest first. Each entry is (commit subject, authored date,
# {path: full new contents}). Commit 1 seeds config.py + runner.py; commit 2
# introduces UNIQUE_TOKEN_PICKAXE in config.py only; commit 3 edits runner.py
# only — so path filters and pickaxe each match a known subset.
COMMIT_1_SUBJECT = "Initial commit"
COMMIT_2_SUBJECT = "Add unique token"
COMMIT_3_SUBJECT = "Double the retries"

PICKAXE_TOKEN = "UNIQUE_TOKEN_PICKAXE"
AUTHOR_NAME = "Ada Lovelace"

_CONFIG_V1 = "MAX_RETRIES = 3\nTIMEOUT_SECONDS = 30\n"
_CONFIG_V2 = f"MAX_RETRIES = 3\nTIMEOUT_SECONDS = 30\n{PICKAXE_TOKEN} = 1\n"
_RUNNER_V1 = "def run():\n    return MAX_RETRIES\n"
_RUNNER_V2 = "def run():\n    return MAX_RETRIES * 2\n"


def _git_env(date: str | None = None) -> dict[str, str]:
    # Strip the ambient repo-redirect vars (e.g. GIT_DIR set when these tests run
    # from a git pre-push hook) so the build targets the tmp repo, not HALO's.
    env = {k: v for k, v in os.environ.items() if k not in _REPO_REDIRECT_GIT_ENV}
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_AUTHOR_NAME": AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": "ada@example.com",
            "GIT_COMMITTER_NAME": AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": "ada@example.com",
        }
    )
    if date is not None:
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
    return env


def _git(root: Path, *args: str, date: str | None = None) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        env=_git_env(date),
    )


def build_git_repo(tmp_path: Path) -> Path:
    """Create a 3-commit git work tree under ``tmp_path`` and return its root."""
    root = tmp_path / "gitrepo"
    root.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(root)],
        check=True,
        capture_output=True,
        text=True,
        env=_git_env(),
    )

    (root / "config.py").write_text(_CONFIG_V1)
    (root / "runner.py").write_text(_RUNNER_V1)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", COMMIT_1_SUBJECT, date="2021-01-01T00:00:00")

    (root / "config.py").write_text(_CONFIG_V2)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", COMMIT_2_SUBJECT, date="2021-02-01T00:00:00")

    (root / "runner.py").write_text(_RUNNER_V2)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", COMMIT_3_SUBJECT, date="2021-03-01T00:00:00")

    return root


def build_empty_git_repo(tmp_path: Path) -> Path:
    """Create a git work tree with no commits (unborn HEAD) and return its root."""
    root = tmp_path / "emptygit"
    root.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(root)],
        check=True,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    return root
