from __future__ import annotations

from pathlib import Path

import pytest

import engine.git.git_repo as git_repo_module
from engine.code.models import FileContent
from engine.git.git_repo import GitRepo
from engine.git.models import BlameLine, GitBlame, GitDiff, GitLog, GitShow
from tests.unit.git.git_fixture import (
    AUTHOR_NAME,
    COMMIT_1,
    COMMIT_2,
    COMMIT_3,
    PICKAXE_TOKEN,
    build_empty_git_repo,
    build_git_repo,
)


def _repo(tmp_path: Path) -> GitRepo:
    repo = GitRepo.open(build_git_repo(tmp_path))
    assert repo is not None
    return repo


def _log(repo: GitRepo, **overrides: object) -> GitLog:
    """Call ``repo.log`` filling every keyword (all required) with a no-op default."""
    args: dict[str, object] = {
        "max_commits": 50,
        "since": None,
        "until": None,
        "ref_range": None,
        "path": None,
        "pickaxe_string": None,
        "pickaxe_regex": None,
    }
    args.update(overrides)
    return repo.log(**args)  # type: ignore[arg-type]


# --- open --------------------------------------------------------------------


def test_open_returns_repo_for_work_tree(tmp_path: Path) -> None:
    root = build_git_repo(tmp_path)
    repo = GitRepo.open(root)
    assert repo is not None
    assert repo.root == root.resolve()


def test_open_returns_none_for_non_git_dir(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "file.txt").write_text("hi\n")
    assert GitRepo.open(plain) is None


def test_open_returns_none_when_git_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(git_repo_module, "find_git", lambda: None)
    assert GitRepo.open(build_git_repo(tmp_path)) is None


# --- log ---------------------------------------------------------------------


def test_log_orders_newest_first(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert _log(repo) == GitLog(
        commits=[COMMIT_3, COMMIT_2, COMMIT_1], returned_count=3, has_more=False
    )


def test_log_since_until_window(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = _log(repo, since="2021-01-15T00:00:00", until="2021-02-15T00:00:00")
    assert result == GitLog(commits=[COMMIT_2], returned_count=1, has_more=False)


def test_log_max_commits_has_more(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert _log(repo, max_commits=1) == GitLog(commits=[COMMIT_3], returned_count=1, has_more=True)


def test_log_path_filter(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    # Commit 2 only touched config.py, so runner.py history is commits 3 and 1.
    assert _log(repo, path="runner.py") == GitLog(
        commits=[COMMIT_3, COMMIT_1], returned_count=2, has_more=False
    )


def test_log_pickaxe_string(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert _log(repo, pickaxe_string=PICKAXE_TOKEN) == GitLog(
        commits=[COMMIT_2], returned_count=1, has_more=False
    )


def test_log_pickaxe_regex(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert _log(repo, pickaxe_regex=r"UNIQUE_TOKEN_[A-Z]+") == GitLog(
        commits=[COMMIT_2], returned_count=1, has_more=False
    )


def test_log_bad_ref_range_raises(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="git failed"):
        _log(repo, ref_range="no-such-ref")


def test_log_empty_repo_returns_empty(tmp_path: Path) -> None:
    repo = GitRepo.open(build_empty_git_repo(tmp_path))
    assert repo is not None
    assert _log(repo) == GitLog(commits=[], returned_count=0, has_more=False)


def test_log_invalid_pickaxe_regex_raises(tmp_path: Path) -> None:
    # An invalid regex makes git exit non-zero with no output, like an unborn HEAD;
    # on a repo that HAS commits this must surface as an error, not an empty log.
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="git failed"):
        _log(repo, pickaxe_regex="(")


# --- show --------------------------------------------------------------------


def test_show_stat_summary(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = repo.show(ref=COMMIT_2.short_sha, path=None, include_patch=False)
    assert result == GitShow(
        commit=COMMIT_2,
        body=" config.py | 1 +\n 1 file changed, 1 insertion(+)",
        truncated=False,
    )


def test_show_patch_body(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = repo.show(ref=COMMIT_2.short_sha, path=None, include_patch=True)
    # The full unified-diff body carries opaque blob-index shas; assert the commit
    # and the added line rather than pinning the whole patch.
    assert result.commit == COMMIT_2
    assert result.truncated is False
    assert f"+{PICKAXE_TOKEN} = 1" in result.body


def test_show_patch_truncates_on_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(git_repo_module, "RESPONSE_CHAR_BUDGET", 20)
    repo = _repo(tmp_path)
    result = repo.show(ref=COMMIT_2.short_sha, path=None, include_patch=True)
    assert result.commit == COMMIT_2
    assert result.truncated is True


def test_show_bad_ref_raises(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="git failed"):
        repo.show(ref="no-such-ref", path=None, include_patch=False)


# --- diff --------------------------------------------------------------------


def test_diff_stat(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = repo.diff(
        from_ref=COMMIT_1.short_sha, to_ref=COMMIT_3.short_sha, path=None, stat_only=True
    )
    assert result == GitDiff(
        diff=" config.py | 1 +\n runner.py | 2 +-\n 2 files changed, 2 insertions(+), 1 deletion(-)",
        stat_only=True,
        truncated=False,
    )


def test_diff_patch(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = repo.diff(
        from_ref=COMMIT_1.short_sha, to_ref=COMMIT_3.short_sha, path=None, stat_only=False
    )
    # Full patch carries opaque blob-index shas; assert flags and the changed lines.
    assert result.stat_only is False
    assert result.truncated is False
    assert f"+{PICKAXE_TOKEN} = 1" in result.diff
    assert "return MAX_RETRIES * 2" in result.diff


def test_diff_path_filter(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = repo.diff(
        from_ref=COMMIT_1.short_sha, to_ref=COMMIT_3.short_sha, path="runner.py", stat_only=True
    )
    assert result == GitDiff(
        diff=" runner.py | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)",
        stat_only=True,
        truncated=False,
    )


# --- blame -------------------------------------------------------------------


def test_blame_attributes_lines(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = repo.blame(path="config.py", start_line=1, end_line=3, ref=None)
    assert result == GitBlame(
        path="config.py",
        lines=[
            BlameLine(
                line_number=1,
                short_sha=COMMIT_1.short_sha,
                author=AUTHOR_NAME,
                summary=COMMIT_1.subject,
                line_text="MAX_RETRIES = 3",
            ),
            BlameLine(
                line_number=2,
                short_sha=COMMIT_1.short_sha,
                author=AUTHOR_NAME,
                summary=COMMIT_1.subject,
                line_text="TIMEOUT_SECONDS = 30",
            ),
            BlameLine(
                line_number=3,
                short_sha=COMMIT_2.short_sha,
                author=AUTHOR_NAME,
                summary=COMMIT_2.subject,
                line_text=f"{PICKAXE_TOKEN} = 1",
            ),
        ],
        returned_count=3,
        truncated=False,
    )


def test_blame_window_clamped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(git_repo_module, "_BLAME_MAX_LINES", 2)
    repo = _repo(tmp_path)
    result = repo.blame(path="config.py", start_line=1, end_line=3, ref=None)
    assert result == GitBlame(
        path="config.py",
        lines=[
            BlameLine(
                line_number=1,
                short_sha=COMMIT_1.short_sha,
                author=AUTHOR_NAME,
                summary=COMMIT_1.subject,
                line_text="MAX_RETRIES = 3",
            ),
            BlameLine(
                line_number=2,
                short_sha=COMMIT_1.short_sha,
                author=AUTHOR_NAME,
                summary=COMMIT_1.subject,
                line_text="TIMEOUT_SECONDS = 30",
            ),
        ],
        returned_count=2,
        truncated=True,
    )


def test_blame_confinement_raises(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="outside the repo root"):
        repo.blame(path="../escape.py", start_line=1, end_line=1, ref=None)


# --- read_at_ref -------------------------------------------------------------


def test_read_at_ref_reads_historical_content(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    # config.py at commit 1 had no pickaxe token; only 2 lines.
    result = repo.read_at_ref(ref=COMMIT_1.short_sha, path="config.py", offset=1, limit=500)
    assert result == FileContent(
        path="config.py",
        content="     1\tMAX_RETRIES = 3\n     2\tTIMEOUT_SECONDS = 30",
        start_line=1,
        end_line=2,
        total_line_count=2,
        truncated=False,
    )


def test_read_at_ref_reads_head(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    # At HEAD the pickaxe token line exists (3 lines).
    result = repo.read_at_ref(ref="HEAD", path="config.py", offset=1, limit=500)
    assert result == FileContent(
        path="config.py",
        content="     1\tMAX_RETRIES = 3\n     2\tTIMEOUT_SECONDS = 30\n     3\tUNIQUE_TOKEN_PICKAXE = 1",
        start_line=1,
        end_line=3,
        total_line_count=3,
        truncated=False,
    )


def test_read_at_ref_confinement_raises(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="outside the repo root"):
        repo.read_at_ref(ref="HEAD", path="/etc/hosts", offset=1, limit=10)


def test_read_at_ref_bad_ref_raises(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="git failed"):
        repo.read_at_ref(ref="no-such-ref", path="config.py", offset=1, limit=10)


# --- ref injection guard -----------------------------------------------------


def test_dash_prefixed_ref_rejected_show(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="must not start with"):
        repo.show(ref="--output=/tmp/x", path=None, include_patch=False)


def test_dash_prefixed_ref_rejected_blame(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="must not start with"):
        repo.blame(path="config.py", start_line=1, end_line=1, ref="--reverse")


def test_dash_prefixed_ref_rejected_diff(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="must not start with"):
        repo.diff(from_ref="-x", to_ref="HEAD", path=None, stat_only=True)


# --- ambient git env isolation -----------------------------------------------


def test_ignores_inherited_git_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An ambient GIT_DIR (e.g. set by a git hook) must not redirect the tools to another repo."""
    root = build_git_repo(tmp_path)
    other_parent = tmp_path / "other_parent"
    other_parent.mkdir()
    other = build_git_repo(other_parent)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(other / ".git" / "index"))

    repo = GitRepo.open(root)
    assert repo is not None
    assert repo.root == root.resolve()
    # Reads ``root``'s history, not whatever GIT_DIR points at.
    assert _log(repo) == GitLog(
        commits=[COMMIT_3, COMMIT_2, COMMIT_1], returned_count=3, has_more=False
    )
