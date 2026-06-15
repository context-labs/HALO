from __future__ import annotations

from pathlib import Path

import pytest

import engine.git.git_repo as git_repo_module
from engine.git.git_repo import GitRepo
from tests.unit.git.git_fixture import (
    AUTHOR_NAME,
    COMMIT_1_SUBJECT,
    COMMIT_2_SUBJECT,
    COMMIT_3_SUBJECT,
    PICKAXE_TOKEN,
    build_empty_git_repo,
    build_git_repo,
)


def _repo(tmp_path: Path) -> GitRepo:
    repo = GitRepo.open(build_git_repo(tmp_path))
    assert repo is not None
    return repo


def _short_sha(repo: GitRepo, subject: str) -> str:
    """Look up a commit's short sha by its subject (commits are unique here)."""
    commits = repo.log(
        max_commits=50,
        since=None,
        until=None,
        ref_range=None,
        path=None,
        pickaxe_string=None,
        pickaxe_regex=None,
    ).commits
    by_subject = {c.subject: c.short_sha for c in commits}
    return by_subject[subject]


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
    result = repo.log(
        max_commits=50,
        since=None,
        until=None,
        ref_range=None,
        path=None,
        pickaxe_string=None,
        pickaxe_regex=None,
    )
    assert [c.subject for c in result.commits] == [
        COMMIT_3_SUBJECT,
        COMMIT_2_SUBJECT,
        COMMIT_1_SUBJECT,
    ]
    assert [c.author for c in result.commits] == [AUTHOR_NAME, AUTHOR_NAME, AUTHOR_NAME]
    assert result.commits[0].authored_at.startswith("2021-03-01")
    assert result.returned_count == 3
    assert result.has_more is False


def test_log_since_until_window(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = repo.log(
        max_commits=50,
        since="2021-01-15T00:00:00",
        until="2021-02-15T00:00:00",
        ref_range=None,
        path=None,
        pickaxe_string=None,
        pickaxe_regex=None,
    )
    assert [c.subject for c in result.commits] == [COMMIT_2_SUBJECT]


def test_log_max_commits_has_more(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = repo.log(
        max_commits=1,
        since=None,
        until=None,
        ref_range=None,
        path=None,
        pickaxe_string=None,
        pickaxe_regex=None,
    )
    assert [c.subject for c in result.commits] == [COMMIT_3_SUBJECT]
    assert result.returned_count == 1
    assert result.has_more is True


def test_log_path_filter(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = repo.log(
        max_commits=50,
        since=None,
        until=None,
        ref_range=None,
        path="runner.py",
        pickaxe_string=None,
        pickaxe_regex=None,
    )
    # Commit 2 only touched config.py, so runner.py history is commits 3 and 1.
    assert [c.subject for c in result.commits] == [COMMIT_3_SUBJECT, COMMIT_1_SUBJECT]


def test_log_pickaxe_string(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = repo.log(
        max_commits=50,
        since=None,
        until=None,
        ref_range=None,
        path=None,
        pickaxe_string=PICKAXE_TOKEN,
        pickaxe_regex=None,
    )
    assert [c.subject for c in result.commits] == [COMMIT_2_SUBJECT]


def test_log_pickaxe_regex(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = repo.log(
        max_commits=50,
        since=None,
        until=None,
        ref_range=None,
        path=None,
        pickaxe_string=None,
        pickaxe_regex=r"UNIQUE_TOKEN_[A-Z]+",
    )
    assert [c.subject for c in result.commits] == [COMMIT_2_SUBJECT]


def test_log_bad_ref_range_raises(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="git failed"):
        repo.log(
            max_commits=50,
            since=None,
            until=None,
            ref_range="no-such-ref",
            path=None,
            pickaxe_string=None,
            pickaxe_regex=None,
        )


def test_log_empty_repo_returns_empty(tmp_path: Path) -> None:
    repo = GitRepo.open(build_empty_git_repo(tmp_path))
    assert repo is not None
    result = repo.log(
        max_commits=50,
        since=None,
        until=None,
        ref_range=None,
        path=None,
        pickaxe_string=None,
        pickaxe_regex=None,
    )
    assert result.commits == []
    assert result.returned_count == 0
    assert result.has_more is False


# --- show --------------------------------------------------------------------


def test_show_stat_summary(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    sha = _short_sha(repo, COMMIT_2_SUBJECT)
    result = repo.show(ref=sha, path=None, include_patch=False)
    assert result.commit.subject == COMMIT_2_SUBJECT
    assert result.commit.short_sha == sha
    assert "config.py" in result.body
    assert "runner.py" not in result.body
    assert result.truncated is False


def test_show_patch_body(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    sha = _short_sha(repo, COMMIT_2_SUBJECT)
    result = repo.show(ref=sha, path=None, include_patch=True)
    assert f"+{PICKAXE_TOKEN} = 1" in result.body
    assert result.truncated is False


def test_show_patch_truncates_on_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(git_repo_module, "_SHOW_PATCH_CHAR_BUDGET", 20)
    repo = _repo(tmp_path)
    sha = _short_sha(repo, COMMIT_2_SUBJECT)
    result = repo.show(ref=sha, path=None, include_patch=True)
    assert result.truncated is True


def test_show_bad_ref_raises(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="git failed"):
        repo.show(ref="no-such-ref", path=None, include_patch=False)


# --- diff --------------------------------------------------------------------


def test_diff_stat(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = _short_sha(repo, COMMIT_1_SUBJECT)
    last = _short_sha(repo, COMMIT_3_SUBJECT)
    result = repo.diff(from_ref=first, to_ref=last, path=None, stat_only=True)
    assert result.stat_only is True
    assert "config.py" in result.diff
    assert "runner.py" in result.diff
    assert result.truncated is False


def test_diff_patch(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = _short_sha(repo, COMMIT_1_SUBJECT)
    last = _short_sha(repo, COMMIT_3_SUBJECT)
    result = repo.diff(from_ref=first, to_ref=last, path=None, stat_only=False)
    assert result.stat_only is False
    assert f"+{PICKAXE_TOKEN} = 1" in result.diff
    assert "return MAX_RETRIES * 2" in result.diff


def test_diff_path_filter(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = _short_sha(repo, COMMIT_1_SUBJECT)
    last = _short_sha(repo, COMMIT_3_SUBJECT)
    result = repo.diff(from_ref=first, to_ref=last, path="runner.py", stat_only=True)
    assert "runner.py" in result.diff
    assert "config.py" not in result.diff


# --- blame -------------------------------------------------------------------


def test_blame_attributes_lines(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = _short_sha(repo, COMMIT_1_SUBJECT)
    second = _short_sha(repo, COMMIT_2_SUBJECT)
    result = repo.blame(path="config.py", start_line=1, end_line=3, ref=None)
    assert result.path == "config.py"
    assert result.returned_count == 3
    assert result.truncated is False
    assert result.lines[0].line_number == 1
    assert result.lines[0].line_text == "MAX_RETRIES = 3"
    assert result.lines[0].short_sha == first
    assert result.lines[0].summary == COMMIT_1_SUBJECT
    assert result.lines[0].author == AUTHOR_NAME
    # Line 3 (the pickaxe token) came in with commit 2.
    assert result.lines[2].line_number == 3
    assert result.lines[2].short_sha == second
    assert result.lines[2].summary == COMMIT_2_SUBJECT


def test_blame_window_clamped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(git_repo_module, "_BLAME_MAX_LINES", 2)
    repo = _repo(tmp_path)
    result = repo.blame(path="config.py", start_line=1, end_line=3, ref=None)
    assert result.truncated is True
    assert result.returned_count == 2
    assert [line.line_number for line in result.lines] == [1, 2]


def test_blame_confinement_raises(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="outside the repo root"):
        repo.blame(path="../escape.py", start_line=1, end_line=1, ref=None)


# --- read_at_ref -------------------------------------------------------------


def test_read_at_ref_reads_historical_content(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = _short_sha(repo, COMMIT_1_SUBJECT)
    result = repo.read_at_ref(ref=first, path="config.py", offset=1, limit=500)
    # config.py at commit 1 had no pickaxe token; only 2 lines.
    assert result.content == "     1\tMAX_RETRIES = 3\n     2\tTIMEOUT_SECONDS = 30"
    assert result.total_line_count == 2
    assert result.start_line == 1
    assert result.end_line == 2
    assert result.truncated is False


def test_read_at_ref_head_has_new_line(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = repo.read_at_ref(ref="HEAD", path="config.py", offset=1, limit=500)
    # At HEAD the pickaxe token line exists (3 lines).
    assert result.total_line_count == 3
    assert f"{PICKAXE_TOKEN} = 1" in result.content


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
    result = repo.log(
        max_commits=50,
        since=None,
        until=None,
        ref_range=None,
        path=None,
        pickaxe_string=None,
        pickaxe_regex=None,
    )
    # Reads ``root``'s history (3 commits), not whatever GIT_DIR points at.
    assert [c.subject for c in result.commits] == [
        COMMIT_3_SUBJECT,
        COMMIT_2_SUBJECT,
        COMMIT_1_SUBJECT,
    ]
