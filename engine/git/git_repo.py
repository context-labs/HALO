from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Generator, Iterable
from pathlib import Path

from engine.code._limits import BINARY_SNIFF_BYTES, RESPONSE_CHAR_BUDGET
from engine.code._paths import confine_path
from engine.code._subprocess import stream_subprocess_lines
from engine.code._textwindow import render_numbered_window
from engine.code.models import FileContent
from engine.git.models import (
    BlameLine,
    CommitSummary,
    GitBlame,
    GitDiff,
    GitLog,
    GitShow,
)

logger = logging.getLogger(__name__)

# Unit-separator-delimited log/show format: full sha, author, author ISO-8601-
# strict date, subject. \x1f never appears in commit metadata, so one commit
# parses unambiguously from one stdout line. The short sha is derived by
# truncating %H (not git's dynamic %h) so log, show, and blame always print the
# SAME abbreviation for a commit — the agent can cross-reference them verbatim.
_GIT_FIELD_SEP = "\x1f"
_LOG_FORMAT = _GIT_FIELD_SEP.join(["%H", "%an", "%aI", "%s"])

# Fixed short-sha length: long enough to stay unambiguous in realistic repos and
# resolvable when the agent passes it back to show/diff/blame.
_SHORT_SHA_LEN = 12

_SUBJECT_CAP_CHARS = 500
_BLAME_LINE_CAP_CHARS = 500
_BLAME_MAX_LINES = 2000

# Read-only subcommands only. Asserted in ``_git_stream`` so a future edit can't
# smuggle a mutating subcommand through the shared runner.
_ALLOWED_SUBCOMMANDS = frozenset({"log", "show", "diff", "blame", "rev-parse"})

# Environment variables that redirect git to a DIFFERENT repository than the one
# at ``-C <root>``. They are set when git invokes a hook (or by some CI setups),
# and inheriting them would make the read-only tools silently operate on the
# wrong repo. We strip them so ``-C <root>`` + filesystem discovery is always
# authoritative. (User git config such as diff settings is left intact.)
_REPO_REDIRECT_GIT_ENV = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
        "GIT_CEILING_DIRECTORIES",
    }
)


def find_git() -> str | None:
    """Locate the system ``git`` binary (no pip dependency — git is never pip-installed)."""
    return shutil.which("git")


def _clean_git_env() -> dict[str, str]:
    """The current environment minus the variables that would redirect git to another repo."""
    return {k: v for k, v in os.environ.items() if k not in _REPO_REDIRECT_GIT_ENV}


def _truncate(text: str, cap: int) -> str:
    """Cap a single line/subject at ``cap`` chars with a marker."""
    if len(text) <= cap:
        return text
    return f"{text[:cap]}... [HALO truncated: original {len(text)} chars]"


def _canonical_iso(value: str) -> str:
    """Normalize git's ``%aI`` author date to a canonical ISO-8601 string.

    git renders a zero UTC offset as ``Z`` in newer versions and ``+00:00`` in
    older ones; pin it to ``+00:00`` so the field is identical regardless of the
    host's git version. Pure string work (not ``datetime.fromisoformat``, which
    only accepts the ``Z`` suffix on Python 3.11+). Non-zero offsets pass through.
    """
    if value.endswith("Z"):
        return f"{value[:-1]}+00:00"
    return value


def _reject_binary(lines: Iterable[str], path: str) -> Generator[str, None, None]:
    """Pass ``lines`` through, raising like ``read_file`` if the head holds a NUL.

    ``git show`` streams a blob decoded with ``errors="replace"``; a NUL byte is
    valid UTF-8 (U+0000) so it survives as ``\\x00`` rather than a replacement
    char. A NUL in the leading content marks a binary blob — rejecting it keeps
    ``git_read_file`` consistent with ``CodeRepo.read``'s working-tree guard
    instead of returning mojibake.
    """
    sniffed = 0
    for line in lines:
        if sniffed < BINARY_SNIFF_BYTES:
            if "\x00" in line:
                raise ValueError(f"binary file: {path!r}; git_read_file only supports text files")
            sniffed += len(line)
        yield line


def _validated_ref(ref: str) -> str:
    """Reject refs that could be parsed as git options (argument-injection guard).

    A leading ``-`` is the only way a positional ref turns into an option;
    legitimate refs/revisions (shas, branch/tag names, ``HEAD``, ``HEAD~1``,
    ``A..B``) never start with one. ``log``/``show``/``diff`` additionally pass
    ``--end-of-options``; ``git blame`` does not support that flag, so this
    portable check is the uniform guard across every ref-accepting command.
    """
    if ref.startswith("-"):
        raise ValueError(f"invalid git ref {ref!r}: must not start with '-'")
    return ref


class GitRepo:
    """Read-only git view of a local checkout for the trace-analysis agent.

    STRICTLY non-mutating: only ``log`` / ``show`` / ``diff`` / ``blame`` /
    ``read_at_ref``. Never commit/checkout/reset/push/add. Additive — ``open()``
    returns ``None`` (not raise) when the path isn't a git work tree or git is
    missing, so the run proceeds with the code/trace tools and just skips git.
    Every invocation runs ``git -C <root> --no-pager`` with a fixed read-only
    subcommand; agent input only ever lands in value/ref/path positions, refs are
    guarded with ``--end-of-options`` and separated from paths with ``--``, and
    paths are confined to the repo root.
    """

    def __init__(self, *, root: Path, git_executable: str) -> None:
        self._root = root
        self._git_executable = git_executable

    @classmethod
    def open(cls, repo_path: Path) -> "GitRepo | None":
        """Return a GitRepo if ``repo_path`` is a git work tree and git is available, else None.

        Never raises — git tooling is an enhancement layered on the same
        ``repo_path`` the code tools use. Logs the reason at INFO when it skips.
        """
        git = find_git()
        if git is None:
            logger.info("git tools disabled: git not found on PATH")
            return None
        root = Path(repo_path).resolve()
        try:
            proc = subprocess.run(
                [git, "-C", str(root), "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                env=_clean_git_env(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.info("git tools disabled: %s", exc)
            return None
        if proc.returncode != 0 or proc.stdout.strip() != "true":
            logger.info("git tools disabled: %s is not a git work tree", root)
            return None
        logger.info("git repo opened at %s (git: %s)", root, git)
        return cls(root=root, git_executable=git)

    @property
    def root(self) -> Path:
        """The resolved repository root all paths are confined to."""
        return self._root

    def _git_stream(self, tail: list[str]) -> Generator[str, None, None]:
        """Stream stdout of a read-only ``git -C <root> --no-pager <tail>`` invocation.

        Asserts the subcommand is read-only. git exits 0 on success and 128 on a
        bad ref/path (we never pass ``--exit-code``/``--quiet``), so the error
        floor is 1: any non-zero with no output surfaces as ``ValueError``.
        """
        assert tail and tail[0] in _ALLOWED_SUBCOMMANDS, f"refusing non-read-only git: {tail[:1]}"
        argv = [self._git_executable, "-C", str(self._root), "--no-pager", *tail]
        return stream_subprocess_lines(
            argv,
            cwd=self._root,
            error_label="git",
            error_returncode_floor=1,
            env=_clean_git_env(),
        )

    def _has_commits(self) -> bool:
        """True if HEAD resolves to a commit; False on an unborn HEAD (a repo with no commits yet).

        Uses ``rev-parse`` directly (not ``_git_stream``) because an unborn HEAD
        exits 1 with no output, which the floor-1 streamer would surface as an
        error — here that exit *is* the signal we want.
        """
        proc = subprocess.run(
            [
                self._git_executable,
                "-C",
                str(self._root),
                "rev-parse",
                "--verify",
                "--quiet",
                "HEAD",
            ],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=_clean_git_env(),
        )
        return proc.returncode == 0

    def _confined_rel(self, path: str) -> str:
        """Confine ``path`` to the repo root and return it repo-relative POSIX (for git ``-- <path>``)."""
        resolved = confine_path(self._root, path)
        return resolved.relative_to(self._root).as_posix()

    def _collect_capped(self, stream: Generator[str, None, None], budget: int) -> tuple[str, bool]:
        """Join stream lines until ``budget`` chars, then stop (terminating git). Returns (text, truncated)."""
        chunks: list[str] = []
        used = 0
        truncated = False
        try:
            for line in stream:
                if used + len(line) > budget:
                    truncated = True
                    break
                chunks.append(line)
                used += len(line)
        finally:
            stream.close()
        return "".join(chunks).rstrip("\n"), truncated

    def _parse_commit(self, line: str) -> CommitSummary | None:
        """Parse one ``_LOG_FORMAT`` line into a CommitSummary (None if malformed)."""
        fields = line.rstrip("\n").split(_GIT_FIELD_SEP)
        if len(fields) != 4:
            return None
        full, author, authored_at, subject = fields
        return CommitSummary(
            full_sha=full,
            short_sha=full[:_SHORT_SHA_LEN],
            author=author,
            authored_at=_canonical_iso(authored_at),
            subject=_truncate(subject, _SUBJECT_CAP_CHARS),
        )

    def log(
        self,
        *,
        max_commits: int,
        since: str | None,
        until: str | None,
        ref_range: str | None,
        path: str | None,
        pickaxe_string: str | None,
        pickaxe_regex: str | None,
    ) -> GitLog:
        """List commits (newest first), bounded by ``max_commits`` with a ``has_more`` flag.

        On an empty repo git exits 0 with no output, so this returns an empty log
        rather than erroring.
        """
        tail = ["log", "--no-color", "-n", str(max_commits + 1), f"--format={_LOG_FORMAT}"]
        if since is not None:
            tail.append(f"--since={since}")
        if until is not None:
            tail.append(f"--until={until}")
        if pickaxe_string is not None:
            tail += ["-S", pickaxe_string]
        elif pickaxe_regex is not None:
            # ``-E`` makes ``-G`` use POSIX extended regex (``+``, ``|``, ``()``
            # without backslashes) instead of the basic-regex default.
            tail += ["-E", "-G", pickaxe_regex]
        tail.append("--end-of-options")
        if ref_range is not None:
            tail.append(_validated_ref(ref_range))
        if path is not None:
            tail += ["--", self._confined_rel(path)]

        commits: list[CommitSummary] = []
        has_more = False
        stream = self._git_stream(tail)
        try:
            for line in stream:
                if not line.strip():
                    continue
                parsed = self._parse_commit(line)
                if parsed is None:
                    continue
                if len(commits) < max_commits:
                    commits.append(parsed)
                    continue
                has_more = True
                break
        except ValueError as exc:
            # ``git log`` exits non-zero with no output for an unborn HEAD (a repo
            # with no commits yet) — an empty history, not a failure. Every other
            # such exit IS a real error (bad ``ref_range``, invalid ``pickaxe_regex``,
            # ...) and must surface, so swallow only when there genuinely are no
            # commits.
            if self._has_commits():
                raise
            logger.warning(
                "git log found no commits (unborn HEAD); returning empty history: %s", exc
            )
            return GitLog(commits=[], returned_count=0, has_more=False)
        finally:
            stream.close()
        return GitLog(commits=commits, returned_count=len(commits), has_more=has_more)

    def show(self, *, ref: str, path: str | None, include_patch: bool) -> GitShow:
        """Show one commit: metadata + a ``--stat`` summary (default) or size-capped patch."""
        tail = ["show", "--no-color", "-n", "1", f"--format={_LOG_FORMAT}"]
        if not include_patch:
            tail.append("--stat")
        tail += ["--end-of-options", _validated_ref(ref)]
        if path is not None:
            tail += ["--", self._confined_rel(path)]

        stream = self._git_stream(tail)
        # First line is the commit summary; the rest is the stat/patch body.
        summary_line = next(stream, "")
        commit = self._parse_commit(summary_line)
        if commit is None:
            stream.close()
            raise ValueError(f"git failed: could not parse commit for ref {ref!r}")
        body, truncated = self._collect_capped(stream, RESPONSE_CHAR_BUDGET)
        return GitShow(commit=commit, body=body.lstrip("\n"), truncated=truncated)

    def diff(self, *, from_ref: str, to_ref: str, path: str | None, stat_only: bool) -> GitDiff:
        """Diff two refs: a ``--stat`` summary (default) or a size-capped patch."""
        tail = ["diff", "--no-color"]
        if stat_only:
            tail.append("--stat")
        tail += ["--end-of-options", f"{_validated_ref(from_ref)}..{_validated_ref(to_ref)}"]
        if path is not None:
            tail += ["--", self._confined_rel(path)]
        diff_text, truncated = self._collect_capped(self._git_stream(tail), RESPONSE_CHAR_BUDGET)
        return GitDiff(diff=diff_text, stat_only=stat_only, truncated=truncated)

    def blame(self, *, path: str, start_line: int, end_line: int, ref: str | None) -> GitBlame:
        """Blame a line range of a file: per line, the commit/author that last changed it."""
        rel = self._confined_rel(path)
        truncated = False
        if end_line - start_line + 1 > _BLAME_MAX_LINES:
            end_line = start_line + _BLAME_MAX_LINES - 1
            truncated = True

        # ``git blame`` does not accept ``--end-of-options``; ``_validated_ref``
        # (leading-dash rejection) is the injection guard here, and ``--``
        # separates the path from the optional rev.
        tail = ["blame", "--line-porcelain", "-L", f"{start_line},{end_line}"]
        if ref is not None:
            tail.append(_validated_ref(ref))
        tail += ["--", rel]

        lines: list[BlameLine] = []
        meta: dict[str, tuple[str, str]] = {}  # full sha -> (author, summary)
        cur_sha: str | None = None
        cur_final = 0
        cur_author = ""
        cur_summary = ""
        stream = self._git_stream(tail)
        try:
            for raw in stream:
                line = raw.rstrip("\n")
                if line.startswith("\t"):
                    # Content line for the current commit/line.
                    if cur_sha is not None:
                        author, summary = meta.get(cur_sha, (cur_author, cur_summary))
                        lines.append(
                            BlameLine(
                                line_number=cur_final,
                                short_sha=cur_sha[:_SHORT_SHA_LEN],
                                author=author,
                                summary=_truncate(summary, _SUBJECT_CAP_CHARS),
                                line_text=_truncate(line[1:], _BLAME_LINE_CAP_CHARS),
                            )
                        )
                    cur_sha = None
                    cur_author = ""
                    cur_summary = ""
                    continue
                parts = line.split(" ")
                if len(parts[0]) == 40 and len(parts) >= 3 and parts[2].isdigit():
                    cur_sha = parts[0]
                    cur_final = int(parts[2])
                elif line.startswith("author ") and cur_sha is not None:
                    cur_author = line[len("author ") :]
                    meta.setdefault(cur_sha, (cur_author, cur_summary))
                elif line.startswith("summary ") and cur_sha is not None:
                    cur_summary = line[len("summary ") :]
                    prev = meta.get(cur_sha, (cur_author, ""))
                    meta[cur_sha] = (prev[0] or cur_author, cur_summary)
        finally:
            stream.close()
        return GitBlame(path=rel, lines=lines, returned_count=len(lines), truncated=truncated)

    def read_at_ref(self, *, ref: str, path: str, offset: int, limit: int) -> FileContent:
        """Read a file's contents as of ``ref`` (``git show <ref>:<path>``) as numbered lines.

        Lets the agent read code *as it ran* at a historical commit, not the
        working-tree version. Shares the numbered-window renderer with read_file.
        """
        rel = self._confined_rel(path)
        stream = self._git_stream(["show", "--end-of-options", f"{_validated_ref(ref)}:{rel}"])
        try:
            return render_numbered_window(
                _reject_binary(stream, rel), path=rel, offset=offset, limit=limit
            )
        finally:
            stream.close()
