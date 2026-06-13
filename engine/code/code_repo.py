from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from engine.code.models import (
    FileContent,
    GlobFileEntry,
    GlobMatches,
    GrepMatches,
    GrepMatchRecord,
)

logger = logging.getLogger(__name__)

# Directories never worth searching: VCS metadata, dependency vendoring,
# build/output trees, and tool caches. Pruned by the pure-Python glob/grep/tree
# paths. Ripgrep additionally honours ``.gitignore`` natively (with ``.git``
# always excluded via ``-g '!.git/'``); the Python paths fall back to this fixed
# set rather than parse ``.gitignore`` (a faithful implementation would need a
# new dependency for marginal benefit). The asymmetry is intentional and visible
# in the ``grep engine: ...`` startup log line.
_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        ".tox",
        ".eggs",
        "dist",
        "build",
        "target",
    }
)

# Pure-Python grep skips files larger than this — large generated/vendored blobs
# blow scan time for no analytical value (ripgrep has its own size heuristics).
_GREP_MAX_FILE_BYTES = 1_000_000

# How many leading bytes to sniff for a NUL when deciding a file is binary.
_BINARY_SNIFF_BYTES = 8192

# Per-match line truncation, so one pathological minified line can't flood the
# model's context with a single result.
_GREP_LINE_TEXT_CAP_CHARS = 500

# read_file caps: per-line and per-call. The response budget mirrors
# ``_VIEW_TRACE_RESPONSE_BYTES_BUDGET`` in trace_store.py — a comfortable
# fraction of even a modest context window.
_READ_LINE_CAP_CHARS = 2000
_READ_RESPONSE_CHAR_BUDGET = 150_000

# Repo-tree snapshot caps so the embedded map stays bounded on large repos.
_TREE_MAX_DEPTH = 4
_TREE_MAX_ENTRIES = 500


def _compile_regex_or_raise(regex_pattern: str) -> re.Pattern[str]:
    """Compile a regex string or raise ``ValueError`` with a clear, model-actionable message."""
    try:
        return re.compile(regex_pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern {regex_pattern!r}: {exc}") from exc


def _is_excluded(parts: tuple[str, ...]) -> bool:
    """True if any path segment is an always-excluded directory name."""
    return any(part in _EXCLUDED_DIRS for part in parts)


def _looks_binary(blob: bytes) -> bool:
    """Heuristic: a NUL byte in the leading bytes means binary (matches ripgrep's default)."""
    return b"\x00" in blob[:_BINARY_SNIFF_BYTES]


class CodeRepo:
    """Read-only, path-confined view of a local source checkout for agent code tools.

    Owns the primitives the code tools expose — ``glob`` (file discovery),
    ``grep`` (regex content search), ``read`` (numbered file contents), and
    ``tree`` (a directory overview, served by the ``view_repo_tree`` tool).

    Every path argument is resolved and confined to ``root`` (symlink escapes
    included), so an agent cannot read outside the repo. ``grep`` shells out to
    ripgrep when available (honouring ``.gitignore``) and falls back to a pure-
    Python scan otherwise; ``glob``/``read``/``tree`` are always pure Python and
    prune a fixed set of excluded directories. There is no persistent index —
    repos are small enough for on-demand scans. ``tree`` is rendered lazily on
    first access and cached for the rest of the run.
    """

    def __init__(self, *, root: Path, rg_executable: str | None) -> None:
        self._root = root
        self._rg_executable = rg_executable
        self._tree: str | None = None

    @classmethod
    def open(cls, repo_path: Path) -> "CodeRepo":
        """Resolve and validate ``repo_path`` and detect ripgrep. Fails fast.

        Raises ``FileNotFoundError`` if the path does not exist and
        ``NotADirectoryError`` if it is not a directory. Runs before any LLM
        call so a bad ``--repo-path`` surfaces immediately, not mid-run. The
        tree is not rendered here — it is built lazily on first ``view_repo_tree``.
        """
        root = Path(repo_path).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(f"repo_path is not a directory: {root}")
        rg_executable = shutil.which("rg")
        logger.info(
            "code repo opened at %s (grep engine: %s)",
            root,
            "ripgrep" if rg_executable else "python",
        )
        return cls(root=root, rg_executable=rg_executable)

    @property
    def root(self) -> Path:
        """The resolved repository root all paths are confined to."""
        return self._root

    @property
    def tree(self) -> str:
        """The depth/entry-capped directory overview, rendered once and cached for the run."""
        if self._tree is None:
            self._tree = _render_tree(self._root)
        return self._tree

    def _resolve_confined(self, path: str) -> Path:
        """Resolve ``path`` (relative to root, or an absolute path already inside root) within the repo.

        ``.resolve()`` follows symlinks before the containment check, so a
        symlink pointing outside the repo is rejected. Raises ``ValueError``
        with a model-actionable message on escape.
        """
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise ValueError(
                f"path {path!r} resolves outside the repo root; pass a path relative "
                "to the repo root (see glob_files/grep_files output)"
            )
        return resolved

    def _validate_glob_pattern(self, pattern: str) -> None:
        """Reject absolute patterns and ``..`` traversal so a glob can't escape the root."""
        if Path(pattern).is_absolute():
            raise ValueError(
                f"glob pattern {pattern!r} must be relative to the repo root, not absolute"
            )
        if ".." in Path(pattern).parts:
            raise ValueError(f"glob pattern {pattern!r} must not contain '..' segments")

    def glob(self, pattern: str, max_results: int) -> GlobMatches:
        """Return repo files matching ``pattern`` (relative POSIX paths + sizes), excluded dirs pruned.

        ``has_more`` is true when more files matched than ``max_results``.
        """
        self._validate_glob_pattern(pattern)
        matched: list[Path] = []
        for candidate in self._root.glob(pattern):
            # Skip symlinks entirely (matches ripgrep's default --no-follow):
            # a symlink could point outside the repo and leak content, and an
            # in-repo symlink is just a duplicate of its target.
            if candidate.is_symlink() or not candidate.is_file():
                continue
            rel = candidate.relative_to(self._root)
            if _is_excluded(rel.parts):
                continue
            matched.append(candidate)
        matched.sort()
        capped = matched[:max_results]
        files = [
            GlobFileEntry(
                path=p.relative_to(self._root).as_posix(),
                size_bytes=p.stat().st_size,
            )
            for p in capped
        ]
        return GlobMatches(
            files=files,
            returned_count=len(files),
            has_more=len(matched) > len(capped),
        )

    def grep(self, regex_pattern: str, glob_pattern: str | None, max_matches: int) -> GrepMatches:
        """Regex-search file contents across the repo (optionally filtered by ``glob_pattern``).

        Validates the regex up front. Uses ripgrep when available, else a pure-
        Python scan. Returns up to ``max_matches`` records with 1-based line
        numbers and per-line-truncated text; ``has_more`` is true when more
        matches existed than were returned.
        """
        _compile_regex_or_raise(regex_pattern)
        if glob_pattern is not None:
            self._validate_glob_pattern(glob_pattern)
        if self._rg_executable is not None:
            return self._grep_ripgrep(regex_pattern, glob_pattern, max_matches)
        return self._grep_python(regex_pattern, glob_pattern, max_matches)

    def _grep_ripgrep(
        self, regex_pattern: str, glob_pattern: str | None, max_matches: int
    ) -> GrepMatches:
        """Run ripgrep from the repo root and parse ``path:line:text`` output into match records."""
        assert self._rg_executable is not None
        args = [
            self._rg_executable,
            "--line-number",
            "--no-heading",
            "--color=never",
            "--hidden",
        ]
        # Exclude the same baseline dirs as the pure-Python path so both engines
        # share a floor; ``--hidden`` keeps dotfiles like .github/ searchable
        # while these globs prune VCS/build/cache trees. Ripgrep additionally
        # honours .gitignore on top of this.
        for excluded in sorted(_EXCLUDED_DIRS):
            args += ["-g", f"!{excluded}/"]
        if glob_pattern is not None:
            args += ["-g", glob_pattern]
        args += ["-e", regex_pattern, "."]
        completed = subprocess.run(
            args,
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        # rg exit codes: 0 = matches, 1 = no matches, >=2 = error.
        if completed.returncode >= 2:
            raise ValueError(f"grep failed: {completed.stderr.strip()}")

        matches: list[GrepMatchRecord] = []
        total = 0
        for line in completed.stdout.splitlines():
            parsed = self._parse_ripgrep_line(line)
            if parsed is None:
                continue
            total += 1
            if len(matches) < max_matches:
                matches.append(parsed)
        return GrepMatches(
            matches=matches,
            returned_match_count=len(matches),
            has_more=total > len(matches),
        )

    def _parse_ripgrep_line(self, line: str) -> GrepMatchRecord | None:
        """Parse one ``path:line:text`` ripgrep output line into a match record (None if malformed)."""
        path_str, sep1, rest = line.partition(":")
        if sep1 == "":
            return None
        line_str, sep2, text = rest.partition(":")
        if sep2 == "" or not line_str.isdigit():
            return None
        rel = Path(path_str)
        # rg prints paths relative to cwd (the repo root); normalise the leading "./".
        path = rel.as_posix()
        return GrepMatchRecord(
            path=path,
            line_number=int(line_str),
            line_text=_truncate_line(text),
        )

    def _grep_python(
        self, regex_pattern: str, glob_pattern: str | None, max_matches: int
    ) -> GrepMatches:
        """Pure-Python regex scan over repo files (no ripgrep), pruning excluded dirs and binary/large files."""
        pattern = _compile_regex_or_raise(regex_pattern)
        candidates = self._candidate_files(glob_pattern)

        matches: list[GrepMatchRecord] = []
        total = 0
        for file_path in candidates:
            try:
                blob = file_path.read_bytes()
            except OSError:
                continue
            if len(blob) > _GREP_MAX_FILE_BYTES or _looks_binary(blob):
                continue
            rel = file_path.relative_to(self._root).as_posix()
            text = blob.decode("utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    total += 1
                    if len(matches) < max_matches:
                        matches.append(
                            GrepMatchRecord(
                                path=rel,
                                line_number=line_number,
                                line_text=_truncate_line(line),
                            )
                        )
        return GrepMatches(
            matches=matches,
            returned_match_count=len(matches),
            has_more=total > len(matches),
        )

    def _candidate_files(self, glob_pattern: str | None) -> list[Path]:
        """Enumerate scannable files for the pure-Python grep, pruning excluded dirs.

        With a ``glob_pattern`` set, reuse ``glob`` (already excluded-dir-aware).
        Without one, walk the tree pruning excluded directories in place.
        """
        if glob_pattern is not None:
            return sorted(
                self._root / entry.path
                for entry in self.glob(glob_pattern, max_results=_TREE_MAX_ENTRIES * 100).files
            )
        found: list[Path] = []
        # ``os.walk`` does not follow symlinked directories (followlinks=False
        # by default); we additionally skip symlinked files so a symlink can't
        # leak content from outside the repo (matches ripgrep's default).
        for dirpath, dirnames, filenames in os.walk(self._root):
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in _EXCLUDED_DIRS and not (Path(dirpath) / d).is_symlink()
            )
            for name in sorted(filenames):
                file_path = Path(dirpath) / name
                if file_path.is_symlink():
                    continue
                found.append(file_path)
        return found

    def read(self, path: str, offset: int, limit: int) -> FileContent:
        """Return a 1-based ``[offset, offset+limit)`` window of ``path`` as ``cat -n`` numbered lines.

        Confines the path, rejects non-files and binary files, decodes UTF-8
        with replacement, caps each line at ``_READ_LINE_CAP_CHARS`` and total
        output at ``_READ_RESPONSE_CHAR_BUDGET``. ``truncated`` flags any clip.
        """
        resolved = self._resolve_confined(path)
        if not resolved.is_file():
            raise ValueError(f"not a file: {path!r}")
        blob = resolved.read_bytes()
        if _looks_binary(blob):
            raise ValueError(f"binary file: {path!r}; read_file only supports text files")

        text = blob.decode("utf-8", errors="replace")
        lines = text.splitlines()
        total_line_count = len(lines)
        if total_line_count == 0:
            return FileContent(
                path=resolved.relative_to(self._root).as_posix(),
                content="",
                start_line=offset,
                end_line=0,
                total_line_count=0,
                truncated=False,
            )

        start_index = offset - 1
        window = lines[start_index : start_index + limit]

        rendered: list[str] = []
        # ``truncated`` means output was clipped *within* the requested window —
        # a line hit the per-line cap, or the response budget cut the window
        # short. It does NOT flag a window that simply doesn't span the whole
        # file: the caller sees that from ``total_line_count`` vs ``end_line``.
        truncated = False
        used_chars = 0
        last_line_number = offset - 1
        for i, line in enumerate(window):
            line_number = offset + i
            if len(line) > _READ_LINE_CAP_CHARS:
                line = (
                    f"{line[:_READ_LINE_CAP_CHARS]}... [HALO truncated: original {len(line)} chars]"
                )
                truncated = True
            entry = f"{line_number:6d}\t{line}"
            if used_chars + len(entry) > _READ_RESPONSE_CHAR_BUDGET:
                truncated = True
                break
            rendered.append(entry)
            used_chars += len(entry) + 1
            last_line_number = line_number

        return FileContent(
            path=resolved.relative_to(self._root).as_posix(),
            content="\n".join(rendered),
            start_line=offset,
            end_line=max(last_line_number, 0),
            total_line_count=total_line_count,
            truncated=truncated,
        )


def _truncate_line(text: str) -> str:
    """Cap a single matched line at ``_GREP_LINE_TEXT_CAP_CHARS`` with a marker."""
    if len(text) <= _GREP_LINE_TEXT_CAP_CHARS:
        return text
    return f"{text[:_GREP_LINE_TEXT_CAP_CHARS]}... [HALO truncated: original {len(text)} chars]"


def _render_tree(root: Path) -> str:
    """Render a dirs-first, depth/entry-capped file tree of ``root`` as an indented string.

    Excluded directories are pruned. Stops at ``_TREE_MAX_DEPTH`` levels and
    ``_TREE_MAX_ENTRIES`` total entries, marking each cap explicitly so the
    model knows the map is partial and should fall back to ``glob_files``.
    """
    lines: list[str] = [f"{root.name}/"]
    state = {"count": 0, "entry_capped": False}

    def walk(directory: Path, depth: int) -> None:
        if state["entry_capped"]:
            return
        try:
            entries = sorted(
                os.scandir(directory),
                key=lambda e: (not e.is_dir(follow_symlinks=False), e.name),
            )
        except OSError:
            return
        for entry in entries:
            if state["entry_capped"]:
                return
            # Skip symlinks for consistency with glob/grep (and so the map never
            # advertises a path read_file would reject as escaping the root).
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False) and entry.name in _EXCLUDED_DIRS:
                continue
            if state["count"] >= _TREE_MAX_ENTRIES:
                state["entry_capped"] = True
                lines.append(f"{'  ' * depth}... (entry cap of {_TREE_MAX_ENTRIES} reached)")
                return
            state["count"] += 1
            indent = "  " * depth
            is_dir = entry.is_dir(follow_symlinks=False)
            lines.append(f"{indent}{entry.name}{'/' if is_dir else ''}")
            if is_dir:
                if depth + 1 >= _TREE_MAX_DEPTH:
                    lines.append(f"{'  ' * (depth + 1)}... (depth cap reached)")
                else:
                    walk(Path(entry.path), depth + 1)

    walk(root, 1)
    return "\n".join(lines)
