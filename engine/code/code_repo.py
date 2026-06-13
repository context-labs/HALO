from __future__ import annotations

import logging
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
from engine.errors import EngineDependencyError

logger = logging.getLogger(__name__)

# Baseline directories to exclude on top of whatever ``.gitignore`` says: VCS
# metadata, dependency vendoring, build/output trees, and tool caches. Fed to
# ripgrep as ``-g '!<dir>/'`` so a repo with no (or an incomplete) ``.gitignore``
# still doesn't surface this junk. Ripgrep honours ``.gitignore`` natively on
# top of these.
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

# How many leading bytes to sniff for a NUL when deciding a file is binary
# (read_file only — ripgrep does its own binary detection for glob/grep/tree).
_BINARY_SNIFF_BYTES = 8192

# Per-match line truncation, so one pathological minified line can't flood the
# model's context with a single result.
_GREP_LINE_TEXT_CAP_CHARS = 500

# read_file caps: per-line and per-call. The response budget mirrors
# ``_VIEW_TRACE_RESPONSE_BYTES_BUDGET`` in trace_store.py — a comfortable
# fraction of even a modest context window.
_READ_LINE_CAP_CHARS = 2000
_READ_RESPONSE_CHAR_BUDGET = 150_000

# Repo-tree overview caps so the map stays bounded on large repos.
_TREE_MAX_DEPTH = 4
_TREE_MAX_ENTRIES = 500

_RIPGREP_INSTALL_HINT = (
    "ripgrep (rg) is required to analyze a code repository but was not found on PATH. "
    "Install it (`brew install ripgrep`, `apt-get install ripgrep`, or `pip install ripgrep`) "
    "and re-run."
)


def _looks_binary(blob: bytes) -> bool:
    """Heuristic: a NUL byte in the leading bytes means binary (matches ripgrep's default)."""
    return b"\x00" in blob[:_BINARY_SNIFF_BYTES]


class CodeRepo:
    """Read-only, ripgrep-backed view of a local source checkout for agent code tools.

    Owns the primitives the code tools expose — ``glob`` (file discovery),
    ``grep`` (regex content search), ``tree`` (a directory overview, served by
    ``view_repo_tree``), and ``read`` (numbered file contents).

    ``glob``/``grep``/``tree`` all run through **ripgrep**, so ``.gitignore`` is
    honoured natively and consistently and symlinks aren't followed (rg's
    default), keeping discovery confined to the repo. Ripgrep is therefore a
    hard requirement — ``open`` fails fast if ``rg`` isn't on PATH. ``read`` is
    the one pure-Python primitive (explicit path access); it resolves and
    confines the path to ``root`` and rejects binary files. There is no
    persistent index. ``tree`` is rendered lazily on first access and cached for
    the rest of the run.
    """

    def __init__(self, *, root: Path, rg_executable: str) -> None:
        self._root = root
        self._rg_executable = rg_executable
        self._tree: str | None = None

    @classmethod
    def open(cls, repo_path: Path) -> "CodeRepo":
        """Resolve and validate ``repo_path`` and locate ripgrep. Fails fast.

        Raises ``FileNotFoundError`` if the path does not exist,
        ``NotADirectoryError`` if it is not a directory, and
        ``EngineDependencyError`` if ``rg`` is not on PATH. Runs before any LLM
        call so a bad ``--repo-path`` or a missing ripgrep surfaces immediately,
        not mid-run. The tree is not rendered here — it is built lazily on first
        ``view_repo_tree``.
        """
        root = Path(repo_path).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(f"repo_path is not a directory: {root}")
        rg_executable = shutil.which("rg")
        if rg_executable is None:
            raise EngineDependencyError(_RIPGREP_INSTALL_HINT)
        logger.info("code repo opened at %s (ripgrep: %s)", root, rg_executable)
        return cls(root=root, rg_executable=rg_executable)

    @property
    def root(self) -> Path:
        """The resolved repository root all paths are confined to."""
        return self._root

    @property
    def tree(self) -> str:
        """The depth/entry-capped directory overview, rendered once and cached for the run."""
        if self._tree is None:
            self._tree = _build_tree(self._root.name, self._rg_files(glob_pattern=None))
        return self._tree

    def _exclude_glob_args(self) -> list[str]:
        """Baseline ``-g '!<dir>/'`` excludes shared by every ripgrep invocation."""
        args: list[str] = []
        for excluded in sorted(_EXCLUDED_DIRS):
            args += ["-g", f"!{excluded}/"]
        return args

    def _rg_files(self, *, glob_pattern: str | None) -> list[str]:
        """List repo files via ``rg --files`` (honours .gitignore), optionally filtered by a glob.

        Returns sorted relative POSIX paths. Raises ``ValueError`` (surfaced to
        the model) if ripgrep errors, e.g. on a malformed glob.
        """
        args = [
            self._rg_executable,
            "--files",
            "--hidden",
            "--no-require-git",
            *self._exclude_glob_args(),
        ]
        if glob_pattern is not None:
            args += ["-g", glob_pattern]
        completed = subprocess.run(args, cwd=self._root, capture_output=True, text=True)
        # rg --files exit codes: 0 = files listed, 1 = none matched, >=2 = error.
        if completed.returncode >= 2:
            raise ValueError(f"glob failed: {completed.stderr.strip()}")
        return sorted(line for line in completed.stdout.splitlines() if line)

    def glob(self, pattern: str, max_results: int) -> GlobMatches:
        """Return repo files matching ``pattern`` (relative POSIX paths + sizes), via ``rg --files``.

        ``pattern`` is gitignore-style: a pattern without ``/`` matches at any
        depth (``*.py`` → all .py), one with ``/`` is anchored (``engine/*.py``).
        Results honour ``.gitignore``. ``has_more`` is true when more files
        matched than ``max_results``.
        """
        matched = self._rg_files(glob_pattern=pattern)
        capped = matched[:max_results]
        files = [GlobFileEntry(path=p, size_bytes=(self._root / p).stat().st_size) for p in capped]
        return GlobMatches(
            files=files,
            returned_count=len(files),
            has_more=len(matched) > len(capped),
        )

    def grep(self, regex_pattern: str, glob_pattern: str | None, max_matches: int) -> GrepMatches:
        """Regex-search file contents across the repo via ripgrep (honours .gitignore).

        ``glob_pattern`` optionally confines the search to matching files.
        Returns up to ``max_matches`` records with 1-based line numbers and
        per-line-truncated text; ``has_more`` is true when more matches existed
        than were returned. Ripgrep owns regex validation — a bad pattern raises
        ``ValueError`` carrying rg's message, surfaced to the model.
        """
        args = [
            self._rg_executable,
            "--line-number",
            "--no-heading",
            "--color=never",
            "--hidden",
            "--no-require-git",
            *self._exclude_glob_args(),
        ]
        if glob_pattern is not None:
            args += ["-g", glob_pattern]
        args += ["-e", regex_pattern, "."]
        completed = subprocess.run(args, cwd=self._root, capture_output=True, text=True)
        # rg exit codes: 0 = matches, 1 = no matches, >=2 = error (e.g. bad regex).
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
        # rg prints paths relative to cwd (the repo root); normalise the leading "./".
        path = Path(path_str).as_posix()
        return GrepMatchRecord(
            path=path,
            line_number=int(line_str),
            line_text=_truncate_line(text),
        )

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


def _build_tree(root_name: str, paths: list[str]) -> str:
    """Render a dirs-first, depth/entry-capped tree from a sorted list of relative file paths.

    ``paths`` comes from ``rg --files`` (already .gitignore-honoured), so only
    directories that contain non-ignored files appear. Stops at ``_TREE_MAX_DEPTH``
    levels and ``_TREE_MAX_ENTRIES`` total entries, marking each cap explicitly so
    the model knows the map is partial and should fall back to ``glob_files``.
    """
    # Nested dict: dir -> {child: ...}; file -> None.
    tree: dict = {}
    for path in paths:
        parts = path.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault(parts[-1], None)

    lines: list[str] = [f"{root_name}/"]
    state = {"count": 0, "entry_capped": False}

    def walk(node: dict, depth: int) -> None:
        if state["entry_capped"]:
            return
        # Directories first (dict values), then files (None values), each alphabetical.
        for name, child in sorted(node.items(), key=lambda kv: (kv[1] is None, kv[0])):
            if state["entry_capped"]:
                return
            if state["count"] >= _TREE_MAX_ENTRIES:
                state["entry_capped"] = True
                lines.append(f"{'  ' * depth}... (entry cap of {_TREE_MAX_ENTRIES} reached)")
                return
            state["count"] += 1
            is_dir = child is not None
            lines.append(f"{'  ' * depth}{name}{'/' if is_dir else ''}")
            if is_dir:
                if depth + 1 >= _TREE_MAX_DEPTH:
                    lines.append(f"{'  ' * (depth + 1)}... (depth cap reached)")
                else:
                    walk(child, depth + 1)

    walk(tree, 1)
    return "\n".join(lines)
