from __future__ import annotations

import json
import logging
import shutil
import sysconfig
from collections.abc import Generator
from pathlib import Path

from engine.code._limits import BINARY_SNIFF_BYTES
from engine.code._paths import confine_path
from engine.code._subprocess import stream_subprocess_lines
from engine.code._textwindow import render_numbered_window
from engine.code.models import (
    FileContent,
    GlobFileEntry,
    GlobMatches,
    GrepMatches,
    GrepMatchRecord,
)
from engine.errors import EngineDependencyError

logger = logging.getLogger(__name__)


def find_ripgrep() -> str | None:
    """Locate the ripgrep binary, or ``None`` if unavailable.

    The pip ``ripgrep`` wheel installs ``rg`` into the interpreter's *scripts*
    directory (``.venv/bin`` in a venv), which is NOT necessarily on ``PATH`` —
    e.g. running ``.venv/bin/python`` or the installed ``.venv/bin/halo`` entry
    point without activating the venv. So check the scripts dir first (finds the
    pip-installed binary regardless of PATH), then fall back to a system ``rg``.
    """
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        for name in ("rg", "rg.exe"):
            candidate = Path(scripts_dir) / name
            if candidate.is_file():
                return str(candidate)
    return shutil.which("rg")


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
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        ".tox",
        ".eggs",
    }
)

# Per-match line truncation, so one pathological minified line can't flood the
# model's context with a single result.
_GREP_LINE_TEXT_CAP_CHARS = 500

# Repo-tree overview caps so the map stays bounded on large repos.
_TREE_MAX_DEPTH = 4
_TREE_MAX_ENTRIES = 500
# Hard cap on how many paths the tree reads from ripgrep before building. The
# render only shows _TREE_MAX_ENTRIES, so reading far more is wasted memory/time
# on huge repos. Generously above the entry cap: any repo with fewer files is
# unaffected (reads them all); larger repos get the sorted-first slice, which the
# entry-cap marker already flags as partial.
_TREE_MAX_PATHS = 10_000

_RIPGREP_INSTALL_HINT = (
    "ripgrep (rg) is required to analyze a code repository but was not found on PATH. "
    "Install it (`brew install ripgrep`, `apt-get install ripgrep`, or `pip install ripgrep`) "
    "and re-run."
)


def _looks_binary(blob: bytes) -> bool:
    """Heuristic: a NUL byte in the leading bytes means binary (matches ripgrep's default)."""
    return b"\x00" in blob[:BINARY_SNIFF_BYTES]


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
        rg_executable = find_ripgrep()
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
        """The depth/entry-capped directory overview, rendered once and cached for the run.

        Streams ``rg --files --sort path`` and stops after ``_TREE_MAX_PATHS`` so
        a huge repo can't materialize an unbounded path list / tree dict — the
        render only ever shows ``_TREE_MAX_ENTRIES`` anyway. ``--sort path`` keeps
        the bounded slice deterministic (alphabetically-first); repos under the
        cap read fully and are unaffected.
        """
        if self._tree is None:
            args = self._rg_files_args(glob_pattern=None, sort=True)
            paths: list[str] = []
            stream = self._rg_line_stream(args)
            try:
                for line in stream:
                    path = line.rstrip("\n")
                    if not path:
                        continue
                    paths.append(path)
                    if len(paths) >= _TREE_MAX_PATHS:
                        break
            finally:
                stream.close()
            self._tree = _build_tree(self._root.name, paths)
        return self._tree

    def _exclude_glob_args(self) -> list[str]:
        """Baseline ``-g '!<dir>/'`` excludes shared by every ripgrep invocation.

        Ripgrep glob precedence is last-match-wins, so these must be appended
        *after* any caller-supplied ``-g`` pattern — otherwise a broad pattern
        like ``**/*`` would re-include ``.git``/``node_modules``.
        """
        args: list[str] = []
        for excluded in sorted(_EXCLUDED_DIRS):
            args += ["-g", f"!{excluded}/"]
        return args

    def _rg_files_args(self, *, glob_pattern: str | None, sort: bool) -> list[str]:
        """Build the ``rg --files`` argv (honours .gitignore), with baseline excludes last.

        ``sort=True`` adds ``--sort path`` so a streamed, early-stopped consumer
        still sees the lexicographically-smallest paths first.
        """
        args = [self._rg_executable, "--files", "--hidden", "--no-require-git"]
        if sort:
            args += ["--sort", "path"]
        if glob_pattern is not None:
            args += ["-g", glob_pattern]
        args += self._exclude_glob_args()
        return args

    def _rg_line_stream(self, args: list[str]) -> Generator[str, None, None]:
        """Stream ripgrep stdout lines (see ``stream_subprocess_lines`` for semantics).

        rg exit codes: 0 = ok, 1 = nothing matched, >=2 = error — so the error
        floor is 2 (a "no matches" exit isn't a failure).
        """
        return stream_subprocess_lines(
            args,
            cwd=self._root,
            error_label="ripgrep",
        )

    def glob(self, pattern: str, max_results: int) -> GlobMatches:
        """Return repo files matching ``pattern`` (relative POSIX paths + sizes), via ``rg --files``.

        ``pattern`` is gitignore-style: a pattern without ``/`` matches at any
        depth (``*.py`` → all .py), one with ``/`` is anchored (``engine/*.py``).
        Results honour ``.gitignore``. Streams ``rg --files --sort path`` and
        stops after ``max_results`` (+1 to set ``has_more``), so a broad pattern
        on a large repo never materializes the full path list — the sorted
        ordering means the returned slice is still the smallest ``max_results``.
        """
        args = self._rg_files_args(glob_pattern=pattern, sort=True)
        files: list[GlobFileEntry] = []
        has_more = False
        stream = self._rg_line_stream(args)
        try:
            for line in stream:
                path = line.rstrip("\n")
                if not path:
                    continue
                try:
                    size = (self._root / path).stat().st_size
                except OSError as exc:
                    # rg listed it, but it's unstattable now — removed/moved
                    # between the listing and here (a race), or a dangling link.
                    # Skip it rather than failing the whole glob.
                    logger.warning(
                        "skipping %r: listed by ripgrep but not stattable: %s", path, exc
                    )
                    continue
                if len(files) < max_results:
                    files.append(GlobFileEntry(path=path, size_bytes=size))
                    continue
                has_more = True
                break
        finally:
            stream.close()
        return GlobMatches(files=files, returned_count=len(files), has_more=has_more)

    def grep(self, regex_pattern: str, glob_pattern: str | None, max_matches: int) -> GrepMatches:
        """Regex-search file contents across the repo via ripgrep (honours .gitignore).

        ``glob_pattern`` optionally confines the search to matching files.
        Returns up to ``max_matches`` records with 1-based line numbers and
        per-line-truncated text; ``has_more`` is true when more matches existed
        than were returned. Streams rg's output and stops one match past the cap
        so a broad pattern can't buffer megabytes. Ripgrep owns regex validation
        — a bad pattern raises ``ValueError`` carrying rg's message.
        """
        # ``--json`` gives unambiguous per-match records (path, line number, text
        # as separate fields), so a repo path containing a ``:`` parses correctly
        # — unlike the ``path:line:text`` text format.
        args = [self._rg_executable, "--json", "--hidden", "--no-require-git"]
        if glob_pattern is not None:
            args += ["-g", glob_pattern]
        args += self._exclude_glob_args()
        args += ["-e", regex_pattern, "."]

        matches: list[GrepMatchRecord] = []
        has_more = False
        stream = self._rg_line_stream(args)
        try:
            for line in stream:
                parsed = self._parse_rg_json_match(line)
                if parsed is None:
                    continue
                if len(matches) < max_matches:
                    matches.append(parsed)
                    continue
                has_more = True
                break
        finally:
            stream.close()
        return GrepMatches(
            matches=matches,
            returned_match_count=len(matches),
            has_more=has_more,
        )

    def _parse_rg_json_match(self, line: str) -> GrepMatchRecord | None:
        """Parse one ``rg --json`` output line into a match record (None unless it's a ``match`` event).

        ``rg --json`` interleaves ``begin``/``match``/``end``/``summary`` events;
        only ``match`` carries a hit. Path/line/text are separate fields, so a
        path containing ``:`` is handled correctly. A non-UTF-8 path or text is
        reported by rg as a ``bytes`` field (no ``text``); such matches are
        skipped rather than guessed at.
        """
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("skipping malformed ripgrep --json line: %s", exc)
            return None
        if not isinstance(event, dict) or event.get("type") != "match":
            return None
        data = event.get("data", {})
        path_text = (data.get("path") or {}).get("text")
        line_number = data.get("line_number")
        if path_text is None or not isinstance(line_number, int):
            return None
        text = (data.get("lines") or {}).get("text", "") or ""
        # rg includes the line's trailing newline (and CR on CRLF); drop it.
        text = text[:-1] if text.endswith("\n") else text
        if text.endswith("\r"):
            text = text[:-1]
        # rg prints paths relative to cwd (the repo root); normalise the leading "./".
        return GrepMatchRecord(
            path=Path(path_text).as_posix(),
            line_number=line_number,
            line_text=_truncate_line(text),
        )

    def _resolve_confined(self, path: str) -> Path:
        """Resolve ``path`` within the repo root (shared canonical check); raise on escape."""
        return confine_path(self._root, path)

    def read(self, path: str, offset: int, limit: int) -> FileContent:
        """Return a 1-based ``[offset, offset+limit)`` window of ``path`` as ``cat -n`` numbered lines.

        Confines the path and rejects non-files and binary files (sniffing only
        the file head). Streams the file line-by-line — constant memory rather
        than loading and decoding the whole file — so a small window over a
        multi-megabyte file is cheap. Windowing/numbering/caps are shared with
        ``git_read_file`` via ``render_numbered_window``.
        """
        resolved = self._resolve_confined(path)
        if not resolved.is_file():
            raise ValueError(f"not a file: {path!r}")
        rel = resolved.relative_to(self._root).as_posix()

        # Sniff only the head for a NUL — avoid reading the whole file to classify it.
        with resolved.open("rb") as fh:
            if _looks_binary(fh.read(BINARY_SNIFF_BYTES)):
                raise ValueError(f"binary file: {path!r}; read_file only supports text files")

        # ``newline="\n"`` splits on ``\n`` only (matching ripgrep's line counting);
        # the per-line CR/LF terminator is stripped by the renderer.
        with resolved.open("r", encoding="utf-8", errors="replace", newline="\n") as fh:
            return render_numbered_window(fh, path=rel, offset=offset, limit=limit)


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
