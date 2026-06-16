from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from engine.code._limits import READ_LIMIT_DEFAULT, READ_LIMIT_MAX, READ_OFFSET_DEFAULT


class GlobFilesArguments(BaseModel):
    """Tool arguments for ``glob_files``: a glob pattern relative to the repo root plus a result cap."""

    model_config = ConfigDict(extra="forbid")

    pattern: str
    max_results: int = Field(default=200, ge=1, le=1000)


class GlobFileEntry(BaseModel):
    """One matched file: repo-root-relative POSIX path plus byte size.

    ``size_bytes`` lets the model decide whether to read a file whole or page
    through it with ``read_file``'s ``offset``/``limit``, the same way
    ``raw_jsonl_bytes`` sizes traces before ``view_trace``.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int = Field(ge=0)


class GlobMatches(BaseModel):
    """Bounded list of files matching a glob.

    ``returned_count`` is how many entries are in ``files`` (capped by
    ``max_results``); ``has_more`` is true when more files matched than were
    returned — narrow the pattern rather than raising ``max_results``.
    """

    model_config = ConfigDict(extra="forbid")

    files: list[GlobFileEntry]
    returned_count: int = Field(ge=0)
    has_more: bool


class GlobFilesResult(BaseModel):
    """Result envelope for ``glob_files`` — wraps a GlobMatches under ``result``."""

    model_config = ConfigDict(extra="forbid")

    result: GlobMatches


class GrepFilesArguments(BaseModel):
    """Tool arguments for ``grep_files``: a regex over file contents, an optional file filter, and a match cap.

    ``regex_pattern`` is a regex compiled internally; invalid patterns fail fast.
    ``glob_pattern`` confines the search to files matching that glob (relative to
    the repo root). ``max_matches`` caps the number of returned ``GrepMatchRecord``s.
    """

    model_config = ConfigDict(extra="forbid")

    regex_pattern: str
    glob_pattern: str | None = None
    max_matches: int = Field(default=50, ge=1, le=500)


class GrepMatchRecord(BaseModel):
    """One regex match: the file it was found in, the 1-based line number, and the (truncated) line text."""

    model_config = ConfigDict(extra="forbid")

    path: str
    line_number: int = Field(ge=1)
    line_text: str


class GrepMatches(BaseModel):
    """Bounded regex match records across the repo.

    ``returned_match_count`` is how many records are in ``matches`` (capped by
    ``max_matches``); ``has_more`` is true when more matches existed than were
    returned — refine the regex rather than raising ``max_matches``.
    """

    model_config = ConfigDict(extra="forbid")

    matches: list[GrepMatchRecord]
    returned_match_count: int = Field(ge=0)
    has_more: bool


class GrepFilesResult(BaseModel):
    """Result envelope for ``grep_files`` — wraps a GrepMatches under ``result``."""

    model_config = ConfigDict(extra="forbid")

    result: GrepMatches


class ReadFileArguments(BaseModel):
    """Tool arguments for ``read_file``: a repo-relative path plus a 1-based line window.

    ``offset`` is the 1-based first line to read; ``limit`` caps how many lines
    are returned. Use ``glob_files``/``grep_files`` to discover paths and line
    numbers, then read a window around them.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    offset: int = Field(default=READ_OFFSET_DEFAULT, ge=1)
    limit: int = Field(default=READ_LIMIT_DEFAULT, ge=1, le=READ_LIMIT_MAX)


class FileContent(BaseModel):
    """A window of a file's contents as ``cat -n`` numbered lines.

    ``start_line``/``end_line`` are the 1-based inclusive bounds of the returned
    window, and are both ``0`` when the window is empty (an empty file, or an
    ``offset`` past the last line). ``total_line_count`` is the file's full
    length so the caller can page with ``offset``. ``truncated`` is true when a
    per-line cap or the per-call response budget clipped output — request a
    narrower window or grep for what you need.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    content: str
    start_line: int = Field(ge=0)
    end_line: int = Field(ge=0)
    total_line_count: int = Field(ge=0)
    truncated: bool


class ReadFileResult(BaseModel):
    """Result envelope for ``read_file`` — wraps a FileContent under ``result``."""

    model_config = ConfigDict(extra="forbid")

    result: FileContent


class ViewRepoTreeArguments(BaseModel):
    """Tool arguments for ``view_repo_tree``: none — returns the whole-repo, depth/entry-capped tree."""

    model_config = ConfigDict(extra="forbid")


class RepoTree(BaseModel):
    """A directory-tree overview of the repository, rooted at ``root``.

    ``tree`` is depth/entry-capped, with explicit markers where a cap clipped it
    — fall back to ``glob_files`` for anything not shown.
    """

    model_config = ConfigDict(extra="forbid")

    root: str
    tree: str


class ViewRepoTreeResult(BaseModel):
    """Result envelope for ``view_repo_tree`` — wraps a RepoTree under ``result``."""

    model_config = ConfigDict(extra="forbid")

    result: RepoTree
