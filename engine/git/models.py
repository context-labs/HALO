from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine.code._limits import READ_LIMIT_DEFAULT, READ_LIMIT_MAX, READ_OFFSET_DEFAULT
from engine.code.models import FileContent


class CommitSummary(BaseModel):
    """One commit's metadata, as returned by ``git_log`` and ``git_show``."""

    model_config = ConfigDict(extra="forbid")

    full_sha: str
    short_sha: str
    author: str
    authored_at: str  # canonical ISO-8601 (zero offset always ``+00:00``, never ``Z``)
    subject: str


class BlameLine(BaseModel):
    """One blamed line: the commit/author that last changed it, plus the line text."""

    model_config = ConfigDict(extra="forbid")

    line_number: int = Field(ge=1)
    short_sha: str
    author: str
    summary: str  # subject of the commit that last touched the line
    line_text: str


# --- git_log -----------------------------------------------------------------


class GitLogArguments(BaseModel):
    """Tool arguments for ``git_log``: bounded commit listing with optional filters.

    ``since``/``until`` window by date (ISO-8601 like a trace timestamp, or git
    relative forms). ``pickaxe_string`` (``-S``) finds commits that changed the
    number of occurrences of an exact string; ``pickaxe_regex`` (``-G``) matches
    diff content by POSIX extended regex — at most one. ``ref_range`` is a ref or
    ``A..B`` range; ``path`` confines to a file/dir.
    """

    model_config = ConfigDict(extra="forbid")

    max_commits: int = Field(default=50, ge=1, le=500)
    since: str | None = None
    until: str | None = None
    ref_range: str | None = None
    path: str | None = None
    pickaxe_string: str | None = None
    pickaxe_regex: str | None = None

    @model_validator(mode="after")
    def _one_pickaxe(self) -> "GitLogArguments":
        if self.pickaxe_string is not None and self.pickaxe_regex is not None:
            raise ValueError("pass at most one of pickaxe_string / pickaxe_regex")
        return self


class GitLog(BaseModel):
    """Bounded list of commits, newest first."""

    model_config = ConfigDict(extra="forbid")

    commits: list[CommitSummary]
    returned_count: int = Field(ge=0)
    has_more: bool


class GitLogResult(BaseModel):
    """Result envelope for ``git_log``."""

    model_config = ConfigDict(extra="forbid")

    result: GitLog


# --- git_show ----------------------------------------------------------------


class GitShowArguments(BaseModel):
    """Tool arguments for ``git_show``: inspect one commit (stat by default, optional patch)."""

    model_config = ConfigDict(extra="forbid")

    ref: str
    path: str | None = None
    include_patch: bool = False


class GitShow(BaseModel):
    """A commit's metadata plus its ``--stat`` summary or (size-capped) patch body."""

    model_config = ConfigDict(extra="forbid")

    commit: CommitSummary
    body: str
    truncated: bool


class GitShowResult(BaseModel):
    """Result envelope for ``git_show``."""

    model_config = ConfigDict(extra="forbid")

    result: GitShow


# --- git_diff ----------------------------------------------------------------


class GitDiffArguments(BaseModel):
    """Tool arguments for ``git_diff``: compare two refs (stat by default, optional patch)."""

    model_config = ConfigDict(extra="forbid")

    from_ref: str
    to_ref: str
    path: str | None = None
    stat_only: bool = True


class GitDiff(BaseModel):
    """The ``--stat`` summary or (size-capped) patch between two refs."""

    model_config = ConfigDict(extra="forbid")

    diff: str
    stat_only: bool
    truncated: bool


class GitDiffResult(BaseModel):
    """Result envelope for ``git_diff``."""

    model_config = ConfigDict(extra="forbid")

    result: GitDiff


# --- git_blame ---------------------------------------------------------------


class GitBlameArguments(BaseModel):
    """Tool arguments for ``git_blame``: attribute a file's line range to commits."""

    model_config = ConfigDict(extra="forbid")

    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    ref: str | None = None  # defaults to the current checkout / HEAD

    @model_validator(mode="after")
    def _ordered(self) -> "GitBlameArguments":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class GitBlame(BaseModel):
    """Per-line blame records for the requested window."""

    model_config = ConfigDict(extra="forbid")

    path: str
    lines: list[BlameLine]
    returned_count: int = Field(ge=0)
    truncated: bool  # true when the window was clamped to the line cap


class GitBlameResult(BaseModel):
    """Result envelope for ``git_blame``."""

    model_config = ConfigDict(extra="forbid")

    result: GitBlame


# --- git_read_file -----------------------------------------------------------


class GitReadFileArguments(BaseModel):
    """Tool arguments for ``git_read_file``: read a file's contents as of a commit."""

    model_config = ConfigDict(extra="forbid")

    ref: str
    path: str
    offset: int = Field(default=READ_OFFSET_DEFAULT, ge=1)
    limit: int = Field(default=READ_LIMIT_DEFAULT, ge=1, le=READ_LIMIT_MAX)


class GitReadFileResult(BaseModel):
    """Result envelope for ``git_read_file`` — wraps the shared FileContent."""

    model_config = ConfigDict(extra="forbid")

    result: FileContent
