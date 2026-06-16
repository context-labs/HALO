from __future__ import annotations

from collections.abc import Iterable

from engine.code._limits import LINE_CHAR_CAP, RESPONSE_CHAR_BUDGET
from engine.code.models import FileContent


def render_numbered_window(
    lines: Iterable[str],
    *,
    path: str,
    offset: int,
    limit: int,
) -> FileContent:
    """Render a 1-based ``[offset, offset+limit)`` window of ``lines`` as ``cat -n`` numbered text.

    ``lines`` is any iterable of (possibly newline-terminated) lines — a streamed
    file handle (``read_file``) or a subprocess's stdout (``git_read_file``); it is
    consumed once, so memory stays bounded to the window. Line numbering is
    ``\\n``-based, matching ripgrep, so file reads and ``grep_files`` agree on line
    numbers. Each line is capped at ``LINE_CHAR_CAP`` and total output at
    ``RESPONSE_CHAR_BUDGET``; ``truncated`` flags either clip. ``truncated``
    means output was clipped *within* the requested window — it does NOT flag a
    window that simply doesn't span the whole file (the caller sees that from
    ``total_line_count`` vs ``end_line``). ``start_line``/``end_line`` are ``0``
    when the window is empty.
    """
    end_exclusive = offset + limit
    rendered: list[str] = []
    truncated = False
    budget_spent = False
    used_chars = 0
    start_line = 0
    end_line = 0
    total_line_count = 0
    for line_number, raw_line in enumerate(lines, start=1):
        total_line_count = line_number
        if not (offset <= line_number < end_exclusive):
            continue
        if budget_spent:
            # The window's char budget is gone; keep iterating only to finish
            # counting total_line_count. (A per-line cap does NOT stop here — the
            # capped line is still emitted and shorter later lines can follow.)
            continue
        line = raw_line[:-1] if raw_line.endswith("\n") else raw_line
        if line.endswith("\r"):
            line = line[:-1]
        if len(line) > LINE_CHAR_CAP:
            line = f"{line[:LINE_CHAR_CAP]}... [HALO truncated: original {len(line)} chars]"
            truncated = True
        entry = f"{line_number:6d}\t{line}"
        if used_chars + len(entry) > RESPONSE_CHAR_BUDGET:
            truncated = True
            budget_spent = True
            continue
        rendered.append(entry)
        used_chars += len(entry) + 1
        if start_line == 0:
            start_line = line_number
        end_line = line_number

    return FileContent(
        path=path,
        content="\n".join(rendered),
        start_line=start_line,
        end_line=end_line,
        total_line_count=total_line_count,
        truncated=truncated,
    )
