"""Shared numeric bounds for the read/grep file tools and the git history tools.

A dependency-free leaf so both ``engine.code`` and ``engine.git`` models and
helpers import a single source of truth (``engine.code.models`` can't live below
``engine.code._textwindow``, which imports it — hence a standalone module).
"""

from __future__ import annotations

# Read-window bounds shared by the code and git ``read_file`` tools (argument
# defaults/caps) and the numbered-window renderer.
READ_OFFSET_DEFAULT = 1
READ_LIMIT_DEFAULT = 500
READ_LIMIT_MAX = 2000

# Per-line character cap and per-call response character budget applied when
# rendering file windows (read_file / git_read_file) and git diff/show bodies.
LINE_CHAR_CAP = 2000
RESPONSE_CHAR_BUDGET = 150_000

# Leading bytes/chars sniffed for a NUL when classifying a file as binary —
# shared by read_file (working-tree bytes) and git_read_file (git-show stream).
BINARY_SNIFF_BYTES = 8192
