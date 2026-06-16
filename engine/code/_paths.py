from __future__ import annotations

from pathlib import Path


def confine_path(root: Path, path: str) -> Path:
    """Resolve ``path`` (relative to ``root``, or an absolute path already inside it) within ``root``.

    ``.resolve()`` follows symlinks before the containment check, so a symlink
    pointing outside the repo is rejected. Raises ``ValueError`` with a
    model-actionable message on escape. Shared by the code and git repo views so
    there is a single canonical confinement check.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"path {path!r} resolves outside the repo root; pass a path relative to the repo root"
        )
    return resolved
