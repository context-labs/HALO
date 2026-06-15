from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Generator
from pathlib import Path

logger = logging.getLogger(__name__)

# Cap on how much of the subprocess's stderr we retain (for the error message);
# excess is still drained so the process never blocks, just not kept.
_DEFAULT_STDERR_CAP_CHARS = 64 * 1024


def stream_subprocess_lines(
    argv: list[str],
    *,
    cwd: Path,
    error_label: str,
    stderr_cap: int = _DEFAULT_STDERR_CAP_CHARS,
    error_returncode_floor: int = 2,
    env: dict[str, str] | None = None,
) -> Generator[str, None, None]:
    """Run ``argv`` in ``cwd`` and yield stdout lines, draining stderr concurrently.

    Streaming lets callers stop early (so memory stays bounded to what they keep)
    instead of buffering all output. stderr is drained in a thread so a chatty
    process (e.g. ripgrep file-open warnings, git diagnostics) can't fill its pipe
    buffer and deadlock against our stdout read; capture is capped at
    ``stderr_cap``, excess drained and discarded.

    ``encoding="utf-8", errors="replace"`` decodes output explicitly rather than
    via the process locale, so a non-UTF-8 locale or odd bytes can't raise
    ``UnicodeDecodeError``.

    If the process exits with a code ``>= error_returncode_floor`` and produced
    *no* output, raises ``ValueError(f"{error_label} failed: <stderr>")`` (surfaced
    to the model). If it errored only after yielding output, the partial results
    the caller already built are kept (a warning is logged) rather than discarded.
    If the caller stops early (breaks the loop / closes the generator), the
    process is terminated and no error is raised.

    ``error_returncode_floor`` is the lowest exit code treated as an error: ``2``
    for ripgrep (1 = "no matches", not an error), ``1`` for git (our read-only git
    commands return 0 on success and 128 on a bad ref, never passing
    ``--exit-code``/``--quiet`` that would make a non-zero a non-error).

    ``env`` replaces the child's environment entirely when given (``None``
    inherits the parent's). Callers pass it to strip ambient variables that
    would change which resource the command targets (e.g. git's ``GIT_DIR``).
    """
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    stderr_capture: list[str] = []

    def _drain_stderr() -> None:
        pipe = proc.stderr
        assert pipe is not None
        captured = 0
        for chunk in iter(lambda: pipe.read(8192), ""):
            if captured < stderr_cap:
                stderr_capture.append(chunk)
                captured += len(chunk)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()
    yielded = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            yielded = True
            yield line
        # Natural end — the caller consumed everything.
        returncode = proc.wait()
        stderr_thread.join()
        if returncode >= error_returncode_floor:
            stderr = "".join(stderr_capture).strip()
            if not yielded:
                raise ValueError(f"{error_label} failed: {stderr}")
            # Errored only after emitting output (rare): keep the partial results
            # the caller built rather than failing the whole tool.
            logger.warning(
                "%s exited %d after producing output; returning partial results: %s",
                error_label,
                returncode,
                stderr,
            )
    finally:
        # Runs on natural end, error, or GeneratorExit (caller stopped early).
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        stderr_thread.join()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
