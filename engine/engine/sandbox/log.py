from __future__ import annotations

import logging
import sys

_logger = logging.getLogger(__name__)


def log_unavailable(*, diagnostic: str, remediation: str) -> None:
    """Emit the sandbox-unavailable warning to ``logging`` and ``stderr``.

    Called by client resolvers when they cannot return a working client. The
    warning is intentionally visible in every common deployment surface
    (CLI, library import, container logs) so operators see why ``run_code``
    is missing.
    """
    warning = (
        "HALO run_code disabled: sandbox unavailable.\n\n"
        f"Reason:\n  {diagnostic}\n\n"
        f"How to fix:\n  {remediation}\n\n"
        "The engine will continue without exposing run_code to the agent."
    )
    _logger.warning(warning)
    print(warning, file=sys.stderr, flush=True)
