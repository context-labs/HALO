"""FastAPI server + static UI for the halo RLM agent.

The server exposes two endpoints:

* ``GET /api/overview`` — lightweight dataset summary used to render the header.
* ``GET /api/ask?question=...`` — server-sent-event stream of ``AgentEvent``s.

The static UI in ``web/static`` consumes ``/api/ask`` as an EventSource and
renders a live view of the agent's tool calls, results, and final answer.
"""

from __future__ import annotations

from web.server import build_app, run

__all__ = ["build_app", "run"]
