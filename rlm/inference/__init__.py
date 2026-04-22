"""Agent harness that answers natural-language questions about a trace dataset.

The harness wraps an LLM with a toolkit for interrogating the index + raw JSONL:
overview stats, filtered sampling, deep-dive on a single trace, content search,
and RLM-style recursive synthesis over a sub-population. It's descriptor-driven,
so every dataset registered in ``catalog/`` is supported out of the box.
"""

from __future__ import annotations

from inference.config import InferenceConfig
from inference.harness import AgentEvent, build_system_prompt, run_agent

__all__ = ["AgentEvent", "InferenceConfig", "build_system_prompt", "run_agent"]
