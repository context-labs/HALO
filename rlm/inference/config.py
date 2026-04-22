"""Agent-harness configuration.

Controls *which models* drive the HALO RLM analyst and *how* it's allowed to
spend tool calls. The choice of *which dataset* to interrogate is now
descriptor-driven (see ``catalog/`` and ``registry.py``) — it's passed per
request rather than baked into this config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.hypers import Hypers

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class InferenceConfig(Hypers):
    """Configuration for the HALO RLM analyst.

    Attributes:
        default_dataset_id: Dataset used when the caller doesn't specify one
            (CLI ``--dataset`` or HTTP ``dataset_id``).
        index_dir: Directory where summary indexes are stored, resolved
            relative to the project root.
        model: LLM model driving the top-level agent loop.
        synth_model: LLM model used inside the ``synthesize`` tool. Can be
            the same as ``model`` or a smaller/cheaper model.
        max_turns: Cap on the number of tool-calling turns the agent may take.
        sample_cap: Hard cap on how many traces a single tool may return.
        synth_trace_cap: Hard cap on traces passed to ``synthesize`` at once.
        synth_chars_per_trace: Per-trace character budget inside ``synthesize``.
    """

    default_dataset_id: str = "grepfruit_pathrich"
    index_dir: Path = Path("data")
    model: str = "gpt-5.4"
    synth_model: str = "gpt-5.4-mini"
    max_turns: int = 16
    sample_cap: int = 50
    synth_trace_cap: int = 25
    synth_chars_per_trace: int = 6000
    # After each turn, keep only the last N tool-result messages in full
    # inside the conversation history sent to the LLM. Older tool
    # messages are compacted to a one-line summary + an ``r_N`` key that
    # the analyst can pass to ``inspect_result`` to pull the full body
    # back on demand. Prevents context/cost explosion on long multi-turn
    # conversations (the RLM-style context-offloading trick).
    compact_keep_recent: int = 6
    # Maximum recursion depth for ``ask_subagent``. The top-level analyst
    # runs at depth 0. At depth=0 the tool is available; sub-agents spawned
    # at depth=1 (i.e. have depth == max_depth) have the tool pruned from
    # their schema so they can't spawn further layers. Keep this at 1 for
    # latency; increase for research runs that want deep decomposition.
    max_depth: int = 1
    # Sub-agents get a lower turn cap than the top-level analyst so
    # decomposition doesn't blow up total latency.
    subagent_max_turns: int = 8

    def init(self) -> None:
        """Resolve the index directory relative to the project root."""
        if not self.index_dir.is_absolute():
            self.index_dir = (PROJECT_ROOT / self.index_dir).resolve()
        self.index_dir.mkdir(parents=True, exist_ok=True)
