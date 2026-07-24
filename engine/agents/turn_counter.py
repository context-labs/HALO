"""Per-turn nudge injection for HALO agents.

Tells the agent which turn it is on and how many remain so it can
self-pace within its ``maximum_turns`` budget. Wired into the OpenAI
Agents SDK runner via ``RunConfig.call_model_input_filter``; one filter
instance is constructed per ``runner.run()`` call (i.e. per agent
execution) and increments its counter on each invocation.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.run_config import CallModelData, ModelInputData
from openai.types.responses import EasyInputMessageParam


def _render_nudge(*, current: int, maximum: int, is_root: bool) -> str:
    """Render the per-turn nudge text.

    ``remaining`` is INCLUSIVE of the current turn — at turn N of N,
    remaining is 1 ("this is the last turn"). At turn 1 of N, remaining
    is N.
    """
    remaining = max(0, maximum - current + 1)
    if remaining >= 4:
        return f"[HALO: turn {current} of {maximum}]"
    if remaining == 3:
        return f"[HALO: turn {current} of {maximum} — 3 turns left]"
    if remaining == 2:
        return f"[HALO: turn {current} of {maximum} — 2 turns left]"
    if remaining == 1:
        if is_root:
            return (
                f"[HALO: turn {current} of {maximum} — last turn. "
                f"Call the `final_answer` tool with your complete final answer now.]"
            )
        return (
            f"[HALO: turn {current} of {maximum} — last turn. "
            f"Produce your final concise answer now.]"
        )
    return f"[HALO: turn {current} of {maximum} — over budget]"


@dataclass
class TurnCounterInputFilter:
    """Append a per-turn nudge to the SDK-side input before each LLM call.

    Constructed once per ``runner.run()`` call. The SDK invokes
    ``__call__`` immediately before every model request inside its turn
    loop; the returned ``ModelInputData`` replaces what gets sent.

    The nudge is appended as a single new ``role="user"`` item at the
    tail of the input list. The caller's list is not mutated — we build
    a new list — so retries inside the SDK that re-read the prior input
    see no side effects from the filter.
    """

    max_turns: int
    is_root: bool
    _current: int = 0

    def __call__(self, data: CallModelData) -> ModelInputData:
        self._current += 1
        nudge: EasyInputMessageParam = {
            "role": "user",
            "content": _render_nudge(
                current=self._current,
                maximum=self.max_turns,
                is_root=self.is_root,
            ),
        }
        return ModelInputData(
            input=[*data.model_data.input, nudge],
            instructions=data.model_data.instructions,
        )
