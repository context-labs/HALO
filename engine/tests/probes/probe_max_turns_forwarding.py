"""Probe: AgentConfig.maximum_turns forwarding to Runner.run_streamed.

Pathways probed:
  1. Default ``maximum_turns=4`` should be forwarded as ``max_turns=4`` to
     ``Runner.run_streamed``. The engine's documented contract.
  2. A custom ``maximum_turns`` value (e.g. 1) should be forwarded as well.
  3. ``runner.calls[0]`` records every kwarg the engine passed.

Why this matters: ``Runner.run_streamed`` uses ``max_turns`` to bound how many
LLM-tool-loop iterations the SDK runs internally. If the engine forgets to
forward it, the SDK falls back to its default (currently 10), silently
ignoring the user's ``AgentConfig.maximum_turns``.

Inspect: ``runner.calls[i]`` keys.
"""

from __future__ import annotations

import asyncio
import sys

from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig
from tests.probes.probe_kit import (
    FakeRunner,
    make_assistant_text,
    make_default_config,
    run_with_fake,
)

_FAILURES: list[str] = []


def _check(condition: bool, description: str, observed: str = "") -> None:
    if condition:
        print(f"PASS: {description}")
    else:
        suffix = f" — observed: {observed}" if observed else ""
        print(f"FAIL: {description}{suffix}")
        _FAILURES.append(description)


async def probe_default_maximum_turns_forwarded() -> None:
    """make_default_config sets maximum_turns=4. The engine should forward
    it as a max_turns kwarg into runner.run_streamed."""
    runner = FakeRunner(
        [make_assistant_text("ok\n<final/>", item_id="m1")],
    )
    cfg = make_default_config()
    result = await run_with_fake(runner, config=cfg)

    _check(result.error is None,
           "default-max-turns: completes without error",
           observed=f"error={type(result.error).__name__ if result.error else None}")
    _check(len(runner.calls) == 1,
           "default-max-turns: runner called exactly once",
           observed=f"calls={len(runner.calls)}")
    if runner.calls:
        kwargs = runner.calls[0]
        _check("max_turns" in kwargs,
               "default-max-turns: max_turns kwarg present in run_streamed call",
               observed=f"kwargs={list(kwargs.keys())}")
        _check(kwargs.get("max_turns") == cfg.root_agent.maximum_turns,
               f"default-max-turns: max_turns == {cfg.root_agent.maximum_turns}",
               observed=f"max_turns={kwargs.get('max_turns')}")


async def probe_custom_maximum_turns_forwarded() -> None:
    """Custom maximum_turns=1 should be forwarded distinctly from the default."""
    agent = AgentConfig(
        name="root",
        instructions="Be brief.",
        model=ModelConfig(name="gpt-5.4-mini"),
        maximum_turns=1,
    )
    cfg = EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub"}),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        maximum_depth=0,
    )
    runner = FakeRunner(
        [make_assistant_text("done\n<final/>", item_id="m1")],
    )
    result = await run_with_fake(runner, config=cfg)

    _check(result.error is None,
           "custom-max-turns: completes without error",
           observed=f"error={type(result.error).__name__ if result.error else None}")
    if runner.calls:
        kwargs = runner.calls[0]
        _check(kwargs.get("max_turns") == 1,
               "custom-max-turns: max_turns == 1 forwarded",
               observed=f"max_turns={kwargs.get('max_turns')!r} keys={list(kwargs.keys())}")


async def main() -> int:
    await probe_default_maximum_turns_forwarded()
    await probe_custom_maximum_turns_forwarded()

    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) failed:")
        for desc in _FAILURES:
            print(f"  - {desc}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
