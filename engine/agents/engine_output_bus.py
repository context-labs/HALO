from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from engine.models.engine_output import EngineStreamEvent


class _BusSignal:
    """Sentinel pushed onto the bus queue to signal close (error=None) or fail (error set)."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error


class EngineOutputBus:
    """Single run-wide async queue that interleaves output from every agent in the run.

    Every agent (root, subagent, grandchild) emits to the same bus, so callers see a
    monotonically-sequenced stream of items even when subagents run in parallel. ``emit``
    assigns the sequence under a lock; the bus deliberately does not retain a replay
    buffer — collection is the caller's responsibility (see ``run_engine_async``).
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[EngineStreamEvent | _BusSignal] = asyncio.Queue()
        self._next_sequence = 0
        self._lock = asyncio.Lock()

    async def emit(self, item: EngineStreamEvent) -> EngineStreamEvent:
        """Assign the next sequence number under a lock and enqueue the item.

        Returns the sequenced copy so the caller can record start/end sequence ranges.
        """
        async with self._lock:
            sequenced = item.model_copy(update={"sequence": self._next_sequence})
            self._next_sequence += 1
        await self._queue.put(sequenced)
        return sequenced

    async def close(self) -> None:
        """Mark the run successful so ``stream()`` returns once drained."""
        await self._queue.put(_BusSignal())

    async def fail(self, error: BaseException) -> None:
        """Mark the run failed so ``stream()`` raises after yielding any already-emitted items."""
        await self._queue.put(_BusSignal(error=error))

    async def stream(self) -> AsyncIterator[EngineStreamEvent]:
        """Yield events in emission order until close (clean stop) or fail (raise)."""
        while True:
            event = await self._queue.get()
            if isinstance(event, _BusSignal):
                if event.error is not None:
                    raise event.error
                return
            yield event
