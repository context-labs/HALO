from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from engine.models.engine_output import AgentOutputItem, AgentTextDelta, EngineStreamEvent


class _BusSignal:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error


class EngineOutputBus:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[EngineStreamEvent | _BusSignal] = asyncio.Queue()
        self._next_sequence = 0
        self._lock = asyncio.Lock()

    async def emit(self, item: EngineStreamEvent) -> EngineStreamEvent:
        async with self._lock:
            sequenced = item.model_copy(update={"sequence": self._next_sequence})
            self._next_sequence += 1
        await self._queue.put(sequenced)
        return sequenced

    async def close(self) -> None:
        await self._queue.put(_BusSignal())

    async def fail(self, error: BaseException) -> None:
        await self._queue.put(_BusSignal(error=error))

    async def stream(self) -> AsyncIterator[EngineStreamEvent]:
        while True:
            event = await self._queue.get()
            if isinstance(event, _BusSignal):
                if event.error is not None:
                    raise event.error
                return
            yield event
