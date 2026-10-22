"""Buffered audit delivery: a bounded queue drained into the real sender by one background worker.

A record is dropped with a warning when the queue is full; close() drains the queue first.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from indykite.intent_agent.audit.message import AuditMessage

if TYPE_CHECKING:
    from indykite.intent_agent.audit.sender import Sender

logger = logging.getLogger(__name__)

_STOP = object()


class AsyncSender:
    """Wraps another sender behind a bounded asyncio.Queue and a single worker task.

    The worker starts on the first send(), since it needs a running event loop.
    """

    def __init__(self, inner: "Sender", *, buffer_size: int) -> None:
        """Wrap `inner` behind a queue of `buffer_size` records."""
        self._inner = inner
        self._queue: asyncio.Queue[AuditMessage | object] = asyncio.Queue(maxsize=buffer_size)
        self._worker: asyncio.Task[None] | None = None
        self._closing = False

    @property
    def buffer_size(self) -> int:
        """Capacity of the queue, for diagnostics."""
        return self._queue.maxsize

    async def send(self, message: AuditMessage) -> None:
        """Enqueue without waiting; drop with a warning if the queue is full or the sender is closing."""
        if self._closing:
            logger.warning("audit: dropping payload while sender is stopping")
            return
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("audit: dropping payload, async queue is full", extra={"buffer_size": self.buffer_size})

    async def _run(self) -> None:
        """Drain the queue into the inner sender until the stop marker arrives."""
        while True:
            item = await self._queue.get()
            if not isinstance(item, AuditMessage):
                return
            try:
                await self._inner.send(item)
            except Exception:  # skipcq: PYL-W0703 - one bad record must not kill the worker; logged with traceback
                logger.exception("audit: async delivery failed")

    async def close(self) -> None:
        """Stop accepting records, let the worker drain what's queued, then close the inner sender."""
        self._closing = True
        if self._worker is not None:
            await self._queue.put(_STOP)
            await self._worker
            self._worker = None
        await self._inner.close()
