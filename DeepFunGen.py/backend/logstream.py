"""Async broker and logging bridge for UI streamed events."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator, Optional

from .models import LogEvent


class LogBroker:
    """Bounded queue for log events with drop-on-overflow semantics."""

    def __init__(self, max_events: int = 256) -> None:
        self._queue: asyncio.Queue[LogEvent] = asyncio.Queue(max_events)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Register the event loop that owns the queue for thread-safe publishing."""

        if loop is None:
            loop = asyncio.get_running_loop()
        self._loop = loop

    def _enqueue(self, event: LogEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                _ = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def publish(self, event: LogEvent) -> None:
        loop = self._loop
        if loop is None:
            try:
                self.bind_loop()
                loop = self._loop
            except RuntimeError:
                loop = None
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._enqueue, event)
        else:
            self._enqueue(event)

    async def next_event(self) -> LogEvent:
        return await self._queue.get()

    async def iter_events(self) -> AsyncIterator[LogEvent]:
        while True:
            yield await self.next_event()


class LogRelayHandler(logging.Handler):
    """Logging handler that mirrors records into the broker."""

    def __init__(self, broker: LogBroker) -> None:
        super().__init__()
        self.broker = broker

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - logging bridge
        try:
            message = self.format(record) if self.formatter else record.getMessage()
            event = LogEvent(
                timestamp=datetime.utcfromtimestamp(record.created),
                job_id=getattr(record, "job_id", None),
                job_name=getattr(record, "job_name", None),
                level=record.levelname,
                message=message,
            )
            self.broker.publish(event)
        except Exception:  # noqa: broad-except - logging must never raise
            self.handleError(record)


__all__ = ["LogBroker", "LogRelayHandler"]
