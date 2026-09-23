"""Redis Streams-backed A2A QueueManager (`RedisQueueManager`), env-scoped like `task_store.py`.

Currently wired but inert: `DefaultRequestHandlerV2` never calls `queue_manager`.
"""

import asyncio
import contextlib
import logging
from collections import deque
from types import TracebackType
from typing import Self
from uuid import uuid4

from a2a.server.events import Event, EventQueue, EventQueueLegacy, NoTaskQueue, QueueManager
from a2a.types.a2a_pb2 import Message, Task, TaskArtifactUpdateEvent, TaskStatusUpdateEvent
from a2a.utils._async_queue_compat import QueueShutDown
from redis.asyncio import Redis
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)

# How long a lagging tap()/get() reader has to finish draining a closed queue before its
# stream is removed for good - an open "grace period length" decision, so constructor-
# overridable rather than a magic number buried in close().
_DEFAULT_CLOSE_GRACE_PERIOD_SECONDS = 300

# Fallback for a queue/manager built directly, without a caller-supplied dequeue block duration.
_DEFAULT_DEQUEUE_BLOCK_MILLISECONDS = 1000

# Fallback without a caller-supplied reclaim idle threshold - see dequeue_event()'s
# _reclaim_pending_entry docstring for what this bounds.
_DEFAULT_RECLAIM_MIN_IDLE_MILLISECONDS = 30_000

# The producer-side group every create_or_tap() first call creates - get() resolves to this
# one, mirroring InMemoryQueueManager.get() returning the stored object, not a tap of it.
_PRODUCER_GROUP = "producer"

_QUEUE_CLOSED_MESSAGE = "Queue is closed."

# Redis's error prefix for "XGROUP CREATE lost a race - group already exists" (see
# create_or_tap) - distinguishes that expected race from any other ResponseError, which must
# still propagate.
_BUSYGROUP_ERROR_PREFIX = "BUSYGROUP"

_EVENT_TYPES: dict[str, type[Event]] = {
    cls.__name__: cls for cls in (Message, Task, TaskStatusUpdateEvent, TaskArtifactUpdateEvent)
}


def _decode(value: bytes | str) -> str:
    """Normalize one Streams field/id to str, regardless of whether the wrapped redis.asyncio
    client was built with decode_responses - mirrors storage.py's _decode_member.
    """
    return value.decode() if isinstance(value, bytes) else value


def _encode_event(event: Event) -> dict[str, bytes]:
    """Encode one Event as XADD fields: its concrete type name (to pick the right class back
    apart on read) plus its serialized protobuf bytes.
    """
    return {"type": type(event).__name__.encode(), "data": event.SerializeToString()}


def _decode_event(fields: dict[bytes | str, bytes | str]) -> Event:
    """Reverse _encode_event: look up the concrete Event subclass by name and parse it back."""
    normalized = {_decode(key): value for key, value in fields.items()}
    event_cls = _EVENT_TYPES[_decode(normalized["type"])]
    data = normalized["data"]
    return event_cls.FromString(data.encode() if isinstance(data, str) else data)


class RedisEventQueue(EventQueue):
    """An EventQueueLegacy-shaped view onto one Redis Streams consumer group, backed by
    XADD/XREADGROUP/XACK instead of an in-process asyncio.Queue.

    Subclasses EventQueue (the ABC, just enqueue_event), not EventQueueLegacy - its methods are
    all implemented against a private asyncio.Queue/_children list a Redis backing would
    override entirely anyway. Matches its public shape structurally instead, which is what
    QueueManager/EventConsumer actually rely on.
    """

    def __init__(  # noqa: PLR0913 - each knob is independently meaningful and caller-configurable.
        self,
        client: Redis,
        stream_key: str,
        group_name: str,
        *,
        consumer_name: str | None = None,
        dequeue_block_milliseconds: int = _DEFAULT_DEQUEUE_BLOCK_MILLISECONDS,
        reclaim_min_idle_milliseconds: int = _DEFAULT_RECLAIM_MIN_IDLE_MILLISECONDS,
    ) -> None:
        """Bind to one (stream_key, group_name) pair; no Redis call is made until first use.

        consumer_name is this reader's identity *within* group_name - defaults to a fresh uuid4
        so two readers of the same group are never merged into one (breaks XPENDING/XCLAIM).
        """
        self._client = client
        self._stream_key = stream_key
        self._group_name = group_name
        self._consumer_name = consumer_name if consumer_name is not None else f"consumer-{uuid4()}"
        self._dequeue_block_milliseconds = dequeue_block_milliseconds
        self._reclaim_min_idle_milliseconds = reclaim_min_idle_milliseconds
        self._closed = False
        # FIFO of dequeued-but-not-yet-acked ids (review: a single scalar loses every id but
        # the latest once two dequeues happen before either is acked) - task_done() pops oldest.
        self._pending_acks: deque[str] = deque()
        self._background_acks: set[asyncio.Task[int]] = set()

    async def __aenter__(self) -> Self:
        """Enters the async context manager, returning the queue itself."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exits the async context manager, ensuring close() is called."""
        await self.close()

    async def enqueue_event(self, event: Event) -> None:
        """XADD event onto the stream - a no-op (with a warning) once this queue is closed,
        matching EventQueueLegacy.enqueue_event's own closed-queue behavior.
        """
        if self._closed:
            logger.warning("Queue is closed. Event will not be enqueued.")
            return
        await self._client.xadd(self._stream_key, _encode_event(event))

    async def dequeue_event(self) -> Event:
        """Block (up to self._dequeue_block_milliseconds at a time) for the next new entry on
        this consumer group, or raise QueueShutDown once closed - matching EventQueueLegacy's
        own contract.
        """
        if self._closed:
            raise QueueShutDown(_QUEUE_CLOSED_MESSAGE)
        reclaimed = await self._reclaim_pending_entry()
        if reclaimed is not None:
            entry_id, fields = reclaimed
            self._pending_acks.append(entry_id)
            return _decode_event(fields)
        response = await self._client.xreadgroup(
            self._group_name,
            self._consumer_name,
            {self._stream_key: ">"},
            count=1,
            # BLOCK 0 means wait forever to Redis, not "don't wait" - floor at 1ms so a
            # sub-millisecond setting can't truncate into an indefinite block.
            block=max(self._dequeue_block_milliseconds, 1),
        )
        if not response:
            if self._closed:
                raise QueueShutDown(_QUEUE_CLOSED_MESSAGE)
            raise asyncio.QueueEmpty
        [[_stream_key, entries]] = response
        entry_id, fields = entries[0]
        self._pending_acks.append(_decode(entry_id))
        return _decode_event(fields)

    async def _reclaim_pending_entry(self) -> tuple[str, dict[bytes | str, bytes | str]] | None:
        """XAUTOCLAIM one entry idle >= reclaim_min_idle_milliseconds - recovers a never-acked
        entry from a crashed consumer without stealing a live one's still-in-flight work.
        """
        _next_cursor, claimed, _deleted = await self._client.xautoclaim(
            self._stream_key,
            self._group_name,
            self._consumer_name,
            min_idle_time=self._reclaim_min_idle_milliseconds,
            start_id="0-0",
            count=1,
        )
        if not claimed:
            return None
        entry_id, fields = claimed[0]
        return _decode(entry_id), fields

    def task_done(self) -> None:
        """Acknowledge (XACK) the oldest not-yet-acked dequeued entry, as a background task -
        matches asyncio.Queue.task_done()'s one-call-per-get() contract, oldest-first, rather
        than assuming the most recent dequeue_event() is the one being marked done. Best-effort:
        one still pending at process exit is only recovered later, by _reclaim_pending_entry.
        """
        if not self._pending_acks:
            return
        entry_id = self._pending_acks.popleft()
        task = asyncio.create_task(self._client.xack(self._stream_key, self._group_name, entry_id))
        self._background_acks.add(task)
        task.add_done_callback(self._background_acks.discard)

    async def close(self, immediate: bool = False) -> None:  # noqa: ARG002, FBT001, FBT002 - must match EventQueueLegacy.close's positional-bool call sites
        """Wait for any in-flight task_done() ack, then destroy this queue's own consumer
        group - destroying it first would discard a still-pending entry, never redelivered.
        """
        self._closed = True
        if self._background_acks:
            await asyncio.gather(*self._background_acks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await self._client.xgroup_destroy(self._stream_key, self._group_name)

    def is_closed(self) -> bool:
        """Checks if the queue is closed."""
        return self._closed


class RedisQueueManager(QueueManager):
    """Redis Streams-backed QueueManager, scoped by a caller-supplied env_id only."""

    def __init__(  # noqa: PLR0913 - each knob is independently meaningful and caller-configurable.
        self,
        client: Redis,
        *,
        env_id: str,
        close_grace_period_seconds: float = _DEFAULT_CLOSE_GRACE_PERIOD_SECONDS,
        dequeue_block_milliseconds: int = _DEFAULT_DEQUEUE_BLOCK_MILLISECONDS,
        reclaim_min_idle_milliseconds: int = _DEFAULT_RECLAIM_MIN_IDLE_MILLISECONDS,
        key_prefix: str = "ikg",
    ) -> None:
        """Store the client/env_id/grace period; no Redis call is made until first use.

        key_prefix namespaces every key this manager writes (default "ikg", matching this
        repo's own usage) - override it if a different consumer shares the same Redis.
        """
        self._client = client
        self._env_id = env_id
        self._close_grace_period_seconds = close_grace_period_seconds
        self._dequeue_block_milliseconds = dequeue_block_milliseconds
        self._reclaim_min_idle_milliseconds = reclaim_min_idle_milliseconds
        self._key_prefix = key_prefix

    def _stream_key(self, task_id: str) -> str:
        """`<key_prefix>_queue:{<env_id>}:<task_id>` - the `{...}` is a literal Redis Cluster
        hash-tag, matching task_store.py's key format for the same env-only scoping.
        """
        return f"{self._key_prefix}_queue:{{{self._env_id}}}:{task_id}"

    async def _exists(self, task_id: str) -> bool:
        """Whether this task_id's stream exists at all - tap()'s own existence check; a tap's
        throwaway group is independent of whatever else lives on the stream.
        """
        return bool(await self._client.exists(self._stream_key(task_id)))

    async def _producer_group_exists(self, task_id: str) -> bool:
        """Whether task_id's stream AND its producer group both still exist - unlike _exists()
        alone, this catches a stream that outlived a prior EventQueue.close() destroying its
        producer group (EXPIRE's grace period hasn't elapsed yet) - see get()/create_or_tap().
        """
        try:
            groups = await self._client.xinfo_groups(self._stream_key(task_id))
        except ResponseError:
            return False
        return any(_decode(group["name"]) == _PRODUCER_GROUP for group in groups)

    async def _new_tap(self, task_id: str) -> RedisEventQueue:
        """Create a fresh consumer group positioned at the stream's current end ($) so this
        caller only sees events from this point forward (EventQueueLegacy.tap()'s own contract).
        The SDK's one caller always sends the task's full state from task_store separately first.
        """
        group_name = f"tap-{uuid4()}"
        await self._client.xgroup_create(self._stream_key(task_id), group_name, id="$")
        return RedisEventQueue(
            self._client,
            self._stream_key(task_id),
            group_name,
            dequeue_block_milliseconds=self._dequeue_block_milliseconds,
            reclaim_min_idle_milliseconds=self._reclaim_min_idle_milliseconds,
        )

    async def add(self, _task_id: str, _queue: EventQueueLegacy) -> None:
        """Unsupported: can't route an already-constructed queue's methods through Redis.

        Raises:
            NotImplementedError: always.
        """
        msg = "RedisQueueManager.add() is not supported"
        raise NotImplementedError(msg)

    async def get(self, task_id: str) -> EventQueueLegacy | None:
        """Return a queue bound to task_id's producer group, or None if no usable queue exists -
        including when the stream outlived a prior close() that already destroyed the group.
        """
        if not await self._producer_group_exists(task_id):
            return None
        return RedisEventQueue(
            self._client,
            self._stream_key(task_id),
            _PRODUCER_GROUP,
            dequeue_block_milliseconds=self._dequeue_block_milliseconds,
            reclaim_min_idle_milliseconds=self._reclaim_min_idle_milliseconds,
        )

    async def tap(self, task_id: str) -> EventQueueLegacy | None:
        """Tap an existing queue, or None if task_id has no queue at all yet."""
        if not await self._exists(task_id):
            return None
        return await self._new_tap(task_id)

    async def close(self, task_id: str) -> None:
        """Retire task_id's stream on a grace period (EXPIRE, not an immediate DEL) so a
        lagging tap()/get() reader has close_grace_period_seconds left to finish draining it.

        Raises:
            NoTaskQueue: if task_id has no queue at all.
        """
        if not await self._exists(task_id):
            raise NoTaskQueue
        # EXPIRE takes whole seconds - floor at 1 so a sub-second grace period doesn't round
        # down to an immediate expiry.
        await self._client.expire(self._stream_key(task_id), max(round(self._close_grace_period_seconds), 1))

    async def create_or_tap(self, task_id: str) -> EventQueueLegacy:
        """Create task_id's producer queue if it doesn't exist yet - including if its stream
        outlived a prior close() that already destroyed the producer group - otherwise tap the
        existing one.

        _producer_group_exists()/xgroup_create() below aren't atomic: two replicas can race to
        create the same task_id. The loser gets BUSYGROUP, caught here and treated as
        already-exists. Any other ResponseError still propagates.
        """
        if await self._producer_group_exists(task_id):
            return await self._new_tap(task_id)
        try:
            await self._client.xgroup_create(self._stream_key(task_id), _PRODUCER_GROUP, id="$", mkstream=True)
        except ResponseError as exc:
            if not str(exc).startswith(_BUSYGROUP_ERROR_PREFIX):
                raise
            return await self._new_tap(task_id)
        return RedisEventQueue(
            self._client,
            self._stream_key(task_id),
            _PRODUCER_GROUP,
            dequeue_block_milliseconds=self._dequeue_block_milliseconds,
            reclaim_min_idle_milliseconds=self._reclaim_min_idle_milliseconds,
        )
