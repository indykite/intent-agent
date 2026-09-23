"""Tests for RedisQueueManager/RedisEventQueue: create_or_tap/get/tap/close against the
QueueManager ABC's documented contract, and an enqueue/dequeue round trip over a real
consumer group - all against fakeredis, no live Redis needed.
"""

import asyncio

import pytest
from a2a.server.events import EventQueue, NoTaskQueue
from a2a.types.a2a_pb2 import Task
from a2a.utils._async_queue_compat import QueueShutDown
from fakeredis import FakeAsyncRedis
from redis.exceptions import ResponseError

from indykite.a2a.redis.queue_manager import _PRODUCER_GROUP, RedisEventQueue, RedisQueueManager

_ENV_ID = "env-1"
_GRACE_PERIOD_SECONDS = 120.0
_CUSTOM_DEQUEUE_BLOCK_MILLISECONDS = 250
_TWO_DISTINCT_READERS = 2


def _manager(**kwargs):
    """A fresh RedisQueueManager over a fakeredis client."""
    return RedisQueueManager(FakeAsyncRedis(), env_id=_ENV_ID, **kwargs)


def test_key_prefix_is_configurable_and_defaults_to_ikg():
    """key_prefix namespaces the stream key this manager writes - defaults to "ikg" but a
    different consumer sharing the same Redis must be able to override it."""

    async def _run():
        manager = _manager(key_prefix="other")

        await manager.create_or_tap("t1")

        assert await manager._client.exists("other_queue:{env-1}:t1")
        assert not await manager._client.exists("ikg_queue:{env-1}:t1")

    asyncio.run(_run())


def test_create_or_tap_creates_a_queue_for_a_new_task_id():
    """The first create_or_tap() for a task_id must create its queue - proven by get()
    afterwards finding it, where it wouldn't have before.
    """

    async def _run():
        manager = _manager()
        assert await manager.get("t1") is None

        queue = await manager.create_or_tap("t1")

        assert queue is not None
        assert await manager.get("t1") is not None

    asyncio.run(_run())


def test_create_or_tap_returns_an_event_queue_instance():
    """RedisEventQueue must be an EventQueue (the a2a SDK's ABC), not just duck-typed to it -
    see queue_manager.py's RedisEventQueue docstring for why it isn't EventQueueLegacy too.
    """

    async def _run():
        manager = _manager()
        queue = await manager.create_or_tap("t1")
        assert isinstance(queue, EventQueue)

    asyncio.run(_run())


def test_second_create_or_tap_taps_rather_than_erroring():
    """A second create_or_tap() on the same task_id must succeed (tapping the existing queue),
    not raise - unlike add(), which now always raises NotImplementedError (see
    test_add_always_raises_not_implemented).
    """

    async def _run():
        manager = _manager()
        await manager.create_or_tap("t1")

        second = await manager.create_or_tap("t1")

        assert second is not None

    asyncio.run(_run())


def test_create_or_tap_handles_a_busygroup_race_from_a_concurrent_creator(mocker):
    """The _producer_group_exists() check and xgroup_create() aren't atomic - two replicas can
    race to create the same brand-new task_id. The loser must still get a working tap, not an
    uncaught BUSYGROUP ResponseError. _producer_group_exists() is mocked on the "loser" to force
    it down the create branch after the "winner" has already created the group for real.
    """

    async def _run():
        client = FakeAsyncRedis()
        winner = RedisQueueManager(client, env_id=_ENV_ID)
        await winner.create_or_tap("t1")

        loser = RedisQueueManager(client, env_id=_ENV_ID)
        mocker.patch.object(loser, "_producer_group_exists", return_value=False)

        queue = await loser.create_or_tap("t1")

        assert queue is not None

    asyncio.run(_run())


def test_create_or_tap_still_raises_a_non_busygroup_response_error(mocker):
    """The BUSYGROUP handling above must not swallow every ResponseError - only the specific
    expected race. Any other Redis-reported failure on xgroup_create must still propagate.
    """

    async def _run():
        manager = _manager()
        mocker.patch.object(manager, "_producer_group_exists", return_value=False)
        mocker.patch.object(manager._client, "xgroup_create", side_effect=ResponseError("WRONGTYPE unrelated failure"))

        with pytest.raises(ResponseError, match="WRONGTYPE"):
            await manager.create_or_tap("t1")

    asyncio.run(_run())


def test_get_returns_none_for_a_nonexistent_task_id():
    """get() on a task_id with no queue at all must return None, not raise."""

    async def _run():
        manager = _manager()
        assert await manager.get("missing") is None

    asyncio.run(_run())


def test_get_assigns_each_queue_instance_its_own_redis_consumer_identity():
    """Two get() calls for the same task_id (simulating two replicas reading the same producer
    group) must not collapse into one Redis consumer - XINFO CONSUMERS would then be unable to
    attribute a pending entry to the reader that actually owns it.
    """

    async def _run():
        manager = _manager()
        await manager.create_or_tap("t1")
        reader_a = await manager.get("t1")
        reader_b = await manager.get("t1")

        assert reader_a._consumer_name != reader_b._consumer_name

        await reader_a.enqueue_event(Task(id="e1"))
        await reader_a.enqueue_event(Task(id="e2"))
        await reader_a.dequeue_event()
        await reader_b.dequeue_event()

        consumers = await manager._client.xinfo_consumers(manager._stream_key("t1"), "producer")
        assert len(consumers) == _TWO_DISTINCT_READERS

    asyncio.run(_run())


def test_dequeue_block_milliseconds_is_configurable(mocker):
    """RedisQueueManager's dequeue_block_milliseconds must reach the real XREADGROUP call on
    every queue it returns (get()/tap()/create_or_tap()'s own tap path), not just the default.
    """

    async def _run():
        manager = _manager(dequeue_block_milliseconds=_CUSTOM_DEQUEUE_BLOCK_MILLISECONDS)
        mocker.patch.object(manager._client, "xreadgroup", new=mocker.AsyncMock(return_value=None))
        queue = await manager.create_or_tap("t1")

        with pytest.raises(asyncio.QueueEmpty):
            await queue.dequeue_event()

        assert manager._client.xreadgroup.call_args.kwargs["block"] == _CUSTOM_DEQUEUE_BLOCK_MILLISECONDS

    asyncio.run(_run())


def test_dequeue_block_milliseconds_is_floored_at_one(mocker):
    """BLOCK 0 means "wait forever" to Redis - a sub-millisecond setting truncating to 0 must
    not reach XREADGROUP, or dequeue_event() would hang instead of polling.
    """

    async def _run():
        manager = _manager(dequeue_block_milliseconds=0)
        mocker.patch.object(manager._client, "xreadgroup", new=mocker.AsyncMock(return_value=None))
        queue = await manager.create_or_tap("t1")

        with pytest.raises(asyncio.QueueEmpty):
            await queue.dequeue_event()

        assert manager._client.xreadgroup.call_args.kwargs["block"] == 1

    asyncio.run(_run())


def test_tap_returns_none_for_a_nonexistent_task_id():
    """tap() (unlike create_or_tap()) must return None when there's nothing to tap, per the
    QueueManager ABC's documented distinction between the two.
    """

    async def _run():
        manager = _manager()
        assert await manager.tap("missing") is None

    asyncio.run(_run())


def test_tap_returns_a_queue_for_an_existing_task_id():
    """tap() on a task_id that already has a queue must return one, not None."""

    async def _run():
        manager = _manager()
        await manager.create_or_tap("t1")

        tapped = await manager.tap("t1")

        assert tapped is not None

    asyncio.run(_run())


def test_add_always_raises_not_implemented():
    """add() must fail loudly for any task_id - registering an already-constructed queue so
    get()/tap() route through it isn't possible against Redis.
    """

    async def _run():
        manager = _manager()
        queue = await manager.create_or_tap("t1")

        with pytest.raises(NotImplementedError):
            await manager.add("t2", queue)

    asyncio.run(_run())


def test_close_raises_no_task_queue_for_a_nonexistent_task_id():
    """close() on a task_id with no queue at all must raise NoTaskQueue, per the ABC."""

    async def _run():
        manager = _manager()
        with pytest.raises(NoTaskQueue):
            await manager.close("missing")

    asyncio.run(_run())


def test_close_does_not_immediately_delete_the_stream():
    """close() must leave the underlying stream key present with a grace-period TTL set, not
    delete it outright - a lagging tap()/get() reader needs time to finish draining it.
    """

    async def _run():
        manager = _manager(close_grace_period_seconds=_GRACE_PERIOD_SECONDS)
        client = manager._client
        await manager.create_or_tap("t1")

        await manager.close("t1")

        assert await client.exists(manager._stream_key("t1"))
        ttl = await client.ttl(manager._stream_key("t1"))
        assert 0 < ttl <= _GRACE_PERIOD_SECONDS

    asyncio.run(_run())


def test_get_returns_none_after_the_producer_group_was_closed_but_stream_lingers():
    """EventQueue.close() destroys the producer group but leaves the stream alive (its own
    EXPIRE grace period, set by a separate manager.close() call, hasn't elapsed). get() must
    not hand back a queue bound to that now-missing group - its first dequeue_event() would
    fail with Redis NOGROUP instead of behaving like no queue exists.
    """

    async def _run():
        manager = _manager()
        producer_queue = await manager.create_or_tap("t1")
        await producer_queue.close()

        assert await manager._client.exists(manager._stream_key("t1"))  # stream still lingers
        assert await manager.get("t1") is None

    asyncio.run(_run())


def test_create_or_tap_restores_the_producer_group_after_it_was_closed():
    """Same lingering-stream scenario as above: create_or_tap() must recreate the PRODUCER
    group (consistent with "create if no usable queue exists"), not create an unrelated tap
    group that leaves the producer group permanently missing.
    """

    async def _run():
        manager = _manager()
        producer_queue = await manager.create_or_tap("t1")
        await producer_queue.close()

        restored = await manager.create_or_tap("t1")

        assert restored._group_name == _PRODUCER_GROUP
        await restored.enqueue_event(Task(id="e1"))
        assert await restored.dequeue_event() == Task(id="e1")

    asyncio.run(_run())


def test_enqueue_then_dequeue_round_trips_an_event():
    """An event enqueued on the producer queue must be dequeued back out equal, via the real
    XADD/XREADGROUP path.
    """

    async def _run():
        manager = _manager()
        queue = await manager.create_or_tap("t1")
        task = Task(id="task-1", context_id="ctx-1")

        await queue.enqueue_event(task)
        dequeued = await queue.dequeue_event()

        assert dequeued == task
        queue.task_done()

    asyncio.run(_run())


def test_task_done_acks_entries_in_dequeue_order_not_just_the_latest():
    """Two dequeue_event() calls before either is acked - a valid pattern asyncio.Queue itself
    supports regardless of get()/task_done() ordering - must each get their own XACK (review:
    a single last_entry_id scalar loses the first entry's id once the second overwrites it,
    leaving it permanently pending and eventually redelivered as a duplicate).
    """

    async def _run():
        manager = _manager()
        queue = await manager.create_or_tap("t1")
        task1 = Task(id="task-1")
        task2 = Task(id="task-2")
        await queue.enqueue_event(task1)
        await queue.enqueue_event(task2)

        dequeued1 = await queue.dequeue_event()
        dequeued2 = await queue.dequeue_event()
        assert dequeued1 == task1
        assert dequeued2 == task2

        queue.task_done()
        queue.task_done()
        await asyncio.gather(*queue._background_acks)

        pending = await manager._client.xpending(manager._stream_key("t1"), queue._group_name)
        assert pending["pending"] == 0

    asyncio.run(_run())


def test_dequeue_reclaims_an_entry_abandoned_by_a_crashed_consumer():
    """A crashed consumer's never-acked entry must be recoverable via XAUTOCLAIM, not
    permanently lost - reclaim_min_idle_milliseconds=0 so this test needn't actually wait.
    """

    async def _run():
        manager = _manager()
        crashed_reader = await manager.create_or_tap("t1")
        task = Task(id="abandoned")
        await crashed_reader.enqueue_event(task)
        await crashed_reader.dequeue_event()  # delivered, then "crashes" - task_done() never runs

        # A second reader of the SAME group the crashed consumer read under - a fresh tap()
        # would create its own, separate group and never see "abandoned" at all.
        fresh_reader = RedisEventQueue(
            manager._client, manager._stream_key("t1"), crashed_reader._group_name, reclaim_min_idle_milliseconds=0
        )

        recovered = await fresh_reader.dequeue_event()

        assert recovered == task

    asyncio.run(_run())


def test_dequeue_does_not_reclaim_an_entry_still_within_its_idle_threshold():
    """A second reader must not steal an entry a live consumer is still processing, just
    hasn't acked yet - the idle threshold is what distinguishes it from an abandoned one.
    """

    async def _run():
        manager = _manager()
        live_reader = await manager.create_or_tap("t1")
        await live_reader.enqueue_event(Task(id="still-in-flight"))
        await live_reader.dequeue_event()  # delivered; live_reader just hasn't acked yet

        other_reader = RedisEventQueue(manager._client, manager._stream_key("t1"), live_reader._group_name)

        with pytest.raises(asyncio.QueueEmpty):
            await other_reader.dequeue_event()

    asyncio.run(_run())


def test_tap_only_sees_events_enqueued_after_it_was_created():
    """A tap() must not see events already on the stream before it was created - only ones
    enqueued from that point forward (EventQueueLegacy.tap()'s documented contract).
    """

    async def _run():
        manager = _manager()
        producer = await manager.create_or_tap("t1")
        await producer.enqueue_event(Task(id="before-tap"))

        tapped = await manager.tap("t1")
        await producer.enqueue_event(Task(id="after-tap"))

        dequeued = await tapped.dequeue_event()
        assert dequeued == Task(id="after-tap")

    asyncio.run(_run())


def test_dequeue_raises_queue_shut_down_once_closed():
    """dequeue_event() on an already-closed queue must raise QueueShutDown, matching
    EventQueueLegacy's own contract so EventConsumer's handling is identical either way.
    """

    async def _run():
        manager = _manager()
        queue = await manager.create_or_tap("t1")
        await queue.close()

        with pytest.raises(QueueShutDown):
            await queue.dequeue_event()

    asyncio.run(_run())


def test_enqueue_after_close_is_a_noop():
    """enqueue_event() on an already-closed queue must not raise, and must not add an entry -
    matching EventQueueLegacy's own closed-queue behavior."""

    async def _run():
        manager = _manager()
        queue = await manager.create_or_tap("t1")
        await queue.close()

        await queue.enqueue_event(Task(id="dropped"))

        assert await manager._client.xlen(manager._stream_key("t1")) == 0

    asyncio.run(_run())


def test_close_waits_for_an_in_flight_ack_before_destroying_the_group(mocker):
    """close() must wait for task_done()'s background XACK, or destroying the group mid-flight
    loses that entry's pending state permanently (never redelivered).
    """

    async def _run():
        manager = _manager()
        queue = await manager.create_or_tap("t1")
        await queue.enqueue_event(Task(id="task-1"))
        await queue.dequeue_event()

        real_xack = manager._client.xack
        ack_completed = False

        async def _slow_xack(*args, **kwargs):
            nonlocal ack_completed
            await asyncio.sleep(0.05)
            result = await real_xack(*args, **kwargs)
            ack_completed = True
            return result

        mocker.patch.object(manager._client, "xack", new=_slow_xack)

        queue.task_done()
        await queue.close()

        assert ack_completed
        assert queue._background_acks == set()

    asyncio.run(_run())
