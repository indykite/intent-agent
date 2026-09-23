"""Tests for the StorageBackend abstraction: MemoryStorageBackend and RedisStorageBackend
(including its Redis-only sorted-set/WATCH helpers, faked with fakeredis so no real Redis
server is needed).
"""

import asyncio

import pytest
from fakeredis import FakeAsyncRedis

from indykite.redis.storage import MemoryStorageBackend, RedisStorageBackend, build_redis_client

_TTL_SECONDS = 60.0
_EXPECTED_TTL_CEILING_SECONDS = 6
_EXPECTED_SORTED_SET_SIZE = 3
_EXPECTED_SOCKET_TIMEOUT_SECONDS = 2.5


def _redis_backend():
    """A fresh RedisStorageBackend wrapping an in-memory fake Redis client."""
    return RedisStorageBackend(FakeAsyncRedis())


def test_memory_backend_round_trips_a_value():
    """set() then get() must return the same bytes."""

    async def _run():
        backend = MemoryStorageBackend()
        await backend.set("key", b"value")
        assert await backend.get("key") == b"value"

    asyncio.run(_run())


def test_memory_backend_get_returns_none_for_missing_key():
    """A key never written must read back as None, not raise."""

    async def _run():
        backend = MemoryStorageBackend()
        assert await backend.get("missing") is None

    asyncio.run(_run())


def test_memory_backend_delete_is_a_noop_for_missing_key():
    """delete() on an absent key must not raise."""

    async def _run():
        backend = MemoryStorageBackend()
        await backend.delete("missing")

    asyncio.run(_run())


def test_memory_backend_delete_removes_an_existing_key():
    """delete() on a present key must make a later get() return None."""

    async def _run():
        backend = MemoryStorageBackend()
        await backend.set("key", b"value")
        await backend.delete("key")
        assert await backend.get("key") is None

    asyncio.run(_run())


def test_memory_backend_with_no_ttl_never_expires(mocker):
    """ttl_seconds=None (the default) must never evict, no matter how much time passes."""

    async def _run():
        backend = MemoryStorageBackend()
        await backend.set("key", b"value")
        mocker.patch("indykite.redis.storage.time.monotonic", return_value=1_000_000.0)
        assert await backend.get("key") == b"value"

    asyncio.run(_run())


def test_memory_backend_evicts_on_read_past_ttl(mocker):
    """An entry read after its ttl_seconds has elapsed (monotonic clock) must evict and read
    back as None - eviction happens lazily, on the read itself, not via a background sweep.
    """
    monotonic = mocker.patch("indykite.redis.storage.time.monotonic")
    monotonic.return_value = 0.0

    async def _run():
        backend = MemoryStorageBackend()
        await backend.set("key", b"value", ttl_seconds=_TTL_SECONDS)
        assert await backend.get("key") == b"value"

        monotonic.return_value = _TTL_SECONDS + 1
        assert await backend.get("key") is None
        # Evicted, not just reported as expired - a second read must still be None with no
        # lingering internal entry.
        assert backend._store == {}

    asyncio.run(_run())


def test_memory_backend_keeps_a_value_still_inside_its_ttl(mocker):
    """A read before ttl_seconds has elapsed must still return the value."""
    monotonic = mocker.patch("indykite.redis.storage.time.monotonic")
    monotonic.return_value = 0.0

    async def _run():
        backend = MemoryStorageBackend()
        await backend.set("key", b"value", ttl_seconds=_TTL_SECONDS)
        monotonic.return_value = _TTL_SECONDS - 1
        assert await backend.get("key") == b"value"

    asyncio.run(_run())


def test_memory_backend_aclose_is_a_noop():
    """MemoryStorageBackend holds no real resource, so aclose() (the StorageBackend ABC's
    default no-op) must not raise.
    """

    async def _run():
        backend = MemoryStorageBackend()
        await backend.aclose()  # must not raise

    asyncio.run(_run())


def test_redis_backend_aclose_closes_the_client(mocker):
    """aclose() must close the wrapped redis.asyncio client."""

    async def _run():
        client = FakeAsyncRedis()
        client.aclose = mocker.AsyncMock(wraps=client.aclose)
        backend = RedisStorageBackend(client)

        await backend.aclose()

        client.aclose.assert_awaited_once()

    asyncio.run(_run())


def test_redis_backend_round_trips_a_value():
    """set() then get() against Redis must return the same bytes."""

    async def _run():
        backend = _redis_backend()
        await backend.set("key", b"value")
        assert await backend.get("key") == b"value"

    asyncio.run(_run())


def test_redis_backend_get_returns_none_for_missing_key():
    """A key never written must read back as None."""

    async def _run():
        backend = _redis_backend()
        assert await backend.get("missing") is None

    asyncio.run(_run())


def test_redis_backend_delete_removes_an_existing_key():
    """delete() must remove the key from Redis."""

    async def _run():
        backend = _redis_backend()
        await backend.set("key", b"value")
        await backend.delete("key")
        assert await backend.get("key") is None

    asyncio.run(_run())


def test_redis_backend_applies_ttl_as_a_real_expiry():
    """ttl_seconds must become a real Redis EXPIRE, even for a fractional-second value (which
    would otherwise raise DataError if passed to redis-py as a bare float).
    """

    async def _run():
        backend = _redis_backend()
        await backend.set("key", b"value", ttl_seconds=5.5)
        ttl = await backend._client.ttl("key")
        assert 0 < ttl <= _EXPECTED_TTL_CEILING_SECONDS

    asyncio.run(_run())


def test_redis_backend_with_no_ttl_sets_no_expiry():
    """ttl_seconds=None (the default) must leave the key with no TTL at all."""

    async def _run():
        backend = _redis_backend()
        await backend.set("key", b"value")
        assert await backend._client.ttl("key") == -1

    asyncio.run(_run())


def test_redis_backend_sorted_set_read_helpers_round_trip():
    """zrevrange/zrevrank/zcard must all agree over one sorted set, seeded directly via the
    raw client since RedisStorageBackend no longer exposes a standalone zadd/zrem."""

    async def _run():
        backend = _redis_backend()
        await backend._client.zadd("idx", {"a": 1.0, "b": 2.0, "c": 3.0})
        assert await backend.zcard("idx") == _EXPECTED_SORTED_SET_SIZE
        assert await backend.zrevrange("idx", 0, -1) == ["c", "b", "a"]
        assert await backend.zrevrank("idx", "b") == 1
        assert await backend.zrevrank("idx", "missing") is None

        await backend._client.zrem("idx", "b")
        assert await backend.zrevrange("idx", 0, -1) == ["c", "a"]

    asyncio.run(_run())


def test_redis_backend_zrevrangebyscore_bounds_by_score():
    """zrevrangebyscore must return only members within [min_score, max_score], highest first."""

    async def _run():
        backend = _redis_backend()
        await backend._client.zadd("idx", {"a": 1.0, "b": 2.0, "c": 3.0})
        assert await backend.zrevrangebyscore("idx", 3.0, 2.0) == ["c", "b"]
        assert await backend.zrevrangebyscore("idx", float("inf"), float("-inf")) == ["c", "b", "a"]

    asyncio.run(_run())


def test_redis_backend_watch_set_writes_primary_and_index_atomically():
    """A watch_set() call writing a primary key and an index entry together must land both -
    the actual multi-key atomic write RedisTaskStore.save() depends on.
    """

    async def _run():
        backend = _redis_backend()
        await backend.watch_set(
            "primary:task-1",
            b"payload",
            ttl_seconds=None,
            build_writes=lambda _old: [("index", "task-1", 42.0)],
            build_cleanup=lambda _old: [],
        )

        assert await backend.get("primary:task-1") == b"payload"
        assert await backend.zrevrank("index", "task-1") == 0

    asyncio.run(_run())


def test_redis_backend_watch_set_build_extra_deletes_removes_a_plain_key():
    """watch_set()'s build_extra_deletes must remove a plain key atomically with the primary
    write - this is RedisTaskStore.save()'s own stale context-lookup-key cleanup (review).
    """

    async def _run():
        backend = _redis_backend()
        await backend.set("lookup:task-2", b"ctx-old")

        await backend.watch_set(
            "primary:task-2",
            b"payload",
            ttl_seconds=None,
            build_writes=lambda _old: [],
            build_cleanup=lambda _old: [],
            build_extra_deletes=lambda _old: ["lookup:task-2"],
        )

        assert await backend.get("primary:task-2") == b"payload"
        assert await backend.get("lookup:task-2") is None

    asyncio.run(_run())


_BOOM_MESSAGE = "boom"


def _raise_boom(_old_value):
    raise RuntimeError(_BOOM_MESSAGE)


def test_redis_backend_watch_set_discards_queued_writes_on_exception():
    """If a build_writes callback raises, the primary SET queued ahead of it in the same
    transaction must not be applied either - atomic or nothing, not partially applied.
    """

    async def _run():
        backend = _redis_backend()
        with pytest.raises(RuntimeError):
            await backend.watch_set(
                "primary:task-2",
                b"payload",
                ttl_seconds=None,
                build_writes=_raise_boom,
                build_cleanup=lambda _old: [],
            )

        assert await backend.get("primary:task-2") is None

    asyncio.run(_run())


def test_redis_backend_watch_delete_removes_primary_and_index_entry_together():
    """watch_delete() removing the primary key and an index entry together - this is
    RedisTaskStore's delete() path (primary key + index entries) collapsed to one index.
    """

    async def _run():
        backend = _redis_backend()
        await backend.watch_set(
            "primary:task-3",
            b"payload",
            ttl_seconds=None,
            build_writes=lambda _old: [("index", "task-3", 7.0)],
            build_cleanup=lambda _old: [],
        )

        deleted = await backend.watch_delete("primary:task-3", build_cleanup=lambda _old: [("index", "task-3")])

        assert deleted == b"payload"
        assert await backend.get("primary:task-3") is None
        assert await backend.zcard("index") == 0

    asyncio.run(_run())


def test_redis_backend_watch_delete_also_removes_plain_keys():
    """watch_delete()'s also_delete must remove plain (non-zset) keys too, atomically with the
    primary key - this is RedisTaskStore.delete()'s own context-lookup key cleanup (review).
    """

    async def _run():
        backend = _redis_backend()
        await backend.set("primary:task-4", b"payload")
        await backend.set("lookup:task-4", b"ctx-1")

        deleted = await backend.watch_delete(
            "primary:task-4", build_cleanup=lambda _old: [], also_delete=["lookup:task-4"]
        )

        assert deleted == b"payload"
        assert await backend.get("primary:task-4") is None
        assert await backend.get("lookup:task-4") is None

    asyncio.run(_run())


def test_build_redis_client_configures_socket_timeouts():
    """redis.asyncio.Redis.from_url() defaults socket_timeout/socket_connect_timeout to None
    (no bound at all) unless given explicitly - a blackholed/unreachable Redis would otherwise
    hang any call on this client indefinitely. build_redis_client must set both.
    """
    client = build_redis_client("redis://localhost:6379", socket_timeout_seconds=_EXPECTED_SOCKET_TIMEOUT_SECONDS)

    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["socket_timeout"] == _EXPECTED_SOCKET_TIMEOUT_SECONDS
    assert kwargs["socket_connect_timeout"] == _EXPECTED_SOCKET_TIMEOUT_SECONDS
