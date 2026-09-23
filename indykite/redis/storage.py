"""Key/value storage abstraction shared by SchemaClient's schema cache and RedisTaskStore's
A2A task store - just get/set/delete on opaque bytes, with an optional hard TTL.
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from redis.asyncio import Redis

if TYPE_CHECKING:
    from redis.asyncio.client import Pipeline


class StorageBackend(ABC):
    """Opaque byte-value key/value store: get/set/delete, with an optional hard TTL.

    Deliberately minimal. Concrete backends may expose extra, backend-specific methods (see
    RedisStorageBackend's sorted-set/WATCH helpers) outside this ABC. aclose() is a no-op by
    default - only RedisStorageBackend has something to actually close.
    """

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Return the value stored at key, or None if absent or expired."""

    @abstractmethod
    async def set(self, key: str, value: bytes, *, ttl_seconds: float | None = None) -> None:
        """Store value at key. ttl_seconds=None (the default) means no expiry."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove key. A no-op, not an error, if it's already absent."""

    async def aclose(self) -> None:
        """Release this backend's resources, if it holds any real ones. A no-op by default -
        most backends (e.g. MemoryStorageBackend) have nothing to release.
        """
        return


class MemoryStorageBackend(StorageBackend):
    """In-process, single-instance backend: a plain dict of key -> (value, expires_at).

    expires_at is a time.monotonic() deadline, or None for no expiry - monotonic so a clock
    adjustment can't throw off expiry math. Checked and evicted lazily on get(), no background
    sweep.
    """

    def __init__(self) -> None:
        """Start with an empty store."""
        self._store: dict[str, tuple[bytes, float | None]] = {}

    async def get(self, key: str) -> bytes | None:
        """Return the value at key, evicting it first if its TTL has passed."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.monotonic() >= expires_at:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: bytes, *, ttl_seconds: float | None = None) -> None:
        """Store value at key, with an absolute monotonic expiry when ttl_seconds is given."""
        expires_at = time.monotonic() + ttl_seconds if ttl_seconds is not None else None
        self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        """Remove key if present."""
        self._store.pop(key, None)


def _decode_member(member: bytes | str) -> str:
    """Normalize one sorted-set member to str, regardless of whether the wrapped redis.asyncio
    client was built with decode_responses - this module doesn't assume either way.
    """
    return member.decode() if isinstance(member, bytes) else member


class RedisStorageBackend(StorageBackend):
    """Redis-backed StorageBackend, wrapping an already-constructed redis.asyncio.Redis client.

    Also exposes sorted-set read helpers and WATCH-based atomic read-then-write helpers
    (watch_set/watch_delete/zrem_if_key_absent) outside the StorageBackend contract - only
    RedisTaskStore needs them.
    """

    def __init__(self, client: Redis) -> None:
        """Wrap the given client; no connection is made here, redis.asyncio connects lazily."""
        self._client = client

    async def aclose(self) -> None:
        """Close the wrapped client's connection pool."""
        await self._client.aclose()

    async def get(self, key: str) -> bytes | None:
        """Return the raw bytes stored at key, or None if absent."""
        return await self._client.get(key)

    async def set(self, key: str, value: bytes, *, ttl_seconds: float | None = None) -> None:
        """SET key to value; EX ttl_seconds when given (redis-py needs a timedelta for a
        fractional-second EX, a bare float raises DataError).
        """
        if ttl_seconds is not None:
            await self._client.set(key, value, ex=timedelta(seconds=ttl_seconds))
        else:
            await self._client.set(key, value)

    async def delete(self, key: str) -> None:
        """DELETE key."""
        await self._client.delete(key)

    async def zrevrangebyscore(self, key: str, max_score: float, min_score: float) -> list[str]:
        """ZREVRANGEBYSCORE: members with min_score <= score <= max_score, highest score first."""
        members = await self._client.zrevrangebyscore(key, max_score, min_score)
        return [_decode_member(m) for m in members]

    async def zrevrange(self, key: str, start: int, stop: int) -> list[str]:
        """ZREVRANGE: members by descending-score rank, inclusive of both start and stop."""
        members = await self._client.zrevrange(key, start, stop)
        return [_decode_member(m) for m in members]

    async def zrevrank(self, key: str, member: str) -> int | None:
        """ZREVRANK: member's 0-based rank in descending-score order, or None if absent."""
        return await self._client.zrevrank(key, member)

    async def zcard(self, key: str) -> int:
        """ZCARD: the number of members in the sorted set."""
        return await self._client.zcard(key)

    async def watch_delete(
        self, key: str, build_cleanup: Callable[[bytes], list[tuple[str, str]]], *, also_delete: list[str] = ()
    ) -> bytes | None:
        """WATCH key, and if it holds a value, atomically DELETE it, every (zset_key, member)
        pair build_cleanup(current_value) derives from it, and every key in also_delete - via
        redis-py's Redis.transaction() (WATCH/MULTI/EXEC, retried on a concurrent write to key).
        Returns the deleted value, or None if key was already absent (also_delete skipped too).

        A read-then-conditionally-write decision: a concurrent write to key retries the whole
        attempt against the fresh value, so the record deleted and the index entries removed
        always correspond to each other.
        """

        async def _txn(pipe: "Pipeline") -> bytes | None:
            encoded = await pipe.get(key)
            pipe.multi()
            if encoded is None:
                return None
            pipe.delete(key)
            for zset_key, member in build_cleanup(encoded):
                pipe.zrem(zset_key, member)
            for extra_key in also_delete:
                pipe.delete(extra_key)
            return encoded

        return await self._client.transaction(_txn, key, value_from_callable=True)

    async def zrem_if_key_absent(
        self, key: str, targets: list[tuple[str, str]], *, also_delete: list[str] = ()
    ) -> None:
        """WATCH key; if still absent, ZREM every target and DELETE also_delete, atomically -
        guards against a concurrent save() recreating key (same race class as watch_delete).
        """

        async def _txn(pipe: "Pipeline") -> None:
            encoded = await pipe.get(key)
            pipe.multi()
            if encoded is None:
                for zset_key, member in targets:
                    pipe.zrem(zset_key, member)
                for extra_key in also_delete:
                    pipe.delete(extra_key)

        await self._client.transaction(_txn, key)

    async def watch_set(  # noqa: PLR0913 - each build_* callback is independently meaningful
        self,
        key: str,
        value: bytes,
        *,
        ttl_seconds: float | None,
        build_writes: Callable[[bytes | None], list[tuple[str, str, float]]],
        build_cleanup: Callable[[bytes | None], list[tuple[str, str]]],
        build_extra_sets: Callable[[bytes | None], list[tuple[str, bytes, float | None]]] = lambda _old: [],
        build_extra_deletes: Callable[[bytes | None], list[str]] = lambda _old: [],
    ) -> None:
        """WATCH key; SET it, then apply build_writes/build_cleanup/build_extra_sets/
        build_extra_deletes(old_value) atomically - retried on a concurrent write, so every
        callback sees the current value.
        """

        async def _txn(pipe: "Pipeline") -> None:
            old_value = await pipe.get(key)
            pipe.multi()
            if ttl_seconds is not None:
                pipe.set(key, value, ex=timedelta(seconds=ttl_seconds))
            else:
                pipe.set(key, value)
            for zset_key, member, score in build_writes(old_value):
                pipe.zadd(zset_key, {member: score})
            for zset_key, member in build_cleanup(old_value):
                pipe.zrem(zset_key, member)
            for extra_key, extra_value, extra_ttl in build_extra_sets(old_value):
                if extra_ttl is not None:
                    pipe.set(extra_key, extra_value, ex=timedelta(seconds=extra_ttl))
                else:
                    pipe.set(extra_key, extra_value)
            for extra_key in build_extra_deletes(old_value):
                pipe.delete(extra_key)

        await self._client.transaction(_txn, key)


def build_redis_client(url: str, *, socket_timeout_seconds: float) -> Redis:
    """Construct a redis.asyncio.Redis client from a DSN - the one place this package builds
    a raw client, so callers (app.py, schema.py, policy.py) never need to import
    redis.asyncio themselves. `url` is passed as-is, preserving username/password/db/TLS scheme.

    socket_timeout_seconds bounds both socket_timeout and socket_connect_timeout - from_url()
    defaults both to None (no bound at all) otherwise, hanging any call indefinitely against a
    blackholed/unreachable Redis.
    """
    return Redis.from_url(url, socket_timeout=socket_timeout_seconds, socket_connect_timeout=socket_timeout_seconds)


def build_redis_storage_backend(url: str, *, socket_timeout_seconds: float) -> RedisStorageBackend:
    """Construct a RedisStorageBackend from a DSN - see build_redis_client."""
    return RedisStorageBackend(build_redis_client(url, socket_timeout_seconds=socket_timeout_seconds))


class _SyncBridge:
    """One persistent background event loop every sync-bridged call (run_sync) runs on - avoids
    rebinding a long-lived redis.asyncio connection to a new loop each call."""

    def __init__(self) -> None:
        """Start the background loop immediately so it's ready before the first call."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        """Submit coro to the persistent loop and block this thread until it completes."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


_bridge = _SyncBridge()


def run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run one StorageBackend coroutine to completion from a synchronous caller (SchemaClient/
    PolicyClient's public APIs stay sync) - see _SyncBridge for why not a plain asyncio.run().
    """
    return _bridge.run(coro)
