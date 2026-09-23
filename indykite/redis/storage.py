"""Key/value storage abstraction shared by SchemaClient/PolicyClient's cache and the A2A
EvictingTaskStore - just get/set/delete on opaque bytes, with an optional hard TTL.

SchemaClient/PolicyClient run synchronously on IntentAgent's own ThreadPoolExecutor, so this
backend is a plain redis.Redis client, not redis.asyncio - no event loop to bridge into.
"""

import time
from abc import ABC, abstractmethod

from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.retry import Retry

# One retry, no backoff, on ConnectionError only (not TimeoutError/socket.timeout, redis-py's
# own default) - from_url() applies no retry otherwise, so a single transient reset would
# surface as a full cache-backend failure. TimeoutError is deliberately excluded: that's what a
# blackholed connect raises, and retrying it doubles the worst-case stall
# REDIS_SOCKET_TIMEOUT_SECONDS bounds (confirmed - it broke
# test_blackholed_redis_stays_within_the_request_budget). Callers already fall back to a direct,
# uncached fetch on any failure here, so a longer retry policy wouldn't help anyway.
_CONNECTION_RETRY_ATTEMPTS = 1
_CONNECTION_RETRY = Retry(NoBackoff(), _CONNECTION_RETRY_ATTEMPTS, supported_errors=(RedisConnectionError,))

_MIN_TTL_MILLISECONDS = 1


class StorageBackend(ABC):
    """Opaque byte-value key/value store: get/set/delete, with an optional hard TTL.

    Deliberately minimal. close() is a no-op by default - only RedisStorageBackend has
    something to actually close.
    """

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Return the value stored at key, or None if absent or expired."""

    @abstractmethod
    def set(self, key: str, value: bytes, *, ttl_seconds: float | None = None) -> None:
        """Store value at key. ttl_seconds=None (the default) means no expiry."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove key. A no-op, not an error, if it's already absent."""

    def close(self) -> None:
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

    def get(self, key: str) -> bytes | None:
        """Return the value at key, evicting it first if its TTL has passed."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.monotonic() >= expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: bytes, *, ttl_seconds: float | None = None) -> None:
        """Store value at key, with an absolute monotonic expiry when ttl_seconds is given."""
        expires_at = time.monotonic() + ttl_seconds if ttl_seconds is not None else None
        self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        """Remove key if present."""
        self._store.pop(key, None)


class RedisStorageBackend(StorageBackend):
    """Redis-backed StorageBackend, wrapping an already-constructed redis.Redis client."""

    def __init__(self, client: Redis) -> None:
        """Wrap the given client; no connection is made here, redis-py connects lazily."""
        self._client = client

    def close(self) -> None:
        """Close the wrapped client's connection pool."""
        self._client.close()

    def get(self, key: str) -> bytes | None:
        """Return the raw bytes stored at key, or None if absent."""
        return self._client.get(key)

    def set(self, key: str, value: bytes, *, ttl_seconds: float | None = None) -> None:
        """SET key to value, with PX ttl_seconds in milliseconds (clamped to 1ms) when given -
        Redis's EX is integer-second only, so a sub-second TTL would truncate to the EX 0
        Redis rejects.
        """
        if ttl_seconds is not None:
            milliseconds = max(_MIN_TTL_MILLISECONDS, round(ttl_seconds * 1000))
            self._client.set(key, value, px=milliseconds)
        else:
            self._client.set(key, value)

    def delete(self, key: str) -> None:
        """DELETE key."""
        self._client.delete(key)


def build_redis_client(url: str, *, socket_timeout_seconds: float) -> Redis:
    """Construct a redis.Redis client from a DSN - the one place this package builds a raw
    client, so callers (schema.py, policy.py) never need to import redis themselves. `url` is
    passed as-is, preserving username/password/db/TLS scheme.

    socket_timeout_seconds bounds both socket_timeout and socket_connect_timeout - from_url()
    defaults both to None (no bound at all) otherwise, hanging any call indefinitely against a
    blackholed/unreachable Redis. retry=_CONNECTION_RETRY absorbs one transient connection
    blip (see its own comment) without materially changing that worst-case bound.
    """
    return Redis.from_url(
        url,
        socket_timeout=socket_timeout_seconds,
        socket_connect_timeout=socket_timeout_seconds,
        retry=_CONNECTION_RETRY,
    )


def build_redis_storage_backend(url: str, *, socket_timeout_seconds: float) -> RedisStorageBackend:
    """Construct a RedisStorageBackend from a DSN - see build_redis_client."""
    return RedisStorageBackend(build_redis_client(url, socket_timeout_seconds=socket_timeout_seconds))
