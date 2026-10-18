"""Tests for the StorageBackend abstraction: MemoryStorageBackend and RedisStorageBackend,
faked with fakeredis so no real Redis server is needed.
"""

from fakeredis import FakeRedis
from redis.backoff import NoBackoff
from redis.exceptions import ConnectionError as RedisConnectionError

from indykite.redis.storage import MemoryStorageBackend, RedisStorageBackend, build_redis_client

_TTL_SECONDS = 60.0
_EXPECTED_TTL_CEILING_SECONDS = 6
_EXPECTED_SOCKET_TIMEOUT_SECONDS = 2.5


def _redis_backend():
    """A fresh RedisStorageBackend wrapping an in-memory fake Redis client."""
    return RedisStorageBackend(FakeRedis())


def test_memory_backend_round_trips_a_value():
    """set() then get() must return the same bytes."""
    backend = MemoryStorageBackend()
    backend.set("key", b"value")
    assert backend.get("key") == b"value"


def test_memory_backend_get_returns_none_for_missing_key():
    """A key never written must read back as None, not raise."""
    backend = MemoryStorageBackend()
    assert backend.get("missing") is None


def test_memory_backend_delete_is_a_noop_for_missing_key():
    """delete() on an absent key must not raise."""
    backend = MemoryStorageBackend()
    backend.delete("missing")


def test_memory_backend_delete_removes_an_existing_key():
    """delete() on a present key must make a later get() return None."""
    backend = MemoryStorageBackend()
    backend.set("key", b"value")
    backend.delete("key")
    assert backend.get("key") is None


def test_memory_backend_with_no_ttl_never_expires(mocker):
    """ttl_seconds=None (the default) must never evict, no matter how much time passes."""
    backend = MemoryStorageBackend()
    backend.set("key", b"value")
    mocker.patch("indykite.redis.storage.time.monotonic", return_value=1_000_000.0)
    assert backend.get("key") == b"value"


def test_memory_backend_evicts_on_read_past_ttl(mocker):
    """An entry read after its ttl_seconds has elapsed (monotonic clock) must evict and read
    back as None - eviction happens lazily, on the read itself, not via a background sweep.
    """
    monotonic = mocker.patch("indykite.redis.storage.time.monotonic")
    monotonic.return_value = 0.0

    backend = MemoryStorageBackend()
    backend.set("key", b"value", ttl_seconds=_TTL_SECONDS)
    assert backend.get("key") == b"value"

    monotonic.return_value = _TTL_SECONDS + 1
    assert backend.get("key") is None
    # Evicted, not just reported as expired - a second read must still be None with no
    # lingering internal entry.
    assert backend._store == {}


def test_memory_backend_keeps_a_value_still_inside_its_ttl(mocker):
    """A read before ttl_seconds has elapsed must still return the value."""
    monotonic = mocker.patch("indykite.redis.storage.time.monotonic")
    monotonic.return_value = 0.0

    backend = MemoryStorageBackend()
    backend.set("key", b"value", ttl_seconds=_TTL_SECONDS)
    monotonic.return_value = _TTL_SECONDS - 1
    assert backend.get("key") == b"value"


def test_memory_backend_close_is_a_noop():
    """MemoryStorageBackend holds no real resource, so close() (the StorageBackend ABC's
    default no-op) must not raise.
    """
    backend = MemoryStorageBackend()
    backend.close()  # must not raise


def test_redis_backend_close_closes_the_client(mocker):
    """close() must close the wrapped redis client."""
    client = FakeRedis()
    client.close = mocker.Mock(wraps=client.close)
    backend = RedisStorageBackend(client)

    backend.close()

    client.close.assert_called_once()


def test_redis_backend_round_trips_a_value():
    """set() then get() against Redis must return the same bytes."""
    backend = _redis_backend()
    backend.set("key", b"value")
    assert backend.get("key") == b"value"


def test_redis_backend_get_returns_none_for_missing_key():
    """A key never written must read back as None."""
    backend = _redis_backend()
    assert backend.get("missing") is None


def test_redis_backend_delete_removes_an_existing_key():
    """delete() must remove the key from Redis."""
    backend = _redis_backend()
    backend.set("key", b"value")
    backend.delete("key")
    assert backend.get("key") is None


def test_redis_backend_applies_ttl_as_a_real_expiry():
    """ttl_seconds must become a real Redis EXPIRE, even for a fractional-second value (which
    would otherwise raise DataError if passed to redis-py as a bare float).
    """
    backend = _redis_backend()
    backend.set("key", b"value", ttl_seconds=5.5)
    ttl = backend._client.ttl("key")
    assert 0 < ttl <= _EXPECTED_TTL_CEILING_SECONDS


def test_redis_backend_with_no_ttl_sets_no_expiry():
    """ttl_seconds=None (the default) must leave the key with no TTL at all."""
    backend = _redis_backend()
    backend.set("key", b"value")
    assert backend._client.ttl("key") == -1


def test_redis_backend_applies_a_sub_second_ttl_without_error():
    """A sub-second ttl_seconds (e.g. a misconfigured *_REFRESH_TTL_MINUTES below 1/60) must
    not raise: Redis's EX is integer-second only, so redis-py's ex=timedelta(...) truncates
    anything under a second to EX 0, which Redis rejects as an invalid expire time - PX
    (millisecond) expiry must be used instead.
    """
    backend = _redis_backend()
    backend.set("key", b"value", ttl_seconds=0.5)
    assert backend._client.pttl("key") > 0
    assert backend.get("key") == b"value"


def test_redis_backend_clamps_a_near_zero_ttl_to_one_millisecond():
    """A TTL that rounds to 0 milliseconds must still clamp to at least 1ms (PX 0 is rejected
    by Redis the same way EX 0 is), rather than passing the rounded-to-zero value through.
    """
    backend = _redis_backend()
    backend.set("key", b"value", ttl_seconds=0.0001)
    assert backend._client.pttl("key") >= 0


def test_build_redis_client_configures_socket_timeouts():
    """redis.Redis.from_url() defaults socket_timeout/socket_connect_timeout to None (no bound
    at all) unless given explicitly - a blackholed/unreachable Redis would otherwise hang any
    call on this client indefinitely. build_redis_client must set both.
    """
    client = build_redis_client("redis://localhost:6379", socket_timeout_seconds=_EXPECTED_SOCKET_TIMEOUT_SECONDS)

    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["socket_timeout"] == _EXPECTED_SOCKET_TIMEOUT_SECONDS
    assert kwargs["socket_connect_timeout"] == _EXPECTED_SOCKET_TIMEOUT_SECONDS


def test_build_redis_client_configures_one_connection_retry():
    """from_url() builds a ConnectionPool with no Retry by default, so a single dropped packet
    would surface as a full cache-backend failure; one retry, no backoff, scoped to
    ConnectionError only - not TimeoutError, which would double a blackholed Redis's connect
    stall (see storage.py's _CONNECTION_RETRY).
    """
    client = build_redis_client("redis://localhost:6379", socket_timeout_seconds=2.0)

    retry = client.connection_pool.connection_kwargs["retry"]
    assert retry._retries == 1  # redis-py's Retry exposes no public accessor
    assert isinstance(retry._backoff, NoBackoff)
    assert retry._supported_errors == (RedisConnectionError,)
