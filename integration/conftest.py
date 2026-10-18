"""Shared real-Redis reachability check, extracted once three test modules needed the same
probe instead of duplicating it.
"""

import os

import redis
import redis.exceptions

REAL_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
# Bounds the probe so an unreachable REDIS_URL skips instead of hanging pytest collection.
_PROBE_TIMEOUT_SECONDS = 2.0


def real_redis_reachable() -> bool:
    """True if a real Redis actually answers PING at REAL_REDIS_URL."""
    try:
        client = redis.Redis.from_url(
            REAL_REDIS_URL, socket_timeout=_PROBE_TIMEOUT_SECONDS, socket_connect_timeout=_PROBE_TIMEOUT_SECONDS
        )
        try:
            return client.ping()
        finally:
            client.close()
    except redis.exceptions.RedisError:
        return False
