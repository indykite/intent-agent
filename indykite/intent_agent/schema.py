"""Fetching and caching the IndyKite graph schema over HTTP."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx2
import jwt
from dogpile.cache.api import NO_VALUE, CacheMutex, KeyType, ValuePayload
from dogpile.cache.backends.redis import RedisBackend
from dogpile.cache.region import AsyncCreator, CacheRegion, make_region

from indykite.intent_agent.config import Settings
from indykite.intent_agent.dependency_status import IKG, REDIS, DependencyStatus, describe_failure
from indykite.intent_agent.errors import InvalidAppAgentKeyError

logger = logging.getLogger(__name__)

_SECONDS_PER_MINUTE = 60


def _make_async_creation_runner(executor: ThreadPoolExecutor) -> AsyncCreator:
    """Build a dogpile async_creation_runner that refreshes an expired entry on `executor`.

    Callers get the stale value immediately, so a request deadline can't cancel the refresh and
    lose it. The runner releases dogpile's mutex itself and never raises.
    """

    def _async_creation_runner(
        cache: CacheRegion, key: KeyType, creator: Callable[[], ValuePayload], mutex: CacheMutex
    ) -> None:
        def _refresh() -> None:
            try:
                schema = creator()
                if schema is not None:
                    cache.set(key, schema)
                else:
                    logger.error("schema_refresh_failed - keeping last known good schema")
            except Exception:
                # Fire-and-forget thread: an uncaught exception (incl. from cache.set) would vanish unlogged.
                logger.exception("schema_refresh_failed - keeping last known good schema")
            finally:
                mutex.release()

        executor.submit(_refresh)

    return _async_creation_runner


def build_schema_cache(settings: Settings, executor: ThreadPoolExecutor) -> CacheRegion:
    """Build the schema cache region (Redis or memory, per CACHE_BACKEND).

    Entries expire after SCHEMA_REFRESH_TTL_MINUTES and are refreshed on demand on `executor`.
    """
    ttl_seconds = settings.schema_refresh_ttl_minutes * _SECONDS_PER_MINUTE
    logger.info("Using cache backend %r with TTL %d seconds", settings.cache_backend, ttl_seconds)
    async_creation_runner = _make_async_creation_runner(executor)
    if settings.cache_backend == "redis":
        return make_region(async_creation_runner=async_creation_runner).configure(
            "dogpile.cache.redis",
            expiration_time=ttl_seconds,
            # `url`, not host/port, so credentials, db and TLS scheme in the DSN are kept.
            arguments={"url": str(settings.redis_url)},
        )
    return make_region(async_creation_runner=async_creation_runner).configure(
        "dogpile.cache.memory", expiration_time=ttl_seconds
    )


def app_agent_key_gid(app_agent_key: str) -> str | None:
    """Return the `sub` claim (environment GID) of an IK_APP_AGENT_KEY JWT, unverified.

    Scopes per-environment cache keys. Returns None for a blank key; raises
    InvalidAppAgentKeyError for a non-blank key that isn't a JWT with a string `sub`.
    """
    if not app_agent_key:
        return None
    message = "IK_APP_AGENT_KEY is not a valid JWT with a 'sub' claim"
    try:
        claims = jwt.decode(app_agent_key, options={"verify_signature": False})
        gid = claims.get("sub")
    except jwt.exceptions.DecodeError as exc:
        logger.exception("IK_APP_AGENT_KEY is not a well-formed JWT")
        raise InvalidAppAgentKeyError(message) from exc
    if not isinstance(gid, str):
        logger.error("IK_APP_AGENT_KEY decoded fine but is missing a string 'sub' claim: %r", gid)
        raise InvalidAppAgentKeyError(message)
    return gid


def _cache_key(app_agent_key: str) -> str:
    """Build the schema cache key scoped to this IndyKite environment."""
    gid = app_agent_key_gid(app_agent_key)
    return "ikg_schema:unconfigured" if gid is None else f"ikg_schema:{gid}"


class SchemaClient:
    """Fetches and caches the IndyKite schema for one base_url/app_agent_key pair."""

    def __init__(
        self,
        base_url: str,
        app_agent_key: str,
        *,
        timeout_seconds: float = 30.0,
        cache: CacheRegion,
        dependency_status: DependencyStatus | None = None,
    ) -> None:
        """Store the connection details; no request is made until load()/preload().

        dependency_status (when given) is told how every fetch went (`ikg`) and, with a Redis
        cache, how every cache access went (`redis`) - what /health reports.
        """
        self._base_url = base_url
        self._app_agent_key = app_agent_key
        self._timeout_seconds = timeout_seconds
        self._cache = cache
        self._cache_key = _cache_key(app_agent_key)
        self._status = dependency_status
        self._cache_is_redis = isinstance(getattr(cache, "actual_backend", None), RedisBackend)

    def load(self) -> dict[str, Any] | None:
        """Fetch the graph schema from the IndyKite platform; None on any failure, never raises."""
        if not self._base_url or not self._app_agent_key:
            logger.warning("IK base URL/app-agent key not set - schema unavailable")
            return None
        url = f"{self._base_url.rstrip('/')}/data-schema/v1"
        logger.debug("Requesting schema from %s", url)
        headers = {"X-IK-ClientKey": self._app_agent_key}
        try:
            with httpx2.Client(timeout=self._timeout_seconds) as client:
                resp = client.get(url, headers=headers)
                logger.debug("Schema response: status=%s body[:500]=%r", resp.status_code, resp.text[:500])
                resp.raise_for_status()
                schema = resp.json()
        except (httpx2.HTTPError, ValueError) as exc:
            logger.exception("Failed to load data schema from %s", url)
            if self._status is not None:
                self._status.record_error(IKG, describe_failure(exc))
            return None
        if self._status is not None:
            self._status.record_ok(IKG)
        logger.info("Loaded data schema from %s", url)
        return schema

    def preload(self) -> None:
        """Fetch and cache the schema once, at startup."""
        schema = self.load()
        if schema is not None:
            self._cache.set(self._cache_key, schema)

    def get_cached(self) -> dict[str, Any] | None:
        """Return the cached schema, fetching it on a cold cache; None if unavailable, never raises.

        An expired entry is served stale while it refreshes in the background, so a failed
        refresh never loses it. With Redis this is network I/O, so it must not run on the event loop.
        """
        try:
            cached = self.peek_cached()
            if self._status is not None and self._cache_is_redis:
                self._status.record_ok(REDIS)
            if cached is not None:
                return cached
            return self._cache.get_or_create(
                self._cache_key, self.load, should_cache_fn=lambda schema: schema is not None
            )
        except Exception as exc:
            logger.exception("Cache backend failed while fetching the schema")
            if self._status is not None and self._cache_is_redis:
                self._status.record_error(REDIS, describe_failure(exc))
            return None

    def peek_cached(self) -> dict[str, Any] | None:
        """Return the cached schema if present (None on a miss), never calling load().

        With Redis this is network I/O, so it must not run on the event loop.
        """
        cached = self._cache.get(self._cache_key)
        logger.debug("cache_key=%s hit=%s", self._cache_key, cached is not NO_VALUE)
        return None if cached is NO_VALUE else cached
