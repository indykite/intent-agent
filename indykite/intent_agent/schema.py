"""Fetching and caching the IndyKite graph schema over HTTP."""

import logging
from typing import Any

import httpx2
import jwt
from dogpile.cache.api import NO_VALUE
from dogpile.cache.region import CacheRegion, make_region

from indykite.intent_agent.config import Settings
from indykite.intent_agent.errors import InvalidAppAgentKeyError

logger = logging.getLogger(__name__)

_SECONDS_PER_MINUTE = 60


def build_schema_cache(settings: Settings) -> CacheRegion:
    """Pick the schema cache backend named by CACHE_BACKEND: Redis, or in-process memory
    (the default) - single-instance deployments never need Redis running. Settings rejects
    CACHE_BACKEND=redis with no REDIS_URL at startup, so redis_url is guaranteed set here.

    Sets expiration_time from SCHEMA_REFRESH_TTL_MINUTES so SchemaClient.get_cached()
    refreshes on-demand once an entry ages past it, instead of a per-instance background
    timer - see get_cached()'s docstring for why that matters with multiple instances.
    """
    ttl_seconds = settings.schema_refresh_ttl_minutes * _SECONDS_PER_MINUTE
    logger.info("Using cache backend %r with TTL %d seconds", settings.cache_backend, ttl_seconds)
    if settings.cache_backend == "redis":
        return make_region().configure(
            "dogpile.cache.redis",
            expiration_time=ttl_seconds,
            # `url` (not host/port) so dogpile hands the DSN as-is to redis.StrictRedis.from_url,
            # preserving username/password/db/TLS scheme instead of dropping them.
            arguments={"url": str(settings.redis_url)},
        )
    return make_region().configure("dogpile.cache.memory", expiration_time=ttl_seconds)


def app_agent_key_gid(app_agent_key: str) -> str | None:
    """Extract the stable, non-secret GID (the `sub` claim) from an IK_APP_AGENT_KEY JWT,
    without verifying its signature (never anything that would require the signing key) - the
    one piece every per-environment cache key (schema, policies) is scoped by, since a single
    shared key would let clients pointed at different IndyKite environments silently overwrite
    each other's cached data if they happen to share a Redis backend/region.

    None for a blank app_agent_key - the supported "IK not configured" state (see
    SchemaClient.load()), which callers map to their own stable placeholder instead of
    decoding, since there's nothing to decode and nothing will ever be fetched/cached under it
    anyway. Any non-blank app_agent_key must be a valid JWT with a string `sub` claim - a
    malformed one is a real configuration error and must fail the process loudly at startup
    with a clear InvalidAppAgentKeyError, not silently degrade to a less specific cache key
    while continuing to attempt a fetch with a broken credential.
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
    """Build a cache key scoped to this specific IndyKite environment - see
    app_agent_key_gid() for why. A blank app_agent_key hashes to a stable placeholder.
    """
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
    ) -> None:
        """Store the connection details; no request is made until load()/preload().

        cache has no default here - build_schema_cache above is the single place that
        decides Redis-vs-Memory, so it's the one place that constructs a default.
        """
        self._base_url = base_url
        self._app_agent_key = app_agent_key
        self._timeout_seconds = timeout_seconds
        self._cache = cache
        self._cache_key = _cache_key(app_agent_key)

    def load(self) -> dict[str, Any] | None:
        """Fetch the graph schema from the Indykite platform, or None on failure."""
        if not self._base_url or not self._app_agent_key:
            logger.warning("IK base URL/app-agent key not set - schema unavailable")
            return None
        url = f"{self._base_url.rstrip('/')}/data-schema/v1"
        logger.debug("Requesting schema from %s", url)
        headers = {"X-IK-ClientKey": self._app_agent_key}
        try:
            with httpx2.Client(timeout=self._timeout_seconds) as client:
                resp = client.get(url, headers=headers)
                # A raw look at what the platform actually sent back, truncated like
                # SCHEMA_PREVIEW_CHARS elsewhere - diagnoses an unexpected/empty schema (or a
                # non-JSON error page) without guessing from the parsed summary line alone.
                logger.debug("Schema response: status=%s body[:500]=%r", resp.status_code, resp.text[:500])
                resp.raise_for_status()
        except httpx2.HTTPError:
            logger.exception("Failed to load data schema from %s", url)
            return None
        logger.info("Loaded data schema from %s", url)
        return resp.json()

    def preload(self) -> None:
        """Fetch and cache the schema once, at startup."""
        schema = self.load()
        if schema is not None:
            self._cache.set(self._cache_key, schema)

    def get_cached(self) -> dict[str, Any] | None:
        """Return the cached schema. A cache miss - never fetched, or the cache
        region's TTL has expired the entry - triggers an on-demand refresh here,
        driven by the request that happens to observe it, rather than a per-instance
        background timer: with multiple agent instances behind a load balancer, a
        timer would have every instance refresh in near-lockstep instead of only
        the one handling traffic doing the work.

        Per TDD 4.4, a failed refresh must never lose the last known-good schema -
        falls back to the expired-but-still-present value instead of returning None,
        so a transient platform outage doesn't take down an already-running agent.
        """
        cached = self.peek_cached()
        if cached is not None:
            return cached
        schema = self.load()
        if schema is not None:
            self._cache.set(self._cache_key, schema)
            return schema
        logger.error("schema_refresh_failed - keeping last known good schema")
        stale = self._cache.get(self._cache_key, ignore_expiration=True)
        return None if stale is NO_VALUE else stale

    def peek_cached(self) -> dict[str, Any] | None:
        """Return the cached schema if present, without ever calling load(). A cheap, bounded
        cache lookup (a dict or Redis round-trip, not the platform's synchronous HTTP call),
        safe for a caller to run directly on the event loop instead of paying a thread-pool
        dispatch for every request regardless of whether a fetch is actually needed (review) -
        only a genuine miss (None here) needs get_cached()'s slower load()-on-miss path.
        """
        cached = self._cache.get(self._cache_key)
        logger.debug("cache_key=%s hit=%s", self._cache_key, cached is not NO_VALUE)
        return None if cached is NO_VALUE else cached
