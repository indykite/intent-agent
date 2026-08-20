"""Fetching and caching the IndyKite graph schema over HTTP."""

import logging
from typing import Any

import httpx2
import jwt
from dogpile.cache.api import NO_VALUE
from dogpile.cache.region import CacheRegion

from indykite.intent_agent.errors import InvalidAppAgentKeyError

logger = logging.getLogger(__name__)


def _cache_key(app_agent_key: str) -> str:
    """Build a cache key scoped to this specific IndyKite environment.

    A single shared key would let multiple SchemaClient instances - e.g. pointing at different
    IndyKite environments - silently overwrite each other's schema if they happen to share a
    Redis backend/region.

    A blank app_agent_key is a supported "IK not configured" state (see SchemaClient.load()) -
    hashes to a stable placeholder instead of decoding, since there's nothing to decode and
    no schema will ever be fetched/cached under it anyway. Any non-blank app_agent_key must be
    a valid JWT with a `sub` claim (a stable, non-secret GID, read without verifying the
    signature - never anything that would require the signing key) - a malformed one is a
    real configuration error and must fail the process loudly at startup with a clear
    InvalidAppAgentKeyError, not silently fall back to a less specific cache key while
    continuing to attempt a schema fetch with a broken credential.
    """
    if not app_agent_key:
        return "ikg_schema:unconfigured"
    message = "IK_APP_AGENT_KEY is not a valid JWT with a 'sub' claim"
    try:
        claims = jwt.decode(app_agent_key, options={"verify_signature": False})
        gid = claims.get("sub")
    except jwt.exceptions.DecodeError as exc:
        raise InvalidAppAgentKeyError(message) from exc
    if not isinstance(gid, str):
        raise InvalidAppAgentKeyError(message)
    return f"ikg_schema:{gid}"


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

        cache has no default here - agent.py's _build_schema_cache is the single place
        that decides Redis-vs-Memory, so it's the one place that constructs a default.
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
        headers = {"X-IK-ClientKey": self._app_agent_key}
        try:
            with httpx2.Client(timeout=self._timeout_seconds) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
        except httpx2.HTTPError:
            logger.warning("Failed to load data schema from %s", url)
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
        cached = self._cache.get(self._cache_key)
        logger.debug("cache_key=%s hit=%s", self._cache_key, cached is not NO_VALUE)
        if cached is not NO_VALUE:
            return cached
        schema = self.load()
        if schema is not None:
            self._cache.set(self._cache_key, schema)
            return schema
        logger.error("schema_refresh_failed - keeping last known good schema")
        stale = self._cache.get(self._cache_key, ignore_expiration=True)
        return None if stale is NO_VALUE else stale
