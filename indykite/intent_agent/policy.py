"""Fetching and caching AuthZEN policies over HTTP.

Mirrors schema.py's SchemaClient closely (same X-IK-ClientKey auth, same on-demand-refresh
cache semantics), with two differences forced by the endpoint itself: policies are scoped to a
subject_type (GET .../access/v1/policies?subject_type=...), and that subject_type is only known
once a request arrives carrying it (see PolicyClient.get_cached) - there is no single
deployment-wide value to preload at cold start the way the schema has, so this client has no
preload() at all, and its cache key includes the subject_type alongside the environment scope.
"""

import logging
from typing import Any

import httpx2
from dogpile.cache.api import NO_VALUE
from dogpile.cache.region import CacheRegion, make_region

from indykite.intent_agent.config import Settings
from indykite.intent_agent.errors import InvalidAppAgentKeyError
from indykite.intent_agent.schema import app_agent_key_gid

logger = logging.getLogger(__name__)

_SECONDS_PER_MINUTE = 60


def build_policy_cache(settings: Settings) -> CacheRegion:
    """Pick the policy cache backend named by CACHE_BACKEND - same backend as the schema
    cache (Redis, or in-process memory), just its own region so a policy entry's TTL/eviction
    never collides with a schema entry's.

    Sets expiration_time from POLICY_REFRESH_TTL_MINUTES so PolicyClient.get_cached()
    refreshes on-demand once an entry ages past it, same on-demand-not-a-timer rationale as
    build_schema_cache (multiple instances behind a load balancer must not all refresh in
    lockstep).
    """
    ttl_seconds = settings.policy_refresh_ttl_minutes * _SECONDS_PER_MINUTE
    if settings.cache_backend == "redis":
        return make_region().configure(
            "dogpile.cache.redis",
            expiration_time=ttl_seconds,
            arguments={"url": str(settings.redis_url)},
        )
    return make_region().configure("dogpile.cache.memory", expiration_time=ttl_seconds)


def _cache_key(app_agent_key: str, subject_type: str) -> str:
    """Build a cache key scoped to this IndyKite environment *and* subject_type - two requests
    for different subject types must never read each other's policies, the same non-collision
    rule schema.py's _cache_key applies to environments alone.
    """
    gid = app_agent_key_gid(app_agent_key)
    scope = "unconfigured" if gid is None else gid
    return f"ikg_policies:{scope}:{subject_type}"


class PolicyClient:
    """Fetches and caches AuthZEN policies for one base_url/app_agent_key pair, per subject_type."""

    def __init__(
        self,
        base_url: str,
        app_agent_key: str,
        *,
        timeout_seconds: float = 30.0,
        retry_attempts: int = 3,
        cache: CacheRegion,
    ) -> None:
        """Store the connection details; no request is made until get_cached()."""
        self._base_url = base_url
        self._app_agent_key = app_agent_key
        self._timeout_seconds = timeout_seconds
        self._retry_attempts = retry_attempts
        self._cache = cache

    def _load(self, subject_type: str) -> list[dict[str, Any]] | None:
        """Fetch this subject_type's policies from the Indykite platform, or None on failure.

        Retries up to retry_attempts times (bounded by timeout_seconds each) before giving up -
        a transient platform blip must not cost the caller a policy-derived answer it could
        have had a moment later. Every attempt's failure is logged; only the final one at
        WARNING, since a policies outage is never fatal to extraction (run_extraction proceeds
        without policy context, see subject_binding._bind_subject/_policy_fetch_failed).
        """
        if not self._base_url or not self._app_agent_key:
            logger.warning("IK base URL/app-agent key not set - policies unavailable")
            return None
        url = f"{self._base_url.rstrip('/')}/access/v1/policies"
        headers = {"X-IK-ClientKey": self._app_agent_key}
        params = {"subject_type": subject_type}
        last_error: httpx2.HTTPError | None = None
        for attempt in range(1, self._retry_attempts + 1):
            logger.debug("Requesting policies from %s (attempt %d/%d)", url, attempt, self._retry_attempts)
            try:
                with httpx2.Client(timeout=self._timeout_seconds) as client:
                    resp = client.get(url, headers=headers, params=params)
                    logger.debug("Policy response: status=%s body[:500]=%r", resp.status_code, resp.text[:500])
                    resp.raise_for_status()
            except httpx2.HTTPError as exc:
                last_error = exc
                continue
            results = resp.json().get("results", [])
            logger.info("Loaded %d polic%s from %s", len(results), "y" if len(results) == 1 else "ies", url)
            return results
        logger.warning("Failed to load policies from %s after %d attempts: %s", url, self._retry_attempts, last_error)
        return None

    def get_cached(self, subject_type: str) -> list[dict[str, Any]] | None:
        """Return the cached policies for this subject_type. A cache miss - never fetched, or
        the cache region's TTL has expired the entry - triggers an on-demand refresh here, the
        same request-driven-not-a-timer refresh schema.get_cached() uses.

        Unlike the schema, a failed refresh here falls back to None, not a stale value: there
        is no "last known good" to preserve on the very first request for a given subject_type
        (no preload happened), and by the time a later request's entry could go stale, a fresh
        attempt (with its own retry budget) is exactly as cheap as serving something stale
        would be - policies are used at request scope, unlike the schema's process-lifetime
        assumption of "some schema is better than none".
        """
        try:
            cache_key = _cache_key(self._app_agent_key, subject_type)
        except InvalidAppAgentKeyError:
            logger.warning("IK_APP_AGENT_KEY is not a valid JWT - policies unavailable")
            return None
        cached = self._cache.get(cache_key)
        logger.debug("cache_key=%s hit=%s", cache_key, cached is not NO_VALUE)
        if cached is not NO_VALUE:
            return cached
        policies = self._load(subject_type)
        if policies is not None:
            self._cache.set(cache_key, policies)
        return policies
