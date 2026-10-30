"""Fetching and caching AuthZEN policies over HTTP, per subject_type.

Like schema.py's SchemaClient, but the subject_type is only known per request, so there is no
cold-start preload and the cache key includes the subject_type.
"""

import logging
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import httpx2
from dogpile.cache.api import NO_VALUE
from dogpile.cache.backends.redis import RedisBackend
from dogpile.cache.region import CacheRegion, make_region

from indykite.intent_agent.config import Settings
from indykite.intent_agent.dependency_status import IKG, REDIS, DependencyStatus, describe_failure
from indykite.intent_agent.errors import InvalidAppAgentKeyError
from indykite.intent_agent.schema import app_agent_key_gid

logger = logging.getLogger(__name__)

_SECONDS_PER_MINUTE = 60
# 408 and 429 are the 4xx codes a later attempt can plausibly clear.
_RETRYABLE_CLIENT_ERROR_STATUSES = frozenset({HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.TOO_MANY_REQUESTS})
_CLIENT_ERROR_STATUS_RANGE = range(HTTPStatus.BAD_REQUEST, HTTPStatus.INTERNAL_SERVER_ERROR)
# Failures are cached briefly so an outage doesn't cost every request the full retry budget,
# but far shorter than the success TTL so a recovered platform is retried soon.
_FAILURE_CACHE_SECONDS = 30.0


def _is_client_error(exc: httpx2.HTTPStatusError) -> bool:
    """True for a 4xx that retrying will not fix (408 and 429 are retried like a 5xx)."""
    status = exc.response.status_code
    return status in _CLIENT_ERROR_STATUS_RANGE and status not in _RETRYABLE_CLIENT_ERROR_STATUSES


def _is_answered(exc: Exception) -> bool:
    """True if the platform answered with a client error that isn't an auth rejection."""
    return (
        isinstance(exc, httpx2.HTTPStatusError)
        and _is_client_error(exc)
        and exc.response.status_code not in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}
    )


def _failure_backoff_seconds(cache: CacheRegion) -> float:
    """Return _FAILURE_CACHE_SECONDS capped at the region's success TTL.

    Otherwise a TTL under 30 s would expire a cached failure before its backoff ends.
    """
    ttl = cache.expiration_time
    return _FAILURE_CACHE_SECONDS if ttl is None else min(_FAILURE_CACHE_SECONDS, ttl)


@dataclass(frozen=True, slots=True)
class _FailedFetch:
    """A cached fetch failure with its own short expiry (dogpile has one TTL per region).

    expires_at is wall-clock time.time(), not monotonic, because the entry may be shared across
    hosts through Redis.
    """

    expires_at: float
    reason: str


@dataclass(frozen=True, slots=True)
class PolicyFetchOptions:
    """How long one policy fetch attempt may take, and how many attempts one fetch makes."""

    timeout_seconds: float = 30.0
    retry_attempts: int = 3


def build_policy_cache(settings: Settings) -> CacheRegion:
    """Build the policy cache region (Redis or memory, per CACHE_BACKEND).

    Entries expire after POLICY_REFRESH_TTL_MINUTES and are refreshed on demand by get_cached().
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
    """Build a cache key scoped to this IndyKite environment and subject_type.

    Unambiguous because a node type label never contains ":".
    """
    gid = app_agent_key_gid(app_agent_key)
    scope = "unconfigured" if gid is None else gid
    return f"ikg_policies:{scope}:{subject_type}"


_ERROR_BODY_PREVIEW_CHARS = 500


def _parse_results(resp: httpx2.Response) -> list[dict[str, Any]]:
    """Return the `results` list of a 2xx policy response.

    Raises ValueError on invalid JSON and TypeError on any other shape, so a malformed body
    counts as a failed attempt. The body preview is truncated because it is logged at WARNING.
    """
    body = resp.json()
    if not isinstance(body, dict):
        msg = f"policy response body is not a JSON object: {body!r:.{_ERROR_BODY_PREVIEW_CHARS}}"
        raise TypeError(msg)
    results = body.get("results")
    if not isinstance(results, list):
        msg = f"policy response 'results' is missing or not an array: {body!r:.{_ERROR_BODY_PREVIEW_CHARS}}"
        raise TypeError(msg)
    return results


class PolicyClient:
    """Fetches and caches AuthZEN policies for one base_url/app_agent_key pair, per subject_type."""

    def __init__(
        self,
        base_url: str,
        app_agent_key: str,
        *,
        fetch: PolicyFetchOptions | None = None,
        cache: CacheRegion,
        dependency_status: DependencyStatus | None = None,
    ) -> None:
        """Store the connection details; no request is made until get_cached().

        dependency_status (when given) is told how every fetch went (`ikg`) and, with a Redis
        cache, how every cache access went (`redis`) - what /health reports.
        """
        fetch = fetch or PolicyFetchOptions()
        self._base_url = base_url
        self._app_agent_key = app_agent_key
        self._timeout_seconds = fetch.timeout_seconds
        self._retry_attempts = fetch.retry_attempts
        self._cache = cache
        self._status = dependency_status

    def _load(self, subject_type: str) -> tuple[list[dict[str, Any]] | None, str]:
        """Fetch this subject_type's policies from the IndyKite platform.

        Returns (policies, "") on success or (None, reason) on failure; never raises. Retries up
        to retry_attempts times, except a non-retryable 4xx, which fails after one attempt.
        """
        if not self._base_url or not self._app_agent_key:
            reason = "IK base URL/app-agent key not set"
            logger.warning("%s - policies unavailable", reason)
            return None, reason
        url = f"{self._base_url.rstrip('/')}/access/v1/policies"
        headers = {"X-IK-ClientKey": self._app_agent_key}
        params = {"subject_type": subject_type}
        last_error: Exception | None = None
        attempt = 0
        for attempt in range(1, self._retry_attempts + 1):
            logger.debug("Requesting policies from %s (attempt %d/%d)", url, attempt, self._retry_attempts)
            try:
                with httpx2.Client(timeout=self._timeout_seconds) as client:
                    resp = client.get(url, headers=headers, params=params)
                    logger.debug("Policy response: status=%s body[:500]=%r", resp.status_code, resp.text[:500])
                    resp.raise_for_status()
                    results = _parse_results(resp)
            except httpx2.HTTPStatusError as exc:
                last_error = exc
                if _is_client_error(exc):
                    logger.warning(
                        "Policy fetch from %s got %s - a client error will not succeed on retry: %s",
                        url,
                        exc.response.status_code,
                        exc,
                    )
                    break
                continue
            except (httpx2.HTTPError, ValueError, TypeError) as exc:
                last_error = exc
                continue
            if self._status is not None:
                self._status.record_ok(IKG)
            logger.info("Loaded %d polic%s from %s", len(results), "y" if len(results) == 1 else "ies", url)
            return results, ""
        reason = f"{last_error}"
        # A 4xx other than 401/403 (e.g. no policies for this subject type) means the platform
        # answered, so it says nothing against `ikg`; a rejected key or an outage does.
        if self._status is not None and last_error is not None and not _is_answered(last_error):
            self._status.record_error(IKG, describe_failure(last_error))
        logger.warning(
            "Failed to load policies from %s after %d attempt%s: %s",
            url,
            attempt,
            "" if attempt == 1 else "s",
            reason,
        )
        return None, reason

    def _record_cache(self, *, error: Exception | None = None) -> None:
        """Tell /health how the latest cache access went - only a Redis cache is a dependency."""
        if self._status is None or not isinstance(getattr(self._cache, "actual_backend", None), RedisBackend):
            return
        if error is None:
            self._status.record_ok(REDIS)
        else:
            self._status.record_error(REDIS, describe_failure(error))

    def get_cached(self, subject_type: str) -> tuple[list[dict[str, Any]] | None, bool]:
        """Return (policies, fetch_failed) for subject_type, refreshing on a cache miss; never raises.

        A failed refresh serves the last known-good policies if cached (fetch_failed still True),
        otherwise caches a short-lived _FailedFetch so an outage doesn't retry on every request.
        A cache backend error is treated as a failed fetch.
        """
        try:
            resolved, cached_result = self.peek_cached(subject_type)
            if resolved:
                self._record_cache()
                return cached_result, cached_result is None
            cache_key = _cache_key(self._app_agent_key, subject_type)
            # The region TTL would still treat an elapsed _FailedFetch as fresh, so evict it first.
            current = self._cache.get(cache_key, ignore_expiration=True)
            if isinstance(current, _FailedFetch) and time.time() >= current.expires_at:
                self._cache.delete(cache_key)

            fetch_failed = False

            def _fetch_or_cache_failure() -> list[dict[str, Any]] | _FailedFetch:
                nonlocal fetch_failed
                policies, reason = self._load(subject_type)
                if policies is not None:
                    return policies
                fetch_failed = True
                if isinstance(current, list):
                    logger.warning(
                        "Policy fetch failed for subject_type %r, serving last known-good policies instead: %s",
                        subject_type,
                        reason,
                    )
                    return current
                logger.warning(
                    "Policy fetch failed for subject_type %r, no prior policies to fall back on: %s",
                    subject_type,
                    reason,
                )
                return _FailedFetch(expires_at=time.time() + _failure_backoff_seconds(self._cache), reason=reason)

            result = self._cache.get_or_create(cache_key, _fetch_or_cache_failure)
        except Exception as exc:
            logger.exception("Cache backend failed while fetching policies for subject_type %r", subject_type)
            self._record_cache(error=exc)
            return None, True
        self._record_cache()
        if isinstance(result, _FailedFetch):
            return None, True
        return result, fetch_failed

    def peek_cached(self, subject_type: str) -> tuple[bool, list[dict[str, Any]] | None]:
        """Resolve subject_type's policies from the cache alone, never calling _load().

        Returns (True, policies-or-None) on a hit or an active failure backoff, (False, None) on a
        miss. With Redis this is network I/O, so it must not run on the event loop.
        """
        try:
            cache_key = _cache_key(self._app_agent_key, subject_type)
        except InvalidAppAgentKeyError:
            logger.warning("IK_APP_AGENT_KEY is not a valid JWT - policies unavailable")
            return True, None
        cached = self._cache.get(cache_key)
        if isinstance(cached, _FailedFetch):
            if time.time() < cached.expires_at:
                logger.debug("cache_key=%s hit=recent-failure (backoff active)", cache_key)
                return True, None
            return False, None
        logger.debug("cache_key=%s hit=%s", cache_key, cached is not NO_VALUE)
        if cached is not NO_VALUE:
            return True, cached
        return False, None
