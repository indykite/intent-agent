"""Fetching and caching AuthZEN policies over HTTP.

Mirrors schema.py's SchemaClient closely (same X-IK-ClientKey auth, same on-demand-refresh
cache semantics), with two differences forced by the endpoint itself: policies are scoped to a
subject_type (GET .../access/v1/policies?subject_type=...), and that subject_type is only known
once a request arrives carrying it (see PolicyClient.get_cached) - there is no single
deployment-wide value to preload at cold start the way the schema has, so this client has no
preload() at all, and its cache key includes the subject_type alongside the environment scope.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx2
from dogpile.cache.api import NO_VALUE
from dogpile.cache.region import CacheRegion, make_region

from indykite.intent_agent.config import Settings
from indykite.intent_agent.errors import InvalidAppAgentKeyError
from indykite.intent_agent.schema import app_agent_key_gid

logger = logging.getLogger(__name__)

_SECONDS_PER_MINUTE = 60
# 4xx codes that are transient despite the 4xx range in general meaning "don't retry" - 408
# (the platform itself timed out on that attempt) and 429 (rate-limited) both describe conditions
# a subsequent attempt can plausibly succeed at, unlike e.g. a 401/403/404, which describe the
# request itself as invalid and would fail identically every time (review).
_RETRYABLE_CLIENT_ERROR_STATUSES = frozenset({408, 429})
_CLIENT_ERROR_STATUS_RANGE = range(400, 500)
# How long a failed fetch is cached before the next request tries again (review: without this,
# every request during an outage pays the full retry_attempts x timeout_seconds cost). Deliberately
# much shorter than the success TTL (POLICY_REFRESH_TTL_MINUTES, minutes) - a recovered platform
# should be retried again soon, not left "marked down" for minutes on the strength of one failure.
_FAILURE_CACHE_SECONDS = 30.0


def _is_client_error(exc: httpx2.HTTPStatusError) -> bool:
    """True for a 4xx - a client error (review: e.g. the AppAgent lacks ReadAuthZConfigs, or a
    404) that retrying identically will not fix, unlike a 5xx or connection issue, which might
    be a transient platform blip worth one more try. 408/429 are the exception (review): both
    describe a condition a subsequent attempt can plausibly clear, so they get the same retry
    treatment as a 5xx instead of failing fast after one attempt.
    """
    status = exc.response.status_code
    return status in _CLIENT_ERROR_STATUS_RANGE and status not in _RETRYABLE_CLIENT_ERROR_STATUSES


def _failure_backoff_seconds(cache: CacheRegion) -> float:
    """_FAILURE_CACHE_SECONDS, but never longer than the region's own success TTL (review): a
    POLICY_REFRESH_TTL_MINUTES configured under 30 seconds would otherwise let dogpile's own
    (shorter) freshness check expire a cached _FailedFetch before its own backoff does -
    peek_cached()'s plain cache.get() would then report a miss early, triggering a fresh fetch
    attempt well before the intended backoff window has actually elapsed, defeating the point
    of caching the failure at all in that configuration.
    """
    ttl = cache.expiration_time
    return _FAILURE_CACHE_SECONDS if ttl is None else min(_FAILURE_CACHE_SECONDS, ttl)


@dataclass(frozen=True, slots=True)
class _FailedFetch:
    """A cached failure (review) - carries its own short expiry rather than the region's normal
    (much longer) success TTL, checked manually in get_cached since dogpile.cache applies one
    expiration_time per region, not per entry.

    expires_at is time.time() (wall clock, review), not time.monotonic() - this entry can be
    written by one process and read by another through a shared Redis-backed region
    (CACHE_BACKEND=redis, multiple instances behind a load balancer). monotonic()'s reference
    point is per-process/per-host and undefined across machines, so comparing one host's
    monotonic reading against a value another host wrote is meaningless - depending on which
    host booted first, the backoff could be ignored entirely or treated as active for the
    region's full (much longer) success TTL. Wall-clock time is what every instance actually
    shares, the same assumption dogpile.cache's own Redis TTL already relies on.

    reason (review) is _load's own short failure summary, carried along so get_cached can log
    the specific cause of a fresh failure right where it happens, not just the fact of it.
    """

    expires_at: float
    reason: str


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
    rule schema.py's _cache_key applies to environments alone. subject_type is a graph schema
    node type label and never contains a ":", so joining it after the environment scope (which
    does, e.g. "gid:AAAA...") with the same delimiter schema.py uses stays unambiguous.
    """
    gid = app_agent_key_gid(app_agent_key)
    scope = "unconfigured" if gid is None else gid
    return f"ikg_policies:{scope}:{subject_type}"


_ERROR_BODY_PREVIEW_CHARS = 500


def _parse_results(resp: httpx2.Response) -> list[dict[str, Any]]:
    """Parse and validate a 2xx policy response's body, raising on anything that isn't the
    documented {"results": [...]} envelope (review: a malformed body must count as a failed
    attempt - retried and, on exhaustion, degrading to None/policy_fetch_failed - not escape
    get_cached() as an uncaught exception, and not silently become a successful empty policy
    set that hides a real problem from the caller). Callers must catch TypeError alongside
    ValueError (resp.json() itself raises the latter, a json.JSONDecodeError, on invalid JSON).

    The body preview in either message is truncated the same as the DEBUG-level response log
    above (review) - unlike that one, _load() logs this exception's own text at WARNING once
    retries are exhausted, a level that is always on; a malformed 2xx response's body, however
    large, must not reach a log at that level unbounded.
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

    def _load(self, subject_type: str) -> tuple[list[dict[str, Any]] | None, str]:
        """Fetch this subject_type's policies from the Indykite platform. Returns
        (policies, "") on success, or (None, reason) on failure - reason is a short,
        human-readable summary of the last error, threaded back into get_cached (review) so a
        cached failure can carry and later log its own specific cause (an 401 vs. a timeout vs.
        a malformed body), not just the fact that something failed.

        Retries up to retry_attempts times (bounded by timeout_seconds each) before giving up -
        a transient platform blip must not cost the caller a policy-derived answer it could
        have had a moment later. A 4xx is the one exception (review): a client error (wrong
        scope, not found, ...) will not succeed on retry, so it fails fast after a single
        attempt instead of spending the same retry budget a transient failure gets. Every
        attempt's failure is logged; only the final one at WARNING, since a policies outage is
        never fatal to extraction (run_extraction proceeds without policy context, see
        subject_binding._bind_subject/_policy_fetch_failed).
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
            logger.info("Loaded %d polic%s from %s", len(results), "y" if len(results) == 1 else "ies", url)
            return results, ""
        reason = f"{last_error}"
        logger.warning(
            "Failed to load policies from %s after %d attempt%s: %s",
            url,
            attempt,
            "" if attempt == 1 else "s",
            reason,
        )
        return None, reason

    def get_cached(self, subject_type: str) -> list[dict[str, Any]] | None:
        """Return the cached policies for this subject_type. A cache miss - never fetched, the
        cache region's TTL has expired the success entry, or a cached failure's own shorter
        backoff has elapsed - triggers an on-demand refresh here, the same
        request-driven-not-a-timer refresh schema.get_cached() uses.

        The refresh itself goes through the cache region's own get_or_create (review) rather
        than a manual get-then-set, so concurrent callers observing the same miss/expired
        backoff coalesce into one _load() behind dogpile's own lock instead of each firing its
        own request - but only within this process (review): that lock is a plain
        threading.Lock unless the Redis backend is configured with distributed_lock=True,
        which build_policy_cache does not set, so it stays per-process even with
        CACHE_BACKEND=redis. Separate instances behind a load balancer still share the same
        cached policies (and the same short failure backoff) once one of them writes it, but
        two instances observing a miss/expired backoff at the same moment can each fire their
        own concurrent fetch rather than coalescing across the fleet.

        A refresh that fails keeps serving the last known-good policies for this subject_type
        if one is cached (review) - an earlier version of this method discarded it in favor of
        a failure marker, reasoning that "a fresh attempt is exactly as cheap as serving
        something stale" (true when failures went uncached), but that stopped being true once
        the backoff below was added: every request during the whole backoff window would then
        get schema-only derivation instead of a policy set that was perfectly usable moments
        earlier. Falling back to the stale value re-caches it (its own freshness clock resets
        to now), so the next actual retry is up to POLICY_REFRESH_TTL_MINUTES away rather than
        on literally the next request the way schema.py's async refresh manages - an accepted
        trade against hammering a down platform on every request, not an oversight. Only when
        there is truly nothing cached yet (the very first request for this subject_type) does a
        failure fall back to the marker below instead.

        That re-cache is one more instance of the no-distributed-lock gap above (review): under
        CACHE_BACKEND=redis with multiple instances, if a different instance's own refresh
        succeeds and writes a fresh value while this instance's _load() is still failing, this
        instance's fallback write of its own (older) stale snapshot can land after and overwrite
        that fresher one. Same accepted risk class as the fetch-stampede case already described
        above, just surfacing here as a stale overwrite instead of a duplicate fetch.

        A failure - the marker case above - is itself cached too, briefly (_failure_backoff_
        seconds, review) - without this, every request during an outage would pay the full
        retry_attempts x timeout_seconds cost again, and, just as importantly, a waiting
        concurrent caller would see nothing cached once it is their turn and retry the fetch
        itself rather than reusing the winner's failure. That still never blocks or raises for
        the caller either way - run_extraction proceeds exactly the same
        (policy_fetch_failed=True) whether this returns fast from a fresh backoff or slow from
        a real attempt. Either fallback logs the specific reason _load() failed with (review),
        not just the fact that it did.

        That short backoff is enforced manually rather than through get_or_create's own
        expiration check, since dogpile applies one expiration_time per region (the much longer
        success TTL), not per entry - so before calling get_or_create, a cached failure whose
        own backoff has already elapsed is explicitly evicted here first; left in place,
        dogpile's own (longer) freshness check would still consider it fresh and hand it back
        without ever regenerating. A harmless no-op when nothing was cached at all.

        That eviction re-reads the entry immediately beforehand rather than trusting
        peek_cached()'s earlier read (review): two callers can both observe the same expired
        failure, and if the first already finished _load() and cached a fresh value by the time
        the second reaches this line, an unconditional delete of "whatever is there" would
        destroy that fresh value and force a second, redundant fetch right on top of a recovery
        that had already happened - the exact fetch stampede get_or_create's coalescing exists
        to prevent. Re-checking immediately before deleting closes that specific case (the
        second caller sees the first's fresh write and skips the delete entirely), though
        without dogpile exposing a public compare-and-delete or its own per-key lock, the
        immediate re-check-then-delete itself is not perfectly atomic - only narrowed to a
        window too small to matter in practice.

        Any cache backend failure (e.g. CACHE_BACKEND=redis genuinely unreachable, not just a
        miss) is treated the same as a policy fetch failure rather than left to escape as an
        unhandled exception (review) - run_extraction proceeds schema-only either way.
        """
        try:
            resolved, cached_result = self.peek_cached(subject_type)
            if resolved:
                return cached_result
            cache_key = _cache_key(self._app_agent_key, subject_type)
            current = self._cache.get(cache_key, ignore_expiration=True)
            if isinstance(current, _FailedFetch) and time.time() >= current.expires_at:
                self._cache.delete(cache_key)

            def _fetch_or_cache_failure() -> list[dict[str, Any]] | _FailedFetch:
                policies, reason = self._load(subject_type)
                if policies is not None:
                    return policies
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
        except Exception:
            logger.exception("Cache backend failed while fetching policies for subject_type %r", subject_type)
            return None
        return None if isinstance(result, _FailedFetch) else result

    def peek_cached(self, subject_type: str) -> tuple[bool, list[dict[str, Any]] | None]:
        """Try to resolve subject_type's policies from the cache alone, without ever calling
        _load(). Not necessarily cheap: with CACHE_BACKEND=redis this is a real network round
        trip, not a bounded in-process lookup, so - like get_cached() itself - it must not be
        called directly on the event loop (review, ENG-9671); see
        IntentAgent.get_cached_policies's docstring.

        Returns (True, value) once resolved purely from the cache - an actual hit, or a
        still-fresh cached failure's own backoff (value is None, same as get_cached()'s own
        fast-reject) - no fetch needed either way. Returns (False, None) on a genuine miss:
        get_cached() must still run (via the executor) to call _load() and populate the cache.
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
