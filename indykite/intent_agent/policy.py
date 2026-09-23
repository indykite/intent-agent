"""Fetching and caching AuthZEN policies over HTTP.

Mirrors schema.py's SchemaClient (same X-IK-ClientKey auth, same on-demand-refresh cache
semantics), except policies are scoped to a subject_type only known once a request carries it
- so this client has no preload(), and its cache key includes subject_type too.

Unlike SchemaClient, refreshes here are fully synchronous/blocking - a stale or cold entry
both go through the same locked fetch-and-fallback path.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from http import HTTPStatus
from typing import Any

import httpx2

from indykite.intent_agent.config import Settings, StorageBackendKind
from indykite.intent_agent.errors import InvalidAppAgentKeyError
from indykite.intent_agent.refresh_gate import RefreshGate
from indykite.intent_agent.schema import IkConnection, app_agent_key_gid
from indykite.redis.storage import MemoryStorageBackend, StorageBackend, build_redis_storage_backend, run_sync

logger = logging.getLogger(__name__)

_SECONDS_PER_MINUTE = 60
# 408/429 are transient despite the 4xx range generally meaning "don't retry" - both describe
# conditions a retry can plausibly clear, unlike e.g. a 401/403/404.
_RETRYABLE_CLIENT_ERROR_STATUSES = frozenset({HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.TOO_MANY_REQUESTS})
_CLIENT_ERROR_STATUS_RANGE = range(HTTPStatus.BAD_REQUEST, HTTPStatus.INTERNAL_SERVER_ERROR)
# How long a failed fetch is cached before retrying - deliberately shorter than the success
# TTL (POLICY_REFRESH_TTL_MINUTES) so a recovered platform is retried soon, not "marked down"
# for minutes on one failure.
_FAILURE_CACHE_SECONDS = 30.0
_ERROR_BODY_PREVIEW_CHARS = 500


def build_policy_cache(settings: Settings) -> StorageBackend:
    """Pick the policy cache backend named by CACHE_BACKEND - same backend as the schema
    cache (Redis, or in-process memory), just its own keys (see _cache_key) so a policy
    entry's TTL/eviction never collides with a schema entry's. Never imports redis.asyncio
    directly - see schema.build_schema_backend's matching docstring.
    """
    if settings.cache_backend == StorageBackendKind.REDIS:
        return build_redis_storage_backend(
            str(settings.redis_url), socket_timeout_seconds=settings.redis_socket_timeout_seconds
        )
    return MemoryStorageBackend()


def _is_client_error(exc: httpx2.HTTPStatusError) -> bool:
    """True for a 4xx - a client error (e.g. the AppAgent lacks ReadAuthZConfigs, or a 404)
    that retrying will not fix, unlike a 5xx. 408/429 are the exception, retried like a 5xx.
    """
    status = exc.response.status_code
    return status in _CLIENT_ERROR_STATUS_RANGE and status not in _RETRYABLE_CLIENT_ERROR_STATUSES


def _failure_backoff_seconds(refresh_ttl_seconds: float) -> float:
    """_FAILURE_CACHE_SECONDS, but never longer than the success TTL - a short
    POLICY_REFRESH_TTL_MINUTES must not let a cached failure outlive it.
    """
    return min(_FAILURE_CACHE_SECONDS, refresh_ttl_seconds)


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


def _parse_results(resp: httpx2.Response) -> list[dict[str, Any]]:
    """Parse and validate a 2xx policy response's body, raising on anything that isn't the
    documented {"results": [...]} envelope - a malformed body must count as a failed attempt,
    not a successful empty policy set. Callers must catch TypeError alongside ValueError
    (resp.json() itself raises the latter on invalid JSON).
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


class _EnvelopeKind(StrEnum):
    """_PolicyEnvelope.kind's values - StrEnum so a typo in a comparison is a NameError, not a
    silently-always-false string comparison. Serializes to the same bare strings either way."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class _PolicyEnvelope:
    """What's actually stored under one policy cache key - a successful fetch's policies, or
    a failure's reason, tagged by `kind` so callers know which without probing which optional
    field is populated. A success entry's `reason` (and a failure entry's `policies`) stays
    unset - not just None but absent from the encoded JSON entirely, matching this envelope's
    pre-dataclass shape so an already-cached value round-trips unchanged across a rollout.
    """

    kind: _EnvelopeKind
    cached_at: float
    policies: list[dict[str, Any]] | None = None
    reason: str | None = None


def _encode_envelope(envelope: _PolicyEnvelope) -> bytes:
    """Serialize a _PolicyEnvelope as compact JSON, dropping whichever of policies/reason is
    unset for this envelope's kind.
    """
    payload = {key: value for key, value in asdict(envelope).items() if value is not None}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _decode_envelope(encoded: bytes) -> _PolicyEnvelope | None:
    """Inverse of _encode_envelope - or None if encoded doesn't decode into this envelope shape
    at all (wrong cache format, a kind's required field - policies/reason - missing or
    mistyped, or cached_at not numeric). Treated as a miss so _resolve() self-heals, rather than
    accepting a payload-invalid entry as a resolved hit or raising later on `time.time() -
    cached_at`.
    """
    try:
        raw = json.loads(encoded)
        if not isinstance(raw, dict):
            message = "unrecognized cache envelope shape"
            raise ValueError(message)  # noqa: TRY301, TRY004 - ValueError kept consistent with the other shape checks here
        kind = _EnvelopeKind(raw.get("kind"))  # raises ValueError for a missing/unrecognized kind
        cached_at = raw["cached_at"]
        policies = raw.get("policies")
        reason = raw.get("reason")
        if not isinstance(cached_at, int | float) or isinstance(cached_at, bool):
            message = "envelope cached_at is not numeric"
            raise ValueError(message)  # noqa: TRY301, TRY004 - ValueError kept consistent with the other shape checks here
        if kind == _EnvelopeKind.SUCCESS and not isinstance(policies, list):
            message = "success envelope missing a policies list"
            raise ValueError(message)  # noqa: TRY301 - caught by the except right below, not left to propagate
        if kind == _EnvelopeKind.FAILURE and not isinstance(reason, str):
            message = "failure envelope missing a reason string"
            raise ValueError(message)  # noqa: TRY301 - caught by the except right below, not left to propagate
    except (ValueError, TypeError, KeyError):
        return None
    else:
        return _PolicyEnvelope(kind=kind, cached_at=cached_at, policies=policies, reason=reason)


def _encode_success(policies: list[dict[str, Any]]) -> bytes:
    return _encode_envelope(_PolicyEnvelope(kind=_EnvelopeKind.SUCCESS, cached_at=time.time(), policies=policies))


def _encode_failure(reason: str) -> bytes:
    return _encode_envelope(_PolicyEnvelope(kind=_EnvelopeKind.FAILURE, cached_at=time.time(), reason=reason))


class PolicyClient:
    """Fetches and caches AuthZEN policies for one base_url/app_agent_key pair, per subject_type."""

    def __init__(
        self,
        connection: IkConnection,
        *,
        timeout_seconds: float = 30.0,
        retry_attempts: int = 3,
        cache: StorageBackend,
        refresh_ttl_seconds: float,
    ) -> None:
        """Store the connection details; no request is made until get_cached()."""
        self._base_url = connection.base_url
        self._app_agent_key = connection.app_agent_key
        self._timeout_seconds = timeout_seconds
        self._retry_attempts = retry_attempts
        self._cache = cache
        self._refresh_ttl_seconds = refresh_ttl_seconds
        self._gate = RefreshGate()

    async def aclose(self) -> None:
        """Close the cache backend's own resources - called from app.py's lifespan on shutdown.
        Routed through run_sync like every other cache call - see SchemaClient.aclose's matching
        docstring for why closing it directly here would raise instead.
        """
        run_sync(self._cache.aclose())

    def _load(self, subject_type: str) -> tuple[list[dict[str, Any]] | None, str]:
        """Fetch this subject_type's policies from the Indykite platform. Returns
        (policies, "") on success, or (None, reason) on failure - reason is a short,
        human-readable summary of the last error, threaded back into get_cached so a cached
        failure can carry and later log its own specific cause (a 401 vs. a timeout vs. a
        malformed body), not just the fact that something failed.

        Retries up to retry_attempts times (bounded by timeout_seconds each) before giving up -
        a transient platform blip must not cost the caller an answer it could have had a
        moment later. A 4xx is the one exception (review): a client error (wrong
        scope, not found, ...) will not succeed on retry, so it fails fast after a single
        attempt instead of spending the same retry budget a transient failure gets. Every
        attempt's failure is logged; only the final one at WARNING - a policies outage fails
        only the requests that arrive during it (run_extraction's policy_unavailable_error,
        unless last known-good policies are cached), never the process.
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

    def _cache_key_for(self, subject_type: str) -> str:
        return _cache_key(self._app_agent_key, subject_type)

    def _get_envelope(self, cache_key: str) -> _PolicyEnvelope | None:
        """Decode cache_key's stored value via _decode_envelope. Treated as a miss, not
        raised, so get_cached() self-heals instead of treating every later call as a backend
        failure.
        """
        encoded = run_sync(self._cache.get(cache_key))
        if encoded is None:
            return None
        envelope = _decode_envelope(encoded)
        if envelope is None:
            logger.warning(
                "Ignoring unreadable cache value for %r (legacy or corrupted) - treating as a miss", cache_key
            )
        return envelope

    def _store_success(self, cache_key: str, policies: list[dict[str, Any]]) -> None:
        """Cache with no physical TTL - a hard expiry could delete the only stale fallback
        value during a traffic gap, breaking the "keep serving the last known-good policies"
        guarantee on get_cached().
        """
        run_sync(self._cache.set(cache_key, _encode_success(policies)))

    def _store_failure(self, cache_key: str, reason: str) -> None:
        """Also no physical TTL, for consistency - get_cached()'s own backoff check against
        cached_at is what governs reuse, not a backend expiry.
        """
        run_sync(self._cache.set(cache_key, _encode_failure(reason)))

    def _resolve(self, cache_key: str) -> tuple[bool, list[dict[str, Any]] | None]:
        """Try to answer purely from the cache: (True, value) once resolved (a fresh success,
        or a still-backing-off cached failure), or (False, None) when a fetch is required.
        Shared by peek_cached (never fetches) and get_cached (fetches when unresolved).
        """
        envelope = self._get_envelope(cache_key)
        if envelope is None:
            return False, None
        if envelope.kind == _EnvelopeKind.FAILURE:
            if time.time() - envelope.cached_at < _failure_backoff_seconds(self._refresh_ttl_seconds):
                return True, None
            return False, None
        if time.time() - envelope.cached_at < self._refresh_ttl_seconds:
            return True, envelope.policies
        return False, None

    def get_cached(self, subject_type: str) -> tuple[list[dict[str, Any]] | None, bool]:
        """Return (policies, fetch_failed) for this subject_type. A cache miss, a stale
        success, or a cached failure past its own backoff triggers a fetch, behind a
        per-cache-key lock (RefreshGate) so concurrent callers coalesce into one _load().

        fetch_failed is True whenever the live fetch (or the cached failure served instead)
        did not succeed - even when policies is still a real, non-None list served from the
        stale-fallback path below.

        A refresh that fails keeps serving the last known-good policies if one is cached,
        re-caching it so the next retry is up to POLICY_REFRESH_TTL_MINUTES away rather than
        on the next request. Only when nothing is cached yet does a failure fall back to a
        failure marker, itself cached briefly (see _failure_backoff_seconds) so an outage
        doesn't pay the full retry cost on every request.

        Any cache backend failure is treated the same as a policy fetch failure rather than
        left to escape as an unhandled exception.
        """
        try:
            resolved, cached_result = self.peek_cached(subject_type)
            if resolved:
                return cached_result, cached_result is None
            # peek_cached already resolved app_agent_key_gid successfully above without
            # raising, so calling it again here (pure function of instance state + subject_type)
            # is safe to assume will not raise either.
            cache_key = self._cache_key_for(subject_type)
            with self._gate.acquire_blocking(cache_key):
                resolved, cached_result = self._resolve(cache_key)
                if resolved:
                    return cached_result, cached_result is None
                current_envelope = self._get_envelope(cache_key)
                current_policies = (
                    current_envelope.policies
                    if current_envelope is not None and current_envelope.kind == _EnvelopeKind.SUCCESS
                    else None
                )
                policies, reason = self._load(subject_type)
                if policies is not None:
                    self._store_success(cache_key, policies)
                    return policies, False
                if current_policies is not None:
                    logger.warning(
                        "Policy fetch failed for subject_type %r, serving last known-good policies instead: %s",
                        subject_type,
                        reason,
                    )
                    self._store_success(cache_key, current_policies)
                    return current_policies, True
                logger.warning(
                    "Policy fetch failed for subject_type %r, no prior policies to fall back on: %s",
                    subject_type,
                    reason,
                )
                self._store_failure(cache_key, reason)
                return None, True
        except Exception:
            logger.exception("Cache backend failed while fetching policies for subject_type %r", subject_type)
            return None, True

    def peek_cached(self, subject_type: str) -> tuple[bool, list[dict[str, Any]] | None]:
        """Try to resolve subject_type's policies from the cache alone, without ever calling
        _load(). Not necessarily cheap: with CACHE_BACKEND=redis this is a real network round
        trip, not a bounded in-process lookup, so - like get_cached() itself - it must not be
        called directly on the event loop; see IntentAgent.get_cached_policies's docstring.

        Returns (True, value) once resolved purely from the cache - an actual hit, or a
        still-fresh cached failure's own backoff (value is None, same as get_cached()'s own
        fast-reject) - no fetch needed either way. Returns (False, None) on a genuine miss:
        get_cached() must still run (via the executor) to call _load() and populate the cache.
        """
        try:
            cache_key = self._cache_key_for(subject_type)
        except InvalidAppAgentKeyError:
            logger.warning("IK_APP_AGENT_KEY is not a valid JWT - policies unavailable")
            return True, None
        resolved, result = self._resolve(cache_key)
        logger.debug("cache_key=%s hit=%s", cache_key, resolved)
        return resolved, result
