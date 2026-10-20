"""Fetching and caching AuthZEN policies over HTTP.

Mirrors schema.py's SchemaClient: same auth, same fresh-key/TTL cache mechanics, and the same
serve-stale-immediately-then-refresh-in-background behavior on a stale hit (only a genuinely
cold entry blocks). Differs in two ways: policies are scoped to a subject_type only known once
a request carries it, so this client has no preload() and its cache key includes subject_type
too; and a fetch failure with nothing cached yet is itself cached briefly (see
_failure_backoff_seconds) to avoid retrying every request during an outage - schema needs no
equivalent, since a failed schema fetch always has something to fall back to after the first
successful one.
"""

import json
import logging
from dataclasses import asdict, dataclass
from enum import StrEnum
from http import HTTPStatus
from typing import Any

import httpx2

from indykite.intent_agent.errors import InvalidAppAgentKeyError
from indykite.intent_agent.refresh_gate import RefreshGate
from indykite.intent_agent.schema import CacheConfig, IkConnection, env_scope

logger = logging.getLogger(__name__)

# 408/429 are transient despite the 4xx range generally meaning "don't retry" - both describe
# conditions a retry can plausibly clear, unlike e.g. a 401/403/404.
_RETRYABLE_CLIENT_ERROR_STATUSES = frozenset({HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.TOO_MANY_REQUESTS})
_CLIENT_ERROR_STATUS_RANGE = range(HTTPStatus.BAD_REQUEST, HTTPStatus.INTERNAL_SERVER_ERROR)
# How long a failed fetch is cached (via the fresh marker's own TTL) before retrying -
# deliberately shorter than the success TTL (POLICY_REFRESH_TTL_MINUTES) so a recovered
# platform is retried soon, not "marked down" for minutes on one failure.
_FAILURE_CACHE_SECONDS = 30.0
_ERROR_BODY_PREVIEW_CHARS = 500
_FRESH_MARKER = b"1"


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
    rule schema.py's _cache_key applies to environments alone.
    """
    return f"ikg:{env_scope(app_agent_key)}:policies:{subject_type}"


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
    field is populated. Staleness isn't tracked here - it lives on the corresponding `:fresh`
    marker key's own backend TTL, the same split schema.py uses.
    """

    kind: _EnvelopeKind
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
    at all (wrong cache format, or a kind's required field - policies/reason - missing or
    mistyped). Treated as a miss so callers self-heal, rather than accepting a payload-invalid
    entry as a resolved hit.
    """
    try:
        raw = json.loads(encoded)
    except (ValueError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        kind = _EnvelopeKind(raw.get("kind"))
    except ValueError:  # a missing/unrecognized kind
        return None
    policies = raw.get("policies")
    reason = raw.get("reason")
    if kind == _EnvelopeKind.SUCCESS and not isinstance(policies, list):
        return None
    if kind == _EnvelopeKind.FAILURE and not isinstance(reason, str):
        return None
    return _PolicyEnvelope(kind=kind, policies=policies, reason=reason)


def _encode_success(policies: list[dict[str, Any]]) -> bytes:
    return _encode_envelope(_PolicyEnvelope(kind=_EnvelopeKind.SUCCESS, policies=policies))


def _encode_failure(reason: str) -> bytes:
    return _encode_envelope(_PolicyEnvelope(kind=_EnvelopeKind.FAILURE, reason=reason))


def _as_result(envelope: _PolicyEnvelope) -> tuple[list[dict[str, Any]] | None, bool]:
    """A resolved envelope's (policies, fetch_failed) - a failure envelope carries no policies."""
    if envelope.kind == _EnvelopeKind.SUCCESS:
        return envelope.policies, False
    return None, True


class PolicyClient:
    """Fetches and caches AuthZEN policies for one base_url/app_agent_key pair, per subject_type."""

    def __init__(
        self,
        connection: IkConnection,
        *,
        timeout_seconds: float = 30.0,
        retry_attempts: int = 3,
        cache_config: CacheConfig,
    ) -> None:
        """Store the connection details; no request is made until get_cached().

        cache_config.executor is the same dedicated fetch executor SchemaClient uses for
        background refreshes. cache_config.cache may be shared with a SchemaClient - this
        class never closes it itself, IntentAgent does once for both.
        """
        self._base_url = connection.base_url
        self._app_agent_key = connection.app_agent_key
        self._timeout_seconds = timeout_seconds
        self._retry_attempts = retry_attempts
        self._cache = cache_config.cache
        self._refresh_ttl_seconds = cache_config.refresh_ttl_seconds
        self._executor = cache_config.executor
        self._gate = RefreshGate()

    def _load(self, subject_type: str) -> tuple[list[dict[str, Any]] | None, str]:
        """Fetch this subject_type's policies from the Indykite platform. Returns
        (policies, "") on success, or (None, reason) on failure - reason is a short,
        human-readable summary of the last error, threaded back into get_cached so a cached
        failure can carry and later log its own specific cause (a 401 vs. a timeout vs. a
        malformed body), not just the fact that something failed.

        Retries up to retry_attempts times (bounded by timeout_seconds each) before giving up -
        a transient platform blip must not cost the caller an answer it could have had a
        moment later. A 4xx is the one exception: a client error (wrong scope, not found, ...)
        will not succeed on retry, so it fails fast after a single
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
        encoded = self._cache.get(cache_key)
        if encoded is None:
            return None
        envelope = _decode_envelope(encoded)
        if envelope is None:
            logger.warning(
                "Ignoring unreadable cache value for %r (legacy or corrupted) - treating as a miss", cache_key
            )
        return envelope

    def _load_envelope(self, cache_key: str) -> tuple[_PolicyEnvelope | None, bool]:
        """(envelope, is_fresh) - mirrors SchemaClient._load_schema(): is_fresh is whether
        `{cache_key}:fresh`'s own backend TTL has elapsed, not anything stored in the envelope.
        """
        envelope = self._get_envelope(cache_key)
        if envelope is None:
            return None, False
        fresh_marker = self._cache.get(f"{cache_key}:fresh")
        return envelope, fresh_marker is not None

    def _store_success(self, cache_key: str, policies: list[dict[str, Any]]) -> None:
        """Value gets no TTL (a hard expiry could delete the only stale fallback value during
        a traffic gap) - freshness lives on `{cache_key}:fresh` instead, mirroring schema.py.
        """
        self._cache.set(cache_key, _encode_success(policies))
        self._cache.set(f"{cache_key}:fresh", _FRESH_MARKER, ttl_seconds=self._refresh_ttl_seconds)

    def _store_failure(self, cache_key: str, reason: str) -> None:
        """Same split as _store_success, but the fresh marker's TTL is the short failure
        backoff instead of the full refresh TTL - get_cached() re-fetches once it elapses.
        """
        self._cache.set(cache_key, _encode_failure(reason))
        backoff_seconds = _failure_backoff_seconds(self._refresh_ttl_seconds)
        self._cache.set(f"{cache_key}:fresh", _FRESH_MARKER, ttl_seconds=backoff_seconds)

    def _fetch_and_store(self, cache_key: str, subject_type: str) -> tuple[list[dict[str, Any]] | None, bool]:
        """Fetch subject_type's policies and cache the outcome - shared by the cold-start
        blocking path and _refresh()'s background path.

        A failed refresh keeps serving the last known-good policies if one is cached,
        re-caching it so the next retry isn't due until POLICY_REFRESH_TTL_MINUTES later. Only
        a failure with nothing cached yet falls back to a failure marker (briefly cached too,
        see _failure_backoff_seconds). Must not raise: a cache-backend error reading the
        current envelope, or a _store_*() failure writing the outcome, is caught and logged
        here - otherwise a cold request would surface as an unhandled exception instead of
        the usual direct-fetch/unavailable fallback, and a background refresh (this also runs
        detached on self._executor via _refresh()) would fail silently.
        """
        try:
            current_envelope = self._get_envelope(cache_key)
        except Exception:
            logger.exception(
                "Cache backend failed while reading current policies for subject_type %r - "
                "proceeding without a last-known-good fallback",
                subject_type,
            )
            current_envelope = None
        current_policies = (
            current_envelope.policies
            if current_envelope is not None and current_envelope.kind == _EnvelopeKind.SUCCESS
            else None
        )
        policies, reason = self._load(subject_type)
        if policies is not None:
            try:
                self._store_success(cache_key, policies)
            except Exception:
                logger.exception(
                    "Failed to cache freshly loaded policies for subject_type %r - serving them uncached this request",
                    subject_type,
                )
            return policies, False
        if current_policies is not None:
            logger.warning(
                "Policy fetch failed for subject_type %r, serving last known-good policies instead: %s",
                subject_type,
                reason,
            )
            try:
                self._store_success(cache_key, current_policies)
            except Exception:
                logger.exception("Failed to re-cache last known-good policies for subject_type %r", subject_type)
            return current_policies, True
        logger.warning(
            "Policy fetch failed for subject_type %r, no prior policies to fall back on: %s",
            subject_type,
            reason,
        )
        try:
            self._store_failure(cache_key, reason)
        except Exception:
            logger.exception("Failed to cache policy failure marker for subject_type %r", subject_type)
        return None, True

    def _refresh(self, cache_key: str, subject_type: str) -> None:
        """Reload and re-store - run on self._executor by RefreshGate, never the caller's own
        thread. Mirrors SchemaClient._refresh()."""
        self._fetch_and_store(cache_key, subject_type)

    def _direct_fetch(self, subject_type: str) -> tuple[list[dict[str, Any]] | None, bool]:
        """Fetch subject_type's policies with no cache interaction at all, not even an attempt
        to store the result - used when the cache backend has already proven unreachable this
        call, so retrying it (even just to look up a last-known-good fallback) would only
        repeat the same failure. Mirrors SchemaClient.get_cached()'s `return self.load()`.
        """
        policies, _reason = self._load(subject_type)
        return policies, policies is None

    def get_cached(self, subject_type: str) -> tuple[list[dict[str, Any]] | None, bool]:
        """Return (policies, fetch_failed) for this subject_type.

        A fresh success returns immediately. A stale success or cached failure also returns
        immediately, dispatching a background refresh (RefreshGate) instead - only a
        genuinely cold entry blocks. fetch_failed is True whenever the live fetch (or the
        cached failure served instead) did not succeed, even if policies is a real list served
        from the stale-fallback path. A cache-backend error falls through to a direct, uncached
        fetch - same posture as SchemaClient.get_cached().
        """
        try:
            cache_key = self._cache_key_for(subject_type)
        except InvalidAppAgentKeyError:
            logger.warning("IK_APP_AGENT_KEY is not a valid JWT - policies unavailable")
            return None, True
        try:
            envelope, fresh = self._load_envelope(cache_key)
        except Exception:
            logger.exception(
                "Cache backend failed while reading policies for subject_type %r - falling back to a direct fetch",
                subject_type,
            )
            return self._direct_fetch(subject_type)
        if envelope is not None:
            logger.debug("cache_key=%s hit=True fresh=%s", cache_key, fresh)
            if not fresh:
                self._gate.refresh_in_background_if_free(
                    cache_key, lambda: self._refresh(cache_key, subject_type), self._executor
                )
            return _as_result(envelope)
        logger.debug("cache_key=%s hit=False", cache_key)
        with self._gate.acquire_blocking(cache_key):
            try:
                envelope, _fresh = self._load_envelope(cache_key)
            except Exception:
                logger.exception(
                    "Cache backend failed while reading policies for subject_type %r - falling back to a direct fetch",
                    subject_type,
                )
                return self._direct_fetch(subject_type)
            if envelope is not None:
                return _as_result(envelope)
            return self._fetch_and_store(cache_key, subject_type)

    def peek_cached(self, subject_type: str) -> tuple[bool, list[dict[str, Any]] | None]:
        """Try to resolve subject_type's policies from the cache alone, without ever calling
        _load(). Not necessarily cheap: with CACHE_BACKEND=redis this is a real network round
        trip, not a bounded in-process lookup, so - like get_cached() itself - it must not be
        called directly on the event loop; see IntentAgent.get_cached_policies's docstring.

        Returns (True, value) once resolved purely from the cache: a fresh success, or a
        still-fresh cached failure (value is None). A stale entry counts as a miss here,
        unlike get_cached() - mirrors SchemaClient.peek_cached().
        """
        try:
            cache_key = self._cache_key_for(subject_type)
        except InvalidAppAgentKeyError:
            logger.warning("IK_APP_AGENT_KEY is not a valid JWT - policies unavailable")
            return True, None
        envelope, fresh = self._load_envelope(cache_key)
        if envelope is None:
            logger.debug("cache_key=%s hit=False", cache_key)
            return False, None
        logger.debug("cache_key=%s hit=%s", cache_key, fresh)
        if not fresh:
            return False, None
        return (True, envelope.policies) if envelope.kind == _EnvelopeKind.SUCCESS else (True, None)
