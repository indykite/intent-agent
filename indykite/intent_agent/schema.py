"""Fetching and caching the IndyKite graph schema over HTTP."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx2
import jwt

from indykite.intent_agent.config import Settings, StorageBackendKind
from indykite.intent_agent.errors import InvalidAppAgentKeyError
from indykite.intent_agent.refresh_gate import RefreshGate
from indykite.redis.storage import MemoryStorageBackend, StorageBackend, build_redis_storage_backend

logger = logging.getLogger(__name__)

_SECONDS_PER_MINUTE = 60


@dataclass(frozen=True, slots=True)
class IkConnection:
    """base_url + app_agent_key, bundled so SchemaClient/PolicyClient each take one param
    instead of two - keeps __init__ under ruff's PLR0913 without reaching for full Settings.
    """

    base_url: str
    app_agent_key: str


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """cache + refresh_ttl_seconds + executor, bundled like IkConnection bundles base_url/
    app_agent_key, so SchemaClient/PolicyClient each take one param instead of three - cache
    and executor are the same shared instances for both clients, refresh_ttl_seconds differs.
    """

    cache: StorageBackend
    refresh_ttl_seconds: float
    executor: ThreadPoolExecutor


def build_cache_backend(settings: Settings) -> StorageBackend:
    """Pick the cache backend named by CACHE_BACKEND: Redis, or in-process memory (the default)
    - shared by SchemaClient and PolicyClient, which used to each open their own Redis client
    for the same Redis before IntentAgent started building this once for both.

    Never imports redis directly - storage.py is the one place that knows how to build a
    Redis-backed StorageBackend; this module only depends on the StorageBackend ABC.
    """
    logger.info("Using cache backend %r", settings.cache_backend)
    if settings.cache_backend == StorageBackendKind.REDIS:
        return build_redis_storage_backend(
            str(settings.redis_url), socket_timeout_seconds=settings.redis_socket_timeout_seconds
        )
    return MemoryStorageBackend()


def app_agent_key_gid(app_agent_key: str) -> str | None:
    """Extract the stable, non-secret GID (the `sub` claim) from an IK_APP_AGENT_KEY JWT,
    without verifying its signature - the piece every per-environment cache key (schema,
    policies) is scoped by, so clients on different IndyKite environments never overwrite
    each other's cached data on a shared Redis backend.

    None for a blank app_agent_key (the "IK not configured" state, see SchemaClient.load()).
    Any non-blank app_agent_key must be a valid JWT with a string `sub` claim - a malformed
    one is a configuration error and fails the process loudly, rather than silently degrading
    to a less specific cache key.
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


def env_scope(app_agent_key: str) -> str:
    """The per-environment scope every cache key is namespaced under - see app_agent_key_gid()
    for why. Every key lives under `ikg:<scope>:*`, so one Redis ACL rule can isolate one
    instance/credential, unlike the old unrelated `ikg_schema:`/`ikg_policies:` prefixes.

    Raises InvalidAppAgentKeyError for a blank key rather than a shared "unconfigured" scope -
    unreachable today (Settings requires it non-blank), but kept, a future no-key caller would
    silently share a namespace with every other one instead of failing loudly like now.
    """
    gid = app_agent_key_gid(app_agent_key)
    if gid is None:
        message = "IK_APP_AGENT_KEY is required to scope a cache key, but none was configured"
        raise InvalidAppAgentKeyError(message)
    return gid


def _cache_key(app_agent_key: str) -> str:
    """Build the schema cache key scoped to this specific IndyKite environment."""
    return f"ikg:{env_scope(app_agent_key)}:schema"


# The freshness marker's value is never read for its content, only for whether get() still
# returns something at all - see SchemaClient._store()/_load_schema().
_FRESH_MARKER = b"1"


def _decode_schema(encoded: bytes) -> dict[str, Any] | None:
    """The cached schema, or None if encoded isn't a JSON object - treated as a miss so
    get_cached() self-heals, rather than leaving a bad value in place forever."""
    try:
        value = json.loads(encoded)
        if not isinstance(value, dict):
            message = "unrecognized cached schema value shape"
            raise ValueError(message)  # noqa: TRY301, TRY004
    except (ValueError, TypeError):
        logger.warning("Ignoring unreadable cached schema value (legacy or corrupted) - treating as a miss")
        return None
    else:
        return value


class SchemaClient:
    """Fetches and caches the IndyKite schema for one base_url/app_agent_key pair."""

    def __init__(
        self,
        connection: IkConnection,
        *,
        timeout_seconds: float = 30.0,
        cache_config: CacheConfig,
    ) -> None:
        """Store the connection details; no request is made until preload()/get_cached().

        cache_config.refresh_ttl_seconds is how long a cached schema is served before
        get_cached() treats it as due for a refresh. cache_config.executor is where
        get_cached() dispatches a stale entry's background refresh. cache_config.cache may be
        shared with a PolicyClient (see build_cache_backend) - this class never closes it
        itself; see close()'s docstring.

        timeout_seconds/refresh_ttl_seconds are float like every Settings *_timeout_seconds
        field - schema_refresh_ttl_minutes is a fractional PositiveFloat; int would truncate it.
        """
        self._base_url = connection.base_url
        self._app_agent_key = connection.app_agent_key
        self._timeout_seconds = timeout_seconds
        self._cache = cache_config.cache
        self._refresh_ttl_seconds = cache_config.refresh_ttl_seconds
        self._executor = cache_config.executor
        self._cache_key = _cache_key(connection.app_agent_key)
        # Separate key, not an envelope field: its own backend TTL answers "is this stale" -
        # see _store()/_load_schema().
        self._fresh_key = f"{self._cache_key}:fresh"
        self._gate = RefreshGate()

    def load(self) -> dict[str, Any] | None:
        """Fetch the graph schema from the Indykite platform, or None on failure.

        resp.json() is inside the same try/except as the request itself: a malformed 2xx body
        raises ValueError (json.JSONDecodeError), not httpx2.HTTPError, and must count as a
        failed load the same way, not escape uncaught - preload() would otherwise crash the
        process at cold start.
        """
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
                schema = resp.json()
        except (httpx2.HTTPError, ValueError):
            logger.exception("Failed to load data schema from %s", url)
            return None
        logger.info("Loaded data schema from %s", url)
        return schema

    def _store(self, schema: dict[str, Any]) -> None:
        """Schema itself gets no TTL (a hard expiry could delete the last known-good copy
        during a traffic gap, breaking TDD 4.4) - freshness lives on _fresh_key instead.
        """
        self._cache.set(self._cache_key, json.dumps(schema, separators=(",", ":")).encode("utf-8"))
        self._cache.set(self._fresh_key, _FRESH_MARKER, ttl_seconds=self._refresh_ttl_seconds)

    def _load_schema(self) -> tuple[dict[str, Any] | None, bool]:
        """(value, is_fresh) - value is None if absent/corrupted (self-heals as a miss);
        is_fresh is whether the freshness marker's own backend TTL has elapsed - see _store().
        """
        encoded = self._cache.get(self._cache_key)
        if encoded is None:
            return None, False
        fresh_marker = self._cache.get(self._fresh_key)
        value = _decode_schema(encoded)
        return (value, fresh_marker is not None) if value is not None else (None, False)

    def _load_and_store(self) -> None:
        """Fetch and, on success, cache - shared by _refresh() and preload(). Must not raise:
        a _store() failure is caught and logged here, or it would vanish silently for
        _refresh() (detached on self._executor) and crash startup for preload()
        (IntentAgent.preload_schema() has no try/except around it).
        """
        schema = self.load()
        if schema is None:
            logger.error("schema_refresh_failed - keeping last known good schema")
            return
        try:
            self._store(schema)
        except Exception:
            logger.exception("schema_refresh_failed - failed to cache freshly loaded schema, keeping last known good")

    def _refresh(self) -> None:
        """Reload and re-store - run on self._executor by RefreshGate, never the caller's
        own thread."""
        self._load_and_store()

    def preload(self) -> None:
        """Fetch and cache the schema once, at startup."""
        self._load_and_store()

    def get_cached(self) -> dict[str, Any] | None:
        """Return the cached schema.

        Per TDD 4.4, a failed refresh must never lose the last known-good schema. An
        expired-but-present entry is returned immediately, with the refresh dispatched onto
        `self._executor` in the background (RefreshGate) - a synchronous
        refresh-then-fall-back here could get cancelled mid-flight by REQUEST_TIMEOUT_SECONDS
        before reaching its own fallback.

        A cold entry blocks and fetches synchronously instead, coalescing concurrent callers
        via RefreshGate.acquire_blocking.

        Callers must always go through a thread executor: with CACHE_BACKEND=redis this is
        genuinely synchronous network I/O, not safe to call directly on the event loop.

        A cache-backend error falls through to a direct, uncached fetch instead of failing the
        request outright - Redis being down should degrade latency, not availability. Only a
        failure in that direct fetch is finally "schema unavailable".
        """
        try:
            value, fresh = self._load_schema()
        except Exception:
            logger.exception("Cache backend failed while reading the schema - falling back to a direct fetch")
            return self.load()
        if value is not None:
            logger.debug("cache_key=%s hit=True fresh=%s", self._cache_key, fresh)
            if not fresh:
                self._gate.refresh_in_background_if_free(self._cache_key, self._refresh, self._executor)
            return value
        logger.debug("cache_key=%s hit=False", self._cache_key)
        with self._gate.acquire_blocking(self._cache_key):
            try:
                value, _fresh = self._load_schema()
            except Exception:
                logger.exception("Cache backend failed while reading the schema - falling back to a direct fetch")
                return self.load()
            if value is not None:
                return value
            schema = self.load()
            if schema is not None:
                try:
                    self._store(schema)
                except Exception:
                    # A freshly loaded schema must still be served even if caching it fails -
                    # the two-key _store() can partially succeed (primary write landing, only
                    # the fresh marker failing).
                    logger.exception("Failed to cache freshly loaded schema - serving it uncached this request")
            return schema

    def peek_cached(self) -> dict[str, Any] | None:
        """Return the cached schema if present and still fresh, without ever calling load() or
        touching the executor/RefreshGate - lets run_extraction skip the fetch executor
        entirely once the cache already has a fresh answer. A stale-but-present entry counts
        as a miss here (unlike get_cached()).
        """
        value, fresh = self._load_schema()
        if value is None:
            logger.debug("cache_key=%s hit=False", self._cache_key)
            return None
        logger.debug("cache_key=%s hit=%s", self._cache_key, fresh)
        return value if fresh else None
