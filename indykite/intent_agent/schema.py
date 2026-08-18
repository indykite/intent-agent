"""Fetching and caching the IndyKite graph schema over HTTP."""

import logging
import threading
import time
from typing import Any

import httpx2
from dogpile.cache.api import NO_VALUE
from dogpile.cache.region import CacheRegion, make_region

logger = logging.getLogger(__name__)

_CACHE_KEY = "ikg_schema"


class SchemaClient:
    """Fetches and caches the IndyKite schema for one base_url/app_agent_key pair."""

    def __init__(
        self,
        base_url: str,
        app_agent_key: str,
        *,
        timeout_seconds: float = 30.0,
        cache: CacheRegion | None = None,
    ) -> None:
        """Store the connection details; no request is made until load()/preload()."""
        self._base_url = base_url
        self._app_agent_key = app_agent_key
        self._timeout_seconds = timeout_seconds
        self._cache = cache or make_region().configure("dogpile.cache.memory")

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
            self._cache.set(_CACHE_KEY, schema)

    def get_cached(self) -> dict[str, Any] | None:
        """Return the cached schema, fetching it once if not already cached."""
        cached = self._cache.get(_CACHE_KEY)
        if cached is not NO_VALUE:
            return cached
        schema = self.load()
        if schema is not None:
            self._cache.set(_CACHE_KEY, schema)
        return schema

    def refresh(self) -> None:
        """Refresh the cached schema for the background refresh loop.

        Per TDD 4.4, a failed refresh must never clear or overwrite the existing
        cache entry. Callers keep getting the last known-good schema, and the
        failure is only observable via the schema_refresh_failed log line.
        """
        schema = self.load()
        if schema is None:
            logger.error("schema_refresh_failed - keeping last known good schema")
            return
        self._cache.set(_CACHE_KEY, schema)
        logger.info("Schema refreshed")

    def start_background_refresh(self, ttl_seconds: float) -> None:
        """Start a daemon thread that refreshes the cached schema every ttl_seconds."""

        def _loop() -> None:
            while True:
                time.sleep(ttl_seconds)
                self.refresh()

        threading.Thread(target=_loop, daemon=True).start()
