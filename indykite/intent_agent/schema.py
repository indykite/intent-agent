"""Fetching and caching the IndyKite graph schema over HTTP."""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SchemaClient:
    """Fetches and caches the IndyKite schema for one base_url/app_agent_key pair."""

    def __init__(
        self, base_url: str, app_agent_key: str, *, timeout_seconds: float = 30.0
    ) -> None:
        """Store the connection details; no request is made until load()/preload()."""
        self._base_url = base_url
        self._app_agent_key = app_agent_key
        self._timeout_seconds = timeout_seconds
        self._cache: dict[str, Any] | None = None

    def load(self) -> dict[str, Any] | None:
        """Fetch the graph schema from the Indykite platform, or None on failure."""
        if not self._base_url or not self._app_agent_key:
            logger.warning("IK base URL/app-agent key not set - schema unavailable")
            return None
        url = f"{self._base_url.rstrip('/')}/data-schema/v1"
        headers = {"X-IK-ClientKey": self._app_agent_key}
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
        except httpx.HTTPError:
            logger.warning("Failed to load data schema from %s", url)
            return None
        logger.info("Loaded data schema from %s", url)
        return resp.json()

    def preload(self) -> None:
        """Fetch and cache the schema once, at startup."""
        self._cache = self.load()

    def get_cached(self) -> dict[str, Any] | None:
        """Return the cached schema, fetching it once if not already preloaded."""
        if self._cache is None:
            self._cache = self.load()
        return self._cache
