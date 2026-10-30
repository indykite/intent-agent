"""The `/health` report: agent version, the data schema's state and the last known state of every
dependency this deployment uses (IKG, token introspection, Gemini, plus the audit webhook and
Redis when configured).

`/health` never makes a network call: it reads what `DependencyStatus` recorded from the real
calls the process made. `HealthChecker.probe_all` runs one probe per dependency at startup to seed
that state, so a dependency that has seen no traffic yet still has an answer. Failure details name
the status code or exception class only, never a response body or a URL carrying a credential.
"""

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import Any

import httpx2
from redis.asyncio import Redis

from indykite.intent_agent.audit import http as audit_http
from indykite.intent_agent.audit.sender import build_http_config
from indykite.intent_agent.config import Settings
from indykite.intent_agent.dependency_status import (
    AUDIT_FILE,
    AUDIT_WEBHOOK,
    GEMINI,
    IKG,
    INTROSPECTION,
    OK,
    REDIS,
    SCHEMA,
    UNKNOWN,
    DependencyStatus,
    describe_failure,
)
from indykite.intent_agent.version import __version__

logger = logging.getLogger(__name__)

_GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DISABLED = "disabled"


class _ProbeFailedError(Exception):
    """A dependency answered, or didn't, in a way that means it is not usable."""


def _unexpected_status(status: int) -> _ProbeFailedError:
    return _ProbeFailedError(f"unexpected status {status}")


class HealthChecker:
    """Reports the recorded dependency state for one deployment, and seeds it at startup."""

    def __init__(self, settings: Settings, dependency_status: DependencyStatus) -> None:
        """Keep the settings and the store the real calls record into."""
        self._settings = settings
        self._status = dependency_status

    def report(self) -> dict[str, Any]:
        """The health body, from recorded state only - no I/O."""
        checks: dict[str, dict[str, Any]] = {}
        for name, enabled in self._dependencies().items():
            if not enabled:
                checks[name] = {"status": DISABLED}
                continue
            state = self._status.get(name)
            checks[name] = state.to_dict() if state else {"status": UNKNOWN}
        schema_state = self._status.get(SCHEMA)
        schema = schema_state.to_dict() if schema_state else {"status": UNKNOWN}
        healthy = schema["status"] == OK and all(check["status"] in {OK, DISABLED} for check in checks.values())
        return {
            "status": "ok" if healthy else "degraded",
            "service": self._settings.intent_agent_name,
            "version": __version__,
            "schema": schema,
            "checks": dict(sorted(checks.items())),
        }

    def _dependencies(self) -> dict[str, bool]:
        """Every dependency that can appear in the report, and whether this deployment uses it."""
        return {
            IKG: True,
            INTROSPECTION: True,
            GEMINI: True,
            AUDIT_WEBHOOK: self._settings.audit_delivery == "webhook",
            AUDIT_FILE: self._settings.audit_delivery == "file",
            REDIS: self._settings.cache_backend == "redis",
        }

    async def probe_all(self) -> None:
        """Probe every dependency in use once, in parallel, and record the outcome as `startup`
        state. A dependency something already recorded (the schema preload records `ikg`, `schema`
        and, with Redis, `redis`) is not probed, and keeps its state.
        """
        probes: dict[str, Callable[[], Awaitable[None]]] = {
            IKG: self._probe_ikg,
            INTROSPECTION: self._probe_introspection,
            GEMINI: self._probe_gemini,
            AUDIT_WEBHOOK: self._probe_audit_webhook,
            AUDIT_FILE: self._probe_audit_file,
            REDIS: self._probe_redis,
        }
        unrecorded = [name for name, used in self._dependencies().items() if used and self._status.get(name) is None]
        await asyncio.gather(*(self._run_probe(name, probes[name]) for name in unrecorded))

    async def _run_probe(self, name: str, probe: Callable[[], Awaitable[None]]) -> None:
        """Run one probe under the timeout and record its outcome, unless traffic got there first."""
        started = time.monotonic()
        failure = ""
        try:
            async with asyncio.timeout(self._settings.health_check_timeout_seconds):
                await probe()
        except _ProbeFailedError as exc:
            failure = str(exc)
        except TimeoutError:
            failure = "timed out"
        except Exception as exc:
            logger.warning("Startup health probe of %s failed", name, exc_info=True)
            failure = describe_failure(exc)
        logger.info(
            "Startup health probe of %s: %s (%.0f ms)", name, failure or OK, (time.monotonic() - started) * 1000
        )
        if self._status.get(name) is not None:
            return
        if failure:
            self._status.record_error(name, failure, source="startup")
        else:
            self._status.record_ok(name, source="startup")

    async def _probe_ikg(self) -> None:
        """The platform answers the schema endpoint for our app-agent key."""
        url = f"{str(self._settings.indykite_base_url).rstrip('/')}/data-schema/v1"
        headers = {"X-IK-ClientKey": self._settings.ik_app_agent_key.get_secret_value()}
        async with httpx2.AsyncClient(timeout=self._settings.health_check_timeout_seconds) as client:
            response = await client.get(url, headers=headers)
        if response.status_code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            message = f"app agent key rejected ({response.status_code})"
            raise _ProbeFailedError(message)
        if response.status_code != HTTPStatus.OK:
            raise _unexpected_status(response.status_code)

    async def _probe_introspection(self) -> None:
        """The IdP answers an introspection call authenticated with our own client credentials.

        The token is a dummy: an RFC 7662 answer (a JSON object with a boolean `active`, false
        here) proves reachability, that the client credentials are accepted and that the endpoint
        is really an introspection endpoint - not a proxy or login page answering 200, which
        validate_access_token would reject on every request.
        """
        async with httpx2.AsyncClient(timeout=self._settings.health_check_timeout_seconds) as client:
            response = await client.post(
                str(self._settings.introspect_url),
                auth=(self._settings.introspect_client_id, self._settings.introspect_client_secret.get_secret_value()),
                headers={"Accept": "application/json"},
                data={"token": "health-check", "token_type_hint": "access_token"},
            )
        if response.status_code == HTTPStatus.UNAUTHORIZED:
            message = "client credentials rejected (401)"
            raise _ProbeFailedError(message)
        if response.status_code != HTTPStatus.OK:
            raise _unexpected_status(response.status_code)
        try:
            payload = response.json()
        except ValueError as exc:
            message = "response is not JSON"
            raise _ProbeFailedError(message) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("active"), bool):
            message = "response is not an RFC 7662 introspection answer"
            raise _ProbeFailedError(message)

    async def _probe_gemini(self) -> None:
        """Gemini is reachable, accepts our API key, and knows the configured model.

        Reads the model's metadata - no generation, so no tokens are spent.
        """
        model = self._settings.gemini_model.removeprefix("models/")
        async with httpx2.AsyncClient(timeout=self._settings.health_check_timeout_seconds) as client:
            response = await client.get(
                f"{_GEMINI_MODELS_URL}/{model}",
                headers={"x-goog-api-key": self._settings.gemini_api_key.get_secret_value()},
            )
        # Google answers an unknown/malformed key with 400 API_KEY_INVALID, not 401.
        if response.status_code in {HTTPStatus.BAD_REQUEST, HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            message = f"API key rejected ({response.status_code})"
            raise _ProbeFailedError(message)
        if response.status_code == HTTPStatus.NOT_FOUND:
            message = f"model {model!r} not found"
            raise _ProbeFailedError(message)
        if response.status_code != HTTPStatus.OK:
            raise _unexpected_status(response.status_code)

    async def _probe_audit_webhook(self) -> None:
        """The audit receiver answers at all - delivery of a real record is not simulated."""
        status = await audit_http.probe(build_http_config(self._settings))
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
            raise _unexpected_status(status)

    async def _probe_audit_file(self) -> None:
        """The audit directory exists and is writable by this process."""
        path = self._settings.audit_file_storage_path
        if path is None or not path.is_dir() or not os.access(path, os.W_OK):
            message = "audit directory is not writable"
            raise _ProbeFailedError(message)

    async def _probe_redis(self) -> None:
        """Redis answers PING."""
        client = Redis.from_url(
            str(self._settings.redis_url), socket_timeout=self._settings.health_check_timeout_seconds
        )
        try:
            await client.ping()
        finally:
            await client.aclose()
