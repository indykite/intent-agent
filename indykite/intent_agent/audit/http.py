"""Webhook audit delivery: one JSON POST/PUT per record, with mTLS, an API key or no auth.

A failed delivery is logged and dropped; it is never retried and never fails the request.
"""

import logging
import ssl
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Literal

import httpx2

from indykite.intent_agent.audit.message import AuditMessage
from indykite.intent_agent.dependency_status import AUDIT_WEBHOOK, DependencyStatus, describe_failure

logger = logging.getLogger(__name__)

HttpMethod = Literal["POST", "PUT"]
AuthType = Literal["mTLS", "no-auth", "api-key"]


@dataclass(frozen=True, slots=True)
class HttpConfig:
    """Webhook delivery settings."""

    url: str
    method: HttpMethod
    auth_type: AuthType
    api_key: str = ""
    api_key_header: str = "X-API-Key"
    mtls_certificate_file_path: Path | None = None
    mtls_private_key_path: Path | None = None
    timeout_seconds: float = 5.0


class HttpSender:
    """Delivers audit records to a webhook."""

    def __init__(
        self,
        config: HttpConfig,
        *,
        transport: httpx2.AsyncBaseTransport | None = None,
        dependency_status: DependencyStatus | None = None,
    ) -> None:
        """Build the HTTP client for the configured auth type.

        `transport` is for tests only. dependency_status (when given) is told how every delivery
        went - what /health reports.
        """
        self._status = dependency_status
        self._url = config.url
        self._method = config.method
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        verify: ssl.SSLContext | bool = True
        if config.auth_type == "mTLS":
            verify = _mtls_context(config)
            logger.info("started auditing through HTTP with mTLS")
        elif config.auth_type == "api-key":
            if not config.api_key:
                message = "audit: api-key auth requires api_key"
                raise ValueError(message)
            self._headers[config.api_key_header] = config.api_key
            logger.info("started auditing through HTTP with API key", extra={"header": config.api_key_header})
        else:
            logger.info("started auditing through HTTP with no auth")
        self._client = httpx2.AsyncClient(verify=verify, timeout=config.timeout_seconds, transport=transport)

    async def send(self, message: AuditMessage) -> None:
        """Deliver one record. Logs and swallows every failure."""
        try:
            response = await self._client.request(
                self._method,
                self._url,
                content=message.to_json().encode("utf-8"),
                headers=self._headers,
            )
        except httpx2.HTTPError as exc:
            logger.exception("audit: failed to send HTTP request")
            if self._status is not None:
                self._status.record_error(AUDIT_WEBHOOK, describe_failure(exc))
            return
        if response.status_code >= HTTPStatus.BAD_REQUEST:
            logger.error(
                "audit: failed to send HTTP request, unexpected status code",
                extra={"status_code": response.status_code},
            )
        # Only a 2xx is a delivery: the client doesn't follow redirects, so a 3xx never reached
        # the redirect target either.
        if self._status is None:
            return
        if response.is_success:
            self._status.record_ok(AUDIT_WEBHOOK)
        else:
            self._status.record_error(AUDIT_WEBHOOK, f"status {response.status_code}")

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


def _mtls_context(config: HttpConfig) -> ssl.SSLContext:
    """Build a TLS 1.3+ client context presenting the configured certificate."""
    if config.mtls_certificate_file_path is None or config.mtls_private_key_path is None:
        message = "audit: mTLS requires mtls_certificate_file_path and mtls_private_key_path"
        raise ValueError(message)
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(certfile=config.mtls_certificate_file_path, keyfile=config.mtls_private_key_path)
    return context


async def probe(config: HttpConfig) -> int:
    """Send one OPTIONS request to the webhook URL and return its status code.

    No record is delivered, so a 4xx proves the receiver answers, not that it would accept a
    record. Raises httpx2.HTTPError if it can't be reached.
    """
    headers = {config.api_key_header: config.api_key} if config.auth_type == "api-key" else {}
    verify: ssl.SSLContext | bool = _mtls_context(config) if config.auth_type == "mTLS" else True
    async with httpx2.AsyncClient(verify=verify, timeout=config.timeout_seconds) as client:
        response = await client.options(config.url, headers=headers)
    return response.status_code
