"""Webhook audit delivery (the IAG's `audit.http`): one POST/PUT per record with the record
as the JSON body, authenticated with mTLS (TLS 1.3 minimum, like the IAG), an API key header,
or nothing. A failed delivery - connection error, timeout, or a 4xx/5xx from the receiver - is
logged and dropped; it is never retried and never fails the request being audited.
"""

import logging
import ssl
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Literal

import httpx2

from indykite.intent_agent.audit.message import AuditMessage

logger = logging.getLogger(__name__)

HttpMethod = Literal["POST", "PUT"]
AuthType = Literal["mTLS", "no-auth", "api-key"]


@dataclass(frozen=True, slots=True)
class HttpConfig:
    """The IAG's `audit.http.*` keys."""

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

    def __init__(self, config: HttpConfig, *, transport: httpx2.AsyncBaseTransport | None = None) -> None:
        """Build the HTTP client for the configured auth type. `transport` is for tests only."""
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
        except httpx2.HTTPError:
            logger.exception("audit: failed to send HTTP request")
            return
        if response.status_code >= HTTPStatus.BAD_REQUEST:
            logger.error(
                "audit: failed to send HTTP request, unexpected status code",
                extra={"status_code": response.status_code},
            )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


def _mtls_context(config: HttpConfig) -> ssl.SSLContext:
    """A client TLS context presenting the configured certificate, TLS 1.3 minimum like the IAG."""
    if config.mtls_certificate_file_path is None or config.mtls_private_key_path is None:
        message = "audit: mTLS requires mtls_certificate_file_path and mtls_private_key_path"
        raise ValueError(message)
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(certfile=config.mtls_certificate_file_path, keyfile=config.mtls_private_key_path)
    return context
