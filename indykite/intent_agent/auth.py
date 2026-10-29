import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from typing import Any

import httpx2
from a2a.server.context import ServerCallContext
from a2a.server.routes.common import DefaultServerCallContextBuilder
from a2a.types import AgentCard, HTTPAuthSecurityScheme, SecurityRequirement, SecurityScheme, StringList
from starlette.authentication import AuthCredentials, SimpleUser
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.status import (
    HTTP_401_UNAUTHORIZED,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_502_BAD_GATEWAY,
)
from starlette.types import ASGIApp

from indykite.intent_agent.dependency_status import INTROSPECTION, DependencyStatus

logger = logging.getLogger(__name__)

_AUTHORIZATION_REQUIRED_MESSAGE = "Authorization required"
# Request (and call context) state key: `time.perf_counter()` when BearerAuthMiddleware got the request.
REQUEST_STARTED_KEY = "request_started"
_NO_CALLER_MESSAGE = "Token does not identify its caller"
_AUTHORIZATION_HEADER = "Authorization"
_WWW_AUTHENTICATE_HEADER = "WWW-Authenticate"
# The scheme in the 401's WWW-Authenticate challenge and in the agent card (add_bearer_security).
_BEARER_AUTH_SCHEME = "Bearer"
# The agent card's name for that scheme, which its security requirement refers to.
_BEARER_SECURITY_SCHEME = "bearer"


class TokenFailure(Enum):
    """Whose fault a refused access token is; _TOKEN_FAILURE_STATUS maps each to the caller's HTTP status."""

    CALLER = "caller"
    """The caller's token: missing, or the IdP says it is not active. 401."""

    SERVICE = "service"
    """This service: its request to the IdP never got an answer. 500."""

    PROVIDER = "provider"
    """The IdP's answer: any HTTP status instead of an answer (4xx and 5xx alike, our own rejected
    client credentials included), or a body that can't be read as one. 502."""


class TokenValidationError(Exception):
    """Raised when an access token fails validation for any reason; `failure` says whose fault."""

    def __init__(self, message: str, failure: TokenFailure) -> None:
        """Keep the caller-facing message and the failure class it belongs to."""
        super().__init__(message)
        self.failure = failure


def bearer_token(headers: Mapping[str, str]) -> str:
    """Extract the Bearer token from the Authorization header; "" if absent or another scheme."""
    auth_header = headers.get(_AUTHORIZATION_HEADER.lower()) or headers.get(_AUTHORIZATION_HEADER) or ""
    bearer_str = "bearer "
    if auth_header.lower().startswith(bearer_str):
        return auth_header[len(bearer_str) :].strip()
    return ""


@dataclass(frozen=True, slots=True)
class IntrospectionConfig:
    """IdP endpoint, client credentials and expected audience for validate_access_token."""

    introspect_url: str
    client_id: str
    client_secret: str
    expected_audience: str
    timeout_seconds: float = 5.0


async def validate_access_token(token: str, config: IntrospectionConfig) -> dict[str, Any]:
    """Validate an access token via RFC 7662 introspection and return its claims.

    Raises TokenValidationError, classified by TokenFailure, if the token is blank, inactive or
    for another audience, or the IdP call fails.
    """
    if not token.strip():
        message = "No token provided"
        raise TokenValidationError(message, TokenFailure.CALLER)

    try:
        async with httpx2.AsyncClient() as client:
            response = await client.post(
                config.introspect_url,
                auth=(config.client_id, config.client_secret),
                headers={"Accept": "application/json"},
                data={
                    "token": token,
                    "token_type_hint": "access_token",
                },
                timeout=config.timeout_seconds,
            )
    except httpx2.RequestError as exc:
        logger.exception("Token introspection request failed")
        message = "Token introspection request failed"
        raise TokenValidationError(message, TokenFailure.SERVICE) from exc

    # Any non-200 is the IdP's; 401 is singled out only to name our own client credentials.
    if response.status_code == HTTPStatus.UNAUTHORIZED:
        logger.error("Token introspection rejected our own client credentials (introspect_client_id/secret)")
        message = "Introspection endpoint rejected our own client credentials"
        raise TokenValidationError(message, TokenFailure.PROVIDER)

    if response.status_code != HTTPStatus.OK:
        logger.error("Token introspection returned unexpected status %s", response.status_code)
        message = f"Introspection endpoint returned status {response.status_code}"
        raise TokenValidationError(message, TokenFailure.PROVIDER)

    try:
        payload = response.json()
    except ValueError as exc:
        logger.exception("Token introspection returned non-JSON response")
        message = "Introspection endpoint returned invalid response"
        raise TokenValidationError(message, TokenFailure.PROVIDER) from exc

    return _validate_introspection_claims(payload, config.expected_audience)


def _validate_introspection_claims(payload: object, expected_audience: str) -> dict[str, Any]:
    """Validate a parsed introspection response body, raising TokenValidationError on any failure."""
    if not isinstance(payload, dict):
        logger.error("Token introspection returned a non-object JSON body")
        message = "Introspection endpoint returned an unexpected response shape"
        raise TokenValidationError(message, TokenFailure.PROVIDER)

    active = payload.get("active")
    if not isinstance(active, bool):
        logger.error("Token introspection response does not say whether the token is active")
        message = "Introspection endpoint did not say whether the token is active"
        raise TokenValidationError(message, TokenFailure.PROVIDER)
    if not active:
        logger.info("Token introspection: token is not active")
        message = "Token is not active"
        raise TokenValidationError(message, TokenFailure.CALLER)

    # `aud` may be a single string or an array of strings (RFC 7519 4.1.3).
    aud = payload.get("aud")
    audiences = aud if isinstance(aud, list) else [aud]
    if expected_audience not in audiences:
        logger.info(
            "Token introspection: token audience %r does not include expected audience %r",
            audiences,
            expected_audience,
        )
        message = "Token is not valid for this agent"
        raise TokenValidationError(message, TokenFailure.CALLER)

    logger.debug("Token introspection: token audience %r matches expected audience %r", audiences, expected_audience)
    return payload


_TOKEN_FAILURE_STATUS = {
    TokenFailure.CALLER: HTTP_401_UNAUTHORIZED,
    TokenFailure.SERVICE: HTTP_500_INTERNAL_SERVER_ERROR,
    TokenFailure.PROVIDER: HTTP_502_BAD_GATEWAY,
}

RejectionHook = Callable[[Request, str], Awaitable[None]]


def _caller_name(claims: Mapping[str, Any]) -> str:
    """The token's `sub`, else its `client_id`; "" when it has neither, which BearerAuthMiddleware refuses."""
    for claim in ("sub", "client_id"):
        value = claims.get(claim)
        if isinstance(value, str) and value.strip():
            return value
    return ""


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Introspect every request's Bearer token and set `request.user`, the owner the A2A SDK stores tasks under.

    Refusals are passed to `on_rejected` for auditing. `dependency_status` (when given) is told
    whether the IdP answered, which is what /health reports.
    """

    def __init__(
        self,
        app: ASGIApp,
        config: IntrospectionConfig,
        on_rejected: RejectionHook,
        dependency_status: DependencyStatus | None = None,
    ) -> None:
        """Wrap `app`."""
        super().__init__(app)
        self._config = config
        self._on_rejected = on_rejected
        self._status = dependency_status

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Authenticate the request, then pass it on or refuse it."""
        setattr(request.state, REQUEST_STARTED_KEY, time.perf_counter())
        token = bearer_token(request.headers)
        if not token:
            return await self._reject(request, _AUTHORIZATION_REQUIRED_MESSAGE, HTTP_401_UNAUTHORIZED, token_sent=False)
        try:
            claims = await validate_access_token(token, self._config)
        except TokenValidationError as exc:
            self._record_introspection(exc)
            return await self._reject(request, str(exc), _TOKEN_FAILURE_STATUS[exc.failure], token_sent=True)
        self._record_introspection(None)
        caller = _caller_name(claims)
        if not caller:
            logger.info("Token introspection: active token carries neither sub nor client_id")
            return await self._reject(request, _NO_CALLER_MESSAGE, HTTP_401_UNAUTHORIZED, token_sent=True)

        # Starlette's usual scope for an authenticated caller; the A2A SDK only reads `request.user`.
        request.scope["auth"] = AuthCredentials(["authenticated"])
        request.scope["user"] = SimpleUser(caller)
        return await call_next(request)

    def _record_introspection(self, failure: TokenValidationError | None) -> None:
        """An inactive token is the caller's problem - the IdP answered fine; any other refusal is the IdP's or ours."""
        if self._status is None:
            return
        if failure is None or failure.failure is TokenFailure.CALLER:
            self._status.record_ok(INTROSPECTION)
        else:
            self._status.record_error(INTROSPECTION, str(failure))

    async def _reject(self, request: Request, message: str, status: int, *, token_sent: bool) -> Response:
        """Report the refusal first, while the body is still readable, then answer it (RFC 6750 3)."""
        await self._on_rejected(request, message)
        headers = {}
        if status == HTTP_401_UNAUTHORIZED:
            headers[_WWW_AUTHENTICATE_HEADER] = (
                f'{_BEARER_AUTH_SCHEME} error="invalid_token"' if token_sent else _BEARER_AUTH_SCHEME
            )
        return PlainTextResponse(message, status_code=status, headers=headers)


def add_bearer_security(card: AgentCard, expected_audience: str) -> None:
    """Declare on the agent card the Bearer token BearerAuthMiddleware requires, so a client can
    discover it before getting a 401. HTTP Bearer, not OAuth2/OpenID: those would advertise the
    IdP's endpoints, which this service doesn't know - it only introspects. No bearer_format (the
    token isn't decoded here) and no scopes (none are checked).
    """
    card.security_schemes[_BEARER_SECURITY_SCHEME].CopyFrom(
        SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(
                scheme=_BEARER_AUTH_SCHEME,
                description=(
                    "OAuth 2.0 access token, validated by RFC 7662 token introspection; "
                    f"its audience must include '{expected_audience}'."
                ),
            ),
        ),
    )
    card.security_requirements.append(SecurityRequirement(schemes={_BEARER_SECURITY_SCHEME: StringList()}))


class CallContextBuilder(DefaultServerCallContextBuilder):
    """The SDK's builder, also carrying BearerAuthMiddleware's REQUEST_STARTED_KEY into the call context."""

    def build(self, request: Request) -> ServerCallContext:
        """Build the SDK's call context, plus REQUEST_STARTED_KEY when the middleware set it."""
        context = super().build(request)
        started = getattr(request.state, REQUEST_STARTED_KEY, None)
        if started is not None:
            context.state[REQUEST_STARTED_KEY] = started
        return context
