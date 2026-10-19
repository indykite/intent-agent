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

logger = logging.getLogger(__name__)

AUTHORIZATION_REQUIRED_MESSAGE = "Authorization required"
# Request (and call context) state key: `time.perf_counter()` when BearerAuthMiddleware got the request.
REQUEST_STARTED_KEY = "request_started"
_NO_CALLER_MESSAGE = "Token does not identify its caller"
_AUTHORIZATION_HEADER = "Authorization"
_WWW_AUTHENTICATE_HEADER = "WWW-Authenticate"


class TokenFailure(Enum):
    """Whose fault a refused access token is - the three ways the IAG tells its introspection
    failures apart (`idp/client.go`, `proxy/middlewares.go`'s `failureDecision`), so a caller
    isn't told its token is bad when it's this service or the IdP that failed. _TOKEN_FAILURE_STATUS
    maps each to the caller's HTTP status.
    """

    CALLER = "caller"
    """The caller's token: missing, or the IdP says it is not active. 401."""

    SERVICE = "service"
    """This service: its request to the IdP never got an answer. 500."""

    PROVIDER = "provider"
    """The IdP's answer: any status instead of an answer (4xx and 5xx alike, our own rejected client
    credentials included - the IAG's `introspectionStatusRefusal` makes no finer split), or a body
    that can't be read as one. 502."""


class TokenValidationError(Exception):
    """Raised when an access token fails validation for any reason; `failure` says whose fault."""

    def __init__(self, message: str, failure: TokenFailure) -> None:
        """Keep the caller-facing message and the failure class it belongs to."""
        super().__init__(message)
        self.failure = failure


def bearer_token(headers: Mapping[str, str]) -> str:
    """Extract the Bearer token from a request's Authorization header.

    Any non-Bearer scheme (or a missing/malformed header) is treated as no
    token provided - not passed through as-is - so it can't be forwarded to
    the introspection endpoint as if it were a bearer token.
    """
    auth_header = headers.get(_AUTHORIZATION_HEADER.lower()) or headers.get(_AUTHORIZATION_HEADER) or ""
    bearer_str = "bearer "
    if auth_header.lower().startswith(bearer_str):
        return auth_header[len(bearer_str) :].strip()
    return ""


@dataclass(frozen=True, slots=True)
class IntrospectionConfig:
    """Everything validate_access_token needs to reach the IdP and judge its answer, besides
    the token itself
    """

    introspect_url: str
    client_id: str
    client_secret: str
    expected_audience: str
    timeout_seconds: float = 5.0


async def validate_access_token(token: str, config: IntrospectionConfig) -> dict[str, Any]:
    """
    Validate an access token via RFC 7662 Token Introspection against
    the external IdP.

    Sends a single POST /introspect call
    Returns the introspection response body (claims) if the token is active and its `aud`
    includes config.expected_audience.
    Raises TokenValidationError if the token is inactive or scoped to a different audience, or
    the call fails - classified (TokenFailure) the way the IAG's IdP client classifies the same
    failures: only a missing/inactive/wrong-audience token is the caller's, a request that never
    got an answer is this service's, and any status or body that isn't an RFC 7662 answer is the
    IdP's.
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

    # Any status instead of an answer is the IdP's, 4xx and 5xx alike, as in the IAG - the 401 is
    # only singled out so the message and log name what actually needs fixing (RFC 7662 2.3: our
    # own client credentials), not to classify it differently.
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

    # `aud` is legally either a single string or an array of strings (RFC 7519 4.1.3) - RFC 7662
    # itself doesn't constrain its shape beyond "OPTIONAL... case sensitive string(s)", so both
    # forms have to be accepted rather than assuming one.
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
    """The token's `sub`, else its `client_id`; "" when it has neither (refused: a blank owner
    would be shared by every such caller, see BearerAuthMiddleware).
    """
    for claim in ("sub", "client_id"):
        value = claims.get(claim)
        if isinstance(value, str) and value.strip():
            return value
    return ""


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Introspects the Bearer token of every request on the wrapped route and sets `request.user`.
    The A2A SDK stores each task under that user, and a SendMessage naming an existing `taskId`
    only finds the caller's own - so another caller can't join a running task and get its history
    back. Refusals go to `on_rejected` for auditing.
    """

    def __init__(self, app: ASGIApp, config: IntrospectionConfig, on_rejected: RejectionHook) -> None:
        """Wrap `app`."""
        super().__init__(app)
        self._config = config
        self._on_rejected = on_rejected

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Authenticate the request, then pass it on or refuse it."""
        setattr(request.state, REQUEST_STARTED_KEY, time.perf_counter())
        token = bearer_token(request.headers)
        if not token:
            return await self._reject(request, AUTHORIZATION_REQUIRED_MESSAGE, HTTP_401_UNAUTHORIZED, token_sent=False)
        try:
            claims = await validate_access_token(token, self._config)
        except TokenValidationError as exc:
            return await self._reject(request, str(exc), _TOKEN_FAILURE_STATUS[exc.failure], token_sent=True)
        caller = _caller_name(claims)
        if not caller:
            logger.info("Token introspection: active token carries neither sub nor client_id")
            return await self._reject(request, _NO_CALLER_MESSAGE, HTTP_401_UNAUTHORIZED, token_sent=True)

        # Starlette's conventional scope for any authenticated caller (what `@requires("authenticated")`
        # checks). Nothing here checks scopes - the A2A SDK only reads `request.user`.
        request.scope["auth"] = AuthCredentials(["authenticated"])
        request.scope["user"] = SimpleUser(caller)
        return await call_next(request)

    async def _reject(self, request: Request, message: str, status: int, *, token_sent: bool) -> Response:
        """Report the refusal first, while the body is still readable, then answer it (RFC 6750 3)."""
        await self._on_rejected(request, message)
        headers = {}
        if status == HTTP_401_UNAUTHORIZED:
            headers[_WWW_AUTHENTICATE_HEADER] = 'Bearer error="invalid_token"' if token_sent else "Bearer"
        return PlainTextResponse(message, status_code=status, headers=headers)


class CallContextBuilder(DefaultServerCallContextBuilder):
    """The SDK's builder, also carrying BearerAuthMiddleware's REQUEST_STARTED_KEY into the call context."""

    def build(self, request: Request) -> ServerCallContext:
        """Build the SDK's call context, plus REQUEST_STARTED_KEY when the middleware set it."""
        context = super().build(request)
        started = getattr(request.state, REQUEST_STARTED_KEY, None)
        if started is not None:
            context.state[REQUEST_STARTED_KEY] = started
        return context
