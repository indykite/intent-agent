import logging
from enum import Enum
from http import HTTPStatus
from typing import Any

import httpx2
from a2a.server.agent_execution import RequestContext

logger = logging.getLogger(__name__)


class TokenFailure(Enum):
    """Whose fault a refused access token is - the three ways the IAG tells its introspection
    failures apart (`idp/client.go`, `proxy/middlewares.go`'s `failureDecision`), so a caller
    isn't told its token is bad when it's this service or the IdP that failed. IntentExecutor
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


def get_access_token_from_context(context: RequestContext | None) -> str:
    """Extract the Bearer token from the incoming request's Authorization header.

    Any non-Bearer scheme (or a missing/malformed header) is treated as no
    token provided - not passed through as-is - so it can't be forwarded to
    the introspection endpoint as if it were a bearer token.
    """
    if not context or not context.call_context:
        return ""
    req_headers = context.call_context.state.get("headers") or {}
    auth_header = req_headers.get("authorization") or req_headers.get("Authorization") or ""
    bearer_str = "bearer "
    if auth_header.lower().startswith(bearer_str):
        return auth_header[len(bearer_str) :].strip()
    return ""


async def validate_access_token(
    token: str,
    *,
    introspect_url: str,
    client_id: str,
    client_secret: str,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """
    Validate an access token via RFC 7662 Token Introspection against
    the external IdP.

    Sends a single POST /introspect call
    Returns the introspection response body (claims) if the token is active.
    Raises TokenValidationError if the token is inactive, or the call fails - classified
    (TokenFailure) the way the IAG's IdP client classifies the same failures: only a missing or
    inactive token is the caller's, a request that never got an answer is this service's, and
    any status or body that isn't an RFC 7662 answer is the IdP's.
    """
    if not token.strip():
        message = "No token provided"
        raise TokenValidationError(message, TokenFailure.CALLER)

    try:
        async with httpx2.AsyncClient() as client:
            response = await client.post(
                introspect_url,
                auth=(client_id, client_secret),
                headers={"Accept": "application/json"},
                data={
                    "token": token,
                    "token_type_hint": "access_token",
                },
                timeout=timeout_seconds,
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

    return _validate_introspection_claims(payload)


def _validate_introspection_claims(payload: object) -> dict[str, Any]:
    """Validate a parsed introspection response body, raising TokenValidationError on any failure."""
    if not isinstance(payload, dict):
        logger.error("Token introspection returned a non-object JSON body")
        message = "Introspection endpoint returned an unexpected response shape"
        raise TokenValidationError(message, TokenFailure.PROVIDER)

    # Per RFC 7662 Section 2.2: `active` is the one REQUIRED field, a boolean.
    # Everything else (exp, sub, scope, etc.) is OPTIONAL and IdP-dependent.
    # Only an explicit `false` is an answer about the token; an answer that omits the field (or
    # doesn't make it a boolean) is the IdP's fault, not the token's - like the IAG's
    # `introspectionRefusal`, which tells those two apart the same way.
    active = payload.get("active")
    if not isinstance(active, bool):
        logger.error("Token introspection response does not say whether the token is active")
        message = "Introspection endpoint did not say whether the token is active"
        raise TokenValidationError(message, TokenFailure.PROVIDER)
    if not active:
        logger.info("Token introspection: token is not active")
        message = "Token is not active"
        raise TokenValidationError(message, TokenFailure.CALLER)

    # This token authenticates the calling client, not any subject (see
    # IntentExecutor._authenticate) - subject_type/subject_id come from the caller-supplied
    # message metadata instead (IntentExecutor._subject_params), so a `sub` claim is neither
    # read from this payload nor required here. Client-credentials-grant tokens - the expected
    # case now that the caller is a client, not an end user - typically don't carry one at all.
    return payload


def validate_ik_token(ik_token: str) -> dict[str, Any]:
    """
    Validate the `ik_token` via the IndyKite Token Service (ITS) introspection
    endpoint.

    STATUS: BLOCKED. v1.0 requires BOTH the `access_token` (validated
    against the external IdP, see validate_access_token above) AND the
    `ik_token` (validated against ITS). v1.0 cannot ship with only one of
    the two validated.

    However, the ITS introspection endpoint itself does not exist yet.
    This function is a stub so the "two tokens must be validated" shape is
    visible in the code and isn't forgotten, but it cannot be implemented
    until:
      - ITS_INTROSPECT_URL is known (once ITS exposes the endpoint)
      - The request/response shape is confirmed (likely also RFC 7662-style,
        but not confirmed - ITS may have its own contract)
      - Any auth needed to call ITS itself is confirmed

    TODO: implement once ITS introspection endpoint is available. Expected
    shape mirrors validate_access_token() above: POST the ik_token to ITS,
    check an "active"-equivalent field, return the claims/payload.
    """
    message = (
        "ik_token validation via ITS is not yet implemented - blocked on the ITS introspection endpoint. See ENG-8846."
    )
    raise NotImplementedError(message)


def validate_app_agent_token(app_agent_token: str) -> dict[str, Any]:
    """
    Validate the calling agent's AppAgentToken - authenticates the calling
    client itself, independent of whatever access_token/ik_token it presents.

    TODO: implement once the backend team's design and the validation
    mechanism are confirmed. Not wired into agent.py either way.
    """
    message = "AppAgentToken validation is not yet implemented - blocked on the backend team's design."
    raise NotImplementedError(message)
