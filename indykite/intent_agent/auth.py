import logging
from http import HTTPStatus
from typing import Any

import httpx2
from a2a.server.agent_execution import RequestContext

logger = logging.getLogger(__name__)


class TokenValidationError(Exception):
    """Raised when an access token fails validation for any reason."""


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
    Raises TokenValidationError if the token is inactive, or the call fails.
    """
    if not token.strip():
        message = "No token provided"
        raise TokenValidationError(message)

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
        logger.warning("Token introspection request failed: %s", exc)
        message = "Token introspection request failed"
        raise TokenValidationError(message) from exc

    if response.status_code == HTTPStatus.UNAUTHORIZED:
        logger.error("Token introspection rejected our own client credentials (introspect_client_id/secret)")
        message = "Introspection endpoint rejected our own client credentials"
        raise TokenValidationError(message)

    if response.status_code != HTTPStatus.OK:
        logger.warning("Token introspection returned unexpected status %s", response.status_code)
        message = f"Introspection endpoint returned status {response.status_code}"
        raise TokenValidationError(message)

    try:
        payload = response.json()
    except ValueError as exc:
        logger.warning("Token introspection returned non-JSON response")
        message = "Introspection endpoint returned invalid response"
        raise TokenValidationError(message) from exc

    return _validate_introspection_claims(payload)


def _validate_introspection_claims(payload: object) -> dict[str, Any]:
    """Validate a parsed introspection response body, raising TokenValidationError on any failure."""
    if not isinstance(payload, dict):
        logger.warning("Token introspection returned a non-object JSON body")
        message = "Introspection endpoint returned an unexpected response shape"
        raise TokenValidationError(message)

    # Per RFC 7662 Section 2.2: `active` is the one REQUIRED field.
    # Everything else (exp, sub, scope, etc.) is OPTIONAL and IdP-dependent.
    if payload.get("active") is not True:
        logger.info("Token introspection: token is not active")
        message = "Token is not active"
        raise TokenValidationError(message)

    # `sub` is OPTIONAL per RFC 7662, but this system's auth design relies on
    # it as the subject identity for downstream authorization decisions - an
    # active token missing it must be rejected outright, not silently
    # forwarded as an anonymous/no-subject request.
    if not payload.get("sub"):
        logger.info("Token introspection: active token is missing required sub claim")
        message = "Token is missing required sub claim"
        raise TokenValidationError(message)

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
