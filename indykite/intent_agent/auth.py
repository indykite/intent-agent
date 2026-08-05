"""
TODO before this is production-ready:
  - Confirm the real INTROSPECT_URL, INTROSPECT_CLIENT_ID, and
    INTROSPECT_CLIENT_SECRET values for the external IdP (currently unset
    placeholders in Settings)
"""

import logging
from http import HTTPStatus
from typing import Any

import httpx
from a2a.server.agent_execution import RequestContext

logger = logging.getLogger(__name__)


class TokenValidationError(Exception):
    """Raised when an access token fails validation for any reason."""


def get_access_token_from_context(context: RequestContext | None) -> str:
    """Extract the Bearer token from the incoming request's Authorization header."""
    if not context or not context.call_context:
        return ""
    req_headers = context.call_context.state.get("headers") or {}
    auth_header = req_headers.get("authorization") or req_headers.get("Authorization") or ""
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return auth_header.strip()


def validate_access_token(
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
    if not token:
        message = "No token provided"
        raise TokenValidationError(message)

    try:
        response = httpx.post(
            introspect_url,
            auth=(client_id, client_secret),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "token": token,
                "token_type_hint": "access_token",
            },
            timeout=timeout_seconds,
        )
    except httpx.RequestError as exc:
        logger.warning("Token introspection call failed: %s", exc)
        message = "Unable to reach IdP introspection endpoint"
        raise TokenValidationError(message) from exc

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

    # Per RFC 7662 Section 2.2: `active` is the one REQUIRED field.
    # Everything else (exp, sub, scope, etc.) is OPTIONAL and IdP-dependent.
    if not payload.get("active"):
        logger.info("Token introspection: token is not active")
        message = "Token is not active"
        raise TokenValidationError(message)

    # No `act`/`aud` actor-chain check here: that check belongs to the
    # ik_token (see validate_ik_token) - the access_token's `sub` is the
    # original workflow initiator (typically the human) and is not expected
    # to carry an `act` claim.
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

    TODO: this function (not validate_access_token) is where the actor-chain
    check belongs: `aud` must equal the topmost/most-recent `act.sub` in the
    ik_token's introspection response - the ik_token is signed proof of which
    client is actually calling the intent agent. `act` can nest for
    multi-hop delegation; ASSUMPTION (not yet confirmed against a real ITS
    response) is that the outermost `act` is the most recent actor, so
    compare `aud` against `act.sub` at the top level only, not any nested
    act - FLAG for review before relying on this in production.
    """
    message = (
        "ik_token validation via ITS is not yet implemented - blocked on the ITS introspection endpoint. See ENG-8846."
    )
    raise NotImplementedError(message)
