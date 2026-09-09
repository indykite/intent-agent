"""Token validation for incoming A2A requests (TDD 4.3, dual-token design).

Two tokens can arrive on a request, both validated by RFC 7662 Token Introspection with
`client_secret_basic` client authentication - the same wire contract, against two
different servers:

- `access_token` (`Authorization: Bearer ...`): the end user's token from the external
  IdP. Its ``sub`` becomes the extraction's subject id.
- `ik_token` (`X-IK-Token`, raw, no `Bearer` prefix - IAG's `forward.Tokens.SetHeaders`
  sends it that way): the delegation token minted by the IndyKite Token Service (ITS,
  jarvis `services/agent-gateway/token-service`) when IAG runs in token-service mode.
  ITS answers `POST /oauth2/introspect` with the RFC 7662 members its tokens carry
  (`active`, `sub`, `iss`, `aud`, `exp`, `iat`, `jti`, `token_type`) plus the
  RFC 8693 `act` delegation chain; an inactive or foreign token is `{"active": false}`.

Neither validator decodes the JWT locally: the introspection call is what covers signature
and expiry. Only the executor decides whether the `ik_token` leg runs at all (`Settings`:
`its_enabled`).
"""

import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import httpx2
from a2a.server.agent_execution import RequestContext

logger = logging.getLogger(__name__)

DELEGATION_HEADER = "x-ik-token"
_DELEGATION_NOT_AUTHORIZED = "actor delegation not authorized"


class TokenValidationError(Exception):
    """Raised when an access token or ik_token fails validation for any reason."""


@dataclass(frozen=True, slots=True)
class _IntrospectionEndpoint:
    """Where and how to call one RFC 7662 introspection endpoint (client_secret_basic)."""

    url: str
    client_id: str
    client_secret: str
    timeout_seconds: float


def _request_headers(context: RequestContext | None) -> dict[str, str]:
    """The incoming request's headers as the A2A server stashed them (lowercase keys)."""
    if not context or not context.call_context:
        return {}
    return context.call_context.state.get("headers") or {}


def get_access_token_from_context(context: RequestContext | None) -> str:
    """Extract the Bearer token from the incoming request's Authorization header.

    Any non-Bearer scheme (or a missing/malformed header) is treated as no
    token provided - not passed through as-is - so it can't be forwarded to
    the introspection endpoint as if it were a bearer token.
    """
    req_headers = _request_headers(context)
    auth_header = req_headers.get("authorization") or req_headers.get("Authorization") or ""
    bearer_str = "bearer "
    if auth_header.lower().startswith(bearer_str):
        return auth_header[len(bearer_str) :].strip()
    return ""


def get_ik_token_from_context(context: RequestContext | None) -> str:
    """Extract the ik_token from the incoming request's X-IK-Token header.

    The header carries the raw delegation token - no `Bearer` scheme, unlike Authorization
    (IAG sends it without a prefix and the platform reads it "as is") - so the value is
    returned whole, only stripped of surrounding whitespace. A missing header yields "".
    """
    req_headers = _request_headers(context)
    return (req_headers.get(DELEGATION_HEADER) or req_headers.get("X-IK-Token") or "").strip()


async def _introspect(token: str, endpoint: _IntrospectionEndpoint, *, label: str) -> dict[str, Any]:
    """One RFC 7662 introspection round trip: POST the token, return the parsed JSON object.

    `label` only names the leg in log lines and error messages ("access token" /
    "ik_token") so an operator can tell the two apart. Raises TokenValidationError on a
    blank token, a transport failure, a non-200 status, or a body that isn't a JSON object -
    the caller checks the claims themselves.
    """
    if not token.strip():
        message = f"No {label} provided"
        raise TokenValidationError(message)

    try:
        async with httpx2.AsyncClient() as client:
            response = await client.post(
                endpoint.url,
                auth=(endpoint.client_id, endpoint.client_secret),
                headers={"Accept": "application/json"},
                data={
                    "token": token,
                    "token_type_hint": "access_token",
                },
                timeout=endpoint.timeout_seconds,
            )
    except httpx2.RequestError as exc:
        logger.warning("%s introspection request failed: %s", label, exc)
        message = f"{label} introspection request failed"
        raise TokenValidationError(message) from exc

    if response.status_code == HTTPStatus.UNAUTHORIZED:
        logger.error("%s introspection rejected our own client credentials", label)
        message = f"{label} introspection endpoint rejected our own client credentials"
        raise TokenValidationError(message)

    if response.status_code != HTTPStatus.OK:
        logger.warning("%s introspection returned unexpected status %s", label, response.status_code)
        message = f"{label} introspection endpoint returned status {response.status_code}"
        raise TokenValidationError(message)

    try:
        payload = response.json()
    except ValueError as exc:
        logger.warning("%s introspection returned non-JSON response", label)
        message = f"{label} introspection endpoint returned invalid response"
        raise TokenValidationError(message) from exc

    if not isinstance(payload, dict):
        logger.warning("%s introspection returned a non-object JSON body", label)
        message = f"{label} introspection endpoint returned an unexpected response shape"
        raise TokenValidationError(message)

    return payload


def _require_active_subject(payload: dict[str, Any], *, label: str) -> None:
    """Reject an introspection body that isn't an active token with a subject."""
    # Per RFC 7662 Section 2.2: `active` is the one REQUIRED field.
    # Everything else (exp, sub, scope, etc.) is OPTIONAL and server-dependent.
    if payload.get("active") is not True:
        logger.info("%s introspection: token is not active", label)
        message = f"{label} is not active"
        raise TokenValidationError(message)

    # `sub` is OPTIONAL per RFC 7662, but this system's auth design relies on
    # it as the subject identity for downstream authorization decisions - an
    # active token missing it must be rejected outright, not silently
    # forwarded as an anonymous/no-subject request. It also has to be a real
    # string (RFC 7662 2.2 / RFC 7519 4.1.2: a StringOrURI), not merely truthy:
    # a list, an object, a number or whitespace would otherwise be bound into
    # every evaluation as the caller's id.
    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub.strip():
        logger.info("%s introspection: active token is missing a usable sub claim (%r)", label, sub)
        message = f"{label} is missing required sub claim"
        raise TokenValidationError(message)


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
    label = "access token"
    endpoint = _IntrospectionEndpoint(introspect_url, client_id, client_secret, timeout_seconds)
    payload = await _introspect(token, endpoint, label=label)
    _require_active_subject(payload, label=label)
    return payload


async def validate_ik_token(
    ik_token: str,
    *,
    introspect_url: str,
    client_id: str,
    client_secret: str,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """
    Validate the ik_token via RFC 7662 Token Introspection against the
    IndyKite Token Service (ITS).

    Mirrors validate_access_token: a single POST to ITS's `/oauth2/introspect`
    with client_secret_basic. Returns the introspection response body (claims)
    if the token is active - judged on `active` and `sub` only, exactly like the
    access token. The RFC 8693 `act` delegation chain is deliberately *not*
    checked: by the time a request reaches this agent it has passed through IAG,
    which has already validated the chain (TDD owner's decision, same rationale
    as the access token's act claim - see test_auth.py, PR #9). Whether the
    subject matches the access token's is `check_delegated_subject`'s job, since
    it needs both tokens' claims.
    Raises TokenValidationError if the token is inactive, or the call fails.
    """
    label = "ik_token"
    endpoint = _IntrospectionEndpoint(introspect_url, client_id, client_secret, timeout_seconds)
    payload = await _introspect(ik_token, endpoint, label=label)
    _require_active_subject(payload, label=label)
    return payload


def requesting_agent_id(ik_claims: dict[str, Any]) -> str:
    """The requesting agent's id from an ik_token, for logging: the outermost `act.sub`.

    Per RFC 8693 the `act` chain nests inward (`act.act.sub` is the previous hop), so the
    top-level `act.sub` is the actor that made this request. "" if there is none - which is
    not an error, see validate_ik_token.
    """
    act = ik_claims.get("act")
    if not isinstance(act, dict):
        return ""
    actor = act.get("sub")
    if not isinstance(actor, str):
        return ""
    return actor.strip()


def check_delegated_subject(ik_claims: dict[str, Any], *, subject_id: str) -> None:
    """Reject an ik_token whose subject isn't the access token's subject.

    An ITS delegation token is minted for one subject (the exchanged subject token's `sub`);
    a delegation for someone else arriving beside this user's access token means the actor
    holds no delegation from *this* subject - the TDD 4.3 "actor not permitted to act for
    Subject" case. IAG and the platform MCP server apply the same rule on ingress.
    """
    if ik_claims.get("sub") != subject_id:
        logger.info("ik_token subject %r does not match access token subject %r", ik_claims.get("sub"), subject_id)
        raise TokenValidationError(_DELEGATION_NOT_AUTHORIZED)


def validate_app_agent_token(app_agent_token: str) -> dict[str, Any]:
    """
    Validate the calling agent's AppAgentToken - authenticates the calling
    client itself, independent of whatever access_token/ik_token it presents.

    TODO: implement once the backend team's design and the validation
    mechanism are confirmed. Not wired into agent.py either way.
    """
    message = "AppAgentToken validation is not yet implemented - blocked on the backend team's design."
    raise NotImplementedError(message)
