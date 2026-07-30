"""Extracting caller identity (Bearer token, JWT sub claim) from an A2A request."""

import base64
import json

from a2a.server.agent_execution import RequestContext

# A JWT is header.payload.signature.
_JWT_PARTS = 3
# base64 encodes 3 bytes into 4 chars, so payloads are padded to a multiple of 4.
_BASE64_PAD_MODULUS = 4


def get_sub_from_token(token: str) -> str | None:
    """Extract the sub claim from a JWT. Returns None if invalid or missing sub."""
    if not token.strip():
        return None
    parts = token.split(".")
    if len(parts) != _JWT_PARTS:
        return None
    payload_b64 = parts[1]
    padding = _BASE64_PAD_MODULUS - len(payload_b64) % _BASE64_PAD_MODULUS
    if padding != _BASE64_PAD_MODULUS:
        payload_b64 += "=" * padding
    payload_b64 = payload_b64.replace("-", "+").replace("_", "/")
    try:
        payload_bytes = base64.b64decode(payload_b64)
        data = json.loads(payload_bytes)
    except ValueError:
        return None
    sub = data.get("sub")
    return str(sub) if sub is not None else None


def get_access_token_from_context(context: RequestContext | None) -> str:
    """Extract the Bearer token from the incoming request's Authorization header."""
    if not context or not context.call_context:
        return ""
    req_headers = context.call_context.state.get("headers") or {}
    auth = req_headers.get("authorization") or req_headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return auth.strip()
