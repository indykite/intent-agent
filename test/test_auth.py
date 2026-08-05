from unittest.mock import Mock

import httpx
import pytest

from indykite.intent_agent import auth

_INTROSPECT_KWARGS = {
    "introspect_url": "https://idp.example.com/introspect",
    "client_id": "client-id",
    "client_secret": "client-secret",
}


def _context_with_headers(headers: dict) -> Mock:
    """Build a minimal RequestContext-shaped mock carrying the given headers."""
    context = Mock()
    context.call_context.state = {"headers": headers}
    return context


def test_rejects_empty_token():
    """An empty token must be rejected before any introspection call is made."""
    with pytest.raises(auth.TokenValidationError):
        auth.validate_access_token("", **_INTROSPECT_KWARGS)


def test_returns_claims_for_active_token(mocker):
    """An active token's introspection response must be returned as-is."""
    claims = {"active": True, "sub": "user-1"}
    mocker.patch("indykite.intent_agent.auth.httpx.post", return_value=httpx.Response(200, json=claims))

    result = auth.validate_access_token("token", **_INTROSPECT_KWARGS)

    assert result == claims


def test_rejects_inactive_token(mocker):
    """An introspection response with active=False must raise TokenValidationError."""
    mocker.patch(
        "indykite.intent_agent.auth.httpx.post",
        return_value=httpx.Response(200, json={"active": False}),
    )

    with pytest.raises(auth.TokenValidationError):
        auth.validate_access_token("token", **_INTROSPECT_KWARGS)


def test_rejects_non_200_status(mocker):
    """A non-200 introspection response must raise TokenValidationError."""
    mocker.patch("indykite.intent_agent.auth.httpx.post", return_value=httpx.Response(500))

    with pytest.raises(auth.TokenValidationError):
        auth.validate_access_token("token", **_INTROSPECT_KWARGS)


def test_rejects_non_json_response(mocker):
    """A non-JSON introspection response body must raise TokenValidationError."""
    mocker.patch(
        "indykite.intent_agent.auth.httpx.post",
        return_value=httpx.Response(200, content=b"not json"),
    )

    with pytest.raises(auth.TokenValidationError):
        auth.validate_access_token("token", **_INTROSPECT_KWARGS)


def test_wraps_network_failure(mocker):
    """A network failure reaching the IdP must raise TokenValidationError, not leak httpx's own exception."""
    mocker.patch(
        "indykite.intent_agent.auth.httpx.post",
        side_effect=httpx.ConnectError("Connection refused"),
    )

    with pytest.raises(auth.TokenValidationError):
        auth.validate_access_token("token", **_INTROSPECT_KWARGS)


def test_validate_ik_token_is_blocked():
    """validate_ik_token must remain an explicit NotImplementedError until ITS exists."""
    with pytest.raises(NotImplementedError):
        auth.validate_ik_token("ik-token")


def test_get_access_token_from_context_extracts_bearer_token():
    """A Bearer Authorization header must have its token extracted."""
    context = _context_with_headers({"authorization": "Bearer abc123"})
    assert auth.get_access_token_from_context(context) == "abc123"


def test_get_access_token_from_context_no_context():
    """A missing RequestContext must yield an empty token, not raise."""
    assert auth.get_access_token_from_context(None) == ""


def test_get_access_token_from_context_no_call_context():
    """A RequestContext with no call_context must yield an empty token."""
    context = Mock(call_context=None)
    assert auth.get_access_token_from_context(context) == ""
