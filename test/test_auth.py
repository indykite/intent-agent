import asyncio
from unittest.mock import Mock

import httpx2
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


def _mock_async_client(mocker, *, response=None, side_effect=None):
    """Patch httpx2.AsyncClient so `async with ... as client: await client.post(...)` yields/raises as given."""
    client = mocker.MagicMock()
    client.__aenter__ = mocker.AsyncMock(return_value=client)
    client.__aexit__ = mocker.AsyncMock(return_value=False)
    client.post = mocker.AsyncMock(return_value=response, side_effect=side_effect)
    mock_client_cls = mocker.patch("indykite.intent_agent.auth.httpx2.AsyncClient", return_value=client)
    return mock_client_cls, client


@pytest.mark.parametrize("token", ["", "   ", "\t\n"])
def test_rejects_blank_token(token):
    """An empty or whitespace-only token must be rejected before any introspection call is made."""
    with pytest.raises(auth.TokenValidationError):
        asyncio.run(auth.validate_access_token(token, **_INTROSPECT_KWARGS))


def test_returns_claims_for_active_token(mocker):
    """An active token's introspection response must be returned as-is."""
    claims = {"active": True, "sub": "user-1"}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    result = asyncio.run(auth.validate_access_token("token", **_INTROSPECT_KWARGS))

    assert result == claims


def test_accepts_active_token_regardless_of_act_claim(mocker):
    """The act claim is irrelevant to the Intent Agent - delegation is already validated
    upstream by IAG before the request ever reaches it (per team discussion on PR #9). A token
    with an act claim (or without one) must be accepted the same way, based only on active/sub.
    """
    claims = {"active": True, "sub": "user-1", "aud": "someone-else", "act": {"sub": "unrelated-actor"}}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    result = asyncio.run(auth.validate_access_token("token", **_INTROSPECT_KWARGS))

    assert result == claims


def test_rejects_inactive_token(mocker):
    """An introspection response with active=False must raise TokenValidationError."""
    _mock_async_client(mocker, response=httpx2.Response(200, json={"active": False}))

    with pytest.raises(auth.TokenValidationError):
        asyncio.run(auth.validate_access_token("token", **_INTROSPECT_KWARGS))


def test_rejects_active_token_missing_sub(mocker):
    """An active token with no sub claim must be rejected, not proceed with subject_id=None."""
    _mock_async_client(mocker, response=httpx2.Response(200, json={"active": True}))

    with pytest.raises(auth.TokenValidationError):
        asyncio.run(auth.validate_access_token("token", **_INTROSPECT_KWARGS))


def test_rejects_non_200_status(mocker):
    """A non-200 introspection response must raise TokenValidationError."""
    _mock_async_client(mocker, response=httpx2.Response(500))

    with pytest.raises(auth.TokenValidationError):
        asyncio.run(auth.validate_access_token("token", **_INTROSPECT_KWARGS))


def test_rejects_401_as_our_own_credentials_being_invalid(mocker):
    """A 401 from the introspection endpoint means OUR client credentials are invalid (RFC 7662 2.3), not the token."""
    _mock_async_client(mocker, response=httpx2.Response(401))

    with pytest.raises(auth.TokenValidationError, match="our own client credentials"):
        asyncio.run(auth.validate_access_token("token", **_INTROSPECT_KWARGS))


def test_rejects_non_json_response(mocker):
    """A non-JSON introspection response body must raise TokenValidationError."""
    _mock_async_client(mocker, response=httpx2.Response(200, content=b"not json"))

    with pytest.raises(auth.TokenValidationError):
        asyncio.run(auth.validate_access_token("token", **_INTROSPECT_KWARGS))


def test_rejects_non_object_json_response(mocker):
    """A valid JSON body that isn't an object (e.g. a list) must raise TokenValidationError, not AttributeError."""
    _mock_async_client(mocker, response=httpx2.Response(200, json=["not", "an", "object"]))

    with pytest.raises(auth.TokenValidationError):
        asyncio.run(auth.validate_access_token("token", **_INTROSPECT_KWARGS))


def test_wraps_network_failure(mocker):
    """A network failure reaching the IdP must raise TokenValidationError, not leak httpx2's own exception."""
    _mock_async_client(mocker, side_effect=httpx2.ConnectError("Connection refused"))

    with pytest.raises(auth.TokenValidationError):
        asyncio.run(auth.validate_access_token("token", **_INTROSPECT_KWARGS))


def test_validate_ik_token_is_blocked():
    """validate_ik_token must remain an explicit NotImplementedError until ITS exists."""
    with pytest.raises(NotImplementedError):
        auth.validate_ik_token("ik-token")


def test_validate_app_agent_token_is_blocked():
    """validate_app_agent_token must remain an explicit NotImplementedError until validation is confirmed."""
    with pytest.raises(NotImplementedError):
        auth.validate_app_agent_token("app-agent-token")


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


def test_get_access_token_from_context_rejects_non_bearer_scheme():
    """A non-Bearer Authorization header must yield an empty token, not be passed through as-is."""
    context = _context_with_headers({"authorization": "Basic dXNlcjpwYXNz"})
    assert auth.get_access_token_from_context(context) == ""


def test_get_access_token_from_context_uses_capitalized_header_fallback():
    """The capitalized 'Authorization' header key must be used when lowercase 'authorization' is absent."""
    context = _context_with_headers({"Authorization": "Bearer abc123"})
    assert auth.get_access_token_from_context(context) == "abc123"


def test_get_access_token_from_context_scheme_match_is_case_insensitive():
    """The 'Bearer' scheme match must be case-insensitive, not just work by coincidence on the canonical case."""
    context = _context_with_headers({"authorization": "BEARER abc123"})
    assert auth.get_access_token_from_context(context) == "abc123"
