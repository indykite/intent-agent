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
_ITS_INTROSPECT_KWARGS = {
    "introspect_url": "https://its.example.com/oauth2/introspect",
    "client_id": "its-client-id",
    "client_secret": "its-client-secret",
}
# What ITS answers for an active delegation token (introspectionResponse in jarvis
# services/agent-gateway/internal/exchange/introspection.go): RFC 7662 members plus the
# RFC 8693 act chain, outermost actor = the requesting agent.
_ACTIVE_IK_CLAIMS = {
    "active": True,
    "sub": "user-1",
    "iss": "https://token-service.example.com",
    "aud": ["intent-agent"],
    "jti": "tok-1",
    "token_type": "Bearer",
    "act": {"sub": "agent-1", "type": "Agent", "act": {"sub": "agent-0", "type": "Agent"}},
}

# Both validators share the introspection round trip, so the transport/claim edge cases
# are asserted once for each of them.
_VALIDATORS = [
    pytest.param(auth.validate_access_token, _INTROSPECT_KWARGS, id="access_token"),
    pytest.param(auth.validate_ik_token, _ITS_INTROSPECT_KWARGS, id="ik_token"),
]


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


@pytest.mark.parametrize(("validate", "kwargs"), _VALIDATORS)
@pytest.mark.parametrize("token", ["", "   ", "\t\n"])
def test_rejects_blank_token(mocker, validate, kwargs, token):
    """An empty or whitespace-only token must be rejected before any introspection call is made."""
    _mock_client_cls, client = _mock_async_client(mocker)

    with pytest.raises(auth.TokenValidationError):
        asyncio.run(validate(token, **kwargs))

    client.post.assert_not_awaited()


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


@pytest.mark.parametrize(("validate", "kwargs"), _VALIDATORS)
def test_posts_rfc7662_form_with_client_secret_basic(mocker, validate, kwargs):
    """Both legs must speak RFC 7662: form-encoded `token` to the configured URL, HTTP Basic
    client auth, bounded by the given timeout - ITS and the external IdP share this contract.
    """
    _mock_client_cls, client = _mock_async_client(mocker, response=httpx2.Response(200, json=_ACTIVE_IK_CLAIMS))

    asyncio.run(validate("the-token", timeout_seconds=2.5, **kwargs))

    client.post.assert_awaited_once()
    args, call_kwargs = client.post.await_args
    assert args == (kwargs["introspect_url"],)
    assert call_kwargs["auth"] == (kwargs["client_id"], kwargs["client_secret"])
    assert call_kwargs["data"] == {"token": "the-token", "token_type_hint": "access_token"}
    assert call_kwargs["timeout"] == 2.5  # noqa: PLR2004 - the timeout passed in above


@pytest.mark.parametrize(("validate", "kwargs"), _VALIDATORS)
def test_rejects_inactive_token(mocker, validate, kwargs):
    """An introspection response with active=False must raise TokenValidationError."""
    _mock_async_client(mocker, response=httpx2.Response(200, json={"active": False}))

    with pytest.raises(auth.TokenValidationError, match="not active"):
        asyncio.run(validate("token", **kwargs))


@pytest.mark.parametrize(("validate", "kwargs"), _VALIDATORS)
@pytest.mark.parametrize(
    "sub",
    [
        pytest.param(None, id="missing"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param(42, id="number"),
        pytest.param(["user-1"], id="list"),
        pytest.param({"id": "user-1"}, id="object"),
        pytest.param(True, id="bool"),
    ],
)
def test_rejects_active_token_without_usable_sub(mocker, validate, kwargs, sub):
    """An active token whose sub is missing, blank, or not a string must be rejected - the value
    becomes the caller's identity in every evaluation, so "truthy" is not good enough.
    """
    claims = {"active": True, "act": {"sub": "agent-1"}}
    if sub is not None:
        claims["sub"] = sub
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    with pytest.raises(auth.TokenValidationError, match="sub claim"):
        asyncio.run(validate("token", **kwargs))


@pytest.mark.parametrize(("validate", "kwargs"), _VALIDATORS)
def test_rejects_non_200_status(mocker, validate, kwargs):
    """A non-200 introspection response must raise TokenValidationError."""
    _mock_async_client(mocker, response=httpx2.Response(500))

    with pytest.raises(auth.TokenValidationError, match="status 500"):
        asyncio.run(validate("token", **kwargs))


@pytest.mark.parametrize(("validate", "kwargs"), _VALIDATORS)
def test_rejects_401_as_our_own_credentials_being_invalid(mocker, validate, kwargs):
    """A 401 from the introspection endpoint means OUR client credentials are invalid (RFC 7662 2.3), not the token."""
    _mock_async_client(mocker, response=httpx2.Response(401))

    with pytest.raises(auth.TokenValidationError, match="our own client credentials"):
        asyncio.run(validate("token", **kwargs))


@pytest.mark.parametrize(("validate", "kwargs"), _VALIDATORS)
def test_rejects_non_json_response(mocker, validate, kwargs):
    """A non-JSON introspection response body must raise TokenValidationError."""
    _mock_async_client(mocker, response=httpx2.Response(200, content=b"not json"))

    with pytest.raises(auth.TokenValidationError, match="invalid response"):
        asyncio.run(validate("token", **kwargs))


@pytest.mark.parametrize(("validate", "kwargs"), _VALIDATORS)
def test_rejects_non_object_json_response(mocker, validate, kwargs):
    """A valid JSON body that isn't an object (e.g. a list) must raise TokenValidationError, not AttributeError."""
    _mock_async_client(mocker, response=httpx2.Response(200, json=["not", "an", "object"]))

    with pytest.raises(auth.TokenValidationError, match="unexpected response shape"):
        asyncio.run(validate("token", **kwargs))


@pytest.mark.parametrize(("validate", "kwargs"), _VALIDATORS)
def test_wraps_network_failure(mocker, validate, kwargs):
    """A network failure reaching the server must raise TokenValidationError, not leak httpx2's own exception."""
    _mock_async_client(mocker, side_effect=httpx2.ConnectError("Connection refused"))

    with pytest.raises(auth.TokenValidationError, match="request failed"):
        asyncio.run(validate("token", **kwargs))


def test_validate_ik_token_returns_claims_for_active_delegation(mocker):
    """An active ITS delegation token's introspection response must be returned as-is."""
    _mock_async_client(mocker, response=httpx2.Response(200, json=_ACTIVE_IK_CLAIMS))

    result = asyncio.run(auth.validate_ik_token("ik-token", **_ITS_INTROSPECT_KWARGS))

    assert result == _ACTIVE_IK_CLAIMS


_ACT_VARIANTS_WITHOUT_ACTOR = [
    pytest.param(None, id="no_act"),
    pytest.param({}, id="empty_act"),
    pytest.param({"type": "Agent"}, id="act_without_sub"),
    pytest.param({"sub": ""}, id="blank_actor"),
    pytest.param({"sub": "   "}, id="whitespace_actor"),
    pytest.param({"sub": 42}, id="non_string_actor"),
    pytest.param("agent-1", id="act_not_an_object"),
]


def _claims_with_act(act):
    claims = {"active": True, "sub": "user-1"}
    if act is not None:
        claims["act"] = act
    return claims


@pytest.mark.parametrize("act", _ACT_VARIANTS_WITHOUT_ACTOR)
def test_validate_ik_token_accepts_delegation_regardless_of_act_claim(mocker, act):
    """The act chain is irrelevant to the Intent Agent, for the ik_token as for the access
    token: IAG validated the delegation chain before the request got here (TDD owner's
    decision). An active ik_token with a sub is accepted whatever its act looks like.
    """
    claims = _claims_with_act(act)
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    result = asyncio.run(auth.validate_ik_token("ik-token", **_ITS_INTROSPECT_KWARGS))

    assert result == claims


def test_requesting_agent_id_is_outermost_actor():
    """The requesting agent is the outermost act.sub - inner act entries are earlier hops."""
    assert auth.requesting_agent_id(_ACTIVE_IK_CLAIMS) == "agent-1"


def test_requesting_agent_id_strips_surrounding_whitespace():
    """A padded act.sub is returned trimmed, so the log line never shows a padded actor."""
    assert auth.requesting_agent_id({"act": {"sub": "  agent-1\n"}}) == "agent-1"


@pytest.mark.parametrize("act", _ACT_VARIANTS_WITHOUT_ACTOR)
def test_requesting_agent_id_is_empty_without_usable_actor(act):
    """No usable act.sub yields "" for the log line - never a KeyError/TypeError on odd shapes."""
    assert auth.requesting_agent_id(_claims_with_act(act)) == ""


def test_check_delegated_subject_accepts_matching_subject():
    """A delegation minted for the access token's subject passes."""
    auth.check_delegated_subject(_ACTIVE_IK_CLAIMS, subject_id="user-1")


@pytest.mark.parametrize("ik_subject", ["user-2", "", None])
def test_check_delegated_subject_rejects_other_subject(ik_subject):
    """A delegation for another subject (or none) beside this user's access token is the TDD 4.3
    "actor not permitted to act for Subject" case and must fail with that reason.
    """
    claims = {**_ACTIVE_IK_CLAIMS, "sub": ik_subject}
    with pytest.raises(auth.TokenValidationError, match="actor delegation not authorized"):
        auth.check_delegated_subject(claims, subject_id="user-1")


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


def test_get_ik_token_from_context_extracts_raw_header_value():
    """X-IK-Token carries the raw delegation token (no Bearer prefix) - returned whole, whitespace-stripped."""
    context = _context_with_headers({"x-ik-token": "  eyJ.ik.token \n"})
    assert auth.get_ik_token_from_context(context) == "eyJ.ik.token"


def test_get_ik_token_from_context_uses_canonical_case_header_fallback():
    """The 'X-IK-Token' header key must be used when the lowercase key is absent."""
    context = _context_with_headers({"X-IK-Token": "eyJ.ik.token"})
    assert auth.get_ik_token_from_context(context) == "eyJ.ik.token"


def test_get_ik_token_from_context_is_independent_of_authorization():
    """An Authorization header alone must not be mistaken for an ik_token, and vice versa."""
    context = _context_with_headers({"authorization": "Bearer abc123"})
    assert auth.get_ik_token_from_context(context) == ""
    context = _context_with_headers({"x-ik-token": "eyJ.ik.token"})
    assert auth.get_access_token_from_context(context) == ""


@pytest.mark.parametrize("context", [None, Mock(call_context=None)])
def test_get_ik_token_from_context_without_context(context):
    """A missing RequestContext or call_context must yield an empty token, not raise."""
    assert auth.get_ik_token_from_context(context) == ""
