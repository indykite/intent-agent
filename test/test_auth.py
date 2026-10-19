import asyncio
from dataclasses import replace
from http import HTTPStatus

import httpx2
import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from indykite.intent_agent import auth

_INTROSPECT_CONFIG = auth.IntrospectionConfig(
    introspect_url="https://idp.example.com/introspect",
    client_id="client-id",
    client_secret="client-secret",  # noqa: S106 - test fixture value, not a real secret
    expected_audience="intent_agent",
)


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
    """An empty or whitespace-only token must be rejected before any introspection call is made,
    as the caller's fault (TokenFailure.CALLER).
    """
    with pytest.raises(auth.TokenValidationError) as excinfo:
        asyncio.run(auth.validate_access_token(token, _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.CALLER


def test_returns_claims_for_active_token(mocker):
    """An active token's introspection response must be returned as-is."""
    claims = {"active": True, "sub": "user-1", "aud": "intent_agent"}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    result = asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert result == claims


def test_accepts_active_token_regardless_of_act_claim(mocker):
    """The act claim is irrelevant to the Intent Agent - delegation is already validated
    upstream by IAG before the request ever reaches it (per team discussion on PR #9). A token
    with an act claim (or without one) must be accepted the same way, based only on active
    (and, independently, its audience).
    """
    claims = {"active": True, "sub": "user-1", "aud": "intent_agent", "act": {"sub": "unrelated-actor"}}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    result = asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert result == claims


def test_rejects_inactive_token(mocker):
    """An introspection response with active=False must raise TokenValidationError as the
    caller's fault (TokenFailure.CALLER) - the one RFC 7662 answer that is about the token.
    """
    _mock_async_client(mocker, response=httpx2.Response(200, json={"active": False}))

    with pytest.raises(auth.TokenValidationError, match="Token is not active") as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.CALLER


@pytest.mark.parametrize("payload", [{"sub": "user-1"}, {"active": None}, {"active": "true"}, {"active": 1}])
def test_rejects_response_that_does_not_say_whether_active(mocker, payload):
    """An answer that omits `active`, or doesn't make it a boolean, isn't an answer about the token
    at all (RFC 7662 2.2 requires the boolean): it's the IdP's fault (TokenFailure.PROVIDER), not
    an inactive token - the same split the IAG's introspectionRefusal makes.
    """
    _mock_async_client(mocker, response=httpx2.Response(200, json=payload))

    with pytest.raises(auth.TokenValidationError, match="did not say whether") as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.PROVIDER


def test_accepts_active_token_missing_sub(mocker):
    """An active token with no sub claim must be accepted - this token authenticates the
    calling client, not any subject (subject_type/subject_id come from the caller-supplied
    message metadata instead, see IntentExecutor._subject_params), and a client-credentials-grant
    token - the expected case now - typically has no sub claim at all.
    """
    claims = {"active": True, "aud": "intent_agent"}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    result = asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert result == claims


def test_rejects_token_with_wrong_audience(mocker):
    """A token whose `aud` doesn't include this agent's expected audience must be rejected as
    the caller's fault (TokenFailure.CALLER) - it's a real, active token, just not scoped to
    this agent (e.g. reused from a different resource server).
    """
    claims = {"active": True, "aud": "some-other-service"}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    with pytest.raises(auth.TokenValidationError, match="not valid for this agent") as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.CALLER


def test_logs_both_audiences_on_mismatch(mocker, caplog):
    """The server-side log for an audience mismatch must name both the token's actual `aud`
    and the expected one - the caller-facing message stays generic (like "Token is not
    active"), so this diagnostic detail has to live in the log, not be lost entirely.
    """
    claims = {"active": True, "aud": "some-other-service"}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))
    caplog.set_level("INFO")

    with pytest.raises(auth.TokenValidationError):
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert "['some-other-service']" in caplog.text
    assert "'intent_agent'" in caplog.text


def test_rejects_token_missing_audience(mocker):
    """An active token with no `aud` claim at all must be rejected the same as a wrong one -
    a token genuinely scoped to this agent is expected to carry it once audience checking is
    in effect, so a missing claim can't be treated as "audience unconstrained".
    """
    claims = {"active": True}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    with pytest.raises(auth.TokenValidationError, match="not valid for this agent") as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.CALLER


def test_accepts_token_with_audience_as_array_containing_expected(mocker):
    """`aud` may legally be an array of audiences (RFC 7519 4.1.3) - the token must be accepted
    if the expected audience is anywhere in it, not only when it's the sole value.
    """
    claims = {"active": True, "aud": ["some-other-service", "intent_agent"]}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    result = asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert result == claims


def test_rejects_audience_using_the_configured_expected_value(mocker):
    """expected_audience must actually be used, not a hardcoded string - a token scoped to a
    non-default expected audience must be accepted when that audience is configured, and a
    default-scoped one rejected in that same configuration.
    """
    claims = {"active": True, "aud": "custom-resource"}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))
    config = replace(_INTROSPECT_CONFIG, expected_audience="custom-resource")

    result = asyncio.run(auth.validate_access_token("token", config))

    assert result == claims


@pytest.mark.parametrize("status", [500, 503, 400, 404])
def test_rejects_unexpected_status_as_provider_fault(mocker, status):
    """A status instead of an RFC 7662 answer must raise TokenValidationError as the IdP's fault
    (TokenFailure.PROVIDER) - it says nothing about the token. 4xx and 5xx alike: the IAG's
    introspectionStatusRefusal makes no finer split, and neither does this.
    """
    _mock_async_client(mocker, response=httpx2.Response(status))

    with pytest.raises(auth.TokenValidationError, match=f"status {status}") as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.PROVIDER


def test_rejects_401_as_our_own_credentials_being_invalid(mocker):
    """A 401 from the introspection endpoint means OUR client credentials are invalid (RFC 7662
    2.3), not the token - the message must say so. It is still classified like any other status
    instead of an answer (TokenFailure.PROVIDER), as the IAG does; only the message differs.
    """
    _mock_async_client(mocker, response=httpx2.Response(401))

    with pytest.raises(auth.TokenValidationError, match="our own client credentials") as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.PROVIDER


def test_rejects_non_json_response(mocker):
    """A non-JSON introspection response body must raise TokenValidationError as the IdP's fault."""
    _mock_async_client(mocker, response=httpx2.Response(200, content=b"not json"))

    with pytest.raises(auth.TokenValidationError) as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.PROVIDER


def test_rejects_non_object_json_response(mocker):
    """A valid JSON body that isn't an object (e.g. a list) must raise TokenValidationError as the
    IdP's fault, not AttributeError.
    """
    _mock_async_client(mocker, response=httpx2.Response(200, json=["not", "an", "object"]))

    with pytest.raises(auth.TokenValidationError) as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.PROVIDER


@pytest.mark.parametrize(
    "error",
    [httpx2.ConnectError("Connection refused"), httpx2.ReadTimeout("timed out")],
    ids=["connect-error", "timeout"],
)
def test_wraps_network_failure_as_service_fault(mocker, error):
    """A network failure reaching the IdP must raise TokenValidationError, not leak httpx2's own
    exception - and as this service's fault (TokenFailure.SERVICE): it never got an answer, so
    nothing was said about the token.
    """
    _mock_async_client(mocker, side_effect=error)

    with pytest.raises(auth.TokenValidationError, match="request failed") as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.SERVICE


def test_bearer_token_extracts_bearer_token():
    """A Bearer Authorization header must have its token extracted."""
    assert auth.bearer_token({"authorization": "Bearer abc123"}) == "abc123"


def test_bearer_token_no_header():
    """No Authorization header must yield an empty token, not raise."""
    assert auth.bearer_token({}) == ""


def test_bearer_token_rejects_non_bearer_scheme():
    """A non-Bearer Authorization header must yield an empty token, not be passed through as-is."""
    assert auth.bearer_token({"authorization": "Basic dXNlcjpwYXNz"}) == ""


def test_bearer_token_uses_capitalized_header_fallback():
    """The capitalized 'Authorization' header key must be used when lowercase 'authorization' is absent."""
    assert auth.bearer_token({"Authorization": "Bearer abc123"}) == "abc123"


def test_bearer_token_scheme_match_is_case_insensitive():
    """The 'Bearer' scheme match must be case-insensitive, not just work by coincidence on the canonical case."""
    assert auth.bearer_token({"authorization": "BEARER abc123"}) == "abc123"


def _guarded_client(mocker, *, claims=None, failure=None):
    """A route behind BearerAuthMiddleware echoing the caller name, the (body, message) pairs the
    rejection hook saw, and the endpoint mock.
    """
    mocker.patch.object(auth, "validate_access_token", mocker.AsyncMock(return_value=claims, side_effect=failure))
    bodies = []

    async def _on_rejected(request, message):
        bodies.append((await request.body(), message))

    endpoint = mocker.AsyncMock()

    async def _echo_caller(request):
        await endpoint(request)
        return PlainTextResponse(request.user.display_name)

    app = Starlette(
        routes=[
            Route(
                "/",
                _echo_caller,
                methods=["POST"],
                middleware=[Middleware(auth.BearerAuthMiddleware, config=_INTROSPECT_CONFIG, on_rejected=_on_rejected)],
            ),
        ],
    )
    return TestClient(app), bodies, endpoint


@pytest.mark.parametrize(
    ("claims", "caller"),
    [
        ({"active": True, "sub": "user-1", "client_id": "iag"}, "user-1"),
        ({"active": True, "client_id": "iag"}, "iag"),
        ({"active": True, "sub": "  ", "client_id": "iag"}, "iag"),
    ],
    ids=["sub", "client-id-without-sub", "blank-sub"],
)
def test_middleware_makes_the_tokens_caller_the_request_user(mocker, claims, caller):
    """An accepted token's `sub`, else `client_id`, names `request.user`."""
    client, bodies, _endpoint = _guarded_client(mocker, claims=claims)

    response = client.post("/", headers={"Authorization": "Bearer good-token"})

    assert response.status_code == HTTPStatus.OK
    assert response.text == caller
    assert bodies == []


def test_middleware_refuses_a_missing_token_without_introspecting(mocker):
    """No Bearer token is a 401 with a bare `Bearer` challenge (RFC 6750 3), reported to the hook."""
    client, bodies, endpoint = _guarded_client(mocker)

    response = client.post("/", content=b"the body", headers={"Authorization": "Basic dXNlcjpwYXNz"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.text == "Authorization required"
    assert bodies == [(b"the body", "Authorization required")]
    endpoint.assert_not_awaited()
    auth.validate_access_token.assert_not_awaited()


@pytest.mark.parametrize(
    ("failure", "message", "status"),
    [
        (auth.TokenFailure.CALLER, "Token is not active", HTTPStatus.UNAUTHORIZED),
        (auth.TokenFailure.SERVICE, "Token introspection request failed", HTTPStatus.INTERNAL_SERVER_ERROR),
        (auth.TokenFailure.PROVIDER, "Introspection endpoint returned status 503", HTTPStatus.BAD_GATEWAY),
    ],
    ids=["caller-401", "service-500", "provider-502"],
)
def test_middleware_answers_token_failures_by_whose_fault_they_are(mocker, failure, message, status):
    """401/500/502 by TokenFailure, only the 401 with a challenge, each reported to the hook."""
    client, bodies, endpoint = _guarded_client(mocker, failure=auth.TokenValidationError(message, failure))

    response = client.post("/", content=b"{}", headers={"Authorization": "Bearer some-token"})

    assert response.status_code == status
    assert response.text == message
    assert response.headers.get("WWW-Authenticate") == (
        'Bearer error="invalid_token"' if status == HTTPStatus.UNAUTHORIZED else None
    )
    assert bodies == [(b"{}", message)]
    endpoint.assert_not_awaited()


def test_middleware_refuses_a_token_that_names_no_caller(mocker):
    """An active token with neither `sub` nor `client_id` is a 401."""
    client, bodies, endpoint = _guarded_client(mocker, claims={"active": True, "aud": "intent_agent"})

    response = client.post("/", headers={"Authorization": "Bearer good-token"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.headers["WWW-Authenticate"] == 'Bearer error="invalid_token"'
    assert bodies == [(b"", "Token does not identify its caller")]
    endpoint.assert_not_awaited()
