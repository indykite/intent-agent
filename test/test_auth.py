import asyncio
from dataclasses import replace
from http import HTTPStatus

import httpx2
import pytest
from a2a.types import AgentCard
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from indykite.intent_agent import auth
from indykite.intent_agent.dependency_status import INTROSPECTION, DependencyStatus

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
    """A blank token is rejected as the caller's fault before any introspection call."""
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
    """The act claim is ignored: delegation is validated upstream by the IAG."""
    claims = {"active": True, "sub": "user-1", "aud": "intent_agent", "act": {"sub": "unrelated-actor"}}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    result = asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert result == claims


def test_rejects_inactive_token(mocker):
    """active=False is the caller's fault."""
    _mock_async_client(mocker, response=httpx2.Response(200, json={"active": False}))

    with pytest.raises(auth.TokenValidationError, match="Token is not active") as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.CALLER


@pytest.mark.parametrize("payload", [{"sub": "user-1"}, {"active": None}, {"active": "true"}, {"active": 1}])
def test_rejects_response_that_does_not_say_whether_active(mocker, payload):
    """A missing or non-boolean `active` is the IdP's fault, not an inactive token."""
    _mock_async_client(mocker, response=httpx2.Response(200, json=payload))

    with pytest.raises(auth.TokenValidationError, match="did not say whether") as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.PROVIDER


def test_accepts_active_token_missing_sub(mocker):
    """No sub is fine: the token authenticates the calling client, the subject comes from message metadata."""
    claims = {"active": True, "aud": "intent_agent"}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    result = asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert result == claims


def test_rejects_token_with_wrong_audience(mocker):
    """An active token scoped to another audience is the caller's fault."""
    claims = {"active": True, "aud": "some-other-service"}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    with pytest.raises(auth.TokenValidationError, match="not valid for this agent") as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.CALLER


def test_logs_both_audiences_on_mismatch(mocker, caplog):
    """The log names both audiences, since the caller-facing message stays generic."""
    claims = {"active": True, "aud": "some-other-service"}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))
    caplog.set_level("INFO")

    with pytest.raises(auth.TokenValidationError):
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert "['some-other-service']" in caplog.text
    assert "'intent_agent'" in caplog.text


def test_rejects_token_missing_audience(mocker):
    """A missing `aud` is rejected like a wrong one, not treated as unconstrained."""
    claims = {"active": True}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    with pytest.raises(auth.TokenValidationError, match="not valid for this agent") as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.CALLER


def test_accepts_token_with_audience_as_array_containing_expected(mocker):
    """An `aud` array is accepted when it contains the expected audience."""
    claims = {"active": True, "aud": ["some-other-service", "intent_agent"]}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))

    result = asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert result == claims


def test_rejects_audience_using_the_configured_expected_value(mocker):
    """The configured expected_audience is used, not a hardcoded one."""
    claims = {"active": True, "aud": "custom-resource"}
    _mock_async_client(mocker, response=httpx2.Response(200, json=claims))
    config = replace(_INTROSPECT_CONFIG, expected_audience="custom-resource")

    result = asyncio.run(auth.validate_access_token("token", config))

    assert result == claims


@pytest.mark.parametrize("status", [500, 503, 400, 404])
def test_rejects_unexpected_status_as_provider_fault(mocker, status):
    """Any non-200 status, 4xx and 5xx alike, is the IdP's fault."""
    _mock_async_client(mocker, response=httpx2.Response(status))

    with pytest.raises(auth.TokenValidationError, match=f"status {status}") as excinfo:
        asyncio.run(auth.validate_access_token("token", _INTROSPECT_CONFIG))

    assert excinfo.value.failure is auth.TokenFailure.PROVIDER


def test_rejects_401_as_our_own_credentials_being_invalid(mocker):
    """A 401 means our own client credentials are invalid; still classified as PROVIDER."""
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
    """A non-object JSON body is the IdP's fault, not an AttributeError."""
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
    """A network failure is wrapped as this service's fault, not leaked as an httpx2 exception."""
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


def test_add_bearer_security_declares_the_token_the_middleware_requires():
    """The card gets one HTTP Bearer scheme naming the audience, required with no scopes."""
    card = AgentCard(name="intent_agent")

    auth.add_bearer_security(card, "custom-agent")

    (name,) = card.security_schemes
    scheme = card.security_schemes[name]
    assert scheme.WhichOneof("scheme") == "http_auth_security_scheme"
    assert scheme.http_auth_security_scheme.scheme == "Bearer"
    assert not scheme.http_auth_security_scheme.bearer_format
    assert "'custom-agent'" in scheme.http_auth_security_scheme.description
    (requirement,) = card.security_requirements
    assert list(requirement.schemes) == [name]
    assert list(requirement.schemes[name].list) == []


@pytest.mark.parametrize(
    ("claims", "failure", "expected"),
    [
        pytest.param({"active": True, "sub": "iag"}, None, "ok", id="accepted"),
        pytest.param(None, auth.TokenValidationError("nope", auth.TokenFailure.CALLER), "ok", id="caller-fault"),
        pytest.param(None, auth.TokenValidationError("nope", auth.TokenFailure.SERVICE), "error", id="service-fault"),
        pytest.param(None, auth.TokenValidationError("nope", auth.TokenFailure.PROVIDER), "error", id="provider-fault"),
    ],
)
def test_middleware_records_introspection_state(mocker, claims, failure, expected):
    """An inactive token is the caller's fault - the IdP answered fine; any other refusal is the IdP's or ours."""
    status = DependencyStatus()
    mocker.patch.object(auth, "validate_access_token", mocker.AsyncMock(return_value=claims, side_effect=failure))

    async def _endpoint(_request):
        return PlainTextResponse("ok")

    async def _on_rejected(_request, _message):
        return None

    middleware = Middleware(
        auth.BearerAuthMiddleware,
        config=_INTROSPECT_CONFIG,
        on_rejected=_on_rejected,
        dependency_status=status,
    )
    client = TestClient(Starlette(routes=[Route("/", _endpoint, methods=["POST"], middleware=[middleware])]))

    client.post("/", headers={"Authorization": "Bearer t"})

    assert status.get(INTROSPECTION).status == expected
