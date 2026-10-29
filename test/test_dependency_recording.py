"""The real calls record their outcome into DependencyStatus - the state /health reports."""

import asyncio

import httpx2
import jwt
import pytest
from dogpile.cache.api import NO_VALUE
from dogpile.cache.backends.redis import RedisBackend
from dogpile.cache.region import make_region

from indykite.intent_agent import auth
from indykite.intent_agent.agent import IntentAgent
from indykite.intent_agent.audit.http import HttpConfig, HttpSender
from indykite.intent_agent.config import Settings
from indykite.intent_agent.dependency_status import (
    AUDIT_WEBHOOK,
    GEMINI,
    IKG,
    INTROSPECTION,
    REDIS,
    SCHEMA,
    DependencyStatus,
)
from indykite.intent_agent.executor import IntentExecutor
from indykite.intent_agent.policy import PolicyClient
from indykite.intent_agent.schema import SchemaClient

_KEY = jwt.encode({"sub": "gid"}, "s" * 32, algorithm="HS256")
_SHA256_HEX_LENGTH = 64
_SCHEMA = {"graph": {"nodes": {"Person": {}}, "edges": {}}}


def _memory_cache():
    return make_region().configure("dogpile.cache.memory", expiration_time=60)


def _patch_client(mocker, module, *, response=None, error=None):
    client = mocker.MagicMock()
    client.__enter__ = mocker.Mock(return_value=client)
    client.__exit__ = mocker.Mock(return_value=False)
    client.get = mocker.Mock(return_value=response, side_effect=error)
    mocker.patch(f"indykite.intent_agent.{module}.httpx2.Client", return_value=client)


def _response(mocker, code=200, body=None):
    response = mocker.Mock(status_code=code, text="")
    response.json = mocker.Mock(return_value=body if body is not None else {"results": []})
    if code >= httpx2.codes.BAD_REQUEST:
        response.raise_for_status = mocker.Mock(
            side_effect=httpx2.HTTPStatusError("e", request=mocker.Mock(), response=response)
        )
    return response


def test_schema_fetch_records_ikg(mocker):
    status = DependencyStatus()
    client = SchemaClient("https://ikg.example.com", _KEY, cache=_memory_cache(), dependency_status=status)

    _patch_client(mocker, "schema", error=httpx2.ConnectError("x"))
    client.load()
    assert (status.get(IKG).status, status.get(IKG).detail) == ("error", "ConnectError")

    _patch_client(mocker, "schema", response=_response(mocker, body=_SCHEMA))
    client.load()
    assert status.get(IKG).status == "ok"


def test_redis_cache_access_records_redis(mocker):
    status = DependencyStatus()
    cache = mocker.Mock(actual_backend=mocker.Mock(spec=RedisBackend))
    client = SchemaClient("https://ikg.example.com", _KEY, cache=cache, dependency_status=status)

    cache.get = mocker.Mock(side_effect=ConnectionError)
    assert client.get_cached() is None
    assert (status.get(REDIS).status, status.get(REDIS).detail) == ("error", "ConnectionError")

    cache.get = mocker.Mock(return_value=_SCHEMA)
    assert client.get_cached() == _SCHEMA
    assert status.get(REDIS).status == "ok"


def test_memory_cache_records_nothing(mocker):
    status = DependencyStatus()
    SchemaClient("https://ikg.example.com", _KEY, cache=_memory_cache(), dependency_status=status).get_cached()

    assert status.get(REDIS) is None


@pytest.mark.parametrize(
    ("code", "expected"),
    [(200, "ok"), (503, "error"), (401, "error"), (404, None)],
    ids=["ok", "5xx", "auth-rejected", "answered-4xx-says-nothing"],
)
def test_policy_fetch_records_ikg(mocker, code, expected):
    status = DependencyStatus()
    client = PolicyClient("https://ikg.example.com", _KEY, retry_attempts=1, cache=_memory_cache())
    client.dependency_status = status
    _patch_client(mocker, "policy", response=_response(mocker, code))

    client.get_cached("Person")

    assert (status.get(IKG).status if status.get(IKG) else None) == expected


def test_llm_calls_record_gemini(baseline_env, monkeypatch, mocker):
    baseline_env()
    monkeypatch.setenv("IK_APP_AGENT_KEY", _KEY)
    status = DependencyStatus()
    ainvoke = mocker.AsyncMock(side_effect=[ConnectionError, "answer"])
    agent = IntentAgent(Settings(_env_file=None), llm=mocker.Mock(), dependency_status=status)
    llm = mocker.Mock(ainvoke=ainvoke)

    with pytest.raises(ConnectionError):
        asyncio.run(agent._invoke_llm(llm, []))
    assert (status.get(GEMINI).status, status.get(GEMINI).detail) == ("error", "ConnectionError")

    asyncio.run(agent._invoke_llm(llm, []))
    assert status.get(GEMINI).status == "ok"


def test_schema_state_is_recorded(baseline_env, monkeypatch, mocker):
    baseline_env()
    monkeypatch.setenv("IK_APP_AGENT_KEY", _KEY)
    status = DependencyStatus()
    agent = IntentAgent(Settings(_env_file=None), llm=mocker.Mock(), dependency_status=status)

    agent._record_schema(None)
    assert status.get(SCHEMA).status == "error"
    agent._record_schema({"graph": {"nodes": {"_Application": {}}}})
    assert status.get(SCHEMA).detail == "schema fetched but the data model is not populated"
    agent._record_schema(_SCHEMA)
    assert status.get(SCHEMA).status == "ok"
    assert len(status.get(SCHEMA).version) == _SHA256_HEX_LENGTH


@pytest.mark.parametrize(
    ("failure", "expected"),
    [(auth.TokenFailure.CALLER, "ok"), (auth.TokenFailure.SERVICE, "error"), (auth.TokenFailure.PROVIDER, "error")],
)
def test_introspection_outcome_records_token_introspection(baseline_env, mocker, failure, expected):
    """An inactive token is the caller's - the IdP answered fine; anything else is the IdP's or ours."""
    baseline_env()
    status = DependencyStatus()
    agent = mocker.Mock(settings=Settings(_env_file=None), dependency_status=status)
    context = mocker.Mock()
    context.call_context.state = {"headers": {"authorization": "Bearer t"}}
    mocker.patch.object(
        auth, "validate_access_token", mocker.AsyncMock(side_effect=auth.TokenValidationError("nope", failure))
    )

    with pytest.raises(Exception, match="nope"):
        asyncio.run(IntentExecutor(agent, mocker.Mock())._authenticate(context))

    assert status.get(INTROSPECTION).status == expected


def test_audit_delivery_records_audit_webhook(audit_message):
    status = DependencyStatus()
    codes = iter([200, 500])
    sender = HttpSender(
        HttpConfig(url="https://audit.example.com/r", method="POST", auth_type="no-auth"),
        transport=httpx2.MockTransport(lambda _request: httpx2.Response(next(codes))),
        dependency_status=status,
    )

    async def run():
        await sender.send(audit_message())
        first = status.get(AUDIT_WEBHOOK).status
        await sender.send(audit_message())
        await sender.close()
        return first

    assert asyncio.run(run()) == "ok"
    assert (status.get(AUDIT_WEBHOOK).status, status.get(AUDIT_WEBHOOK).detail) == ("error", "status 500")


def test_audit_redirect_is_not_a_delivery(audit_message):
    """The client doesn't follow redirects, so a 3xx never reached the redirect target."""
    status = DependencyStatus()
    sender = HttpSender(
        HttpConfig(url="https://audit.example.com/r", method="POST", auth_type="no-auth"),
        transport=httpx2.MockTransport(lambda _request: httpx2.Response(302, headers={"Location": "https://x"})),
        dependency_status=status,
    )

    async def run():
        await sender.send(audit_message())
        await sender.close()

    asyncio.run(run())

    assert (status.get(AUDIT_WEBHOOK).status, status.get(AUDIT_WEBHOOK).detail) == ("error", "status 302")


def test_redis_policy_cache_access_records_redis(mocker):
    status = DependencyStatus()
    cache = mocker.Mock(actual_backend=mocker.Mock(spec=RedisBackend))
    client = PolicyClient("https://ikg.example.com", _KEY, cache=cache)
    client.dependency_status = status
    cache.get = mocker.Mock(return_value=NO_VALUE)

    cache.get_or_create = mocker.Mock(side_effect=ConnectionError)
    assert client.get_cached("Person") == (None, True)
    assert (status.get(REDIS).status, status.get(REDIS).detail) == ("error", "ConnectionError")

    cache.get_or_create = mocker.Mock(return_value=[])
    assert client.get_cached("Person") == ([], False)
    assert status.get(REDIS).status == "ok"

    # a plain cache hit (no refresh) is a successful Redis access too
    status.record_error(REDIS, "ConnectionError")
    cache.get = mocker.Mock(return_value=[])
    assert client.get_cached("Person") == ([], False)
    assert status.get(REDIS).status == "ok"
