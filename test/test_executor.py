"""IntentExecutor.execute's token-validation branches (TDD 4.1 steps 1-3, 4.3 failure modes).

The pipeline after authentication (schema, LLM, artifact) is exercised elsewhere
(test_agent.py, integration/); here run_extraction is stubbed so each test pins one
authentication outcome: which validators ran, with what, and whether the request was served.
"""

import asyncio
from unittest.mock import Mock

import jwt
import pytest
from starlette.exceptions import HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED

from indykite.intent_agent import auth
from indykite.intent_agent.agent import ExtractionResult, IntentAgent, IntentExecutor
from indykite.intent_agent.audit.message import Outcome
from indykite.intent_agent.config import Settings

_IDP_INTROSPECT_KWARGS = {
    "introspect_url": "https://idp.example.com/introspect",
    "client_id": "client-id",
    "client_secret": "client-secret",
    "timeout_seconds": 5.0,
}
_ACCESS_CLAIMS = {"active": True, "sub": "user-1"}
_IK_CLAIMS = {"active": True, "sub": "user-1", "act": {"sub": "agent-1", "type": "Agent"}}
_ITS_ENV = {
    "ITS_INTROSPECT_URL": "https://its.example.com/oauth2/introspect",
    "ITS_CLIENT_ID": "its-client-id",
    "ITS_CLIENT_SECRET": "its-client-secret",
    "ITS_INTROSPECT_TIMEOUT_SECONDS": "2.5",
}
# Settings requires IK_APP_AGENT_KEY to be set, and SchemaClient requires any non-blank value
# to be a well-formed JWT (see test_schema.py).
_APP_AGENT_KEY = jwt.encode({"sub": "test-app-agent"}, key="", algorithm="none")


def _executor(baseline_env, monkeypatch, mocker, *, its: bool):
    """An IntentExecutor whose agent's run_extraction is a stub and whose audit sender records,
    with or without ITS configured. Returns the executor, the extraction stub and the sender.
    """
    baseline_env()
    monkeypatch.setenv("IK_APP_AGENT_KEY", _APP_AGENT_KEY)
    for env_var in _ITS_ENV:
        monkeypatch.delenv(env_var, raising=False)
    if its:
        for env_var, value in _ITS_ENV.items():
            monkeypatch.setenv(env_var, value)
    settings = Settings(_env_file=None)
    agent = IntentAgent(settings, llm=mocker.Mock(), schema_client=mocker.Mock())
    result = ExtractionResult(text='{"prompt": "rent kitt", "evaluations": []}', outcome=Outcome.UNMATCHED)
    run_extraction = mocker.AsyncMock(return_value=result)
    mocker.patch.object(agent, "run_extraction", run_extraction)
    sender = mocker.Mock(send=mocker.AsyncMock(), close=mocker.AsyncMock())
    return IntentExecutor(agent, sender), run_extraction, sender


def _context(headers: dict[str, str]) -> Mock:
    """A RequestContext-shaped mock carrying the given headers and one text part."""
    context = Mock()
    context.call_context.state = {"headers": headers}
    context.message.parts = [Mock(text="rent kitt")]
    context.current_task = Mock()
    context.task_id = "task-1"
    context.context_id = "ctx-1"
    return context


def _patch_validators(mocker, *, access=_ACCESS_CLAIMS, ik=_IK_CLAIMS):
    """Stub both introspection validators; a dict is returned, an exception is raised."""
    access_kwargs = {"side_effect": access} if isinstance(access, Exception) else {"return_value": access}
    ik_kwargs = {"side_effect": ik} if isinstance(ik, Exception) else {"return_value": ik}
    return (
        mocker.patch("indykite.intent_agent.agent.auth.validate_access_token", mocker.AsyncMock(**access_kwargs)),
        mocker.patch("indykite.intent_agent.agent.auth.validate_ik_token", mocker.AsyncMock(**ik_kwargs)),
    )


async def _accept_event(_event):
    """EventQueue.enqueue_event stand-in: accept everything."""


def _execute(executor, context):
    """Run execute() to completion against a queue that accepts every event."""
    return asyncio.run(executor.execute(context, Mock(enqueue_event=_accept_event)))


def _assert_401(executor, context, *, detail: str):
    """execute() must refuse the request with a 401 whose detail carries the given reason."""
    with pytest.raises(HTTPException) as excinfo:
        _execute(executor, context)
    assert excinfo.value.status_code == HTTP_401_UNAUTHORIZED
    assert detail in excinfo.value.detail


@pytest.mark.parametrize("its", [False, True])
def test_rejects_request_without_bearer_token(baseline_env, monkeypatch, mocker, caplog, its):
    """No Authorization header is a 401 before any introspection call, in both modes - and is
    logged, so the operator sees the rejection the client is told about.
    """
    executor, run_extraction, _sender = _executor(baseline_env, monkeypatch, mocker, its=its)
    validate_access, validate_ik = _patch_validators(mocker)

    with caplog.at_level("INFO"):
        _assert_401(executor, _context({"x-ik-token": "ik-token"}), detail="Authorization required")

    assert "no Bearer token in the Authorization header" in caplog.text
    validate_access.assert_not_awaited()
    validate_ik.assert_not_awaited()
    run_extraction.assert_not_awaited()


@pytest.mark.parametrize("its", [False, True])
def test_rejects_invalid_access_token_before_ik_token(baseline_env, monkeypatch, mocker, its):
    """An access token the IdP rejects is a 401 carrying the reason; the ik_token is never
    introspected for a request whose subject is unknown.
    """
    executor, run_extraction, _sender = _executor(baseline_env, monkeypatch, mocker, its=its)
    validate_access, validate_ik = _patch_validators(
        mocker, access=auth.TokenValidationError("access token is not active")
    )

    _assert_401(
        executor,
        _context({"authorization": "Bearer access-token", "x-ik-token": "ik-token"}),
        detail="access token is not active",
    )

    validate_access.assert_awaited_once()
    validate_ik.assert_not_awaited()
    run_extraction.assert_not_awaited()


def test_access_token_is_introspected_with_idp_settings(baseline_env, monkeypatch, mocker):
    """The external IdP leg must be driven by the INTROSPECT_* settings, token from the header."""
    executor, _run_extraction, _sender = _executor(baseline_env, monkeypatch, mocker, its=False)
    validate_access, _validate_ik = _patch_validators(mocker)

    _execute(executor, _context({"authorization": "Bearer access-token"}))

    validate_access.assert_awaited_once_with("access-token", **_IDP_INTROSPECT_KWARGS)


def test_without_its_the_external_idp_token_alone_authenticates(baseline_env, monkeypatch, mocker):
    """ITS not configured: the access token alone authenticates and its sub is the subject id;
    an X-IK-Token that still arrives (IAG in classic mode never sends one) is ignored, not
    validated - there is nothing configured to validate it against.
    """
    executor, run_extraction, _sender = _executor(baseline_env, monkeypatch, mocker, its=False)
    _validate_access, validate_ik = _patch_validators(mocker)

    _execute(executor, _context({"authorization": "Bearer access-token", "x-ik-token": "ik-token"}))

    validate_ik.assert_not_awaited()
    run_extraction.assert_awaited_once_with("rent kitt", "user-1")


def test_with_its_a_missing_ik_token_is_rejected(baseline_env, monkeypatch, mocker, caplog):
    """ITS configured: a request with no X-IK-Token is a 401 (TDD 3.1: missing either token),
    logged like every other rejection.
    """
    executor, run_extraction, _sender = _executor(baseline_env, monkeypatch, mocker, its=True)
    _validate_access, validate_ik = _patch_validators(mocker)

    with caplog.at_level("INFO"):
        _assert_401(executor, _context({"authorization": "Bearer access-token"}), detail="X-IK-Token required")

    assert "X-IK-Token header missing" in caplog.text
    validate_ik.assert_not_awaited()
    run_extraction.assert_not_awaited()


def test_with_its_an_invalid_ik_token_is_rejected(baseline_env, monkeypatch, mocker):
    """ITS configured: an ik_token ITS rejects is a 401 carrying ITS's reason, no extraction."""
    executor, run_extraction, _sender = _executor(baseline_env, monkeypatch, mocker, its=True)
    _patch_validators(mocker, ik=auth.TokenValidationError("ik_token is not active"))

    _assert_401(
        executor,
        _context({"authorization": "Bearer access-token", "x-ik-token": "ik-token"}),
        detail="ik_token is not active",
    )

    run_extraction.assert_not_awaited()


def test_with_its_unreachable_the_request_fails_closed(baseline_env, monkeypatch, mocker):
    """ITS configured but unreachable is still a 401: falling back to the access token alone
    would let anyone bypass the delegation check by taking ITS down.
    """
    executor, run_extraction, _sender = _executor(baseline_env, monkeypatch, mocker, its=True)
    _patch_validators(mocker, ik=auth.TokenValidationError("ik_token introspection request failed"))

    _assert_401(
        executor,
        _context({"authorization": "Bearer access-token", "x-ik-token": "ik-token"}),
        detail="ik_token introspection request failed",
    )

    run_extraction.assert_not_awaited()


def test_with_its_a_delegation_for_another_subject_is_rejected(baseline_env, monkeypatch, mocker):
    """ITS configured: an active ik_token minted for a different subject than the access
    token's is the TDD 4.3 "actor not permitted to act for Subject" 401.
    """
    executor, run_extraction, _sender = _executor(baseline_env, monkeypatch, mocker, its=True)
    _patch_validators(mocker, ik={**_IK_CLAIMS, "sub": "user-2"})

    _assert_401(
        executor,
        _context({"authorization": "Bearer access-token", "x-ik-token": "ik-token"}),
        detail="actor delegation not authorized",
    )

    run_extraction.assert_not_awaited()


def test_with_its_a_delegation_without_act_is_served(baseline_env, monkeypatch, mocker, caplog):
    """ITS configured: an active, subject-matching ik_token with no act chain is served - the
    executor doesn't judge the chain (IAG did) and only logs that no actor was named.
    """
    executor, run_extraction, _sender = _executor(baseline_env, monkeypatch, mocker, its=True)
    _patch_validators(mocker, ik={"active": True, "sub": "user-1"})

    with caplog.at_level("INFO"):
        _execute(executor, _context({"authorization": "Bearer access-token", "x-ik-token": "ik-token"}))

    run_extraction.assert_awaited_once_with("rent kitt", "user-1")
    assert "agent '(no act claim)' acting for subject 'user-1'" in caplog.text


def test_with_its_both_tokens_valid_serves_the_request(baseline_env, monkeypatch, mocker, caplog):
    """ITS configured, both tokens valid and matching: the ik_token is introspected with the
    ITS_* settings, the subject id still comes from the access token, the requesting agent
    (outermost act.sub) is logged, and extraction runs.
    """
    executor, run_extraction, _sender = _executor(baseline_env, monkeypatch, mocker, its=True)
    _validate_access, validate_ik = _patch_validators(mocker)

    with caplog.at_level("INFO"):
        _execute(executor, _context({"authorization": "Bearer access-token", "x-ik-token": "ik-token"}))

    validate_ik.assert_awaited_once_with(
        "ik-token",
        introspect_url=_ITS_ENV["ITS_INTROSPECT_URL"],
        client_id=_ITS_ENV["ITS_CLIENT_ID"],
        client_secret=_ITS_ENV["ITS_CLIENT_SECRET"],
        timeout_seconds=float(_ITS_ENV["ITS_INTROSPECT_TIMEOUT_SECONDS"]),
    )
    run_extraction.assert_awaited_once_with("rent kitt", "user-1")
    assert "agent 'agent-1' acting for subject 'user-1'" in caplog.text


def test_with_its_a_missing_ik_token_is_audited_as_error(baseline_env, monkeypatch, mocker):
    """A request refused for a missing X-IK-Token still leaves exactly one audit record, as
    `outcome = error` carrying the same reason the caller got (TDD 4.9, like every rejection).
    """
    executor, _run_extraction, sender = _executor(baseline_env, monkeypatch, mocker, its=True)
    _patch_validators(mocker)

    _assert_401(executor, _context({"authorization": "Bearer access-token"}), detail="X-IK-Token required")

    sender.send.assert_awaited_once()
    record = sender.send.await_args.args[0]
    assert record.outcome == Outcome.ERROR
    assert record.message == "X-IK-Token required"
    assert record.subject_id == ""
