import httpx2

from indykite.intent_agent.dependency_status import GEMINI, DependencyStatus, describe_failure


def test_describe_failure_names_the_status_or_class_only():
    request = httpx2.Request("GET", "https://secret.example.com/?key=hunter2")
    status_error = httpx2.HTTPStatusError("boom", request=request, response=httpx2.Response(503, request=request))

    assert describe_failure(status_error) == "status 503"
    assert describe_failure(httpx2.ConnectError("secret-url")) == "ConnectError"


def test_latest_record_wins_and_unrecorded_is_none():
    status = DependencyStatus()
    assert status.get(GEMINI) is None

    status.record_error(GEMINI, "status 429")
    status.record_ok(GEMINI, source="startup")

    state = status.get(GEMINI)
    assert (state.status, state.source, state.detail) == ("ok", "startup", "")


def test_state_dict_omits_empty_detail_and_version():
    status = DependencyStatus()
    status.record_ok(GEMINI)

    assert set(status.get(GEMINI).to_dict()) == {"status", "checked_at", "source"}
