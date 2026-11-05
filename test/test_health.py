import asyncio

import httpx2
import pytest

from indykite.intent_agent.config import Settings
from indykite.intent_agent.dependency_status import (
    AUDIT_FILE,
    AUDIT_WEBHOOK,
    GEMINI,
    IKG,
    INTROSPECTION,
    REDIS,
    SCHEMA,
    DependencyStatus,
)
from indykite.intent_agent.health import DISABLED, HealthChecker
from indykite.intent_agent.version import __version__


@pytest.fixture
def settings(baseline_env):
    baseline_env()
    return Settings(_env_file=None)


_NOT_RFC_7662 = "response is not an RFC 7662 introspection answer"


def _route(mocker, handler):
    """Make every AsyncClient the startup probes build talk to `handler` instead of the network."""
    real = httpx2.AsyncClient
    mocker.patch(
        "indykite.intent_agent.health.httpx2.AsyncClient",
        side_effect=lambda **kwargs: real(transport=httpx2.MockTransport(handler), **kwargs),
    )


def _all_ok(_request):
    return httpx2.Response(200, json={"active": False})


def _seed_all_ok(status, *, extra=(AUDIT_FILE,)):
    for name in (IKG, INTROSPECTION, GEMINI, *extra):
        status.record_ok(name)
    status.record_ok(SCHEMA, version="abc")


def _enable_webhook_and_redis(settings):
    settings.audit_delivery = "webhook"
    settings.audit_http_url = "https://audit.example.com/records"
    settings.audit_http_method = "POST"
    settings.audit_http_auth_type = "no-auth"
    settings.cache_backend = "redis"
    settings.redis_url = "redis://redis:6379"


def test_report_is_unknown_until_something_is_recorded(settings):
    report = HealthChecker(settings, DependencyStatus()).report()

    assert report["status"] == "degraded"
    assert report["schema"] == {"status": "unknown"}
    assert report["checks"]["gemini"] == {"status": "unknown"}


def test_report_reflects_recorded_state_without_any_io(settings):
    status = DependencyStatus()
    _seed_all_ok(status)

    report = HealthChecker(settings, status).report()

    assert report["status"] == "ok"
    assert report["version"] == __version__
    assert report["schema"]["version"] == "abc"
    assert {name: check["status"] for name, check in report["checks"].items()} == {
        AUDIT_FILE: "ok",
        AUDIT_WEBHOOK: DISABLED,
        GEMINI: "ok",
        IKG: "ok",
        REDIS: DISABLED,
        INTROSPECTION: "ok",
    }
    assert report["checks"][GEMINI]["source"] == "traffic"
    assert "checked_at" in report["checks"][GEMINI]


def test_latest_call_wins(settings):
    status = DependencyStatus()
    _seed_all_ok(status)
    status.record_error(GEMINI, "status 429")
    checker = HealthChecker(settings, status)

    assert checker.report()["checks"][GEMINI]["detail"] == "status 429"
    assert checker.report()["status"] == "degraded"

    status.record_ok(GEMINI)

    assert checker.report()["status"] == "ok"
    assert "detail" not in checker.report()["checks"][GEMINI]


def test_schema_error_is_degraded(settings):
    status = DependencyStatus()
    _seed_all_ok(status)
    status.record_error(SCHEMA, "no schema cached - the next request retries the fetch")

    report = HealthChecker(settings, status).report()

    assert report["status"] == "degraded"
    assert report["schema"]["status"] == "error"


def test_probe_all_seeds_startup_state(mocker, settings):
    _route(mocker, _all_ok)
    status = DependencyStatus()

    asyncio.run(HealthChecker(settings, status).probe_all())

    for name in (IKG, INTROSPECTION, GEMINI):
        state = status.get(name)
        assert (state.status, state.source) == ("ok", "startup")
    assert status.get(AUDIT_WEBHOOK) is None
    assert status.get(REDIS) is None


@pytest.mark.parametrize(
    ("host", "code", "detail"),
    [
        pytest.param("generativelanguage.googleapis.com", 400, "API key rejected (400)", id="gemini-bad-key"),
        pytest.param(
            "generativelanguage.googleapis.com",
            404,
            "model 'gemini-3.5-flash-lite' not found",
            id="gemini-unknown-model",
        ),
        pytest.param("ikg.example.com", 403, "app agent key rejected (403)", id="ikg-bad-key"),
        pytest.param("idp.example.com", 401, "client credentials rejected (401)", id="idp-bad-credentials"),
        pytest.param("idp.example.com", 503, "unexpected status 503", id="idp-unavailable"),
    ],
)
def test_probe_all_names_the_failing_dependency(mocker, settings, host, code, detail):
    name = {
        "generativelanguage.googleapis.com": GEMINI,
        "ikg.example.com": IKG,
        "idp.example.com": INTROSPECTION,
    }[host]
    _route(mocker, lambda request: httpx2.Response(code if request.url.host == host else 200, json={"active": False}))
    status = DependencyStatus()

    asyncio.run(HealthChecker(settings, status).probe_all())

    assert status.get(name).detail == detail
    assert [n for n in (IKG, INTROSPECTION, GEMINI) if status.get(n).status == "error"] == [name]


@pytest.mark.parametrize(
    ("response", "detail"),
    [
        pytest.param(httpx2.Response(200, text="<html>login</html>"), "response is not JSON", id="html"),
        pytest.param(httpx2.Response(200, json=["active"]), _NOT_RFC_7662, id="not-object"),
        pytest.param(httpx2.Response(200, json={"active": "no"}), _NOT_RFC_7662, id="active-not-bool"),
        pytest.param(httpx2.Response(200, json={}), _NOT_RFC_7662, id="no-active"),
    ],
)
def test_introspection_probe_needs_a_real_rfc_7662_answer(mocker, settings, response, detail):
    _route(
        mocker,
        lambda request: (
            response if request.url.host == "idp.example.com" else httpx2.Response(200, json={"active": False})
        ),
    )
    status = DependencyStatus()

    asyncio.run(HealthChecker(settings, status).probe_all())

    assert status.get(INTROSPECTION).detail == detail


def test_probe_all_records_exception_class_only(mocker, settings):
    def handler(_request):
        message = "connect failed for secret-url"
        raise httpx2.ConnectError(message)

    _route(mocker, handler)
    status = DependencyStatus()

    asyncio.run(HealthChecker(settings, status).probe_all())

    assert status.get(GEMINI).detail == "ConnectError"


def test_probe_all_times_out_a_slow_dependency(mocker, settings):
    settings.health_check_timeout_seconds = 0.01

    async def slow(_request):
        await asyncio.sleep(1)

    status = DependencyStatus()
    checker = HealthChecker(settings, status)

    asyncio.run(checker._run_probe(GEMINI, lambda: slow(None)))

    assert (status.get(GEMINI).status, status.get(GEMINI).detail) == ("error", "timed out")


def test_probe_all_skips_a_dependency_already_recorded(mocker, settings):
    probed_hosts = []

    def handler(request):
        probed_hosts.append(request.url.host)
        return httpx2.Response(500)

    _route(mocker, handler)
    status = DependencyStatus()
    status.record_ok(GEMINI)

    asyncio.run(HealthChecker(settings, status).probe_all())

    assert "generativelanguage.googleapis.com" not in probed_hosts
    assert (status.get(GEMINI).status, status.get(GEMINI).source) == ("ok", "traffic")
    assert status.get(IKG).status == "error"


@pytest.mark.parametrize(
    ("subdir", "expected"),
    [
        pytest.param("", ("ok", ""), id="writable"),
        pytest.param("missing", ("error", "audit directory is not writable"), id="missing"),
    ],
)
def test_audit_file_probe_checks_the_directory_is_writable(mocker, settings, tmp_path, subdir, expected):
    _route(mocker, _all_ok)
    settings.audit_file_storage_path = tmp_path / subdir
    status = DependencyStatus()

    asyncio.run(HealthChecker(settings, status).probe_all())

    audit = status.get(AUDIT_FILE)
    assert (audit.status, audit.detail, audit.source) == (*expected, "startup")


@pytest.mark.parametrize(
    ("probe_status", "ping_error", "expected"),
    [
        pytest.param(405, None, {AUDIT_WEBHOOK: ("ok", ""), REDIS: ("ok", "")}, id="reachable"),
        pytest.param(
            502,
            ConnectionError,
            {AUDIT_WEBHOOK: ("error", "unexpected status 502"), REDIS: ("error", "ConnectionError")},
            id="down",
        ),
    ],
)
def test_webhook_and_redis_are_probed_when_configured(mocker, settings, probe_status, ping_error, expected):
    _enable_webhook_and_redis(settings)
    _route(mocker, _all_ok)
    probe = mocker.patch("indykite.intent_agent.health.audit_http.probe", mocker.AsyncMock(return_value=probe_status))
    redis = mocker.patch("indykite.intent_agent.health.Redis.from_url")
    redis.return_value.ping = mocker.AsyncMock(side_effect=ping_error)
    redis.return_value.aclose = mocker.AsyncMock()
    status = DependencyStatus()

    asyncio.run(HealthChecker(settings, status).probe_all())

    probe.assert_awaited_once()
    redis.return_value.aclose.assert_awaited_once()
    assert {name: (status.get(name).status, status.get(name).detail) for name in expected} == expected
