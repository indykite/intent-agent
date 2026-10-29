import time
from http import HTTPStatus

import httpx2
from prometheus_client import REGISTRY
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from indykite.intent_agent.__main__ import (
    MetricsMiddleware,
    _start_observability_server,
    build_observability_app,
    main,
)
from indykite.intent_agent.config import Settings
from indykite.intent_agent.logging_config import build_logging_config


def _health_checker(mocker, report=None):
    return mocker.Mock(report=mocker.Mock(return_value=report or {"status": "ok"}))


def _client(mocker):
    return TestClient(build_observability_app(_health_checker(mocker)))


async def _ping(_request):
    return PlainTextResponse("pong")


def _instrumented_client():
    app = Starlette(routes=[Route("/ping", _ping)])
    app.add_middleware(MetricsMiddleware)
    return TestClient(app)


def test_health_returns_the_checker_report(mocker):
    report = {"status": "ok", "version": "1.2.3"}
    app = build_observability_app(_health_checker(mocker, report))
    response = TestClient(app).get("/health")
    assert response.status_code == HTTPStatus.OK
    assert response.json() == report


def test_health_stays_200_when_degraded(mocker):
    """A dependency outage must not fail the liveness/readiness probes - a restart can't fix it."""
    app = build_observability_app(_health_checker(mocker, {"status": "degraded"}))
    response = TestClient(app).get("/health")
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"status": "degraded"}


def test_metrics_returns_prometheus_exposition_format(mocker):
    response = _client(mocker).get("/metrics")
    assert response.status_code == HTTPStatus.OK
    assert "text/plain" in response.headers["content-type"]
    assert "# HELP" in response.text


def test_unknown_path_returns_404(mocker):
    response = _client(mocker).get("/nope")
    assert response.status_code == HTTPStatus.NOT_FOUND


def test_metrics_middleware_records_request_count_and_latency():
    """One request adds one to the count and one latency observation; the counters are process-wide."""
    count_labels = {"method": "GET", "path": "/ping", "status": "200"}
    latency_labels = {"method": "GET", "path": "/ping"}
    count_before = REGISTRY.get_sample_value("http_requests_total", count_labels) or 0.0
    latency_before = REGISTRY.get_sample_value("http_request_duration_seconds_count", latency_labels) or 0.0

    response = _instrumented_client().get("/ping")
    assert response.status_code == HTTPStatus.OK

    assert REGISTRY.get_sample_value("http_requests_total", count_labels) == count_before + 1
    assert REGISTRY.get_sample_value("http_request_duration_seconds_count", latency_labels) == latency_before + 1


_STARTUP_TIMEOUT_SECONDS = 5.0


def _wait_until_started(server):
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started


def test_start_observability_server_binds_and_serves_health_in_background(mocker):
    server = _start_observability_server("127.0.0.1", 0, build_logging_config("INFO"), _health_checker(mocker))
    try:
        _wait_until_started(server)
        port = server.servers[0].sockets[0].getsockname()[1]

        response = httpx2.get(f"http://127.0.0.1:{port}/health")

        assert response.status_code == HTTPStatus.OK
        assert response.json() == {"status": "ok"}
    finally:
        server.should_exit = True


def test_start_observability_server_runs_on_background_daemon_thread(mocker):
    thread_cls = mocker.patch("indykite.intent_agent.__main__.threading.Thread")

    server = _start_observability_server("127.0.0.1", 0, build_logging_config("INFO"), _health_checker(mocker))

    thread_cls.assert_called_once_with(target=server.run, daemon=True)
    thread_cls.return_value.start.assert_called_once()


def test_main_routes_warnings_through_logging_before_serving(monkeypatch, mocker, baseline_env):
    """main() wires logging, captureWarnings, the app, both servers and the middleware."""
    baseline_env()
    # main()'s bare Settings() would read a developer's local .env.
    settings = Settings(_env_file=None)
    mocker.patch("indykite.intent_agent.__main__.Settings", return_value=settings)
    # A real dictConfig would reconfigure loggers process-wide and leak into other tests.
    dict_config = mocker.patch("indykite.intent_agent.__main__.logging.config.dictConfig")
    capture_warnings = mocker.patch("indykite.intent_agent.__main__.logging.captureWarnings")
    fake_agent = mocker.Mock(get_cached_schema=mocker.Mock(return_value=None))
    fake_app = mocker.Mock()
    build_app_mock = mocker.patch("indykite.intent_agent.__main__.build_app", return_value=(fake_app, fake_agent))
    health_checker = mocker.patch("indykite.intent_agent.__main__.HealthChecker").return_value
    health_checker.probe_all = mocker.AsyncMock()
    start_observability_mock = mocker.patch("indykite.intent_agent.__main__._start_observability_server")
    uvicorn_run_mock = mocker.patch("indykite.intent_agent.__main__.uvicorn.run")

    main()

    dict_config.assert_called_once_with(build_logging_config(settings.log_level))
    capture_warnings.assert_called_once_with(capture=True)
    build_app_mock.assert_called_once_with(settings)
    fake_app.add_middleware.assert_called_once_with(MetricsMiddleware)
    health_checker.probe_all.assert_awaited_once()
    start_observability_mock.assert_called_once_with(
        str(settings.bind_host), settings.health_port, mocker.ANY, health_checker
    )
    uvicorn_run_mock.assert_called_once_with(
        fake_app,
        host=str(settings.bind_host),
        port=settings.intent_port,
        log_level=settings.log_level.lower(),
        log_config=mocker.ANY,
    )


def _run_main_with_mocks(mocker, settings):
    mocker.patch("indykite.intent_agent.__main__.Settings", return_value=settings)
    mocker.patch("indykite.intent_agent.__main__.logging.config.dictConfig")
    mocker.patch("indykite.intent_agent.__main__.logging.captureWarnings")
    fake_agent = mocker.Mock(get_cached_schema=mocker.Mock(return_value=None))
    mocker.patch("indykite.intent_agent.__main__.build_app", return_value=(mocker.Mock(), fake_agent))
    mocker.patch("indykite.intent_agent.__main__._start_observability_server")
    mocker.patch("indykite.intent_agent.__main__.uvicorn.run")
    main()


def test_main_warns_when_advertised_url_is_unset(monkeypatch, mocker, caplog, baseline_env):
    baseline_env()
    monkeypatch.delenv("ADVERTISED_URL", raising=False)

    _run_main_with_mocks(mocker, Settings(_env_file=None))

    assert "ADVERTISED_URL is not set" in caplog.text


def test_main_does_not_warn_when_advertised_url_is_set(monkeypatch, mocker, caplog, baseline_env):
    baseline_env()
    monkeypatch.setenv("ADVERTISED_URL", "https://intent-agent.example.com")

    _run_main_with_mocks(mocker, Settings(_env_file=None))

    assert "ADVERTISED_URL is not set" not in caplog.text
