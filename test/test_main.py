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


def _client():
    return TestClient(build_observability_app())


async def _ping(_request):
    return PlainTextResponse("pong")


def _instrumented_client():
    app = Starlette(routes=[Route("/ping", _ping)])
    app.add_middleware(MetricsMiddleware)
    return TestClient(app)


def test_health_returns_ok():
    response = _client().get("/health")
    assert response.status_code == HTTPStatus.OK
    assert response.text == "ok"


def test_metrics_returns_prometheus_exposition_format():
    response = _client().get("/metrics")
    assert response.status_code == HTTPStatus.OK
    assert "text/plain" in response.headers["content-type"]
    assert "# HELP" in response.text


def test_unknown_path_returns_404():
    response = _client().get("/nope")
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


def test_start_observability_server_binds_and_serves_health_in_background():
    server = _start_observability_server("127.0.0.1", 0, build_logging_config("INFO"))
    try:
        _wait_until_started(server)
        port = server.servers[0].sockets[0].getsockname()[1]

        response = httpx2.get(f"http://127.0.0.1:{port}/health")

        assert response.status_code == HTTPStatus.OK
        assert response.text == "ok"
    finally:
        server.should_exit = True


def test_start_observability_server_runs_on_background_daemon_thread(mocker):
    thread_cls = mocker.patch("indykite.intent_agent.__main__.threading.Thread")

    server = _start_observability_server("127.0.0.1", 0, build_logging_config("INFO"))

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
    start_observability_mock = mocker.patch("indykite.intent_agent.__main__._start_observability_server")
    uvicorn_run_mock = mocker.patch("indykite.intent_agent.__main__.uvicorn.run")

    main()

    dict_config.assert_called_once_with(build_logging_config(settings.log_level))
    capture_warnings.assert_called_once_with(capture=True)
    build_app_mock.assert_called_once_with(settings)
    fake_app.add_middleware.assert_called_once_with(MetricsMiddleware)
    start_observability_mock.assert_called_once()
    uvicorn_run_mock.assert_called_once_with(
        fake_app,
        host=str(settings.bind_host),
        port=settings.intent_port,
        log_level=settings.log_level.lower(),
        log_config=mocker.ANY,
    )
