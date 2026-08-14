import time
from http import HTTPStatus

import httpx2
from prometheus_client import generate_latest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from indykite.intent_agent.__main__ import MetricsMiddleware, _start_observability_server, build_observability_app
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
    response = _instrumented_client().get("/ping")
    assert response.status_code == HTTPStatus.OK

    exposition = generate_latest().decode()
    assert 'http_requests_total{method="GET",path="/ping",status="200"} 1.0' in exposition
    assert 'http_request_duration_seconds_count{method="GET",path="/ping"} 1.0' in exposition


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
