"""Entrypoint: `python -m indykite.intent_agent`.

Starts the A2A JSON-RPC app on `INTENT_PORT` and an observability app (`/health` +
`/metrics`, matching this repo's Dockerfile HEALTHCHECK) on `HEALTH_PORT` in a
background thread.
"""

import logging
import threading
import time

import uvicorn
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from indykite.intent_agent.agent import build_app
from indykite.intent_agent.config import Settings

logger = logging.getLogger(__name__)

_UVICORN_REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests served by the intent agent app",
    ["method", "path", "status"],
)
_UVICORN_REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency for the intent agent app",
    ["method", "path"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record per-request count and latency for the main uvicorn app."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Time the request and record it under the route's path (not raw URL, to avoid label cardinality blowup)."""
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        path = request.scope.get("route").path if request.scope.get("route") else request.url.path
        _UVICORN_REQUEST_LATENCY.labels(request.method, path).observe(duration)
        _UVICORN_REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
        return response


async def _health(_request: Request) -> PlainTextResponse:
    """Respond 200 to GET /health for the Dockerfile's HEALTHCHECK."""
    return PlainTextResponse("ok")


async def _metrics(_request: Request) -> Response:
    """Respond with the process's metrics in Prometheus text exposition format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def build_observability_app() -> Starlette:
    """Build the Starlette app serving `/health` and `/metrics`."""
    return Starlette(routes=[Route("/health", _health), Route("/metrics", _metrics)])


def _start_observability_server(bind_host: str, port: int) -> uvicorn.Server:
    """Start the observability app on a background daemon thread and return the server."""
    config = uvicorn.Config(build_observability_app(), host=bind_host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server


def main() -> None:
    """Load settings, build the app, start the observability server, then serve the A2A app."""
    settings = Settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app, agent = build_app(settings)
    app.add_middleware(MetricsMiddleware)

    _start_observability_server(settings.bind_host, settings.health_port)
    logger.info(
        "Starting %s on port %d (observability on %d, LLM: Gemini %s)",
        settings.intent_agent_name,
        settings.intent_port,
        settings.health_port,
        settings.gemini_model,
    )
    logger.info("Data schema: %s", "loaded" if agent.get_cached_schema() else "unavailable")

    uvicorn.run(
        app,
        host=settings.bind_host,
        port=settings.intent_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
