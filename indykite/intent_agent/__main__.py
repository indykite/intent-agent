"""Entrypoint: `python -m indykite.intent_agent`.

Starts the A2A JSON-RPC app on `INTENT_PORT` and a minimal health endpoint on
`HEALTH_PORT` (matching this repo's Dockerfile HEALTHCHECK) in a background thread.
"""

import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer

import uvicorn

from indykite.intent_agent.agent import build_app
from indykite.intent_agent.config import Settings

logger = logging.getLogger(__name__)


class _HealthRequestHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler serving GET /health for the Dockerfile's HEALTHCHECK."""

    def do_GET(self) -> None:
        """Respond 200 to /health, 404 to anything else."""
        if self.path == "/health":
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

    def log_message(self, format_str: str, *args: object) -> None:
        """Silence per-request access logs."""


def _start_health_server(bind_host: str, port: int) -> HTTPServer:
    """Start the health server on a background daemon thread and return it."""
    server = HTTPServer((bind_host, port), _HealthRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    """Load settings, build the app, start the health server, then serve the A2A app."""
    settings = Settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app, agent = build_app(settings)

    _start_health_server(settings.bind_host, settings.health_port)
    logger.info(
        "Starting %s on port %d (health on %d, LLM: Gemini %s)",
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
