"""Entrypoint: `python -m indykite.intent_agent`.

Starts the A2A JSON-RPC app on `INTENT_PORT` and a minimal health endpoint on
`HEALTH_PORT` (matching this repo's Dockerfile HEALTHCHECK) in a background thread.
"""

import logging
import threading
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
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format_str: str, *args: object) -> None:  # noqa: ARG002 - silence per-request access logs
        """Silence per-request access logs."""
        return


def _start_health_server(port: int) -> HTTPServer:
    """Start the health server on a background daemon thread and return it."""
    server = HTTPServer(("0.0.0.0", port), _HealthRequestHandler)  # noqa: S104 - health check must be reachable in-container
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

    _start_health_server(settings.health_port)
    logger.info(
        "Starting %s on port %d (health on %d, LLM: Gemini %s)",
        settings.intent_agent_name,
        settings.intent_port,
        settings.health_port,
        settings.gemini_model,
    )
    logger.info(
        "Data schema: %s", "loaded" if agent.get_cached_schema() else "unavailable"
    )

    uvicorn.run(app, host="0.0.0.0", port=settings.intent_port, log_level="info")  # noqa: S104 - must be reachable in-container


if __name__ == "__main__":
    main()
