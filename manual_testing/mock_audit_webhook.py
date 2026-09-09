"""A throwaway audit webhook receiver for AUDIT_DELIVERY=webhook: logs every record it is
sent and answers 204. Run with `uv run python manual_testing/mock_audit_webhook.py`, then
point the agent at it with AUDIT_HTTP_URL=http://127.0.0.1:9002/audit.

Pass `--fail` to answer 503 instead, to watch the agent log the failed delivery and carry on
serving (a delivery failure must never fail the request being audited). `--port N` listens on
another port (default 9002).
"""

import json
import logging
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("mock_audit_webhook")

_STATUS = HTTPStatus.SERVICE_UNAVAILABLE if "--fail" in sys.argv else HTTPStatus.NO_CONTENT
_PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 9002


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self._receive()

    def do_PUT(self) -> None:
        self._receive()

    def _receive(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        record = json.loads(self.rfile.read(length))
        api_key = self.headers.get("X-API-Key")
        logger.info("%s %s api-key=%r\n%s", self.command, self.path, api_key, json.dumps(record, indent=2))
        self.send_response(_STATUS)
        self.end_headers()


logger.info("audit webhook listening on http://127.0.0.1:%d/audit, answering %d", _PORT, _STATUS.value)
HTTPServer(("127.0.0.1", _PORT), Handler).serve_forever()
