"""A throwaway RFC 7662 introspection stub for running the agent locally without a real IdP.

MOCK_INTROSPECT_MODE picks what every token is answered with, so each class of token failure
(auth.TokenFailure) can be tried by hand:

- ``active`` (default): ``{"active": true, ...}`` - the token is accepted.
- ``inactive``: ``{"active": false}`` - the caller's token is refused (401).
- ``broken``: a 503 instead of an answer - the IdP's fault (502). Stopping the stub altogether
  is the remaining case: no answer at all, this service's fault (500).
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

_MODE = os.environ.get("MOCK_INTROSPECT_MODE", "active")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if _MODE == "broken":
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if _MODE == "inactive":
            payload: dict[str, object] = {"active": False}
        else:
            payload = {
                "active": True,
                "sub": "rebecca",
                "act": {"sub": "rebecca"},
                "aud": "rebecca",
            }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HTTPServer(("127.0.0.1", 9001), Handler).serve_forever()
