"""The last known state of every outside dependency, as seen by the calls this process really made.

`/health` reads this instead of probing: each component records the outcome of its own real calls
(schema/policy fetch -> `ikg`, introspection, LLM invocation -> `gemini`, audit delivery, cache
access -> `redis`), and the startup probes in `health.py` seed it once so a dependency that has
seen no traffic yet still has a state. Failure details name a status code or exception class
only - never a response body or a URL that could carry a credential.
"""

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import httpx2

OK = "ok"
ERROR = "error"
UNKNOWN = "unknown"

Source = Literal["startup", "traffic"]

IKG = "ikg"
INTROSPECTION = "token_introspection"
GEMINI = "gemini"
AUDIT_WEBHOOK = "audit_webhook"
AUDIT_FILE = "audit_file"
REDIS = "redis"
SCHEMA = "schema"


@dataclass(frozen=True, slots=True)
class DependencyState:
    """One dependency's latest outcome."""

    status: str
    checked_at: str
    source: Source
    detail: str = ""
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """The state as it appears in the `/health` body (empty detail/version left out)."""
        body: dict[str, Any] = {"status": self.status, "checked_at": self.checked_at, "source": self.source}
        if self.detail:
            body["detail"] = self.detail
        if self.version:
            body["version"] = self.version
        return body


def describe_failure(exc: BaseException) -> str:
    """A secret-free summary of why a call failed: the HTTP status if there was one, else the exception class."""
    if isinstance(exc, httpx2.HTTPStatusError):
        return f"status {exc.response.status_code}"
    return type(exc).__name__


class DependencyStatus:
    """Thread-safe latest-outcome store - recorded from worker threads (fetches) and the event loop alike."""

    def __init__(self) -> None:
        """Start empty: every dependency is `unknown` until something records it."""
        self._lock = threading.Lock()
        self._states: dict[str, DependencyState] = {}

    def record_ok(self, name: str, *, source: Source = "traffic", detail: str = "", version: str = "") -> None:
        """A call to `name` just succeeded."""
        self._record(name, OK, source, detail, version)

    def record_error(self, name: str, detail: str, *, source: Source = "traffic", version: str = "") -> None:
        """A call to `name` just failed, for the reason in `detail`."""
        self._record(name, ERROR, source, detail, version)

    def get(self, name: str) -> DependencyState | None:
        """The latest outcome recorded for `name`, if any."""
        with self._lock:
            return self._states.get(name)

    def _record(self, name: str, status: str, source: Source, detail: str, version: str) -> None:
        state = DependencyState(status, datetime.now(UTC).isoformat(timespec="seconds"), source, detail, version)
        with self._lock:
            self._states[name] = state
