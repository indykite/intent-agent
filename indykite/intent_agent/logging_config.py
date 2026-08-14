"""Structured JSON logging, applied uniformly to app logs and uvicorn's own logs.

`build_logging_config` returns one dictConfig-compatible mapping that both `main()`
(for the app's own root logger) and every `uvicorn.Config`/`uvicorn.run` call (for
"uvicorn"/"uvicorn.error"/"uvicorn.access") must be given, so a line logged from either
source is the same one-line-JSON-to-stdout shape.
"""

import json
import logging
from typing import Any

# Built from a throwaway LogRecord rather than hardcoded, so it tracks whichever attrs
# the running Python version's LogRecord actually carries (e.g. `taskName`, added in 3.12).
# `color_message` is uvicorn's own extra, carrying an ANSI-colorized dupe of `message` -
# meaningless once rendered as JSON, so it's dropped alongside the stdlib attrs.
_STANDARD_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "color_message",
}


class JsonFormatter(logging.Formatter):
    """Render each log record as one line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Build the JSON payload for one log record, including exception info and extras."""
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        extra = {key: value for key, value in record.__dict__.items() if key not in _STANDARD_RECORD_ATTRS}
        if extra:
            payload["extra"] = extra
        return json.dumps(payload, default=str, ensure_ascii=False)


def build_logging_config(log_level: str) -> dict[str, Any]:
    """Build a dictConfig mapping that writes JSON-formatted log lines to stdout only."""
    handler = {
        "class": "logging.StreamHandler",
        "formatter": "json",
        "stream": "ext://sys.stdout",
    }
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"json": {"()": f"{__name__}.JsonFormatter"}},
        "handlers": {"default": handler},
        "root": {"handlers": ["default"], "level": log_level},
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": log_level, "propagate": False},
            "uvicorn.error": {"level": log_level},
            "uvicorn.access": {"handlers": ["default"], "level": log_level, "propagate": False},
        },
    }
