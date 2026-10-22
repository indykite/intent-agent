"""Structured one-line JSON logging to stdout for both app and uvicorn logs.

Pass `build_logging_config`'s result to `dictConfig` and to every uvicorn config/run call.
"""

import json
import logging
from typing import Any

# Derived from a real LogRecord so it tracks the running Python's attrs; `color_message` is
# uvicorn's ANSI-colored duplicate of `message`.
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
