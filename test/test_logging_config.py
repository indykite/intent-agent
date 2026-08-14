import json
import logging
import sys

from indykite.intent_agent.logging_config import JsonFormatter, build_logging_config


def _make_record(**kwargs):
    defaults = {
        "name": "test.logger",
        "level": logging.INFO,
        "pathname": __file__,
        "lineno": 1,
        "msg": "hello %s",
        "args": ("world",),
        "exc_info": None,
    }
    defaults.update(kwargs)
    return logging.LogRecord(**defaults)


def test_format_produces_json_with_expected_fields():
    payload = json.loads(JsonFormatter().format(_make_record()))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload
    assert "exception" not in payload


def _raise_value_error():
    message = "boom"
    raise ValueError(message)


def test_format_includes_exception_when_present():
    record = _make_record()
    try:
        _raise_value_error()
    except ValueError:
        record.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_format_merges_custom_extra_fields():
    record = _make_record()
    record.subject_id = "abc-123"

    payload = json.loads(JsonFormatter().format(record))
    assert payload["extra"] == {"subject_id": "abc-123"}


def test_format_drops_uvicorn_color_message():
    """`color_message` is uvicorn's own ANSI-colorized dupe of `message` - noise once rendered as JSON."""
    record = _make_record()
    record.color_message = "hello \x1b[36mworld\x1b[0m"

    payload = json.loads(JsonFormatter().format(record))
    assert "extra" not in payload


def test_format_omits_extra_key_when_no_custom_fields():
    payload = json.loads(JsonFormatter().format(_make_record()))
    assert "extra" not in payload


def test_build_logging_config_routes_everything_to_stdout():
    config = build_logging_config("INFO")

    assert config["handlers"]["default"]["stream"] == "ext://sys.stdout"
    assert config["root"]["level"] == "INFO"
    for logger_name in ("uvicorn", "uvicorn.access"):
        assert config["loggers"][logger_name]["handlers"] == ["default"]
        assert config["loggers"][logger_name]["propagate"] is False
