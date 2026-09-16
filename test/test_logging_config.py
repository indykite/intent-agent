import json
import logging
import logging.config
import sys
import warnings

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


def test_captured_warnings_reach_the_json_stdout_handler(capsys):
    """logging.captureWarnings(capture=True) (see __main__.main()) must route warnings.warn()
    through the same JSON stdout handler every other log line uses, at WARNING severity - not
    just that captureWarnings gets called (main()'s own test mocks it out), but that a real
    warning actually becomes one JSON line on stdout. dictConfig mutates the root/uvicorn
    loggers process-wide, and emitting the captured warning creates/touches the "py.warnings"
    logger itself, so this snapshots and restores all of them afterward instead of leaking
    into every other test in the suite.
    """
    affected_loggers = [
        logging.getLogger(name) for name in (None, "uvicorn", "uvicorn.access", "uvicorn.error", "py.warnings")
    ]
    snapshot = [(logger, list(logger.handlers), logger.level, logger.propagate) for logger in affected_loggers]
    try:
        logging.config.dictConfig(build_logging_config("INFO"))
        logging.captureWarnings(capture=True)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("test warning message", UserWarning, stacklevel=2)
    finally:
        logging.captureWarnings(capture=False)
        for logger, handlers, level, propagate in snapshot:
            logger.handlers = handlers
            logger.level = level
            logger.propagate = propagate

    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "py.warnings"
    assert "test warning message" in payload["message"]
