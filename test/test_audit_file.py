import asyncio
import csv
import json
import os
import stat

import pytest

from indykite.intent_agent.audit.file import FileConfig, FileSender
from indykite.intent_agent.audit.message import FIELD_NAMES, Outcome

_ONE_DAY_SECONDS = 24 * 60 * 60
_EXPECTED_ROTATED_FILES = 2
_EXPECTED_ROWS_WITH_HEADER = 3
_OWNER_ONLY_FILE_MODE = 0o600
_OWNER_RW_GROUP_R_DIR_MODE = 0o750
_PREEXISTING_DIR_MODE = 0o700


def _sender(tmp_path, **overrides):
    """A FileSender into tmp_path with time rotation that never fires unless overridden."""
    options = {
        "storage_path": tmp_path / "audit",
        "file_format": "json",
        "rotation_strategy": "time",
        "rotation_interval_seconds": float(_ONE_DAY_SECONDS),
        "rotation_max_bytes": 100 * 1024 * 1024,
    }
    options.update(overrides)
    return FileSender(FileConfig(**options), service="intent_agent")


def _send_all(sender, *messages):
    """Deliver messages in order, then close the sender."""

    async def _run():
        """Send then close, on one event loop."""
        for message in messages:
            await sender.send(message)
        await sender.close()

    asyncio.run(_run())


def _files(tmp_path, suffix):
    """Every audit file written, oldest first (the unixNano in the name sorts chronologically)."""
    return sorted((tmp_path / "audit").glob(f"audit-intent_agent-*.{suffix}"))


def test_creates_storage_path_and_names_files_like_the_iag(tmp_path, audit_message):
    """`audit-<service>-<unixNano>.<format>` in a directory created on demand, file mode 0600."""
    sender = _sender(tmp_path)
    _send_all(sender, audit_message())

    assert (tmp_path / "audit").is_dir()
    files = _files(tmp_path, "json")
    assert len(files) == 1
    assert stat.S_IMODE(files[0].stat().st_mode) == _OWNER_ONLY_FILE_MODE


def test_modes_are_pinned_regardless_of_umask(tmp_path, audit_message):
    """Even under a umask without owner write, the record lands and the modes end up 0750 / 0600."""
    previous_umask = os.umask(0o277)
    try:
        _send_all(_sender(tmp_path), audit_message())
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE((tmp_path / "audit").stat().st_mode) == _OWNER_RW_GROUP_R_DIR_MODE
    files = _files(tmp_path, "json")
    assert len(files) == 1
    assert stat.S_IMODE(files[0].stat().st_mode) == _OWNER_ONLY_FILE_MODE
    assert json.loads(files[0].read_text().splitlines()[0])["request_id"] == "req-1"


def test_existing_storage_directory_keeps_its_mode(tmp_path, audit_message):
    """Only a directory the sender creates gets the pinned mode; an operator's one is left as-is."""
    storage = tmp_path / "audit"
    storage.mkdir(mode=0o700)
    storage.chmod(0o700)

    _send_all(_sender(tmp_path), audit_message())

    assert stat.S_IMODE(storage.stat().st_mode) == _PREEXISTING_DIR_MODE
    assert len(_files(tmp_path, "json")) == 1


def test_json_format_writes_one_record_per_line(tmp_path, audit_message):
    """JSON lines: each record is one parseable object, evaluations kept structured."""
    _send_all(_sender(tmp_path), audit_message(), audit_message(request_id="req-2", outcome=Outcome.UNMATCHED))

    lines = _files(tmp_path, "json")[0].read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["request_id"] for line in lines] == ["req-1", "req-2"]
    assert json.loads(lines[1])["outcome"] == "unmatched"
    assert json.loads(lines[0])["evaluations"][0]["resource"] == {"type": "Car", "id": "kitt"}


def test_csv_format_writes_header_then_rows(tmp_path, audit_message):
    """CSV: the field names as a header row once per file, then one row per record."""
    _send_all(_sender(tmp_path, file_format="csv"), audit_message(), audit_message(request_id="req-2"))

    with _files(tmp_path, "csv")[0].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == _EXPECTED_ROWS_WITH_HEADER
    assert rows[0] == list(FIELD_NAMES)
    assert rows[1][FIELD_NAMES.index("request_id")] == "req-1"
    assert rows[2][FIELD_NAMES.index("request_id")] == "req-2"
    assert json.loads(rows[1][FIELD_NAMES.index("evaluations")]) == audit_message().evaluations


def test_txt_format_writes_bracketed_timestamp_and_key_value_pairs(tmp_path, audit_message):
    """TXT: `[timestamp] field=value ...`, one record per line."""
    _send_all(_sender(tmp_path, file_format="txt"), audit_message(message="no bearer token"))

    line = _files(tmp_path, "txt")[0].read_text(encoding="utf-8")
    assert line.startswith("[2026-09-07T10:15:04.123456Z] request_id=req-1 subject_id=user-1 ")
    assert " outcome=success message=no bearer token\n" in line
    assert " evaluations=[{" in line


def test_rotates_by_size(tmp_path, audit_message):
    """`size`: once the current file holds at least rotation_max_bytes, the next record opens a new one."""
    sender = _sender(tmp_path, rotation_strategy="size", rotation_max_bytes=10)
    _send_all(sender, audit_message(), audit_message(request_id="req-2"))

    files = _files(tmp_path, "json")
    assert len(files) == _EXPECTED_ROTATED_FILES
    assert json.loads(files[0].read_text(encoding="utf-8"))["request_id"] == "req-1"
    assert json.loads(files[1].read_text(encoding="utf-8"))["request_id"] == "req-2"


def test_rotates_by_time(tmp_path, audit_message):
    """`time`: with a zero interval each record lands in a fresh file (the initial one stays empty)."""
    sender = _sender(tmp_path, rotation_interval_seconds=0.0)
    _send_all(sender, audit_message(), audit_message(request_id="req-2"))

    written = [path for path in _files(tmp_path, "json") if path.stat().st_size > 0]
    assert len(written) == _EXPECTED_ROTATED_FILES
    assert [json.loads(path.read_text(encoding="utf-8"))["request_id"] for path in written] == ["req-1", "req-2"]


def test_size_and_time_rotates_on_either(tmp_path, audit_message):
    """`size-and-time`: size alone is enough to rotate (time never elapses here)."""
    sender = _sender(tmp_path, rotation_strategy="size-and-time", rotation_max_bytes=10)
    _send_all(sender, audit_message(), audit_message(request_id="req-2"))

    assert len(_files(tmp_path, "json")) == _EXPECTED_ROTATED_FILES


def test_does_not_rotate_below_thresholds(tmp_path, audit_message):
    """Two small records well inside both limits stay in one file."""
    sender = _sender(tmp_path, rotation_strategy="size-and-time")
    _send_all(sender, audit_message(), audit_message(request_id="req-2"))

    assert len(_files(tmp_path, "json")) == 1


def test_write_failure_is_logged_not_raised(tmp_path, caplog, mocker, audit_message):
    """A disk error is logged and never reaches the request path."""
    caplog.set_level("ERROR")
    sender = _sender(tmp_path)
    mocker.patch.object(FileSender, "_append", side_effect=OSError("disk full"))

    _send_all(sender, audit_message())

    assert "audit: write record" in caplog.text


def test_send_after_close_is_dropped_with_a_warning(tmp_path, caplog, audit_message):
    """Once closed there's no file to append to - the record is dropped, loudly, not crashed on."""
    caplog.set_level("WARNING")
    sender = _sender(tmp_path)
    _send_all(sender)

    asyncio.run(sender.send(audit_message()))

    assert "audit: dropping payload, file sender is closed" in caplog.text
    assert _files(tmp_path, "json")[0].read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("file_format", ["json", "csv", "txt"])
def test_current_path_uses_the_configured_extension(tmp_path, file_format):
    """The file extension is the configured format, so a reader can tell them apart."""
    sender = _sender(tmp_path, file_format=file_format)
    assert sender.current_path is not None
    assert sender.current_path.suffix == f".{file_format}"
    _send_all(sender)
    assert sender.current_path is None


def test_failed_rotation_keeps_the_current_file(tmp_path, caplog, mocker, audit_message):
    """If the successor file can't be opened, the current one stays in service and the record is written."""
    caplog.set_level("ERROR")
    sender = _sender(tmp_path, rotation_strategy="size", rotation_max_bytes=10)
    asyncio.run(sender.send(audit_message()))
    mocker.patch.object(FileSender, "_open_file", side_effect=[OSError("disk full"), mocker.DEFAULT])

    _send_all(sender, audit_message(request_id="req-2"))

    assert "audit: rotate file" in caplog.text
    assert "file sender is closed" not in caplog.text
    files = _files(tmp_path, "json")
    assert len(files) == 1
    assert [json.loads(line)["request_id"] for line in files[0].read_text(encoding="utf-8").splitlines()] == [
        "req-1",
        "req-2",
    ]
