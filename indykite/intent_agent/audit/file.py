"""File audit delivery: rotating `audit-<service>-<unixNano>.<format>` files (json, csv or txt).

Write failures are logged and never raised into the request path.
"""

import asyncio
import csv
import io
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from indykite.intent_agent.audit.message import FIELD_NAMES, AuditMessage
from indykite.intent_agent.dependency_status import AUDIT_FILE, DependencyStatus, describe_failure

logger = logging.getLogger(__name__)

FileFormat = Literal["csv", "json", "txt"]
RotationStrategy = Literal["time", "size", "size-and-time"]

_DIR_MODE = 0o750
_FILE_MODE = 0o600


@dataclass(frozen=True, slots=True)
class FileConfig:
    """File delivery settings."""

    storage_path: Path
    file_format: FileFormat
    rotation_strategy: RotationStrategy
    rotation_interval_seconds: float
    rotation_max_bytes: int


class FileSender:
    """Appends audit records to a rotating file. Safe to call from multiple threads."""

    def __init__(self, config: FileConfig, *, service: str, dependency_status: DependencyStatus | None = None) -> None:
        """Create the storage directory if missing and open the first audit file.

        dependency_status (when given) is told how every write went - what /health reports.
        """
        self._config = config
        self._service = service
        self._status = dependency_status
        self._lock = threading.Lock()
        self._file: io.TextIOWrapper | None = None
        self._opened_at = 0.0
        self._bytes_written = 0
        self._csv_header_written = False
        # mkdir's mode is masked by the umask; pin it only on a directory this sender creates.
        created = not config.storage_path.is_dir()
        config.storage_path.mkdir(mode=_DIR_MODE, parents=True, exist_ok=True)
        if created:
            config.storage_path.chmod(_DIR_MODE)
        self._open_new_file()

    @property
    def current_path(self) -> Path | None:
        """Path of the file records are currently appended to (None once closed)."""
        return Path(self._file.name) if self._file is not None else None

    async def send(self, message: AuditMessage) -> None:
        """Append one record, off the event loop thread so disk latency never stalls a request."""
        await asyncio.to_thread(self._write, message)

    async def close(self) -> None:
        """Flush and close the current file."""
        with self._lock:
            self._close_file()

    def _write(self, message: AuditMessage) -> None:
        """Rotate if due, then append - under the lock, logging (never raising) on disk errors."""
        with self._lock:
            if self._file is None:
                logger.warning("audit: dropping payload, file sender is closed")
                return
            try:
                self._rotate_if_needed()
            except OSError:
                # The current file is still open, so the record is appended to it rather than lost.
                logger.exception("audit: rotate file")
            try:
                self._bytes_written += self._append(message)
            except OSError as exc:
                logger.exception("audit: write record")
                if self._status is not None:
                    self._status.record_error(AUDIT_FILE, describe_failure(exc))
            else:
                if self._status is not None:
                    self._status.record_ok(AUDIT_FILE)

    def _rotate_if_needed(self) -> None:
        """Open a fresh file when the configured strategy's threshold has been crossed."""
        aged_out = time.monotonic() - self._opened_at >= self._config.rotation_interval_seconds
        too_big = self._bytes_written >= self._config.rotation_max_bytes
        if self._config.rotation_strategy == "time":
            rotate = aged_out
        elif self._config.rotation_strategy == "size":
            rotate = too_big
        else:
            rotate = aged_out or too_big
        if rotate:
            # Open the successor first so a failed open leaves the current file in service.
            successor = self._open_file()
            self._close_file()
            self._adopt(successor)

    def _open_new_file(self) -> None:
        """Create and start writing to the next audit file."""
        self._adopt(self._open_file())

    def _open_file(self) -> io.TextIOWrapper:
        """Create and open the next `audit-<service>-<unixNano>.<format>` file, owner-only."""
        name = f"audit-{self._service}-{time.time_ns()}.{self._config.file_format}"
        path = self._config.storage_path / name
        # Never append to a file this process didn't create (unknown perms and content).
        path.touch(mode=_FILE_MODE, exist_ok=False)
        # touch's mode is masked by the umask; records hold prompts, so pin owner-only access.
        path.chmod(_FILE_MODE)
        # newline="" so the csv module controls line endings itself.
        return path.open("a", encoding="utf-8", newline="")

    def _adopt(self, handle: io.TextIOWrapper) -> None:
        """Make `handle` the file records are appended to, resetting the rotation counters."""
        self._file = handle
        self._opened_at = time.monotonic()
        self._bytes_written = 0
        self._csv_header_written = False

    def _close_file(self) -> None:
        """Flush and close the current file, if any."""
        if self._file is None:
            return
        try:
            self._file.flush()
            self._file.close()
        except OSError:
            logger.exception("audit: close file")
        self._file = None

    def _append(self, message: AuditMessage) -> int:
        """Serialize `message` in the configured format, append it, and return the bytes added."""
        if self._file is None:
            return 0
        if self._config.file_format == "json":
            line = message.to_json() + "\n"
        elif self._config.file_format == "csv":
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            if not self._csv_header_written:
                writer.writerow(FIELD_NAMES)
                self._csv_header_written = True
            writer.writerow(message.to_row())
            line = buffer.getvalue()
        else:
            line = _text_line(message)
        self._file.write(line)
        self._file.flush()
        return len(line.encode("utf-8"))


def _text_line(message: AuditMessage) -> str:
    """Render one record as `[timestamp] field=value field=value ...`."""
    row = dict(zip(FIELD_NAMES, message.to_row(), strict=True))
    timestamp = row.pop("timestamp")
    pairs = " ".join(f"{name}={value}" for name, value in row.items())
    return f"[{timestamp}] {pairs}\n"
