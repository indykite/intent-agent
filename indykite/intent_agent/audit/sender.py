"""The audit `Sender` protocol and its factory, picked from AUDIT_DELIVERY (and AUDIT_ASYNC).

There is no no-op sender: every request must produce an audit record.
"""

from typing import Protocol

from indykite.intent_agent.audit.async_sender import AsyncSender
from indykite.intent_agent.audit.file import FileConfig, FileSender
from indykite.intent_agent.audit.http import HttpConfig, HttpSender
from indykite.intent_agent.audit.message import AuditMessage
from indykite.intent_agent.config import Settings
from indykite.intent_agent.dependency_status import DependencyStatus


class Sender(Protocol):
    """What every audit delivery implements."""

    async def send(self, message: AuditMessage) -> None:
        """Deliver one record. Never raises - a delivery failure is logged instead."""

    async def close(self) -> None:
        """Release the underlying resource (file handle, HTTP client, worker task)."""


def _required[T](value: T | None, env_var: str) -> T:
    """Narrow a setting the Settings validators guarantee is set; raise ValueError naming `env_var` if not."""
    if value is None:
        message = f"audit: {env_var} is required"
        raise ValueError(message)
    return value


def build_http_config(settings: Settings) -> HttpConfig:
    """The webhook delivery config for these settings (AUDIT_DELIVERY=webhook)."""
    api_key = settings.audit_http_api_key
    return HttpConfig(
        url=str(_required(settings.audit_http_url, "AUDIT_HTTP_URL")),
        method=_required(settings.audit_http_method, "AUDIT_HTTP_METHOD"),
        auth_type=_required(settings.audit_http_auth_type, "AUDIT_HTTP_AUTH_TYPE"),
        api_key=api_key.get_secret_value() if api_key else "",
        api_key_header=settings.audit_http_api_key_header,
        mtls_certificate_file_path=settings.audit_http_mtls_certificate_file_path,
        mtls_private_key_path=settings.audit_http_mtls_private_key_path,
        timeout_seconds=settings.audit_http_timeout_seconds,
    )


def build_sender(settings: Settings, dependency_status: DependencyStatus | None = None) -> Sender:
    """Construct the sender AUDIT_DELIVERY names. A webhook sender reports each delivery to dependency_status."""
    sender: Sender
    if settings.audit_delivery == "webhook":
        sender = HttpSender(build_http_config(settings), dependency_status=dependency_status)
    else:
        file_config = FileConfig(
            storage_path=_required(settings.audit_file_storage_path, "AUDIT_FILE_STORAGE_PATH"),
            file_format=_required(settings.audit_file_format, "AUDIT_FILE_FORMAT"),
            rotation_strategy=_required(settings.audit_file_rotation_strategy, "AUDIT_FILE_ROTATION_STRATEGY"),
            rotation_interval_seconds=settings.audit_file_rotation_interval_seconds,
            rotation_max_bytes=settings.audit_file_rotation_max_bytes,
        )
        sender = FileSender(file_config, service=settings.intent_agent_name)
    if settings.audit_async:
        return AsyncSender(sender, buffer_size=settings.audit_async_buffer_size)
    return sender
