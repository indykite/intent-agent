import asyncio
import json
from http import HTTPStatus

import httpx2
import pytest

from indykite.intent_agent.audit.http import HttpConfig, HttpSender, probe
from indykite.intent_agent.dependency_status import AUDIT_WEBHOOK, DependencyStatus

_URL = "https://audit.example.com/records"


def _sender_with_transport(responder, **overrides):
    """An HttpSender whose HTTP layer is `responder(request) -> Response` (or raises)."""
    options = {"url": _URL, "method": "POST", "auth_type": "no-auth"}
    options.update(overrides)
    return HttpSender(HttpConfig(**options), transport=httpx2.MockTransport(responder))


def _send_and_close(sender, message):
    """Deliver one record, then close the sender."""

    async def _run():
        """Send then close, on one event loop."""
        await sender.send(message)
        await sender.close()

    asyncio.run(_run())


def test_posts_the_record_as_json_body(audit_message):
    """One request per record: the record is the whole body (no envelope), JSON content type."""
    seen = []

    def responder(request):
        seen.append(request)
        return httpx2.Response(HTTPStatus.OK)

    _send_and_close(_sender_with_transport(responder), audit_message())

    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert str(seen[0].url) == _URL
    assert seen[0].headers["content-type"] == "application/json"
    assert json.loads(seen[0].content) == json.loads(audit_message().to_json())


def test_put_method_is_honoured(audit_message):
    """AUDIT_HTTP_METHOD=PUT sends PUT."""
    seen = []

    def responder(request):
        seen.append(request.method)
        return httpx2.Response(HTTPStatus.NO_CONTENT)

    _send_and_close(_sender_with_transport(responder, method="PUT"), audit_message())

    assert seen == ["PUT"]


def test_api_key_auth_sets_the_configured_header(audit_message):
    """`api-key` auth: the key rides in the configured header."""
    seen = []

    def responder(request):
        seen.append(dict(request.headers))
        return httpx2.Response(HTTPStatus.OK)

    sender = _sender_with_transport(
        responder, auth_type="api-key", api_key="test-audit-key", api_key_header="X-Audit-Key"
    )
    _send_and_close(sender, audit_message())

    assert seen[0]["x-audit-key"] == "test-audit-key"


def test_no_auth_sends_no_credentials(audit_message):
    """`no-auth`: nothing but the content type is added."""
    seen = []

    def responder(request):
        seen.append(dict(request.headers))
        return httpx2.Response(HTTPStatus.OK)

    _send_and_close(_sender_with_transport(responder), audit_message())

    assert "x-api-key" not in seen[0]
    assert "authorization" not in seen[0]


@pytest.mark.parametrize(
    ("auth_type", "message"),
    [("api-key", "api-key auth requires api_key"), ("mTLS", "mTLS requires")],
)
def test_auth_requires_credentials(auth_type, message):
    """api-key auth needs a key; mTLS needs both the certificate and the key."""
    with pytest.raises(ValueError, match=message):
        _sender_with_transport(lambda _request: httpx2.Response(HTTPStatus.OK), auth_type=auth_type)


def test_error_status_is_logged_not_raised(caplog, audit_message):
    """A 4xx/5xx from the receiver is logged with its status, never retried or raised."""
    caplog.set_level("ERROR")

    _send_and_close(
        _sender_with_transport(lambda _request: httpx2.Response(HTTPStatus.SERVICE_UNAVAILABLE)),
        audit_message(),
    )

    record = next(r for r in caplog.records if "unexpected status code" in r.message)
    assert record.status_code == HTTPStatus.SERVICE_UNAVAILABLE


def test_transport_failure_is_logged_not_raised(caplog, audit_message):
    """A connection error is logged and swallowed."""
    caplog.set_level("ERROR")

    def responder(_request):
        message = "connection refused"
        raise httpx2.ConnectError(message)

    _send_and_close(_sender_with_transport(responder), audit_message())

    assert "audit: failed to send HTTP request" in caplog.text


def _self_signed_pair(tmp_path):
    """Write a throwaway self-signed certificate and key (skipped if transitive `cryptography` is gone)."""
    from datetime import UTC, datetime, timedelta

    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "intent-agent-test")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "client.pem"
    key_path = tmp_path / "client-key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


def test_mtls_loads_the_client_certificate(tmp_path, caplog, audit_message):
    """`mTLS`: the certificate and key are loaded at startup and records still flow."""
    caplog.set_level("INFO")
    cert_path, key_path = _self_signed_pair(tmp_path)
    seen = []

    def responder(request):
        seen.append(request.method)
        return httpx2.Response(HTTPStatus.OK)

    sender = _sender_with_transport(
        responder,
        auth_type="mTLS",
        mtls_certificate_file_path=cert_path,
        mtls_private_key_path=key_path,
    )
    _send_and_close(sender, audit_message())

    assert seen == ["POST"]
    assert "started auditing through HTTP with mTLS" in caplog.text


def test_probe_sends_options_with_the_api_key_and_returns_the_status(mocker):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx2.Response(HTTPStatus.METHOD_NOT_ALLOWED)

    real = httpx2.AsyncClient
    mocker.patch(
        "indykite.intent_agent.audit.http.httpx2.AsyncClient",
        side_effect=lambda **kwargs: real(transport=httpx2.MockTransport(handler), **kwargs),
    )
    config = HttpConfig(url="https://audit.example.com/r", method="POST", auth_type="api-key", api_key="k")

    assert asyncio.run(probe(config)) == HTTPStatus.METHOD_NOT_ALLOWED
    assert seen[0].method == "OPTIONS"
    assert seen[0].headers["X-API-Key"] == "k"


@pytest.mark.parametrize(
    ("response", "status", "detail"),
    [
        pytest.param(httpx2.Response(HTTPStatus.NO_CONTENT), "ok", "", id="2xx"),
        pytest.param(httpx2.Response(HTTPStatus.INTERNAL_SERVER_ERROR), "error", "status 500", id="5xx"),
        pytest.param(
            httpx2.Response(HTTPStatus.FOUND, headers={"Location": "https://x"}), "error", "status 302", id="3xx"
        ),
    ],
)
def test_delivery_records_audit_webhook_state(audit_message, response, status, detail):
    """Only a 2xx is a delivery - the client doesn't follow redirects."""
    store = DependencyStatus()
    sender = HttpSender(
        HttpConfig(url=_URL, method="POST", auth_type="no-auth"),
        transport=httpx2.MockTransport(lambda _request: response),
        dependency_status=store,
    )

    _send_and_close(sender, audit_message())

    assert (store.get(AUDIT_WEBHOOK).status, store.get(AUDIT_WEBHOOK).detail) == (status, detail)


def test_unreachable_receiver_records_the_exception_class(audit_message):
    store = DependencyStatus()

    def responder(_request):
        message = "no route"
        raise httpx2.ConnectError(message)

    sender = HttpSender(
        HttpConfig(url=_URL, method="POST", auth_type="no-auth"),
        transport=httpx2.MockTransport(responder),
        dependency_status=store,
    )

    _send_and_close(sender, audit_message())

    assert (store.get(AUDIT_WEBHOOK).status, store.get(AUDIT_WEBHOOK).detail) == ("error", "ConnectError")
