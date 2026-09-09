import asyncio

from indykite.intent_agent.audit.async_sender import AsyncSender

_DEFAULT_BUFFER_SIZE = 512
_RECORDS_TO_DRAIN = 3


def _inner(mocker, **kwargs):
    """A fake inner sender recording what it was handed."""
    return mocker.Mock(send=mocker.AsyncMock(**kwargs), close=mocker.AsyncMock())


def test_delivers_through_the_inner_sender_and_closes_it(mocker, audit_message):
    """The happy path: send returns immediately, the worker delivers, close() closes the inner sender."""
    inner = _inner(mocker)
    sender = AsyncSender(inner, buffer_size=_DEFAULT_BUFFER_SIZE)

    async def _run():
        await sender.send(audit_message())
        await sender.close()

    asyncio.run(_run())

    inner.send.assert_awaited_once_with(audit_message())
    inner.close.assert_awaited_once()


def test_close_drains_queued_records_before_closing(mocker, audit_message):
    """IAG: "buffered audit messages are drained during shutdown before the sender is closed"."""
    delivered = []
    inner = _inner(mocker, side_effect=lambda message: delivered.append(message.request_id))
    sender = AsyncSender(inner, buffer_size=_DEFAULT_BUFFER_SIZE)

    async def _run():
        for index in range(_RECORDS_TO_DRAIN):
            await sender.send(audit_message(request_id=f"req-{index}"))
        await sender.close()

    asyncio.run(_run())

    assert delivered == ["req-0", "req-1", "req-2"]
    assert inner.close.await_count == 1


def test_drops_when_the_queue_is_full(mocker, caplog, audit_message):
    """A full queue drops the record with a warning rather than blocking the request path."""
    caplog.set_level("WARNING")
    release = asyncio.Event()
    delivered = []

    async def _slow_send(message):
        await release.wait()
        delivered.append(message.request_id)

    inner = _inner(mocker, side_effect=_slow_send)
    sender = AsyncSender(inner, buffer_size=1)

    async def _run():
        await sender.send(audit_message(request_id="req-0"))  # taken by the worker, blocks in _slow_send
        await asyncio.sleep(0)
        await sender.send(audit_message(request_id="req-1"))  # fills the one queue slot
        await sender.send(audit_message(request_id="req-2"))  # dropped
        release.set()
        await sender.close()

    asyncio.run(_run())

    assert delivered == ["req-0", "req-1"]
    dropped = [record for record in caplog.records if "async queue is full" in record.message]
    assert len(dropped) == 1
    assert dropped[0].buffer_size == 1


def test_drops_while_stopping(mocker, caplog, audit_message):
    """After close() nothing more is accepted - dropped with a warning, like the IAG."""
    caplog.set_level("WARNING")
    inner = _inner(mocker)
    sender = AsyncSender(inner, buffer_size=_DEFAULT_BUFFER_SIZE)

    async def _run():
        await sender.close()
        await sender.send(audit_message())

    asyncio.run(_run())

    assert "audit: dropping payload while sender is stopping" in caplog.text
    inner.send.assert_not_awaited()


def test_inner_failure_does_not_kill_the_worker(mocker, caplog, audit_message):
    """One record the inner sender chokes on must not stop delivery of the next."""
    caplog.set_level("ERROR")
    inner = _inner(mocker, side_effect=[RuntimeError("boom"), None])
    sender = AsyncSender(inner, buffer_size=_DEFAULT_BUFFER_SIZE)

    async def _run():
        await sender.send(audit_message(request_id="req-0"))
        await sender.send(audit_message(request_id="req-1"))
        await sender.close()

    asyncio.run(_run())

    assert inner.send.await_count == 2  # noqa: PLR2004 - both records reached the inner sender
    assert "audit: async delivery failed" in caplog.text
