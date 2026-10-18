"""Tests for EvictingTaskStore: save/get/delete and terminal-state TTL eviction."""

import asyncio

import pytest
from a2a.server.context import ServerCallContext
from a2a.types.a2a_pb2 import ListTasksRequest, Task, TaskState, TaskStatus
from a2a.utils.errors import UnsupportedOperationError

from indykite.a2a.task_store import EvictingTaskStore

_TERMINAL_TTL_SECONDS = 3600.0
_CONTEXT = ServerCallContext()


def _task(task_id, *, state=TaskState.TASK_STATE_WORKING):
    return Task(id=task_id, status=TaskStatus(state=state))


def test_save_get_delete_round_trip():
    """save() then get() must return an equal task; delete() then get() must return None."""

    async def _run():
        store = EvictingTaskStore(terminal_ttl_seconds=_TERMINAL_TTL_SECONDS)
        task = _task("t1")

        await store.save(task, _CONTEXT)
        assert await store.get("t1", _CONTEXT) == task

        await store.delete("t1", _CONTEXT)
        assert await store.get("t1", _CONTEXT) is None

    asyncio.run(_run())


def test_get_returns_none_for_a_task_never_saved():
    """A task_id that was never saved must read back as None, not raise."""

    async def _run():
        store = EvictingTaskStore(terminal_ttl_seconds=_TERMINAL_TTL_SECONDS)
        assert await store.get("missing", _CONTEXT) is None

    asyncio.run(_run())


def test_delete_is_a_noop_for_a_task_that_does_not_exist():
    """delete() on an absent task_id must not raise."""

    async def _run():
        store = EvictingTaskStore(terminal_ttl_seconds=_TERMINAL_TTL_SECONDS)
        await store.delete("missing", _CONTEXT)

    asyncio.run(_run())


@pytest.mark.parametrize(
    "state",
    [
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_REJECTED,
    ],
)
def test_a_terminal_task_is_evicted_once_its_ttl_elapses(state, mocker):
    """A terminal-state task must be gone from get() after terminal_ttl_seconds - the actual
    memory-bounding behavior this store exists for, not just a TTL value stored unused.
    """
    monotonic = mocker.patch("indykite.redis.storage.time.monotonic")
    monotonic.return_value = 0.0

    async def _run():
        store = EvictingTaskStore(terminal_ttl_seconds=_TERMINAL_TTL_SECONDS)
        await store.save(_task("terminal", state=state), _CONTEXT)
        assert await store.get("terminal", _CONTEXT) is not None

        monotonic.return_value = _TERMINAL_TTL_SECONDS + 1
        assert await store.get("terminal", _CONTEXT) is None

    asyncio.run(_run())


def test_a_non_terminal_task_is_never_evicted(mocker):
    """A WORKING (non-terminal) task must survive indefinitely - still in flight, not finished."""
    monotonic = mocker.patch("indykite.redis.storage.time.monotonic")
    monotonic.return_value = 0.0

    async def _run():
        store = EvictingTaskStore(terminal_ttl_seconds=_TERMINAL_TTL_SECONDS)
        await store.save(_task("in-flight", state=TaskState.TASK_STATE_WORKING), _CONTEXT)

        monotonic.return_value = _TERMINAL_TTL_SECONDS * 1000
        assert await store.get("in-flight", _CONTEXT) is not None

    asyncio.run(_run())


def test_list_raises_unsupported_operation_error():
    """list() must reject every call - ListTasks is out of contract, see the module docstring."""

    async def _run():
        store = EvictingTaskStore(terminal_ttl_seconds=_TERMINAL_TTL_SECONDS)
        with pytest.raises(UnsupportedOperationError):
            await store.list(ListTasksRequest(), _CONTEXT)

    asyncio.run(_run())
