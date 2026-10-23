import asyncio
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock

from a2a.helpers import new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, Message, Part, Role, SendMessageRequest, TaskState
from google.protobuf.struct_pb2 import Struct

from indykite.intent_agent.app import RequestScopedTaskStore, SendMessageOnlyHandler


class _HeldExecutor(AgentExecutor):
    """Completes its task only once `release` is set."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(context.current_task or new_task_from_user_message(context.message))
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        self.started.set()
        await self.release.wait()
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def _handler(executor: AgentExecutor, store: RequestScopedTaskStore) -> SendMessageOnlyHandler:
    card = AgentCard(name="test", capabilities=AgentCapabilities())
    return SendMessageOnlyHandler(agent_executor=executor, task_store=store, agent_card=card, on_refused=AsyncMock())


def _request() -> SendMessageRequest:
    metadata = Struct()
    metadata.update({"subject_type": "Customer", "subject_id": "user-1"})
    message = Message(role=Role.ROLE_USER, message_id="m-1", parts=[Part(text="Can I read doc-1?")], metadata=metadata)
    return SendMessageRequest(message=message)


async def _eventually(check: Callable[[], Awaitable[bool]]) -> bool:
    """Whether `check()` comes true within about a second of polling."""
    for _ in range(100):
        if await check():
            return True
        await asyncio.sleep(0.01)
    return False


def test_a_cancelled_requests_task_is_deleted_once_its_work_finishes():
    async def scenario():
        store = RequestScopedTaskStore()
        saved = []
        save = store.save

        async def _record_save(task, context):
            await save(task, context)
            saved.append(task.id)

        store.save = _record_save
        executor = _HeldExecutor()
        context = ServerCallContext()
        request = asyncio.create_task(_handler(executor, store).on_message_send(_request(), context))
        try:
            await executor.started.wait()
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)

            async def _saved() -> bool:
                return bool(saved)

            async def _deleted() -> bool:
                return await store.get(saved[0], context) is None

            assert await _eventually(_saved)
            assert not await _deleted()  # the work is still running

            executor.release.set()
            assert await _eventually(_deleted)
        finally:
            executor.release.set()

    asyncio.run(scenario())


def test_a_failed_delete_is_logged_and_never_changes_the_response(mocker, caplog):
    async def scenario():
        store = RequestScopedTaskStore()
        mocker.patch.object(store, "delete", AsyncMock(side_effect=RuntimeError("store down")))
        executor = _HeldExecutor()
        executor.release.set()
        return await _handler(executor, store).on_message_send(_request(), ServerCallContext())

    result = asyncio.run(scenario())

    assert result.status.state == TaskState.TASK_STATE_COMPLETED
    assert "Failed to delete task" in caplog.text
