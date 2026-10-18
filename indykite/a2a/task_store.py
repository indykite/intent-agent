"""In-memory A2A TaskStore with short-TTL eviction for terminal tasks.

IAG only calls blocking SendMessage, answered in full by the receiving replica, so no task is
ever read, listed, or canceled later (those are rejected first, see app.py's
_SendMessageOnlyRequestHandler) - a plain per-process store is enough. Unlike the SDK's own
InMemoryTaskStore, which never deletes and keeps every task forever, this evicts a terminal one
terminal_ttl_seconds after it's saved, so memory doesn't grow with every request.
"""

from a2a.server.context import ServerCallContext
from a2a.server.tasks.task_store import TaskStore
from a2a.types.a2a_pb2 import ListTasksRequest, ListTasksResponse, Task, TaskState
from a2a.utils.errors import UnsupportedOperationError

from indykite.redis.storage import MemoryStorageBackend

# Matches the SDK's own terminal-state set (default_request_handler_v2.TERMINAL_TASK_STATES).
_TERMINAL_STATES = frozenset(
    {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_REJECTED,
    },
)


class EvictingTaskStore(TaskStore):
    """Keeps every task in memory, but drops a terminal one terminal_ttl_seconds after it's saved."""

    def __init__(self, *, terminal_ttl_seconds: float) -> None:
        """terminal_ttl_seconds bounds how long a finished task stays around; a non-terminal
        (still in-flight) one is kept with no expiry.
        """
        self._backend = MemoryStorageBackend()
        self._terminal_ttl_seconds = terminal_ttl_seconds

    async def save(self, task: Task, _context: ServerCallContext) -> None:
        """Store the task; a terminal one gets terminal_ttl_seconds, a non-terminal one no TTL.

        MemoryStorageBackend is a plain in-process dict (no I/O), so this is called
        synchronously despite the TaskStore ABC requiring an async signature here.
        """
        ttl_seconds = self._terminal_ttl_seconds if task.status.state in _TERMINAL_STATES else None
        self._backend.set(task.id, task.SerializeToString(), ttl_seconds=ttl_seconds)

    async def get(self, task_id: str, _context: ServerCallContext) -> Task | None:
        """Return the task, or None if it was never saved or has since been evicted."""
        encoded = self._backend.get(task_id)
        return Task.FromString(encoded) if encoded is not None else None

    async def delete(self, task_id: str, _context: ServerCallContext) -> None:
        """Remove task_id - a no-op if it's already gone."""
        self._backend.delete(task_id)

    async def list(self, _params: ListTasksRequest, _context: ServerCallContext) -> ListTasksResponse:
        """Required by the TaskStore ABC, but unreachable - _SendMessageOnlyRequestHandler
        rejects ListTasks before this would ever be called.
        """
        raise UnsupportedOperationError(message="ListTasks is not supported.")
