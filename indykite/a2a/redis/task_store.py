"""Redis-backed A2A TaskStore (`RedisTaskStore`), scoped by a caller-supplied env_id only - not
by caller/subject like the SDK's own reference implementations.
"""

import logging
import time

from a2a.server.context import ServerCallContext
from a2a.server.tasks.task_store import TaskStore
from a2a.types.a2a_pb2 import ListTasksRequest, ListTasksResponse, Task, TaskState
from a2a.utils.constants import DEFAULT_LIST_TASKS_PAGE_SIZE
from a2a.utils.errors import InvalidParamsError
from a2a.utils.task import decode_page_token, encode_page_token, validate_page_size

from indykite.redis.storage import RedisStorageBackend

logger = logging.getLogger(__name__)

_MILLISECONDS_PER_SECOND = 1000

# The context-lookup record outlives the primary record by this factor, so a prune running
# after the primary key has already expired can still recover which context index to clean.
_CONTEXT_LOOKUP_TTL_MULTIPLIER = 2

# Matches the SDK's own terminal-state set (default_request_handler_v2.TERMINAL_TASK_STATES).
_TERMINAL_STATES = frozenset(
    {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_REJECTED,
    },
)


def _decode(value: bytes | str) -> str:
    """Normalize one lookup value to str, regardless of whether the wrapped redis.asyncio
    client was built with decode_responses.
    """
    return value.decode() if isinstance(value, bytes) else value


def _status_score(task: Task) -> float:
    """The ZSET score for one task: its status.timestamp as epoch milliseconds, or the
    current time if it doesn't carry one yet - safer than a fixed 0, which would sort it
    behind everything else.
    """
    if task.HasField("status") and task.status.HasField("timestamp"):
        return task.status.timestamp.ToMilliseconds()
    return time.time() * _MILLISECONDS_PER_SECOND


class RedisTaskStore(TaskStore):
    """Persists A2A `Task`s in Redis, scoped by a caller-supplied env_id (see module docstring)."""

    def __init__(
        self, backend: RedisStorageBackend, *, env_id: str, terminal_ttl_seconds: float, key_prefix: str = "ikg"
    ) -> None:
        """Store the backend/env_id/TTL; no Redis call is made until save()/get()/etc.

        key_prefix namespaces every key this store writes (default "ikg", matching this repo's
        own usage) - override it if a different consumer shares the same Redis.
        """
        self._backend = backend
        self._env_id = env_id
        self._terminal_ttl_seconds = terminal_ttl_seconds
        self._key_prefix = key_prefix

    async def aclose(self) -> None:
        """Close the underlying Redis client - called from app.py's lifespan on shutdown."""
        await self._backend.aclose()

    def _primary_key(self, task_id: str) -> str:
        """`<key_prefix>_task:{<env_id>}:<task_id>` - the `{...}` is a literal Redis Cluster hash-tag."""
        return f"{self._key_prefix}_task:{{{self._env_id}}}:{task_id}"

    def _env_index_key(self) -> str:
        """`<key_prefix>_task_idx:{<env_id>}` - the env-wide ZSET of every task id in this environment."""
        return f"{self._key_prefix}_task_idx:{{{self._env_id}}}"

    def _context_index_key(self, context_id: str) -> str:
        """`<key_prefix>_task_idx:{<env_id>}:ctx:<context_id>` - the same ZSET, scoped to one context_id."""
        return f"{self._key_prefix}_task_idx:{{{self._env_id}}}:ctx:{context_id}"

    def _context_lookup_key(self, task_id: str) -> str:
        """`<key_prefix>_task_ctx:{<env_id>}:<task_id>` - task_id's last-saved context_id, for
        pruning to recover after the primary record has already expired.
        """
        return f"{self._key_prefix}_task_ctx:{{{self._env_id}}}:{task_id}"

    async def save(self, task: Task, _context: ServerCallContext) -> None:
        """Write the primary record, update its indexes, and migrate the context index if
        task_id's context_id changed since its last save (WATCH-protected against a stale read).
        """
        score = _status_score(task)
        ttl_seconds = self._terminal_ttl_seconds if task.status.state in _TERMINAL_STATES else None

        def _build_writes(_old_encoded: bytes | None) -> list[tuple[str, str, float]]:
            writes = [(self._env_index_key(), task.id, score)]
            if task.context_id:
                writes.append((self._context_index_key(task.context_id), task.id, score))
            return writes

        def _build_cleanup(old_encoded: bytes | None) -> list[tuple[str, str]]:
            if old_encoded is None:
                return []
            old_task = Task.FromString(old_encoded)
            if old_task.context_id and old_task.context_id != task.context_id:
                return [(self._context_index_key(old_task.context_id), task.id)]
            return []

        def _build_extra_sets(_old_encoded: bytes | None) -> list[tuple[str, bytes, float | None]]:
            if not task.context_id:
                return []
            lookup_ttl = ttl_seconds * _CONTEXT_LOOKUP_TTL_MULTIPLIER if ttl_seconds is not None else None
            return [(self._context_lookup_key(task.id), task.context_id.encode(), lookup_ttl)]

        def _build_extra_deletes(old_encoded: bytes | None) -> list[str]:
            # Clearing context_id (review) - build_extra_sets above only overwrites the lookup
            # key when the new context_id is truthy, so a clear must delete it explicitly here.
            if old_encoded is None or task.context_id:
                return []
            old_task = Task.FromString(old_encoded)
            return [self._context_lookup_key(task.id)] if old_task.context_id else []

        await self._backend.watch_set(
            self._primary_key(task.id),
            task.SerializeToString(),
            ttl_seconds=ttl_seconds,
            build_writes=_build_writes,
            build_cleanup=_build_cleanup,
            build_extra_sets=_build_extra_sets,
            build_extra_deletes=_build_extra_deletes,
        )

    async def get(self, task_id: str, _context: ServerCallContext) -> Task | None:
        """Return the task, or None if its primary record is absent (never saved, or expired)."""
        encoded = await self._backend.get(self._primary_key(task_id))
        return Task.FromString(encoded) if encoded is not None else None

    async def delete(self, task_id: str, _context: ServerCallContext) -> None:
        """Delete the task's primary record, remove it from both index(es) it could be in, and
        drop its context-lookup reverse-index key - a no-op if it's already gone.

        The lookup key must go here too (review): _prune_expired_candidate's own cleanup of it
        only runs while scanning an index this call has already stripped the membership from.

        Uses RedisStorageBackend.watch_delete rather than a plain read-then-delete: a
        concurrent save() landing in between must not have its replacement silently destroyed
        by a delete meant for the old record.
        """

        def _cleanup_targets(encoded: bytes) -> list[tuple[str, str]]:
            task = Task.FromString(encoded)
            targets = [(self._env_index_key(), task_id)]
            if task.context_id:
                targets.append((self._context_index_key(task.context_id), task_id))
            return targets

        await self._backend.watch_delete(
            self._primary_key(task_id), _cleanup_targets, also_delete=[self._context_lookup_key(task_id)]
        )

    async def _resolve_start_rank(self, index_key: str, page_token: str) -> tuple[int, str]:
        """Resolve a page_token into (starting rank, the task id it resolved to)."""
        start_task_id = decode_page_token(page_token)
        start_rank = await self._backend.zrevrank(index_key, start_task_id)
        if start_rank is None:
            message = f"Invalid page token: {page_token}"
            raise InvalidParamsError(message=message)
        return start_rank + 1, start_task_id

    async def _reanchor_rank(self, index_key: str, seen_ids: list[str], floor_rank: int) -> int:
        """list()'s self-healing resume under concurrent mutation: re-derives rank from the
        most recent still-present id in seen_ids, falling back to floor_rank if all are gone."""
        while seen_ids:
            current_rank = await self._backend.zrevrank(index_key, seen_ids[-1])
            if current_rank is not None:
                return current_rank + 1
            seen_ids.pop()
        return floor_rank

    async def list(self, params: ListTasksRequest, context: ServerCallContext) -> ListTasksResponse:
        """Page through the chosen index (env-wide, or one context_id's) by ZSET rank,
        filtering by status_timestamp_after and status (app-side, since the index has no
        status of its own). total_size is the index's raw size before filtering/pruning - a
        known imprecision.

        Safe under concurrent replicas *within one call*: _reanchor_rank re-derives the scan
        position when a write shifts the index mid-scan, rather than trusting a purely local
        counter. A stale page_token from a concurrently-deleted anchor, across separate calls,
        still raises InvalidParamsError - unchanged, pre-existing behavior this fix doesn't cover.

        An expired candidate is pruned from every index it was in (_prune_expired_candidate) as
        the scan passes it; _prune_stale_tail then sweeps the rest of index_key's own tail.
        """
        index_key = self._context_index_key(params.context_id) if params.context_id else self._env_index_key()
        total_size = await self._backend.zcard(index_key)
        page_size = params.page_size or DEFAULT_LIST_TASKS_PAGE_SIZE
        validate_page_size(page_size)
        min_score = (
            params.status_timestamp_after.ToMilliseconds() if params.HasField("status_timestamp_after") else None
        )

        rank = 0
        seen_ids: list[str] = []
        if params.page_token:
            rank, start_task_id = await self._resolve_start_rank(index_key, params.page_token)
            seen_ids.append(start_task_id)
        # Fallback of last resort if every id in seen_ids turns out concurrently removed too.
        floor_rank = rank

        matched: list[Task] = []
        matched_ids: set[str] = set()
        exhausted = False
        while len(matched) < page_size and not exhausted:
            window_ids = await self._backend.zrevrange(index_key, rank, rank + page_size - 1)
            if not window_ids:
                exhausted = True
                break
            rank += len(window_ids)
            for candidate_id in window_ids:
                if candidate_id in matched_ids:
                    # Only reachable once the full-restart fallback below has kicked in and
                    # re-scanned from floor_rank - already matched earlier this same call.
                    continue
                task = await self.get(candidate_id, context)
                if task is None:
                    await self._prune_expired_candidate(candidate_id, index_key)
                    # Removing a member shifts every lower-ranked (later) member's rank down
                    # by one - rank must shift with it, or the next zrevrange call starts one
                    # past where it should and silently skips a live candidate.
                    rank -= 1
                    continue
                if min_score is not None and _status_score(task) < min_score:
                    # zrevrange yields highest-score-first, so once we're below the lower
                    # bound every later candidate is too - nothing left to gain here.
                    exhausted = True
                    break
                seen_ids.append(candidate_id)
                if params.status and task.status.state != params.status:
                    continue
                matched.append(task)
                matched_ids.add(candidate_id)
                if len(matched) == page_size:
                    break
            else:
                # Window fully consumed, another zrevrange call is coming - re-derive rank
                # rather than trust the local-only counter (review, reproduced: see _reanchor_rank).
                rank = await self._reanchor_rank(index_key, seen_ids, floor_rank)

        next_page_token = encode_page_token(matched[-1].id) if len(matched) == page_size and not exhausted else None
        await self._prune_stale_tail(index_key, context)
        return ListTasksResponse(
            tasks=matched,
            next_page_token=next_page_token or "",
            total_size=total_size,
            page_size=page_size,
        )

    async def _prune_stale_tail(self, index_key: str, context: ServerCallContext) -> None:
        """Prune every member old enough that, were it ever a terminal task, its primary
        record's TTL must already have elapsed - see list()'s docstring for why. Scales with
        the actual backlog rather than a fixed count, so it catches up regardless of expiry
        rate; a non-terminal task's entry just costs one harmless extra get() that confirms
        it's still alive. Independent of pagination: never touches the tasks/total_size/
        next_page_token list() returns, only the index's own size.
        """
        cutoff_score = (time.time() - self._terminal_ttl_seconds) * _MILLISECONDS_PER_SECOND
        stale_candidates = await self._backend.zrevrangebyscore(index_key, cutoff_score, float("-inf"))
        for candidate_id in stale_candidates:
            if await self.get(candidate_id, context) is None:
                await self._prune_expired_candidate(candidate_id, index_key)

    async def _prune_expired_candidate(self, candidate_id: str, index_key: str) -> None:
        """Remove candidate_id from env-wide, index_key, and its last-saved context's index (via
        the reverse lookup) - atomically conditioned on its primary record still being absent.
        """
        lookup_key = self._context_lookup_key(candidate_id)
        other_context_id = await self._backend.get(lookup_key)
        targets = {(self._env_index_key(), candidate_id), (index_key, candidate_id)}
        if other_context_id:
            targets.add((self._context_index_key(_decode(other_context_id)), candidate_id))
        await self._backend.zrem_if_key_absent(self._primary_key(candidate_id), list(targets), also_delete=[lookup_key])
