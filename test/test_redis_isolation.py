"""Pins the Redis-isolation boundary: only indykite/redis/storage.py and
indykite/a2a/redis/queue_manager.py may import the redis package directly - every other
module, including intent_agent's own.
"""

import ast
from pathlib import Path

import indykite.a2a.redis
import indykite.intent_agent
import indykite.redis

_INTENT_AGENT_DIR = Path(indykite.intent_agent.__file__).parent
_A2A_REDIS_DIR = Path(indykite.a2a.redis.__file__).parent
_REDIS_DIR = Path(indykite.redis.__file__).parent
_ALLOWED_TO_IMPORT_REDIS = {"storage.py", "queue_manager.py"}


def _imports_redis(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "redis" for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "redis":
            return True
    return False


def test_only_storage_and_queue_manager_import_redis_directly():
    """Every other module must go through storage.py's factories/ABCs instead of importing
    redis.asyncio/redis.exceptions itself - checked across all three package directories now."""
    offenders = [
        path.name
        for path in sorted(_INTENT_AGENT_DIR.glob("*.py"))
        + sorted(_A2A_REDIS_DIR.glob("*.py"))
        + sorted(_REDIS_DIR.glob("*.py"))
        if path.name not in _ALLOWED_TO_IMPORT_REDIS and _imports_redis(path)
    ]
    assert offenders == []
