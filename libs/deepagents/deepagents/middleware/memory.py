"""Middleware for providing persistent memory tools to an agent.

Memory persists across conversation turns using Redis (default) or InMemory (fallback).
"""

import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)


# =============================================================================
# Store Protocol & Implementations
# =============================================================================


class MemoryStoreProtocol(Protocol):
    """Interface for memory storage backends."""

    async def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        """Get value by key."""
        ...

    async def put(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        """Store value by key."""
        ...

    async def delete(self, namespace: str, key: str) -> None:
        """Delete by key."""
        ...

    async def list_keys(self, namespace: str) -> list[str]:
        """List all keys in namespace."""
        ...


class InMemoryStore:
    """In-memory store for local/testing when Redis unavailable.

    Warning: Data is lost on restart.
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    async def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        return self._data.get(f"{namespace}:{key}")

    async def put(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        self._data[f"{namespace}:{key}"] = value

    async def delete(self, namespace: str, key: str) -> None:
        self._data.pop(f"{namespace}:{key}", None)

    async def list_keys(self, namespace: str) -> list[str]:
        prefix = f"{namespace}:"
        return [k[len(prefix):] for k in self._data if k.startswith(prefix)]


class RedisMemoryStore:
    """Redis-backed memory store.

    Keys are stored as: memory:{namespace}:{key}
    Values are JSON-serialized dictionaries with TTL.
    """

    def __init__(self, redis_url: str, ttl: int = 86400 * 7) -> None:
        """Initialize Redis store.

        Args:
            redis_url: Redis connection URL
            ttl: Time-to-live in seconds (default: 7 days)
        """
        self.redis_url = redis_url
        self.ttl = ttl
        self._client = None

    async def _get_client(self):
        """Lazy-initialize Redis client."""
        if self._client is None:
            import redis.asyncio as redis
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _make_key(self, namespace: str, key: str) -> str:
        return f"memory:{namespace}:{key}"

    async def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        client = await self._get_client()
        data = await client.get(self._make_key(namespace, key))
        return json.loads(data) if data else None

    async def put(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        client = await self._get_client()
        await client.setex(
            self._make_key(namespace, key),
            self.ttl,
            json.dumps(value),
        )

    async def delete(self, namespace: str, key: str) -> None:
        client = await self._get_client()
        await client.delete(self._make_key(namespace, key))

    async def list_keys(self, namespace: str) -> list[str]:
        client = await self._get_client()
        pattern = f"memory:{namespace}:*"
        keys = await client.keys(pattern)
        prefix_len = len(f"memory:{namespace}:")
        return [k[prefix_len:] for k in keys]


async def get_memory_store() -> MemoryStoreProtocol:
    """Get memory store: Redis if available, else InMemory."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            store = RedisMemoryStore(redis_url)
            await store._get_client()  # Test connection
            logger.info("[MEMORY] Using Redis store")
            return store
        except Exception as e:
            logger.warning(f"[MEMORY] Redis unavailable ({e}), using InMemory")

    logger.warning("[MEMORY] Using InMemory store (data lost on restart)")
    return InMemoryStore()


# =============================================================================
# Tool Descriptions
# =============================================================================


SAVE_MEMORY_DESCRIPTION = """Save data to persistent memory.

Memory persists across conversation turns and agent invocations.

Args:
    key: Unique identifier for this memory
    data: Dictionary to save

Returns:
    Confirmation message
"""

GET_MEMORY_DESCRIPTION = """Retrieve data from persistent memory.

Args:
    key: The memory key to retrieve

Returns:
    The saved dictionary, or empty dict if not found
"""

LIST_MEMORIES_DESCRIPTION = """List all available memory keys.

Returns:
    List of memory keys that can be retrieved with get_memory
"""


MEMORY_SYSTEM_PROMPT = """## Memory Tools `save_memory`, `get_memory`, `list_memories`

You have access to persistent memory which survives across conversation turns.

- save_memory: store a dictionary under a key
- get_memory: retrieve previously saved data by key
- list_memories: list all available memory keys
"""


# =============================================================================
# Tool Generators
# =============================================================================


def _save_memory_tool_generator(
    store_factory: Callable[[], Awaitable[MemoryStoreProtocol]],
    namespace_key: str = "conversation_id",
) -> BaseTool:
    """Generate the save_memory tool."""

    async def save_memory(
        key: str,
        data: dict[str, Any],
        runtime: ToolRuntime,
    ) -> str:
        store = await store_factory()
        namespace = runtime.config.get("configurable", {}).get(namespace_key, "default")
        try:
            await store.put(namespace, key, data)
            return f"Saved to memory: {key}"
        except Exception as e:
            return f"Error saving memory: {e}"

    return StructuredTool.from_function(
        name="save_memory",
        description=SAVE_MEMORY_DESCRIPTION,
        func=lambda key, data: None,  # Sync placeholder
        coroutine=save_memory,
    )


def _get_memory_tool_generator(
    store_factory: Callable[[], Awaitable[MemoryStoreProtocol]],
    namespace_key: str = "conversation_id",
) -> BaseTool:
    """Generate the get_memory tool."""

    async def get_memory(
        key: str,
        runtime: ToolRuntime,
    ) -> dict[str, Any]:
        store = await store_factory()
        namespace = runtime.config.get("configurable", {}).get(namespace_key, "default")
        try:
            result = await store.get(namespace, key)
            return result or {}
        except Exception as e:
            return {"_error": str(e)}

    return StructuredTool.from_function(
        name="get_memory",
        description=GET_MEMORY_DESCRIPTION,
        func=lambda key: {},  # Sync placeholder
        coroutine=get_memory,
    )


def _list_memories_tool_generator(
    store_factory: Callable[[], Awaitable[MemoryStoreProtocol]],
    namespace_key: str = "conversation_id",
) -> BaseTool:
    """Generate the list_memories tool."""

    async def list_memories(
        runtime: ToolRuntime,
    ) -> list[str]:
        store = await store_factory()
        namespace = runtime.config.get("configurable", {}).get(namespace_key, "default")
        try:
            return await store.list_keys(namespace)
        except Exception as e:
            return [f"_error: {e}"]

    return StructuredTool.from_function(
        name="list_memories",
        description=LIST_MEMORIES_DESCRIPTION,
        func=lambda: [],  # Sync placeholder
        coroutine=list_memories,
    )


def _get_memory_tools(
    store_factory: Callable[[], Awaitable[MemoryStoreProtocol]],
    namespace_key: str = "conversation_id",
) -> list[BaseTool]:
    """Get all memory tools."""
    return [
        _save_memory_tool_generator(store_factory, namespace_key),
        _get_memory_tool_generator(store_factory, namespace_key),
        _list_memories_tool_generator(store_factory, namespace_key),
    ]


# =============================================================================
# Middleware
# =============================================================================


class MemoryMiddleware(AgentMiddleware):
    """Middleware for providing persistent memory tools to an agent.

    Adds memory tools: save_memory, get_memory, list_memories.
    Uses Redis by default, falls back to InMemory if unavailable.

    Args:
        store: Optional custom store implementation. If not provided,
            uses Redis (REDIS_URL env) or InMemory fallback.
        namespace_key: Config key to use for namespace isolation.
            Default "conversation_id" means each conversation has separate memory.
        system_prompt: Optional custom system prompt override.

    Example:
        ```python
        from deepagents import create_deep_agent

        # Use default Redis/InMemory
        agent = create_deep_agent(
            enable_memory=True,
        )

        # Use custom store
        agent = create_deep_agent(
            middleware=[MemoryMiddleware(store=my_store)],
        )
        ```
    """

    def __init__(
        self,
        *,
        store: MemoryStoreProtocol | None = None,
        namespace_key: str = "conversation_id",
        system_prompt: str | None = None,
    ) -> None:
        self._store = store
        self._store_initialized = store is not None
        self.namespace_key = namespace_key
        self._custom_system_prompt = system_prompt

        # Store factory for lazy initialization
        async def store_factory() -> MemoryStoreProtocol:
            if not self._store_initialized:
                self._store = await get_memory_store()
                self._store_initialized = True
            return self._store

        self.tools = _get_memory_tools(store_factory, namespace_key)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject memory system prompt."""
        system_prompt = self._custom_system_prompt or MEMORY_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = (
                f"{request.system_prompt}\n\n{system_prompt}"
                if request.system_prompt
                else system_prompt
            )
            request = request.override(system_prompt=new_prompt)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version - inject memory system prompt."""
        system_prompt = self._custom_system_prompt or MEMORY_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = (
                f"{request.system_prompt}\n\n{system_prompt}"
                if request.system_prompt
                else system_prompt
            )
            request = request.override(system_prompt=new_prompt)

        return await handler(request)
