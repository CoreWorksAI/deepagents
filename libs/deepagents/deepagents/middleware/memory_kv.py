"""Persistent KV memory middleware for multi-user agents.

Provides save_memory, get_memory, list_memories tools backed by Redis.
Each conversation gets an isolated namespace via the configurable
`namespace_key`.

This is separate from the upstream MemoryMiddleware (AGENTS.md file-based
context loading). Both can be used together — KV memory for runtime state,
AGENTS.md memory for persistent instructions.

Store selection (when no explicit store is passed to `MemoryKVMiddleware`):

- `REDIS_URL` set and Redis reachable → `RedisMemoryStore`.
- `REDIS_URL` unset or Redis unreachable → `NullMemoryStore` with a startup
  warning. Tools remain callable but persist nothing (degraded experience).
  Hosts that want a different fallback can pass `store=...` explicitly.

`InMemoryStore` is provided for unit tests — it is never selected automatically
because silent in-memory persistence hides configuration errors and loses data
on restart.
"""

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger(__name__)


class MemoryStoreProtocol(Protocol):
    """Interface for memory storage backends."""

    async def get(self, namespace: str, key: str) -> dict[str, Any] | None: ...
    async def put(self, namespace: str, key: str, value: dict[str, Any]) -> None: ...
    async def delete(self, namespace: str, key: str) -> None: ...
    async def list_keys(self, namespace: str) -> list[str]: ...


class InMemoryStore:
    """In-memory store for unit tests. Never selected automatically; data is lost on restart."""

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


class NullMemoryStore:
    """No-op store used as the fallback when Redis is unavailable.

    All operations silently succeed without persisting. A single warning is
    logged when this store is selected so operators see the degraded state.
    Chosen over an in-memory fallback because in-memory persistence looks like
    it works but silently loses data across restarts."""

    async def get(self, namespace: str, key: str) -> dict[str, Any] | None:  # noqa: ARG002
        return None

    async def put(self, namespace: str, key: str, value: dict[str, Any]) -> None:  # noqa: ARG002
        return None

    async def delete(self, namespace: str, key: str) -> None:  # noqa: ARG002
        return None

    async def list_keys(self, namespace: str) -> list[str]:  # noqa: ARG002
        return []


class RedisMemoryStore:
    """Redis-backed memory store with TTL support.

    Keys are stored as: memory:{namespace}:{key}
    Values are JSON-serialized dictionaries.
    """

    def __init__(self, redis_url: str, ttl: int = 86400 * 7) -> None:
        self.redis_url = redis_url
        self.ttl = ttl
        self._client = None

    async def _get_client(self):
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
        await client.setex(self._make_key(namespace, key), self.ttl, json.dumps(value))

    async def delete(self, namespace: str, key: str) -> None:
        client = await self._get_client()
        await client.delete(self._make_key(namespace, key))

    async def list_keys(self, namespace: str) -> list[str]:
        client = await self._get_client()
        keys = await client.keys(f"memory:{namespace}:*")
        prefix_len = len(f"memory:{namespace}:")
        return [k[prefix_len:] for k in keys]


async def get_memory_store() -> MemoryStoreProtocol:
    """Resolve the default memory store.

    Returns a `RedisMemoryStore` when `REDIS_URL` is configured and reachable.
    Otherwise returns a `NullMemoryStore` and logs a warning — tools remain
    callable but persist nothing, so callers are in a degraded experience
    rather than a silent failure.
    """
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            store = RedisMemoryStore(redis_url)
            client = await store._get_client()
            await client.ping()  # confirm reachability, not just URL parsing
            logger.info("[MEMORY_KV] Using Redis store")
            return store
        except Exception as e:
            logger.warning(
                "[MEMORY_KV] Redis at %s unreachable (%s); memory tools will no-op "
                "(degraded experience). Configure a reachable REDIS_URL to restore persistence.",
                redis_url, e,
            )
    else:
        logger.warning(
            "[MEMORY_KV] REDIS_URL not set; memory tools will no-op "
            "(degraded experience). Set REDIS_URL to enable persistent memory."
        )
    return NullMemoryStore()


MEMORY_KV_SYSTEM_PROMPT = """## Memory Tools `save_memory`, `get_memory`, `list_memories`

You have access to persistent memory which survives across conversation turns.

- save_memory: store a dictionary under a key
- get_memory: retrieve previously saved data by key
- list_memories: list all available memory keys
"""


class _SaveMemoryInput(BaseModel):
    """Schema for the save_memory tool (runtime is injected separately)."""

    key: str = Field(..., description="Unique identifier for the memory entry.")
    data: dict[str, Any] = Field(..., description="Dictionary payload to persist.")


class _GetMemoryInput(BaseModel):
    """Schema for the get_memory tool (runtime is injected separately)."""

    key: str = Field(..., description="Key of the memory entry to retrieve.")


class _ListMemoriesInput(BaseModel):
    """Schema for the list_memories tool (runtime is injected separately)."""

    prefix: str | None = Field(
        default=None,
        description="Optional prefix; only keys starting with it are returned. None lists all keys.",
    )


def _build_tools(
    store_factory: Callable[[], Awaitable[MemoryStoreProtocol]],
    namespace_key: str,
) -> list[BaseTool]:
    """Build the three memory tools."""

    async def save_memory(key: str, data: dict[str, Any], runtime: ToolRuntime) -> str:
        """Save data to persistent memory."""
        store = await store_factory()
        ns = runtime.config.get("configurable", {}).get(namespace_key, "default")
        try:
            await store.put(ns, key, data)
            return f"Saved to memory: {key}"
        except Exception as e:
            return f"Error saving memory: {e}"

    async def get_memory(key: str, runtime: ToolRuntime) -> dict[str, Any]:
        """Retrieve data from persistent memory."""
        store = await store_factory()
        ns = runtime.config.get("configurable", {}).get(namespace_key, "default")
        try:
            return await store.get(ns, key) or {}
        except Exception as e:
            return {"_error": str(e)}

    async def list_memories(prefix: str | None, runtime: ToolRuntime) -> list[str]:
        """List available memory keys, optionally filtered by prefix."""
        store = await store_factory()
        ns = runtime.config.get("configurable", {}).get(namespace_key, "default")
        try:
            keys = await store.list_keys(ns)
        except Exception as e:
            return [f"_error: {e}"]
        if prefix:
            return [k for k in keys if k.startswith(prefix)]
        return keys

    # These tools are async-only. We declare an explicit `args_schema` so the
    # LLM sees the real input fields (key, data) without the injected
    # `runtime: ToolRuntime` parameter leaking into the tool specification.
    # Passing only `coroutine=` lets `StructuredTool` detect `ToolRuntime` as
    # an injected argument on the coroutine's signature; sync `.invoke()` is
    # unsupported by design (there is no sync store client).
    return [
        StructuredTool.from_function(
            name="save_memory",
            description="Save data to persistent memory.\n\nArgs:\n    key: Unique identifier\n    data: Dictionary to save",
            coroutine=save_memory,
            infer_schema=False,
            args_schema=_SaveMemoryInput,
        ),
        StructuredTool.from_function(
            name="get_memory",
            description="Retrieve data from persistent memory by key.",
            coroutine=get_memory,
            infer_schema=False,
            args_schema=_GetMemoryInput,
        ),
        StructuredTool.from_function(
            name="list_memories",
            description="List available memory keys. Pass an optional prefix to filter.",
            coroutine=list_memories,
            infer_schema=False,
            args_schema=_ListMemoriesInput,
        ),
    ]


class MemoryKVMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Persistent KV memory tools for multi-user agents.

    Adds save_memory, get_memory, list_memories tools backed by Redis or InMemory.
    Each conversation gets isolated storage via the namespace_key config value.

    Args:
        store: Custom store. If None, auto-detects Redis (REDIS_URL) / InMemory.
        namespace_key: Config key for namespace isolation (default: "conversation_id").
        system_prompt: Custom system prompt override.
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
        self._store_lock = asyncio.Lock()
        self.namespace_key = namespace_key
        self._system_prompt = system_prompt or MEMORY_KV_SYSTEM_PROMPT

        async def store_factory() -> MemoryStoreProtocol:
            if not self._store_initialized:
                async with self._store_lock:
                    if not self._store_initialized:  # double-check under lock
                        self._store = await get_memory_store()
                        self._store_initialized = True
            return self._store

        self.tools = _build_tools(store_factory, namespace_key)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        if self._system_prompt:
            new_msg = append_to_system_message(request.system_message, self._system_prompt)
            request = request.override(system_message=new_msg)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        if self._system_prompt:
            new_msg = append_to_system_message(request.system_message, self._system_prompt)
            request = request.override(system_message=new_msg)
        return await handler(request)
