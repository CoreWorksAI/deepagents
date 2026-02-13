"""Middleware for providing semantic memory tools via Mem0.

Mem0 adds automatic fact extraction, semantic search, and knowledge graph
capabilities on top of the base MemoryMiddleware's key-value persistence.

Requires: pip install mem0ai
"""

import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)


# =============================================================================
# Mem0 Client Protocol & Wrapper
# =============================================================================


class Mem0ClientProtocol(Protocol):
    """Interface for Mem0-compatible memory clients.

    This protocol allows swapping in alternative implementations
    (e.g., for testing or custom backends) while keeping the same API.
    """

    def add(
        self,
        messages: list[dict[str, str]] | str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add memories from messages or text.

        Mem0 extracts facts automatically using an LLM, deduplicates
        against existing memories, and stores them.

        Args:
            messages: Chat messages or raw text to extract memories from.
            user_id: Scope memories to a specific user.
            agent_id: Scope memories to a specific agent.
            run_id: Scope memories to a specific run/session.
            metadata: Additional metadata to attach.

        Returns:
            Dict with 'results' key containing created/updated memories.
        """
        ...

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memories by semantic similarity.

        Args:
            query: Natural language search query.
            user_id: Filter to user's memories.
            agent_id: Filter to agent's memories.
            run_id: Filter to run/session memories.
            limit: Max results to return.

        Returns:
            List of memory dicts with 'id', 'memory', 'score' fields.
        """
        ...

    def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all memories for given scope.

        Args:
            user_id: Filter to user's memories.
            agent_id: Filter to agent's memories.
            run_id: Filter to run/session memories.

        Returns:
            List of all memory dicts in scope.
        """
        ...

    def delete(self, memory_id: str) -> dict[str, Any]:
        """Delete a specific memory by ID.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            Confirmation dict.
        """
        ...


class InMemoryMem0Client:
    """In-memory mock of Mem0 client for testing.

    Stores memories as plain strings without LLM-based extraction.
    Useful for unit tests and local development without Mem0 infrastructure.
    """

    def __init__(self) -> None:
        self._memories: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def _scope_key(
        self,
        user_id: str | None,
        agent_id: str | None,
        run_id: str | None,
    ) -> str:
        return f"{user_id or ''}:{agent_id or ''}:{run_id or ''}"

    def add(
        self,
        messages: list[dict[str, str]] | str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._counter += 1
        memory_id = f"mem_{self._counter}"

        if isinstance(messages, str):
            text = messages
        else:
            text = " ".join(m.get("content", "") for m in messages)

        self._memories[memory_id] = {
            "id": memory_id,
            "memory": text,
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "metadata": metadata or {},
        }
        return {"results": [{"id": memory_id, "memory": text, "event": "ADD"}]}

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        query_lower = query.lower()
        results = []
        for mem in self._memories.values():
            if user_id and mem.get("user_id") != user_id:
                continue
            if agent_id and mem.get("agent_id") != agent_id:
                continue
            if run_id and mem.get("run_id") != run_id:
                continue
            # Simple substring match for testing
            if query_lower in mem["memory"].lower():
                results.append({**mem, "score": 0.9})
        return results[:limit]

    def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        results = []
        for mem in self._memories.values():
            if user_id and mem.get("user_id") != user_id:
                continue
            if agent_id and mem.get("agent_id") != agent_id:
                continue
            if run_id and mem.get("run_id") != run_id:
                continue
            results.append(mem)
        return results

    def delete(self, memory_id: str) -> dict[str, Any]:
        self._memories.pop(memory_id, None)
        return {"message": "Memory deleted successfully"}


def _create_mem0_client(config: dict[str, Any] | None = None) -> Mem0ClientProtocol:
    """Create a Mem0 Memory client.

    Attempts to import and initialize mem0. Falls back to InMemoryMem0Client
    if mem0 is not installed.

    Args:
        config: Optional Mem0 configuration dict. If not provided, Mem0
            will use its defaults (which read from env vars like
            OPENAI_API_KEY for the extraction LLM).

    Returns:
        A Mem0-compatible client instance.
    """
    try:
        from mem0 import Memory  # type: ignore[import-untyped]

        if config:
            client = Memory.from_config(config)
        else:
            client = Memory()
        logger.info("[MEM0] Using Mem0 Memory client")
        return client
    except ImportError:
        logger.warning("[MEM0] mem0ai not installed, using InMemoryMem0Client")
        return InMemoryMem0Client()
    except Exception as e:
        logger.warning("[MEM0] Failed to initialize Mem0 (%s), using InMemoryMem0Client", e)
        return InMemoryMem0Client()


# =============================================================================
# Tool Descriptions
# =============================================================================

REMEMBER_DESCRIPTION = """Store a fact or piece of information in semantic memory.

Unlike save_memory (key-value), this tool uses AI to automatically extract
and deduplicate facts. Use this for natural language information that should
be remembered across conversations.

Args:
    content: The information to remember. Can be a natural language statement
        like "The user prefers dark mode" or "The project uses FastAPI".

Returns:
    Confirmation with extracted memory details.
"""

RECALL_DESCRIPTION = """Search semantic memory for relevant information.

Retrieves memories that are semantically similar to the query, even if
the exact words don't match. Use this to find previously stored knowledge
before answering questions or making decisions.

Args:
    query: Natural language search query describing what you're looking for.
    limit: Maximum number of memories to return (default: 5).

Returns:
    List of relevant memories with similarity scores.
"""

FORGET_DESCRIPTION = """Remove a specific memory by its ID.

Use this when information is outdated, incorrect, or the user explicitly
asks you to forget something.

Args:
    memory_id: The ID of the memory to delete (from recall results).

Returns:
    Confirmation message.
"""

LIST_ALL_MEMORIES_DESCRIPTION = """List all stored memories for the current scope.

Returns all memories without filtering by relevance. Useful for reviewing
what the agent knows or for debugging memory contents.

Returns:
    List of all memories with their IDs and content.
"""

MEM0_SYSTEM_PROMPT = """## Semantic Memory Tools `remember`, `recall`, `forget`, `list_all_memories`

You have access to semantic memory powered by Mem0. This is different from
simple key-value storage -- it understands meaning and deduplicates automatically.

- **remember**: Store facts, preferences, or context. The system automatically
  extracts and deduplicates key information.
- **recall**: Search memory by meaning, not just keywords. Ask natural language
  questions like "What does the user prefer for testing?"
- **forget**: Remove outdated or incorrect memories by ID.
- **list_all_memories**: View everything stored in memory.

### When to use semantic memory:
- Store user preferences, project conventions, and learned patterns
- Recall relevant context before answering questions
- Update memory when preferences or facts change
- Forget information the user wants removed

### Memory scoping:
Memories are scoped to user_id, agent_id, and session. User-scoped memories
persist across all conversations. Agent-scoped memories are shared across
users but isolated per agent. Session memories are ephemeral.
"""


# =============================================================================
# Tool Generators
# =============================================================================


def _remember_tool_generator(
    client_factory: Callable[[], Mem0ClientProtocol],
    namespace_key: str = "conversation_id",
) -> BaseTool:
    """Generate the remember tool."""

    def sync_remember(
        content: str,
        runtime: ToolRuntime,
    ) -> str:
        """Store information in semantic memory."""
        client = client_factory()
        config = runtime.config.get("configurable", {})
        user_id = config.get("user_id", "default")
        agent_id = config.get("agent_id")
        run_id = config.get(namespace_key)

        try:
            result = client.add(
                content,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
            )
            memories = result.get("results", [])
            if not memories:
                return "No new memories extracted from the content."
            summary_parts = []
            for mem in memories:
                event = mem.get("event", "ADD")
                text = mem.get("memory", "")
                summary_parts.append(f"[{event}] {text}")
            return "Memories updated:\n" + "\n".join(summary_parts)
        except Exception as e:
            return f"Error storing memory: {e}"

    async def async_remember(
        content: str,
        runtime: ToolRuntime,
    ) -> str:
        """Async wrapper -- delegates to sync since Mem0's API is sync."""
        return sync_remember(content, runtime)

    return StructuredTool.from_function(
        name="remember",
        description=REMEMBER_DESCRIPTION,
        func=sync_remember,
        coroutine=async_remember,
    )


def _recall_tool_generator(
    client_factory: Callable[[], Mem0ClientProtocol],
    namespace_key: str = "conversation_id",
) -> BaseTool:
    """Generate the recall tool."""

    def sync_recall(
        query: str,
        runtime: ToolRuntime,
        limit: int = 5,
    ) -> str:
        """Search semantic memory."""
        client = client_factory()
        config = runtime.config.get("configurable", {})
        user_id = config.get("user_id", "default")
        agent_id = config.get("agent_id")
        run_id = config.get(namespace_key)

        try:
            results = client.search(
                query,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                limit=limit,
            )
            if not results:
                return "No relevant memories found."
            parts = []
            for mem in results:
                mid = mem.get("id", "?")
                text = mem.get("memory", "")
                score = mem.get("score", 0)
                parts.append(f"[{mid}] (score: {score:.2f}) {text}")
            return "Relevant memories:\n" + "\n".join(parts)
        except Exception as e:
            return f"Error searching memory: {e}"

    async def async_recall(
        query: str,
        runtime: ToolRuntime,
        limit: int = 5,
    ) -> str:
        """Async wrapper."""
        return sync_recall(query, runtime, limit)

    return StructuredTool.from_function(
        name="recall",
        description=RECALL_DESCRIPTION,
        func=sync_recall,
        coroutine=async_recall,
    )


def _forget_tool_generator(
    client_factory: Callable[[], Mem0ClientProtocol],
) -> BaseTool:
    """Generate the forget tool."""

    def sync_forget(
        memory_id: str,
        runtime: ToolRuntime,
    ) -> str:
        """Delete a specific memory."""
        client = client_factory()
        try:
            client.delete(memory_id)
            return f"Memory {memory_id} deleted successfully."
        except Exception as e:
            return f"Error deleting memory: {e}"

    async def async_forget(
        memory_id: str,
        runtime: ToolRuntime,
    ) -> str:
        """Async wrapper."""
        return sync_forget(memory_id, runtime)

    return StructuredTool.from_function(
        name="forget",
        description=FORGET_DESCRIPTION,
        func=sync_forget,
        coroutine=async_forget,
    )


def _list_all_memories_tool_generator(
    client_factory: Callable[[], Mem0ClientProtocol],
    namespace_key: str = "conversation_id",
) -> BaseTool:
    """Generate the list_all_memories tool."""

    def sync_list_all(
        runtime: ToolRuntime,
    ) -> str:
        """List all memories in current scope."""
        client = client_factory()
        config = runtime.config.get("configurable", {})
        user_id = config.get("user_id", "default")
        agent_id = config.get("agent_id")
        run_id = config.get(namespace_key)

        try:
            results = client.get_all(
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
            )
            if not results:
                return "No memories stored."
            parts = []
            for mem in results:
                mid = mem.get("id", "?")
                text = mem.get("memory", "")
                parts.append(f"[{mid}] {text}")
            return f"All memories ({len(results)}):\n" + "\n".join(parts)
        except Exception as e:
            return f"Error listing memories: {e}"

    async def async_list_all(
        runtime: ToolRuntime,
    ) -> str:
        """Async wrapper."""
        return sync_list_all(runtime)

    return StructuredTool.from_function(
        name="list_all_memories",
        description=LIST_ALL_MEMORIES_DESCRIPTION,
        func=sync_list_all,
        coroutine=async_list_all,
    )


def _get_mem0_tools(
    client_factory: Callable[[], Mem0ClientProtocol],
    namespace_key: str = "conversation_id",
) -> list[BaseTool]:
    """Get all Mem0 memory tools."""
    return [
        _remember_tool_generator(client_factory, namespace_key),
        _recall_tool_generator(client_factory, namespace_key),
        _forget_tool_generator(client_factory),
        _list_all_memories_tool_generator(client_factory, namespace_key),
    ]


# =============================================================================
# Middleware
# =============================================================================


class Mem0MemoryMiddleware(AgentMiddleware):
    """Middleware for semantic memory via Mem0.

    Provides four tools: remember, recall, forget, list_all_memories.
    Unlike the base MemoryMiddleware (key-value), this middleware offers:

    - **Automatic fact extraction**: Mem0 uses an LLM to extract facts from
      natural language, so the agent doesn't need to structure data.
    - **Semantic search**: Recall memories by meaning, not just exact keys.
    - **Deduplication**: Mem0 automatically detects and merges duplicate facts.
    - **Multi-scope**: Memories can be scoped to user, agent, or session.

    Args:
        client: Optional Mem0-compatible client. If not provided, creates one
            using Mem0's defaults (reads OPENAI_API_KEY from env).
        mem0_config: Optional Mem0 configuration dict for customizing the
            vector store, graph store, LLM, and embedder. Ignored if client
            is provided directly.
        namespace_key: Config key for session scoping. Default "conversation_id".
        system_prompt: Optional custom system prompt override.

    Example:
        ```python
        from deepagents import create_deep_agent
        from deepagents.middleware.mem0_memory import Mem0MemoryMiddleware

        # Use with default Mem0 config (requires OPENAI_API_KEY)
        agent = create_deep_agent(
            middleware=[Mem0MemoryMiddleware()],
        )

        # Use with custom Mem0 config (e.g., Qdrant + Neo4j)
        agent = create_deep_agent(
            middleware=[Mem0MemoryMiddleware(
                mem0_config={
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {"host": "localhost", "port": 6333},
                    },
                    "graph_store": {
                        "provider": "neo4j",
                        "config": {
                            "url": "bolt://localhost:7687",
                            "username": "neo4j",
                            "password": "password",
                        },
                    },
                },
            )],
        )

        # Invoke with user scoping
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "I prefer dark mode"}]},
            config={"configurable": {"user_id": "alice", "conversation_id": "conv-1"}},
        )
        ```
    """

    def __init__(
        self,
        *,
        client: Mem0ClientProtocol | None = None,
        mem0_config: dict[str, Any] | None = None,
        namespace_key: str = "conversation_id",
        system_prompt: str | None = None,
    ) -> None:
        self._client = client
        self._mem0_config = mem0_config
        self._client_initialized = client is not None
        self.namespace_key = namespace_key
        self._custom_system_prompt = system_prompt

        def client_factory() -> Mem0ClientProtocol:
            if not self._client_initialized:
                self._client = _create_mem0_client(self._mem0_config)
                self._client_initialized = True
            return self._client  # type: ignore[return-value]

        self.tools = _get_mem0_tools(client_factory, namespace_key)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject Mem0 system prompt."""
        system_prompt = self._custom_system_prompt or MEM0_SYSTEM_PROMPT

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
        """Async version - inject Mem0 system prompt."""
        system_prompt = self._custom_system_prompt or MEM0_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = (
                f"{request.system_prompt}\n\n{system_prompt}"
                if request.system_prompt
                else system_prompt
            )
            request = request.override(system_prompt=new_prompt)

        return await handler(request)
