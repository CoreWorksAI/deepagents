"""Unit tests for MemoryMiddleware."""

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime

from deepagents.graph import create_deep_agent
from deepagents.middleware.memory import (
    GET_MEMORY_DESCRIPTION,
    LIST_MEMORIES_DESCRIPTION,
    MEMORY_SYSTEM_PROMPT,
    SAVE_MEMORY_DESCRIPTION,
    InMemoryStore,
    MemoryMiddleware,
    RedisMemoryStore,
)


class TestMemoryMiddlewareInit:
    def test_init_default(self):
        """Test MemoryMiddleware initialization with defaults."""
        middleware = MemoryMiddleware()
        assert middleware._store is None
        assert not middleware._store_initialized
        assert middleware.namespace_key == "conversation_id"
        assert middleware._custom_system_prompt is None
        assert len(middleware.tools) == 3

    def test_init_with_custom_store(self):
        """Test MemoryMiddleware with custom store."""
        store = InMemoryStore()
        middleware = MemoryMiddleware(store=store)
        assert middleware._store is store
        assert middleware._store_initialized
        assert len(middleware.tools) == 3

    def test_init_with_custom_namespace_key(self):
        """Test MemoryMiddleware with custom namespace key."""
        middleware = MemoryMiddleware(namespace_key="session_id")
        assert middleware.namespace_key == "session_id"

    def test_init_with_custom_system_prompt(self):
        """Test MemoryMiddleware with custom system prompt."""
        middleware = MemoryMiddleware(system_prompt="Custom memory prompt")
        assert middleware._custom_system_prompt == "Custom memory prompt"

    def test_tools_have_correct_names(self):
        """Test that all three tools have correct names."""
        middleware = MemoryMiddleware()
        tool_names = {tool.name for tool in middleware.tools}
        assert tool_names == {"save_memory", "get_memory", "list_memories"}

    def test_tools_have_correct_descriptions(self):
        """Test that tools have correct descriptions."""
        middleware = MemoryMiddleware()
        tools_by_name = {tool.name: tool for tool in middleware.tools}
        # Compare stripped versions to handle trailing newlines
        assert tools_by_name["save_memory"].description.strip() == SAVE_MEMORY_DESCRIPTION.strip()
        assert tools_by_name["get_memory"].description.strip() == GET_MEMORY_DESCRIPTION.strip()
        assert tools_by_name["list_memories"].description.strip() == LIST_MEMORIES_DESCRIPTION.strip()


class TestInMemoryStore:
    @pytest.mark.asyncio
    async def test_put_and_get(self):
        """Test basic put and get operations."""
        store = InMemoryStore()
        await store.put("ns1", "key1", {"value": "test"})
        result = await store.get("ns1", "key1")
        assert result == {"value": "test"}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test get returns None for nonexistent key."""
        store = InMemoryStore()
        result = await store.get("ns1", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test delete operation."""
        store = InMemoryStore()
        await store.put("ns1", "key1", {"value": "test"})
        await store.delete("ns1", "key1")
        result = await store.get("ns1", "key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """Test delete on nonexistent key doesn't raise."""
        store = InMemoryStore()
        await store.delete("ns1", "nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_list_keys(self):
        """Test list_keys operation."""
        store = InMemoryStore()
        await store.put("ns1", "key1", {"value": "test1"})
        await store.put("ns1", "key2", {"value": "test2"})
        await store.put("ns2", "key3", {"value": "test3"})

        keys = await store.list_keys("ns1")
        assert set(keys) == {"key1", "key2"}

    @pytest.mark.asyncio
    async def test_list_keys_empty(self):
        """Test list_keys returns empty list for empty namespace."""
        store = InMemoryStore()
        keys = await store.list_keys("empty_ns")
        assert keys == []

    @pytest.mark.asyncio
    async def test_namespace_isolation(self):
        """Test that namespaces are isolated."""
        store = InMemoryStore()
        await store.put("ns1", "key1", {"value": "ns1-value"})
        await store.put("ns2", "key1", {"value": "ns2-value"})

        result1 = await store.get("ns1", "key1")
        result2 = await store.get("ns2", "key1")

        assert result1 == {"value": "ns1-value"}
        assert result2 == {"value": "ns2-value"}


class TestRedisMemoryStore:
    def test_init(self):
        """Test RedisMemoryStore initialization."""
        store = RedisMemoryStore("redis://localhost:6379", ttl=3600)
        assert store.redis_url == "redis://localhost:6379"
        assert store.ttl == 3600
        assert store._client is None

    def test_make_key(self):
        """Test key generation."""
        store = RedisMemoryStore("redis://localhost:6379")
        key = store._make_key("ns1", "key1")
        assert key == "memory:ns1:key1"


class TestMemoryMiddlewareAgent:
    def test_agent_with_memory_middleware(self):
        """Test that agent includes memory tools when MemoryMiddleware is added."""
        middleware = [MemoryMiddleware()]
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=middleware, tools=[])
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        assert "save_memory" in agent_tools
        assert "get_memory" in agent_tools
        assert "list_memories" in agent_tools

    def test_agent_without_memory_middleware(self):
        """Test that agent doesn't have memory tools without MemoryMiddleware."""
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=[], tools=[])
        # When no tools are provided, the agent may not have a tools node
        if "tools" in agent.nodes:
            agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
            assert "save_memory" not in agent_tools
            assert "get_memory" not in agent_tools
            assert "list_memories" not in agent_tools
        # If no tools node, that's fine - no memory tools present


class TestCreateDeepAgentMemory:
    def test_enable_memory_false_by_default(self):
        """Test that enable_memory is False by default."""
        agent = create_deep_agent(enable_filesystem=False, enable_todos=False)
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        assert "save_memory" not in agent_tools
        assert "get_memory" not in agent_tools
        assert "list_memories" not in agent_tools

    def test_enable_memory_true(self):
        """Test that enable_memory=True adds memory tools."""
        agent = create_deep_agent(enable_memory=True, enable_filesystem=False, enable_todos=False)
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        assert "save_memory" in agent_tools
        assert "get_memory" in agent_tools
        assert "list_memories" in agent_tools

    def test_enable_memory_with_filesystem(self):
        """Test that memory tools work alongside filesystem tools."""
        agent = create_deep_agent(enable_memory=True, enable_filesystem=True, enable_todos=False)
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        # Memory tools
        assert "save_memory" in agent_tools
        assert "get_memory" in agent_tools
        assert "list_memories" in agent_tools
        # Filesystem tools
        assert "ls" in agent_tools
        assert "read_file" in agent_tools
        assert "write_file" in agent_tools

    def test_enable_memory_with_todos(self):
        """Test that memory tools work alongside todo tools."""
        agent = create_deep_agent(enable_memory=True, enable_filesystem=False, enable_todos=True)
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        # Memory tools
        assert "save_memory" in agent_tools
        assert "get_memory" in agent_tools
        assert "list_memories" in agent_tools
        # Todo tools
        assert "write_todos" in agent_tools


class TestMemorySystemPrompt:
    def test_system_prompt_content(self):
        """Test that default system prompt mentions all tools."""
        assert "save_memory" in MEMORY_SYSTEM_PROMPT
        assert "get_memory" in MEMORY_SYSTEM_PROMPT
        assert "list_memories" in MEMORY_SYSTEM_PROMPT
        assert "persistent memory" in MEMORY_SYSTEM_PROMPT
