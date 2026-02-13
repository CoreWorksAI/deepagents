"""Unit tests for Mem0MemoryMiddleware."""

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime

from deepagents.middleware.mem0_memory import (
    FORGET_DESCRIPTION,
    LIST_ALL_MEMORIES_DESCRIPTION,
    MEM0_SYSTEM_PROMPT,
    RECALL_DESCRIPTION,
    REMEMBER_DESCRIPTION,
    InMemoryMem0Client,
    Mem0MemoryMiddleware,
)


class TestMem0MemoryMiddlewareInit:
    def test_init_default(self):
        """Test Mem0MemoryMiddleware initialization with defaults."""
        middleware = Mem0MemoryMiddleware()
        assert middleware._client is None
        assert not middleware._client_initialized
        assert middleware.namespace_key == "conversation_id"
        assert middleware._custom_system_prompt is None
        assert len(middleware.tools) == 4

    def test_init_with_custom_client(self):
        """Test Mem0MemoryMiddleware with custom client."""
        client = InMemoryMem0Client()
        middleware = Mem0MemoryMiddleware(client=client)
        assert middleware._client is client
        assert middleware._client_initialized
        assert len(middleware.tools) == 4

    def test_init_with_custom_namespace_key(self):
        """Test Mem0MemoryMiddleware with custom namespace key."""
        middleware = Mem0MemoryMiddleware(namespace_key="session_id")
        assert middleware.namespace_key == "session_id"

    def test_init_with_custom_system_prompt(self):
        """Test Mem0MemoryMiddleware with custom system prompt."""
        middleware = Mem0MemoryMiddleware(system_prompt="Custom mem0 prompt")
        assert middleware._custom_system_prompt == "Custom mem0 prompt"

    def test_init_with_mem0_config(self):
        """Test Mem0MemoryMiddleware stores config for deferred init."""
        config = {"vector_store": {"provider": "chroma"}}
        middleware = Mem0MemoryMiddleware(mem0_config=config)
        assert middleware._mem0_config == config
        assert not middleware._client_initialized

    def test_tools_have_correct_names(self):
        """Test that all four tools have correct names."""
        middleware = Mem0MemoryMiddleware()
        tool_names = {tool.name for tool in middleware.tools}
        assert tool_names == {"remember", "recall", "forget", "list_all_memories"}

    def test_tools_have_correct_descriptions(self):
        """Test that tools have correct descriptions."""
        middleware = Mem0MemoryMiddleware()
        tools_by_name = {tool.name: tool for tool in middleware.tools}
        assert tools_by_name["remember"].description.strip() == REMEMBER_DESCRIPTION.strip()
        assert tools_by_name["recall"].description.strip() == RECALL_DESCRIPTION.strip()
        assert tools_by_name["forget"].description.strip() == FORGET_DESCRIPTION.strip()
        assert tools_by_name["list_all_memories"].description.strip() == LIST_ALL_MEMORIES_DESCRIPTION.strip()


class TestInMemoryMem0Client:
    def test_add_string(self):
        """Test adding a string memory."""
        client = InMemoryMem0Client()
        result = client.add("The user likes Python", user_id="alice")
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["event"] == "ADD"
        assert "Python" in result["results"][0]["memory"]

    def test_add_messages(self):
        """Test adding memories from chat messages."""
        client = InMemoryMem0Client()
        messages = [
            {"role": "user", "content": "I prefer dark mode"},
            {"role": "assistant", "content": "Noted!"},
        ]
        result = client.add(messages, user_id="alice")
        assert len(result["results"]) == 1
        assert "dark mode" in result["results"][0]["memory"]

    def test_search_finds_matching(self):
        """Test search returns matching memories."""
        client = InMemoryMem0Client()
        client.add("The user prefers Python for backend", user_id="alice")
        client.add("The project uses React for frontend", user_id="alice")

        results = client.search("Python", user_id="alice")
        assert len(results) == 1
        assert "Python" in results[0]["memory"]

    def test_search_empty(self):
        """Test search returns empty for no matches."""
        client = InMemoryMem0Client()
        client.add("The user likes Python", user_id="alice")

        results = client.search("Java", user_id="alice")
        assert len(results) == 0

    def test_search_respects_user_scope(self):
        """Test search filters by user_id."""
        client = InMemoryMem0Client()
        client.add("Alice likes Python", user_id="alice")
        client.add("Bob likes Java", user_id="bob")

        results = client.search("likes", user_id="alice")
        assert len(results) == 1
        assert "Python" in results[0]["memory"]

    def test_search_limit(self):
        """Test search respects limit parameter."""
        client = InMemoryMem0Client()
        for i in range(10):
            client.add(f"fact number {i}", user_id="alice")

        results = client.search("fact", user_id="alice", limit=3)
        assert len(results) == 3

    def test_get_all(self):
        """Test get_all returns all memories for scope."""
        client = InMemoryMem0Client()
        client.add("fact 1", user_id="alice")
        client.add("fact 2", user_id="alice")
        client.add("fact 3", user_id="bob")

        results = client.get_all(user_id="alice")
        assert len(results) == 2

    def test_get_all_empty(self):
        """Test get_all returns empty for unknown scope."""
        client = InMemoryMem0Client()
        results = client.get_all(user_id="nobody")
        assert len(results) == 0

    def test_delete(self):
        """Test deleting a memory."""
        client = InMemoryMem0Client()
        result = client.add("temporary fact", user_id="alice")
        memory_id = result["results"][0]["id"]

        client.delete(memory_id)
        results = client.get_all(user_id="alice")
        assert len(results) == 0

    def test_delete_nonexistent(self):
        """Test deleting nonexistent memory doesn't raise."""
        client = InMemoryMem0Client()
        result = client.delete("nonexistent_id")
        assert "message" in result

    def test_agent_id_scoping(self):
        """Test memories are scoped by agent_id."""
        client = InMemoryMem0Client()
        client.add("researcher fact", agent_id="researcher")
        client.add("coder fact", agent_id="coder")

        results = client.get_all(agent_id="researcher")
        assert len(results) == 1
        assert "researcher" in results[0]["memory"]

    def test_run_id_scoping(self):
        """Test memories are scoped by run_id."""
        client = InMemoryMem0Client()
        client.add("session 1 fact", run_id="run-1")
        client.add("session 2 fact", run_id="run-2")

        results = client.get_all(run_id="run-1")
        assert len(results) == 1
        assert "session 1" in results[0]["memory"]

    def test_add_with_metadata(self):
        """Test adding memory with metadata."""
        client = InMemoryMem0Client()
        result = client.add(
            "important fact",
            user_id="alice",
            metadata={"importance": "high", "source": "user"},
        )
        memory_id = result["results"][0]["id"]
        all_mems = client.get_all(user_id="alice")
        assert all_mems[0]["metadata"] == {"importance": "high", "source": "user"}


class TestMem0MemoryMiddlewareAgent:
    def test_agent_with_mem0_middleware(self):
        """Test that agent includes mem0 tools when middleware is added."""
        client = InMemoryMem0Client()
        middleware = [Mem0MemoryMiddleware(client=client)]
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=middleware, tools=[])
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        assert "remember" in agent_tools
        assert "recall" in agent_tools
        assert "forget" in agent_tools
        assert "list_all_memories" in agent_tools

    def test_agent_without_mem0_middleware(self):
        """Test that agent doesn't have mem0 tools without middleware."""
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=[], tools=[])
        if "tools" in agent.nodes:
            agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
            assert "remember" not in agent_tools
            assert "recall" not in agent_tools
            assert "forget" not in agent_tools
            assert "list_all_memories" not in agent_tools

    def test_mem0_alongside_base_memory(self):
        """Test Mem0 middleware works alongside base MemoryMiddleware."""
        from deepagents.middleware.memory import InMemoryStore, MemoryMiddleware

        client = InMemoryMem0Client()
        middleware = [
            MemoryMiddleware(store=InMemoryStore()),
            Mem0MemoryMiddleware(client=client),
        ]
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=middleware, tools=[])
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        # Base memory tools
        assert "save_memory" in agent_tools
        assert "get_memory" in agent_tools
        assert "list_memories" in agent_tools
        # Mem0 tools
        assert "remember" in agent_tools
        assert "recall" in agent_tools
        assert "forget" in agent_tools
        assert "list_all_memories" in agent_tools


class TestMem0SystemPrompt:
    def test_system_prompt_content(self):
        """Test that default system prompt mentions all tools."""
        assert "remember" in MEM0_SYSTEM_PROMPT
        assert "recall" in MEM0_SYSTEM_PROMPT
        assert "forget" in MEM0_SYSTEM_PROMPT
        assert "list_all_memories" in MEM0_SYSTEM_PROMPT
        assert "semantic" in MEM0_SYSTEM_PROMPT.lower()

    def test_custom_system_prompt(self):
        """Test that custom system prompt overrides default."""
        middleware = Mem0MemoryMiddleware(system_prompt="My custom prompt")
        assert middleware._custom_system_prompt == "My custom prompt"
