"""Unit tests for MemoryKVMiddleware tool registration.

Covers the invariants that `save_memory`, `get_memory`, and `list_memories`
must satisfy so the hosting agent can invoke them without schema or injection
errors:

- LLM-facing JSON schema exposes only model-controlled fields; `runtime`
  (a ToolRuntime) is injected, not advertised.
- `_injected_args_keys` detects `runtime` so the graph's ToolNode injects it.
- The end-to-end tool-call path through `create_deep_agent` succeeds and
  persists to the store.
"""

from collections.abc import Iterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from deepagents import create_deep_agent
from deepagents.middleware.memory_kv import (
    InMemoryStore,
    MemoryKVMiddleware,
    NullMemoryStore,
    RedisMemoryStore,
    get_memory_store,
)


def _schema_of(tool: Any) -> dict[str, Any]:
    schema = tool.args_schema
    return schema.model_json_schema() if hasattr(schema, "model_json_schema") else dict(schema)


def _tool(name: str) -> Any:
    mw = MemoryKVMiddleware(store=InMemoryStore())
    return next(t for t in mw.tools if t.name == name)


class _ToolCallingFakeModel(BaseChatModel):
    """Minimal chat model that scripts AIMessages and supports `bind_tools`."""

    messages: Iterator[AIMessage]

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ToolCallingFakeModel":  # noqa: ARG002
        return self

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:  # noqa: ARG002
        return ChatResult(generations=[ChatGeneration(message=next(self.messages))])

    async def _agenerate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:  # noqa: ARG002
        return ChatResult(generations=[ChatGeneration(message=next(self.messages))])

    @property
    def _llm_type(self) -> str:
        return "fake_tool_calling"


def _build_agent(scripted: list[AIMessage], store: InMemoryStore) -> Any:
    model = _ToolCallingFakeModel(messages=iter(scripted))
    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt="test",
        enable_memory=False,
        enable_filesystem=False,
        enable_todos=False,
        middleware=[MemoryKVMiddleware(store=store)],
    )


def test_save_memory_schema_exposes_key_and_data_without_runtime() -> None:
    """save_memory advertises key+data to the LLM; runtime stays injected."""
    js = _schema_of(_tool("save_memory"))
    props = set((js.get("properties") or {}).keys())
    required = set(js.get("required") or [])
    assert {"key", "data"}.issubset(props), f"missing key/data in schema; got props={props}"
    assert "runtime" not in props, f"runtime must not appear in LLM schema; got props={props}"
    assert {"key", "data"}.issubset(required), f"key+data must be required; got required={required}"


def test_get_memory_schema_exposes_key_without_runtime() -> None:
    js = _schema_of(_tool("get_memory"))
    props = set((js.get("properties") or {}).keys())
    required = set(js.get("required") or [])
    assert "key" in props
    assert "runtime" not in props
    assert "key" in required


def test_list_memories_schema_accepts_optional_prefix_without_runtime() -> None:
    """list_memories exposes an optional prefix filter; runtime stays injected."""
    js = _schema_of(_tool("list_memories"))
    props = set((js.get("properties") or {}).keys())
    required = set(js.get("required") or [])
    assert "prefix" in props, f"expected optional prefix field; got props={props}"
    assert "runtime" not in props
    assert "prefix" not in required, f"prefix must be optional; got required={required}"


def test_tools_declare_runtime_as_injected() -> None:
    """Without this, ToolNode will not construct/inject ToolRuntime at call time."""
    for name in ("save_memory", "get_memory", "list_memories"):
        tool = _tool(name)
        assert "runtime" in tool._injected_args_keys, (
            f"{name}._injected_args_keys must contain 'runtime'; got {tool._injected_args_keys}"
        )


async def test_save_memory_persists_via_create_deep_agent() -> None:
    """End-to-end: LLM emits save_memory → ToolNode → store.put."""
    store = InMemoryStore()
    ai_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "save_memory",
            "args": {"key": "topic", "data": {"title": "chess"}},
            "id": "call_save",
            "type": "tool_call",
        }],
    )
    ai_done = AIMessage(content="saved.")
    agent = _build_agent([ai_call, ai_done], store)

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="save")]},
        {"configurable": {"conversation_id": "conv-save", "thread_id": "t-save"}},
    )

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs, "expected a ToolMessage for save_memory"
    assert "Saved" in str(tool_msgs[0].content)
    assert await store.get("conv-save", "topic") == {"title": "chess"}


async def test_get_memory_reads_via_create_deep_agent() -> None:
    """End-to-end: LLM emits get_memory → ToolNode → store.get, returns payload."""
    store = InMemoryStore()
    await store.put("conv-get", "topic", {"title": "chess"})
    ai_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "get_memory",
            "args": {"key": "topic"},
            "id": "call_get",
            "type": "tool_call",
        }],
    )
    ai_done = AIMessage(content="read.")
    agent = _build_agent([ai_call, ai_done], store)

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="read")]},
        {"configurable": {"conversation_id": "conv-get", "thread_id": "t-get"}},
    )

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs, "expected a ToolMessage for get_memory"
    assert "chess" in str(tool_msgs[0].content)


async def test_list_memories_via_create_deep_agent() -> None:
    """End-to-end: LLM emits list_memories → ToolNode → returns all keys."""
    store = InMemoryStore()
    await store.put("conv-list", "alpha", {"v": 1})
    await store.put("conv-list", "beta", {"v": 2})
    ai_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "list_memories",
            "args": {},
            "id": "call_list",
            "type": "tool_call",
        }],
    )
    ai_done = AIMessage(content="listed.")
    agent = _build_agent([ai_call, ai_done], store)

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="list")]},
        {"configurable": {"conversation_id": "conv-list", "thread_id": "t-list"}},
    )

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs, "expected a ToolMessage for list_memories"
    body = str(tool_msgs[0].content)
    assert "alpha" in body and "beta" in body


async def test_get_memory_store_falls_back_to_null_without_redis_url(
    caplog: Any, monkeypatch: Any,
) -> None:
    """Without REDIS_URL, `get_memory_store` returns a NullMemoryStore and warns."""
    import logging as _logging
    monkeypatch.delenv("REDIS_URL", raising=False)
    with caplog.at_level(_logging.WARNING, logger="deepagents.middleware.memory_kv"):
        store = await get_memory_store()
    assert isinstance(store, NullMemoryStore)
    assert any("REDIS_URL not set" in r.message for r in caplog.records), (
        f"expected a warning about missing REDIS_URL; got {[r.message for r in caplog.records]}"
    )


async def test_get_memory_store_falls_back_to_null_when_redis_unreachable(
    caplog: Any, monkeypatch: Any,
) -> None:
    """With REDIS_URL pointing at an unreachable host, falls back to Null + warns."""
    import logging as _logging
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1")  # port 1 = unreachable
    with caplog.at_level(_logging.WARNING, logger="deepagents.middleware.memory_kv"):
        store = await get_memory_store()
    assert isinstance(store, NullMemoryStore)
    assert any("unreachable" in r.message for r in caplog.records), (
        f"expected a warning about Redis being unreachable; got {[r.message for r in caplog.records]}"
    )


async def test_null_store_tools_do_not_crash_via_create_deep_agent() -> None:
    """With NullMemoryStore, tools stay callable — they just don't persist."""
    store = NullMemoryStore()
    ai_save = AIMessage(
        content="",
        tool_calls=[{
            "name": "save_memory",
            "args": {"key": "k", "data": {"a": 1}},
            "id": "call_save",
            "type": "tool_call",
        }],
    )
    ai_done = AIMessage(content="ok.")
    agent = _build_agent([ai_save, ai_done], store)

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="save")]},
        {"configurable": {"conversation_id": "null-store", "thread_id": "t"}},
    )
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs, "expected a ToolMessage even when the store is a no-op"
    # Nothing was persisted.
    assert await store.get("null-store", "k") is None


async def test_list_memories_prefix_filter_via_create_deep_agent() -> None:
    """Optional prefix filters the returned keys."""
    store = InMemoryStore()
    await store.put("conv-prefix", "alpha_one", {"v": 1})
    await store.put("conv-prefix", "alpha_two", {"v": 2})
    await store.put("conv-prefix", "other", {"v": 3})
    ai_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "list_memories",
            "args": {"prefix": "alpha_"},
            "id": "call_list_prefix",
            "type": "tool_call",
        }],
    )
    ai_done = AIMessage(content="listed.")
    agent = _build_agent([ai_call, ai_done], store)

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="list")]},
        {"configurable": {"conversation_id": "conv-prefix", "thread_id": "t-prefix"}},
    )

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs, "expected a ToolMessage for list_memories"
    body = str(tool_msgs[0].content)
    assert "alpha_one" in body and "alpha_two" in body
    assert "other" not in body
