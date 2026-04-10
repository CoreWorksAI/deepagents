"""Chess game with AsyncSubAgents — remote agent pattern.

Architecture:
    GameMaster (local orchestrator)
    ├── white_player → AsyncSubAgent (remote via Agent Protocol HTTP)
    ├── black_player → AsyncSubAgent (remote via Agent Protocol HTTP)
    └── scoreboard() tool (local)

This test demonstrates the AsyncSubAgentMiddleware with:
- start_async_task / check_async_task / list_async_tasks tools
- Remote agents running on a LangGraph Platform-compatible server
- enable_filesystem=False / enable_todos=False on the orchestrator

NOTE: This test requires a running Agent Protocol server.
      Skip with: pytest -k "not async_chess"

Run:
    # Terminal 1: start the player server (see examples/async-subagent-server/)
    python examples/async-subagent-server/server.py

    # Terminal 2: run this test
    ANTHROPIC_API_KEY=... pytest tests/integration_tests/test_chess_game_async.py -xvs
"""

import os

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from deepagents.graph import create_deep_agent
from deepagents.middleware.async_subagents import AsyncSubAgent


# --- Shared game state ---

_game_moves: list[str] = []
_scoreboard_msgs: list[str] = []


@tool
def scoreboard(message: str) -> str:
    """Update the game scoreboard with a status message."""
    _scoreboard_msgs.append(message)
    return f"Scoreboard updated: {message}"


@tool
def get_board() -> str:
    """Get the current board state (list of moves played)."""
    if not _game_moves:
        return "No moves yet. Standard starting position."
    return f"Moves so far: {', '.join(_game_moves)}"


@pytest.fixture(autouse=True)
def reset_game_state():
    _game_moves.clear()
    _scoreboard_msgs.clear()
    yield


# The async subagent server URL — skip tests if not set
ASYNC_SERVER_URL = os.getenv("CHESS_ASYNC_SERVER_URL")


@pytest.mark.skipif(
    not ASYNC_SERVER_URL,
    reason="CHESS_ASYNC_SERVER_URL not set — no async subagent server running",
)
@pytest.mark.requires("langchain_anthropic")
class TestChessGameAsync:
    """Integration test: chess with async (remote) subagents.

    Requires a running Agent Protocol server at CHESS_ASYNC_SERVER_URL.
    The server should expose a graph named 'chess_player' that accepts
    messages and returns chess moves.
    """

    def test_async_chess_game(self):
        """GameMaster uses async subagents to coordinate remote chess players.

        This tests the full AsyncSubAgentMiddleware flow:
        1. start_async_task → launches remote agent
        2. check_async_task → polls for completion
        3. Result returned to orchestrator
        """
        agent = create_deep_agent(
            tools=[scoreboard, get_board],
            system_prompt=(
                "You are a chess game master coordinating remote players. "
                "Start an async task for white_player to make an opening move. "
                "Wait briefly, then check the task status. "
                "When done, update the scoreboard. Be brief."
            ),
            subagents=[
                AsyncSubAgent(
                    name="white_player",
                    description="Remote chess player for WHITE. Launches an async task to make a move.",
                    graph_id="chess_player",
                    url=ASYNC_SERVER_URL,
                ),
                AsyncSubAgent(
                    name="black_player",
                    description="Remote chess player for BLACK. Launches an async task to make a move.",
                    graph_id="chess_player",
                    url=ASYNC_SERVER_URL,
                ),
            ],
            enable_filesystem=False,
            enable_todos=False,
        )

        # Verify async tools present, sync tools absent
        tool_names = set(agent.nodes["tools"].bound._tools_by_name.keys())
        assert "start_async_task" in tool_names, "Async subagent tools should be present"
        assert "check_async_task" in tool_names
        assert "list_async_tasks" in tool_names
        assert "ls" not in tool_names, "Filesystem should be disabled"
        assert "write_todos" not in tool_names, "Todos should be disabled"
        # No sync task tool — only async subagents provided + auto general-purpose
        assert "task" in tool_names, "Sync task tool should still exist for general-purpose"

        result = agent.invoke(
            {"messages": [HumanMessage(content="Start a game. White moves first.")]},
        )

        # Verify the orchestrator used async tools
        ai_messages = [m for m in result["messages"] if m.type == "ai"]
        tool_calls = [tc for m in ai_messages for tc in m.tool_calls]
        async_calls = [tc for tc in tool_calls if tc["name"] == "start_async_task"]
        assert len(async_calls) >= 1, f"Expected at least 1 start_async_task call, got {tool_calls}"
        print(f"Async tool calls: {[tc['name'] for tc in tool_calls]}")


@pytest.mark.requires("langchain_anthropic")
class TestAsyncSubAgentToolPresence:
    """Unit-level checks for async subagent middleware wiring.

    These don't need a running server — they just verify the agent
    is assembled correctly.
    """

    def test_async_subagent_tools_added(self):
        """Verify async subagent tools are registered when AsyncSubAgent specs given."""
        from langchain_core.messages import AIMessage
        from tests.unit_tests.chat_model import GenericFakeChatModel

        agent = create_deep_agent(
            model=GenericFakeChatModel(messages=iter([AIMessage(content="done")])),
            subagents=[
                {
                    "name": "remote_worker",
                    "description": "A remote worker",
                    "graph_id": "worker_graph",
                    "url": "http://localhost:9999",
                },
            ],
            enable_filesystem=False,
            enable_todos=False,
        )

        tool_names = set(agent.nodes["tools"].bound._tools_by_name.keys())
        # Async tools
        assert "start_async_task" in tool_names
        assert "check_async_task" in tool_names
        assert "update_async_task" in tool_names
        assert "cancel_async_task" in tool_names
        assert "list_async_tasks" in tool_names
        # Stripped
        assert "ls" not in tool_names
        assert "write_todos" not in tool_names
        # General-purpose sync subagent still present (auto-added)
        assert "task" in tool_names

    def test_mixed_sync_and_async_subagents(self):
        """Verify both sync SubAgent and AsyncSubAgent can coexist."""
        from langchain_core.messages import AIMessage
        from tests.unit_tests.chat_model import GenericFakeChatModel

        agent = create_deep_agent(
            model=GenericFakeChatModel(messages=iter([AIMessage(content="done")])),
            subagents=[
                {
                    "name": "local_analyzer",
                    "description": "Local analysis agent",
                    "system_prompt": "Analyze data.",
                    "model": GenericFakeChatModel(messages=iter([AIMessage(content="analysis done")])),
                    "tools": [],
                },
                {
                    "name": "remote_researcher",
                    "description": "Remote research agent",
                    "graph_id": "research_graph",
                    "url": "http://localhost:9999",
                },
            ],
            enable_filesystem=False,
            enable_todos=False,
        )

        tool_names = set(agent.nodes["tools"].bound._tools_by_name.keys())
        assert "task" in tool_names, "Sync task tool for local_analyzer + general-purpose"
        assert "start_async_task" in tool_names, "Async tools for remote_researcher"
        assert "ls" not in tool_names
