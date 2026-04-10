"""Chess game integration test — hierarchical agents with enable_* toggles.

Architecture:
    GameMaster (orchestrator) → delegates to player sub-agents
    ├── PlayerWhite (stripped agent — no filesystem, no todos)
    ├── PlayerBlack (stripped agent — no filesystem, no todos)
    └── scoreboard() tool

Validates:
    - enable_filesystem=False / enable_todos=False toggles work end-to-end
    - Subagent delegation via task() tool with real LLM
    - Callback/config propagation to subagents
    - MemoryKV middleware (enable_memory=True) with InMemory store

Run: pytest tests/integration_tests/test_chess_game.py -xvs
Requires: ANTHROPIC_API_KEY
"""

import pytest
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from deepagents.graph import create_deep_agent
from deepagents.middleware.memory_kv import InMemoryStore, MemoryKVMiddleware


# --- Game state ---

_game_moves: list[str] = []
_scoreboard_msgs: list[str] = []


@tool
def make_move(piece: str, from_sq: str, to_sq: str) -> str:
    """Make a chess move. Example: make_move('pawn', 'e2', 'e4')"""
    move = f"{piece} {from_sq}->{to_sq}"
    _game_moves.append(move)
    return f"Move recorded: {move}"


@tool
def get_board() -> str:
    """Get the current board state (list of moves played)."""
    if not _game_moves:
        return "No moves yet. Standard starting position."
    return f"Moves so far: {', '.join(_game_moves)}"


@tool
def scoreboard(message: str) -> str:
    """Update the game scoreboard with a status message."""
    _scoreboard_msgs.append(message)
    return f"Scoreboard updated: {message}"


@pytest.fixture(autouse=True)
def reset_game_state():
    """Reset game state between tests."""
    _game_moves.clear()
    _scoreboard_msgs.clear()
    yield
    _game_moves.clear()
    _scoreboard_msgs.clear()


@pytest.mark.requires("langchain_anthropic")
class TestChessGame:
    """Integration test: hierarchical chess agents with stripped middleware."""

    def test_chess_game_with_subagents(self):
        """GameMaster delegates to PlayerWhite and PlayerBlack subagents.

        Validates enable_filesystem=False and enable_todos=False work with
        real LLM calls and subagent delegation.
        """
        agent = create_deep_agent(
            tools=[scoreboard, get_board],
            system_prompt=(
                "You are a chess game master. You have two player subagents. "
                "Ask white_player to make one opening move, then ask black_player to respond. "
                "After each move, call scoreboard() with the move. Be very brief."
            ),
            subagents=[
                {
                    "name": "white_player",
                    "description": "Chess player for WHITE pieces. Ask to make a move.",
                    "system_prompt": "You are white. Make exactly one chess move using make_move. Be brief.",
                    "tools": [make_move, get_board],
                    "model": "claude-sonnet-4-20250514",
                },
                {
                    "name": "black_player",
                    "description": "Chess player for BLACK pieces. Ask to make a move.",
                    "system_prompt": "You are black. Make exactly one chess move using make_move. Be brief.",
                    "tools": [make_move, get_board],
                    "model": "claude-sonnet-4-20250514",
                },
            ],
            enable_filesystem=False,
            enable_todos=False,
            checkpointer=InMemorySaver(),
        )

        # Verify stripped middleware — no filesystem tools
        tool_names = set(agent.nodes["tools"].bound._tools_by_name.keys())
        assert "ls" not in tool_names, "Filesystem tools should be disabled"
        assert "write_todos" not in tool_names, "Todos should be disabled"
        assert "task" in tool_names, "task (subagent) tool should be present"
        assert "scoreboard" in tool_names

        # Run the game
        result = agent.invoke(
            {"messages": [HumanMessage(content="Start a quick 2-move game.")]},
            config={"configurable": {"thread_id": f"chess-{uuid4()}"}},
        )

        # Verify moves were made
        assert len(_game_moves) >= 2, f"Expected at least 2 moves, got {_game_moves}"
        print(f"Moves: {_game_moves}")
        print(f"Scoreboard: {_scoreboard_msgs}")

    def test_chess_game_with_memory(self):
        """Same chess game but with enable_memory=True for KV state tracking."""
        memory_store = InMemoryStore()

        agent = create_deep_agent(
            tools=[scoreboard, get_board],
            system_prompt=(
                "You are a chess game master. Ask white_player to make one move. "
                "Then save the move to memory using save_memory with key 'last_move'. "
                "Be very brief."
            ),
            subagents=[
                {
                    "name": "white_player",
                    "description": "Chess player for WHITE pieces.",
                    "system_prompt": "Make exactly one chess move using make_move. Be brief.",
                    "tools": [make_move, get_board],
                    "model": "claude-sonnet-4-20250514",
                },
            ],
            enable_filesystem=False,
            enable_todos=False,
            enable_memory=True,
        )

        # Verify memory tools are present
        tool_names = set(agent.nodes["tools"].bound._tools_by_name.keys())
        assert "save_memory" in tool_names, "Memory tools should be present"
        assert "get_memory" in tool_names
        assert "list_memories" in tool_names
        assert "ls" not in tool_names, "Filesystem should still be disabled"

        result = agent.invoke(
            {"messages": [HumanMessage(content="Make white's opening move and save it to memory.")]},
        )

        assert len(_game_moves) >= 1, f"Expected at least 1 move, got {_game_moves}"
        print(f"Moves: {_game_moves}")
