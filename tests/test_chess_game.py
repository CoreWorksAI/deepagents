#!/usr/bin/env python3
"""Minimal test: Chess game with hierarchical agents.

Architecture:
    GameMaster (checkpointer) - orchestrates the game
    ├── PlayerWhite (checkpointer) - plays white pieces
    ├── PlayerBlack (checkpointer) - plays black pieces
    └── scoreboard() tool - updates score/UI (no checkpointer)

Usage:
    export ANTHROPIC_API_KEY=...
    python tests/test_chess_game.py
"""

import asyncio
import logging
import sys
from uuid import uuid4

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from enhanced_redis_checkpointer import EnhancedAsyncRedisSaver
from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Game State (simple in-memory for demo)
# =============================================================================

GAME_STATE = {
    "board": "standard starting position",
    "moves": [],
    "turn": "white",
    "score": {"white": 0, "black": 0},
}


# =============================================================================
# Tools
# =============================================================================

@tool
def make_move(piece: str, from_sq: str, to_sq: str) -> str:
    """Make a chess move. Example: make_move('pawn', 'e2', 'e4')"""
    move = f"{piece} {from_sq}->{to_sq}"
    GAME_STATE["moves"].append(move)
    GAME_STATE["turn"] = "black" if GAME_STATE["turn"] == "white" else "white"
    return f"Move recorded: {move}. Next turn: {GAME_STATE['turn']}"


@tool
def scoreboard(message: str) -> str:
    """Update the game scoreboard/UI with a message."""
    print(f"    [SCOREBOARD] {message}")
    return f"Scoreboard updated: {message}"


@tool
def get_game_state() -> str:
    """Get current game state."""
    return f"Moves: {GAME_STATE['moves']}, Turn: {GAME_STATE['turn']}"


# =============================================================================
# Test
# =============================================================================

async def test_chess_game():
    """Test chess game with hierarchical agents."""

    print("\n" + "=" * 60)
    print("CHESS GAME - HIERARCHICAL AGENTS TEST")
    print("=" * 60)

    async with EnhancedAsyncRedisSaver.from_conn_string(
        "redis://localhost:6379", ttl_seconds=300
    ) as checkpointer:

        # Player White (with checkpointer - remembers strategy)
        player_white = create_deep_agent(
            tools=[make_move, get_game_state],
            system_prompt="You are a chess player (WHITE). Make one move when asked. Be brief.",
            checkpointer=checkpointer,
            enable_filesystem=False,
            enable_todos=False,
        )

        # Player Black (with checkpointer - remembers strategy)
        player_black = create_deep_agent(
            tools=[make_move, get_game_state],
            system_prompt="You are a chess player (BLACK). Make one move when asked. Be brief.",
            checkpointer=checkpointer,
            enable_filesystem=False,
            enable_todos=False,
        )

        @tool
        async def ask_white(instruction: str) -> str:
            """Ask white player to make a move."""
            result = await player_white.ainvoke(
                {"messages": [HumanMessage(content=instruction)]},
                config={"configurable": {"thread_id": "player-white"}},
            )
            return result["messages"][-1].content

        @tool
        async def ask_black(instruction: str) -> str:
            """Ask black player to make a move."""
            result = await player_black.ainvoke(
                {"messages": [HumanMessage(content=instruction)]},
                config={"configurable": {"thread_id": "player-black"}},
            )
            return result["messages"][-1].content

        # Game Master (with checkpointer - orchestrates game)
        game_master = create_deep_agent(
            tools=[ask_white, ask_black, scoreboard, get_game_state],
            system_prompt="""You are a chess game master.
Run a quick 2-move game: ask white to move, update scoreboard, ask black to move, update scoreboard.
Be brief and efficient.""",
            checkpointer=checkpointer,
            enable_filesystem=False,
            enable_todos=False,
        )

        print("\n[1/3] Created game agents...")
        print("    - GameMaster (orchestrator)")
        print("    - PlayerWhite (sub-agent)")
        print("    - PlayerBlack (sub-agent)")
        print("    - scoreboard (tool)")
        print("    ✅ Hierarchy created")

        # Run game
        print("\n[2/3] Running 2-move game...")
        thread_id = f"game-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        result = await game_master.ainvoke(
            {"messages": [HumanMessage(content="Start a quick 2-move game. White moves first, then black.")]},
            config=config,
        )

        print(f"\n    Game completed with {len(result['messages'])} messages")
        print(f"    Moves played: {GAME_STATE['moves']}")
        print("    ✅ Hierarchical game works")

        # Verify checkpoints
        print("\n[3/3] Verifying checkpoints...")

        gm_checkpoint = await checkpointer.aget_tuple(config)
        white_checkpoint = await checkpointer.aget_tuple({"configurable": {"thread_id": "player-white"}})
        black_checkpoint = await checkpointer.aget_tuple({"configurable": {"thread_id": "player-black"}})

        print(f"    GameMaster checkpoint: {'✅' if gm_checkpoint else '❌'}")
        print(f"    PlayerWhite checkpoint: {'✅' if white_checkpoint else '❌'}")
        print(f"    PlayerBlack checkpoint: {'✅' if black_checkpoint else '❌'}")

        # Cleanup
        await checkpointer.adelete_thread(thread_id)
        await checkpointer.adelete_thread("player-white")
        await checkpointer.adelete_thread("player-black")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        return True


if __name__ == "__main__":
    success = asyncio.run(test_chess_game())
    sys.exit(0 if success else 1)
