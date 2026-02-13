#!/usr/bin/env python3
"""Chess game with hierarchical agents and Mem0 semantic memory.

This example extends the base chess game to demonstrate how Mem0MemoryMiddleware
adds semantic memory to agents -- players remember strategies, learn from past
games, and the GameMaster tracks player tendencies.

Architecture:
    GameMaster (checkpointer + Mem0 memory) - orchestrates the game
    ├── PlayerWhite (checkpointer + Mem0 memory) - plays white, remembers strategy
    ├── PlayerBlack (checkpointer + Mem0 memory) - plays black, remembers strategy
    └── scoreboard() tool - updates score/UI (no checkpointer)

Memory Flow:
    1. Players use `recall` before each move to check what they know about
       the opponent's tendencies and their own strategy notes.
    2. After each move, players use `remember` to store strategic observations.
    3. The GameMaster uses `remember` to track game patterns and player styles.
    4. In subsequent games (same user_id), agents retrieve and build on past memories.

Setup:
    # 1. Install dependencies
    pip install mem0ai redis

    # 2. Set environment variables
    export ANTHROPIC_API_KEY=sk-ant-...     # For Claude (agent LLM)
    export OPENAI_API_KEY=sk-...            # For Mem0 (fact extraction + embeddings)
    export REDIS_URL=redis://localhost:6379  # Optional: for Redis checkpointer

    # 3. Start Redis (for checkpointing)
    docker run -d -p 6379:6379 redis:latest

    # 4. Run
    python tests/test_chess_game_with_memory.py

What this demonstrates:
    - Mem0 `remember` tool: automatic fact extraction from natural language
    - Mem0 `recall` tool: semantic search for relevant memories
    - Memory scoping: user_id (player identity) + agent_id (agent role)
    - Memory persistence across multiple game sessions
    - Coexistence with Redis checkpointer for state management
"""

import asyncio
import logging
import os
import sys
from uuid import uuid4

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from deepagents import create_deep_agent
from deepagents.middleware.mem0_memory import InMemoryMem0Client, Mem0MemoryMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

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
    "game_number": 1,
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
    """Get current game state including move history."""
    return (
        f"Game #{GAME_STATE['game_number']}, "
        f"Moves: {GAME_STATE['moves']}, "
        f"Turn: {GAME_STATE['turn']}, "
        f"Score: {GAME_STATE['score']}"
    )


# =============================================================================
# Helper: Create agents with Mem0 memory
# =============================================================================

def create_memory_chess_agents(
    checkpointer=None,
    mem0_client=None,
):
    """Create chess agents with Mem0 semantic memory.

    Args:
        checkpointer: Optional Redis/InMemory checkpointer for state persistence.
        mem0_client: Optional Mem0 client. Defaults to InMemoryMem0Client for
            local testing, or real Mem0 client if mem0ai is installed and
            OPENAI_API_KEY is set.

    Returns:
        Tuple of (game_master, player_white_tools, player_black_tools) where
        the tools include ask_white/ask_black wrappers.
    """
    # Use InMemoryMem0Client for demo/testing, or real Mem0 if configured
    if mem0_client is None:
        mem0_client = InMemoryMem0Client()

    # -- Player White --
    # Gets its own Mem0 middleware instance (tools are scoped by agent_id at runtime)
    white_memory = Mem0MemoryMiddleware(
        client=mem0_client,
        system_prompt="""## Memory
You have semantic memory tools. Before each move:
1. Use `recall` to check what you know about the opponent's play style.
2. After your move, use `remember` to note your strategy or observations.

Example:
- recall("opponent tendencies") -> finds past observations
- remember("Opponent favors the Sicilian Defense. I should prepare e4 lines.")
""",
    )

    player_white = create_deep_agent(
        tools=[make_move, get_game_state],
        system_prompt=(
            "You are a chess player (WHITE). You have a strategic, aggressive style. "
            "Use your memory to learn from past games and adapt your strategy. "
            "Make one move when asked. Be brief but note your reasoning."
        ),
        middleware=[white_memory],
        checkpointer=checkpointer,
        enable_filesystem=False,
        enable_todos=False,
    )

    # -- Player Black --
    black_memory = Mem0MemoryMiddleware(
        client=mem0_client,
        system_prompt="""## Memory
You have semantic memory tools. Before each move:
1. Use `recall` to check what you know about the opponent's play style.
2. After your move, use `remember` to note your strategy or observations.

Example:
- recall("opponent's opening") -> finds past observations
- remember("White opened with e4 again. They seem to prefer King's Pawn openings.")
""",
    )

    player_black = create_deep_agent(
        tools=[make_move, get_game_state],
        system_prompt=(
            "You are a chess player (BLACK). You have a defensive, positional style. "
            "Use your memory to learn from past games and adapt your strategy. "
            "Make one move when asked. Be brief but note your reasoning."
        ),
        middleware=[black_memory],
        checkpointer=checkpointer,
        enable_filesystem=False,
        enable_todos=False,
    )

    return player_white, player_black, mem0_client


# =============================================================================
# Demo: Run games with memory
# =============================================================================

async def run_chess_with_memory():
    """Run chess games demonstrating Mem0 memory across sessions."""

    print("\n" + "=" * 70)
    print("CHESS GAME WITH MEM0 SEMANTIC MEMORY")
    print("=" * 70)

    # Shared Mem0 client (in production, this would be backed by a real vector store)
    mem0_client = InMemoryMem0Client()

    # Create players
    player_white, player_black, _ = create_memory_chess_agents(
        mem0_client=mem0_client,
    )

    # Create tools for GameMaster to invoke players
    @tool
    async def ask_white(instruction: str) -> str:
        """Ask white player to make a move. They will consult their memory first."""
        result = await player_white.ainvoke(
            {"messages": [HumanMessage(content=instruction)]},
            config={"configurable": {
                "thread_id": "player-white",
                "user_id": "player-white",
                "agent_id": "chess-white",
            }},
        )
        return result["messages"][-1].content

    @tool
    async def ask_black(instruction: str) -> str:
        """Ask black player to make a move. They will consult their memory first."""
        result = await player_black.ainvoke(
            {"messages": [HumanMessage(content=instruction)]},
            config={"configurable": {
                "thread_id": "player-black",
                "user_id": "player-black",
                "agent_id": "chess-black",
            }},
        )
        return result["messages"][-1].content

    # -- GameMaster with its own Mem0 memory --
    gm_memory = Mem0MemoryMiddleware(
        client=mem0_client,
        system_prompt="""## Memory
You have semantic memory tools. Use them to:
- `remember` observations about player styles after each game
- `recall` past game patterns before starting a new game
- Track which openings each player prefers
""",
    )

    game_master = create_deep_agent(
        tools=[ask_white, ask_black, scoreboard, get_game_state],
        system_prompt=(
            "You are a chess game master. Orchestrate the game between White and Black.\n"
            "Run a quick 2-move game: ask white to move, update scoreboard, ask black "
            "to move, update scoreboard.\n"
            "After the game, use `remember` to note observations about the players.\n"
            "Be brief and efficient."
        ),
        middleware=[gm_memory],
        enable_filesystem=False,
        enable_todos=False,
    )

    # =====================
    # Game 1
    # =====================
    print("\n[GAME 1] Starting first game...")
    print("-" * 40)

    thread_id_1 = f"game-{uuid4()}"
    result = await game_master.ainvoke(
        {"messages": [HumanMessage(
            content="Start a quick 2-move game. White moves first, then black. "
                    "After the game, remember observations about each player's style."
        )]},
        config={"configurable": {
            "thread_id": thread_id_1,
            "user_id": "game-master",
            "agent_id": "chess-gm",
        }},
    )

    print(f"\n    Game 1 completed with {len(result['messages'])} messages")
    print(f"    Moves played: {GAME_STATE['moves']}")

    # Show what got stored in memory
    print("\n    [MEMORY STATE after Game 1]")
    all_memories = mem0_client.get_all()  # Get all memories across all scopes
    for mem in all_memories:
        scope = mem.get("user_id") or mem.get("agent_id") or "global"
        print(f"      [{scope}] {mem['memory'][:80]}...")

    # =====================
    # Game 2 (new thread, same memories)
    # =====================
    print("\n" + "=" * 70)
    print("[GAME 2] Starting second game (agents retain memories from Game 1)...")
    print("-" * 40)

    GAME_STATE["moves"] = []
    GAME_STATE["turn"] = "white"
    GAME_STATE["game_number"] = 2

    thread_id_2 = f"game-{uuid4()}"
    result = await game_master.ainvoke(
        {"messages": [HumanMessage(
            content="Start game 2. Before each move, ask players to recall what they "
                    "learned from the previous game. White moves first, then black."
        )]},
        config={"configurable": {
            "thread_id": thread_id_2,
            "user_id": "game-master",
            "agent_id": "chess-gm",
        }},
    )

    print(f"\n    Game 2 completed with {len(result['messages'])} messages")
    print(f"    Moves played: {GAME_STATE['moves']}")

    # Final memory state
    print("\n    [MEMORY STATE after Game 2]")
    all_memories = mem0_client.get_all()
    for mem in all_memories:
        scope = mem.get("user_id") or mem.get("agent_id") or "global"
        print(f"      [{scope}] {mem['memory'][:80]}...")

    print(f"\n    Total memories stored: {len(all_memories)}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    return True


# =============================================================================
# Simpler standalone demo (no external dependencies)
# =============================================================================

async def run_simple_memory_demo():
    """Minimal demo showing Mem0 memory tools in action without API calls.

    This demo directly exercises the InMemoryMem0Client to show the
    memory lifecycle without requiring any API keys or external services.
    """
    print("\n" + "=" * 70)
    print("MEM0 MEMORY TOOLS - STANDALONE DEMO")
    print("=" * 70)

    client = InMemoryMem0Client()

    # Simulate what the `remember` tool does internally
    print("\n[1] Storing memories (what `remember` does)...")
    client.add("White player opens with e4 consistently", user_id="game-master", agent_id="chess-gm")
    client.add("Black player prefers the Sicilian Defense (c5)", user_id="game-master", agent_id="chess-gm")
    client.add("White is aggressive in the middlegame", user_id="game-master", agent_id="chess-gm")
    client.add("Black plays positionally and avoids early confrontation", user_id="game-master", agent_id="chess-gm")
    print("    Stored 4 observations about players")

    # Simulate what the `recall` tool does
    print("\n[2] Searching memories (what `recall` does)...")
    results = client.search("opening preferences", user_id="game-master", agent_id="chess-gm")
    print(f"    Query: 'opening preferences' -> {len(results)} results")
    for r in results:
        print(f"      [{r['id']}] (score: {r['score']:.2f}) {r['memory']}")

    results = client.search("aggressive", user_id="game-master", agent_id="chess-gm")
    print(f"\n    Query: 'aggressive' -> {len(results)} results")
    for r in results:
        print(f"      [{r['id']}] (score: {r['score']:.2f}) {r['memory']}")

    # Simulate what `list_all_memories` does
    print("\n[3] Listing all memories (what `list_all_memories` does)...")
    all_mems = client.get_all(user_id="game-master", agent_id="chess-gm")
    print(f"    Total: {len(all_mems)} memories")
    for m in all_mems:
        print(f"      [{m['id']}] {m['memory']}")

    # Simulate what `forget` does
    print("\n[4] Deleting outdated memory (what `forget` does)...")
    first_id = all_mems[0]["id"]
    client.delete(first_id)
    remaining = client.get_all(user_id="game-master", agent_id="chess-gm")
    print(f"    Deleted {first_id}, remaining: {len(remaining)} memories")

    # Show scoping
    print("\n[5] Memory scoping demo...")
    client.add("I prefer the King's Indian", user_id="player-black")
    client.add("I always open with d4", user_id="player-white")

    black_mems = client.get_all(user_id="player-black")
    white_mems = client.get_all(user_id="player-white")
    gm_mems = client.get_all(user_id="game-master", agent_id="chess-gm")
    print(f"    Player Black memories: {len(black_mems)}")
    print(f"    Player White memories: {len(white_mems)}")
    print(f"    GameMaster memories:   {len(gm_mems)}")
    print("    -> Memories are isolated by scope!")

    print("\n" + "=" * 70)
    print("STANDALONE DEMO COMPLETE")
    print("=" * 70)
    return True


if __name__ == "__main__":
    # Run the standalone demo (no API keys needed)
    print("Running standalone memory demo (no API keys required)...")
    asyncio.run(run_simple_memory_demo())

    # Optionally run the full agent demo (requires ANTHROPIC_API_KEY)
    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n\nANTHROPIC_API_KEY detected, running full agent demo...")
        asyncio.run(run_chess_with_memory())
    else:
        print("\n\nSkipping full agent demo (set ANTHROPIC_API_KEY to run)")
