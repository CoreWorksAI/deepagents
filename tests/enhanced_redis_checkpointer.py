#!/usr/bin/env python3
"""Enhanced Redis checkpointer with checkpoint history and ACID-like guarantees.

Features:
- Checkpoint history with alist() support
- ACID-like transactions using MULTI/EXEC
- Filtering by metadata, source, step
- No RedisJSON/RediSearch modules required

Usage:
    from enhanced_redis_checkpointer import EnhancedAsyncRedisSaver

    async with EnhancedAsyncRedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
        agent = create_deep_agent(..., checkpointer=checkpointer)
"""

import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple, Type, Union

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
)
import redis.asyncio as redis

logger = logging.getLogger(__name__)


# Lua script for atomic checkpoint storage
# Ensures all keys are written together or none at all
LUA_ATOMIC_CHECKPOINT = """
local checkpoint_key = KEYS[1]
local latest_key = KEYS[2]
local history_key = KEYS[3]
local checkpoint_data = ARGV[1]
local checkpoint_id = ARGV[2]
local score = ARGV[3]
local ttl = tonumber(ARGV[4])

-- Store checkpoint
redis.call('SET', checkpoint_key, checkpoint_data)

-- Update latest pointer
redis.call('SET', latest_key, checkpoint_id)

-- Add to history sorted set (score = timestamp)
redis.call('ZADD', history_key, score, checkpoint_id)

-- Apply TTL if configured
if ttl > 0 then
    redis.call('EXPIRE', checkpoint_key, ttl)
    redis.call('EXPIRE', latest_key, ttl)
    redis.call('EXPIRE', history_key, ttl)
end

return 1
"""

# Lua script for atomic delete
LUA_ATOMIC_DELETE = """
local thread_pattern = KEYS[1]
local keys = redis.call('KEYS', thread_pattern)
for i, key in ipairs(keys) do
    redis.call('DEL', key)
end
return #keys
"""


class EnhancedAsyncRedisSaver(BaseCheckpointSaver):
    """Enhanced async Redis checkpointer with history and ACID-like guarantees.

    Features:
    - Full checkpoint history via sorted sets
    - ACID-like transactions using Lua scripts
    - Filter by metadata, source, step
    - No RedisJSON/RediSearch required
    """

    CHECKPOINT_PREFIX = "ckpt"
    WRITES_PREFIX = "ckpt_w"
    HISTORY_PREFIX = "ckpt_hist"
    LATEST_PREFIX = "ckpt_latest"

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[redis.Redis] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._redis_url = redis_url
        self._redis: Optional[redis.Redis] = redis_client
        self._owns_client = redis_client is None
        self.ttl_seconds = ttl_seconds or 0
        self._atomic_checkpoint_sha: Optional[str] = None
        self._atomic_delete_sha: Optional[str] = None

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        redis_url: str = "redis://localhost:6379",
        ttl_seconds: Optional[int] = None,
    ) -> AsyncIterator["EnhancedAsyncRedisSaver"]:
        """Create checkpointer from connection string."""
        saver = cls(redis_url=redis_url, ttl_seconds=ttl_seconds)
        async with saver:
            yield saver

    async def __aenter__(self) -> "EnhancedAsyncRedisSaver":
        if self._redis is None:
            self._redis = redis.Redis.from_url(
                self._redis_url or "redis://localhost:6379",
                decode_responses=False,  # Keep binary for proper handling
            )
        await self._redis.ping()

        # Register Lua scripts for atomic operations
        self._atomic_checkpoint_sha = await self._redis.script_load(LUA_ATOMIC_CHECKPOINT)
        self._atomic_delete_sha = await self._redis.script_load(LUA_ATOMIC_DELETE)

        logger.info(f"Connected to Redis at {self._redis_url}")
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._owns_client and self._redis:
            await self._redis.aclose()

    def _key(self, *parts: str) -> str:
        """Build a Redis key from parts."""
        return ":".join(str(p) for p in parts)

    def _checkpoint_key(self, thread_id: str, ns: str, checkpoint_id: str) -> str:
        return self._key(self.CHECKPOINT_PREFIX, thread_id, ns or "_", checkpoint_id)

    def _latest_key(self, thread_id: str, ns: str) -> str:
        return self._key(self.LATEST_PREFIX, thread_id, ns or "_")

    def _history_key(self, thread_id: str, ns: str) -> str:
        return self._key(self.HISTORY_PREFIX, thread_id, ns or "_")

    def _writes_key(self, thread_id: str, ns: str, checkpoint_id: str, task_id: str, idx: int) -> str:
        return self._key(self.WRITES_PREFIX, thread_id, ns or "_", checkpoint_id, task_id, idx)

    def _writes_set_key(self, thread_id: str, ns: str, checkpoint_id: str) -> str:
        return self._key(self.WRITES_PREFIX, "set", thread_id, ns or "_", checkpoint_id)

    def _serialize_value(self, value: Any) -> Dict[str, str]:
        """Serialize a value to JSON-safe dict with base64 encoding."""
        type_, data = self.serde.dumps_typed(value)
        return {
            "type": type_,
            "data": base64.b64encode(data).decode("utf-8") if isinstance(data, bytes) else data,
        }

    def _deserialize_value(self, serialized: Dict[str, Any]) -> Any:
        """Deserialize a value from JSON-safe dict."""
        data = serialized["data"]
        if isinstance(data, str):
            data = base64.b64decode(data)
        return self.serde.loads_typed((serialized["type"], data))

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store checkpoint with ACID-like guarantees using Lua script."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        # Serialize channel values
        serialized_channel_values = {
            k: self._serialize_value(v)
            for k, v in checkpoint.get("channel_values", {}).items()
        }

        # Build checkpoint data with metadata for filtering
        checkpoint_data = {
            "v": checkpoint.get("v", 1),
            "id": checkpoint_id,
            "ts": checkpoint.get("ts", ""),
            "channel_values": serialized_channel_values,
            "channel_versions": checkpoint.get("channel_versions", {}),
            "versions_seen": checkpoint.get("versions_seen", {}),
            "parent_checkpoint_id": parent_checkpoint_id,
            "metadata": metadata,
            # Denormalized fields for filtering
            "source": metadata.get("source", ""),
            "step": metadata.get("step", 0),
        }

        # Calculate score for sorted set (timestamp in ms)
        score = int(time.time() * 1000)

        # Execute atomic Lua script
        checkpoint_key = self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        latest_key = self._latest_key(thread_id, checkpoint_ns)
        history_key = self._history_key(thread_id, checkpoint_ns)

        await self._redis.evalsha(
            self._atomic_checkpoint_sha,
            3,  # number of keys
            checkpoint_key,
            latest_key,
            history_key,
            json.dumps(checkpoint_data, default=str),
            checkpoint_id,
            str(score),
            str(self.ttl_seconds),
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store pending writes atomically using MULTI/EXEC transaction."""
        if not writes:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        writes_set_key = self._writes_set_key(thread_id, checkpoint_ns, checkpoint_id)

        # Use MULTI/EXEC for atomic write of all writes
        async with self._redis.pipeline(transaction=True) as pipe:
            for idx, (channel, value) in enumerate(writes):
                write_data = {
                    "task_id": task_id,
                    "task_path": task_path,
                    "idx": WRITES_IDX_MAP.get(channel, idx),
                    "channel": channel,
                    **self._serialize_value(value),
                }

                write_key = self._writes_key(
                    thread_id, checkpoint_ns, checkpoint_id, task_id, idx
                )

                pipe.set(write_key, json.dumps(write_data, default=str))
                pipe.sadd(writes_set_key, write_key)

                if self.ttl_seconds > 0:
                    pipe.expire(write_key, self.ttl_seconds)

            if self.ttl_seconds > 0:
                pipe.expire(writes_set_key, self.ttl_seconds)

            # Execute all commands atomically
            await pipe.execute()

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        if not checkpoint_id:
            # Get latest checkpoint ID from pointer
            latest_key = self._latest_key(thread_id, checkpoint_ns)
            checkpoint_id = await self._redis.get(latest_key)
            if checkpoint_id:
                checkpoint_id = checkpoint_id.decode() if isinstance(checkpoint_id, bytes) else checkpoint_id
            else:
                return None

        # Get checkpoint data
        checkpoint_key = self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        data = await self._redis.get(checkpoint_key)

        if not data:
            return None

        checkpoint_data = json.loads(data)

        # Deserialize channel values
        channel_values = {
            k: self._deserialize_value(v)
            for k, v in checkpoint_data.get("channel_values", {}).items()
        }

        # Build checkpoint
        checkpoint: Checkpoint = {
            "v": checkpoint_data.get("v", 1),
            "id": checkpoint_id,
            "ts": checkpoint_data.get("ts", ""),
            "channel_values": channel_values,
            "channel_versions": checkpoint_data.get("channel_versions", {}),
            "versions_seen": checkpoint_data.get("versions_seen", {}),
            "pending_sends": [],
        }

        # Load pending writes
        pending_writes = await self._load_pending_writes(thread_id, checkpoint_ns, checkpoint_id)

        parent_checkpoint_id = checkpoint_data.get("parent_checkpoint_id")
        parent_config = None
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=checkpoint_data.get("metadata", {}),
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints with filtering support.

        Args:
            config: Filter by thread_id and checkpoint_ns
            filter: Filter by metadata fields (source, step)
            before: Only return checkpoints before this one
            limit: Maximum number of checkpoints to return
        """
        if not config:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        history_key = self._history_key(thread_id, checkpoint_ns)

        # Get checkpoint IDs from sorted set (newest first)
        # ZREVRANGE returns items sorted by score descending
        before_score = "+inf"
        if before:
            before_checkpoint_id = get_checkpoint_id(before)
            if before_checkpoint_id:
                # Get the score of the 'before' checkpoint
                score = await self._redis.zscore(history_key, before_checkpoint_id)
                if score:
                    before_score = f"({score}"  # Exclusive

        # Get checkpoint IDs with scores
        checkpoint_ids = await self._redis.zrevrangebyscore(
            history_key,
            before_score,
            "-inf",
            start=0,
            num=limit or 100,
        )

        if not checkpoint_ids:
            return

        # Fetch all checkpoints in parallel
        pipeline = self._redis.pipeline(transaction=False)
        for cid in checkpoint_ids:
            cid_str = cid.decode() if isinstance(cid, bytes) else cid
            checkpoint_key = self._checkpoint_key(thread_id, checkpoint_ns, cid_str)
            pipeline.get(checkpoint_key)

        results = await pipeline.execute()

        count = 0
        for cid, data in zip(checkpoint_ids, results):
            if not data:
                continue

            checkpoint_data = json.loads(data)

            # Apply filters
            if filter:
                if "source" in filter and checkpoint_data.get("source") != filter["source"]:
                    continue
                if "step" in filter and checkpoint_data.get("step") != filter["step"]:
                    continue

            cid_str = cid.decode() if isinstance(cid, bytes) else cid

            # Deserialize channel values
            channel_values = {
                k: self._deserialize_value(v)
                for k, v in checkpoint_data.get("channel_values", {}).items()
            }

            checkpoint: Checkpoint = {
                "v": checkpoint_data.get("v", 1),
                "id": cid_str,
                "ts": checkpoint_data.get("ts", ""),
                "channel_values": channel_values,
                "channel_versions": checkpoint_data.get("channel_versions", {}),
                "versions_seen": checkpoint_data.get("versions_seen", {}),
                "pending_sends": [],
            }

            pending_writes = await self._load_pending_writes(thread_id, checkpoint_ns, cid_str)

            parent_checkpoint_id = checkpoint_data.get("parent_checkpoint_id")
            parent_config = None
            if parent_checkpoint_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": cid_str,
                    }
                },
                checkpoint=checkpoint,
                metadata=checkpoint_data.get("metadata", {}),
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

            count += 1
            if limit and count >= limit:
                break

    async def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[PendingWrite]:
        """Load pending writes for a checkpoint."""
        writes_set_key = self._writes_set_key(thread_id, checkpoint_ns, checkpoint_id)

        write_keys = await self._redis.smembers(writes_set_key)
        if not write_keys:
            return []

        pipeline = self._redis.pipeline(transaction=False)
        for key in write_keys:
            pipeline.get(key)

        results = await pipeline.execute()

        pending_writes = []
        for data in results:
            if data:
                write_data = json.loads(data)
                value = self._deserialize_value(write_data)
                pending_writes.append(
                    (write_data["task_id"], write_data["channel"], value)
                )

        return pending_writes

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread atomically."""
        # Use pattern matching to find all keys for this thread
        patterns = [
            f"{self.CHECKPOINT_PREFIX}:{thread_id}:*",
            f"{self.LATEST_PREFIX}:{thread_id}:*",
            f"{self.HISTORY_PREFIX}:{thread_id}:*",
            f"{self.WRITES_PREFIX}:{thread_id}:*",
            f"{self.WRITES_PREFIX}:set:{thread_id}:*",
        ]

        async with self._redis.pipeline(transaction=True) as pipe:
            for pattern in patterns:
                # SCAN is safer than KEYS for production
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                    for key in keys:
                        pipe.delete(key)
                    if cursor == 0:
                        break

            await pipe.execute()

    # Sync wrappers for compatibility
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return asyncio.get_event_loop().run_until_complete(self.aget_tuple(config))

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return asyncio.get_event_loop().run_until_complete(
            self.aput(config, checkpoint, metadata, new_versions)
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return asyncio.get_event_loop().run_until_complete(
            self.aput_writes(config, writes, task_id, task_path)
        )


# Test the enhanced checkpointer
async def test_enhanced_redis_checkpointer():
    """Test checkpoint history and ACID guarantees."""
    from langgraph.graph import StateGraph
    from typing import TypedDict, Annotated
    from operator import add
    from uuid import uuid4

    logging.basicConfig(level=logging.INFO)

    class State(TypedDict):
        messages: Annotated[list[str], add]

    def add_message(state: State) -> dict:
        n = len(state.get("messages", []))
        return {"messages": [f"Message {n + 1}"]}

    builder = StateGraph(State)
    builder.add_node("add_message", add_message)
    builder.set_entry_point("add_message")
    builder.set_finish_point("add_message")

    try:
        async with EnhancedAsyncRedisSaver.from_conn_string(
            "redis://localhost:6379",
            ttl_seconds=3600,
        ) as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)

            thread_id = f"test-enhanced-{uuid4()}"
            config = {"configurable": {"thread_id": thread_id}}

            # Create multiple checkpoints
            print("\n1. Creating checkpoints...")
            for i in range(5):
                result = await graph.ainvoke({"messages": []}, config=config)
                print(f"   Checkpoint {i+1}: {len(result['messages'])} messages")

            # Test checkpoint history
            print("\n2. Testing checkpoint history (alist)...")
            checkpoints = []
            async for cp in checkpointer.alist(config):
                checkpoints.append(cp)
            print(f"   Found {len(checkpoints)} total checkpoints in history")
            # LangGraph creates ~3 checkpoints per invoke (start, node, end)
            # So we expect at least 5 checkpoints (could be more)
            if len(checkpoints) < 5:
                print(f"   ❌ Expected at least 5 checkpoints, got {len(checkpoints)}")
                return False
            print(f"   ✅ History working correctly")

            # Test limit
            print("\n3. Testing limit...")
            limited = []
            async for cp in checkpointer.alist(config, limit=3):
                limited.append(cp)
            print(f"   With limit=3: {len(limited)} checkpoints")
            if len(limited) == 3:
                print(f"   ✅ Limit working correctly")
            else:
                print(f"   ❌ Expected 3 checkpoints, got {len(limited)}")
                return False

            # Test filtering by source
            print("\n3b. Testing filter by source...")
            loop_checkpoints = []
            async for cp in checkpointer.alist(config, filter={"source": "loop"}):
                loop_checkpoints.append(cp)
            print(f"   Found {len(loop_checkpoints)} 'loop' source checkpoints")

            # Test before filter
            print("\n4. Testing 'before' filter...")
            middle_checkpoint = checkpoints[2]
            before_config = middle_checkpoint.config
            before_list = []
            async for cp in checkpointer.alist(config, before=before_config):
                before_list.append(cp)
            print(f"   Checkpoints before middle: {len(before_list)}")

            # Test state persistence
            print("\n5. Verifying state persistence...")
            result = await graph.ainvoke({"messages": []}, config=config)
            if len(result["messages"]) == 6:
                print(f"   ✅ State persisted: {len(result['messages'])} messages")
            else:
                print(f"   ❌ Expected 6 messages, got {len(result['messages'])}")
                return False

            # Cleanup
            print("\n6. Testing atomic delete...")
            await checkpointer.adelete_thread(thread_id)
            deleted_result = await checkpointer.aget_tuple(config)
            if deleted_result is None:
                print("   ✅ Thread deleted successfully")
            else:
                print("   ❌ Thread not deleted")
                return False

            print("\n✅ All tests passed!")
            return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(test_enhanced_redis_checkpointer())
