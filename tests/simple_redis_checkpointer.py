#!/usr/bin/env python3
"""Simple Redis checkpointer that works without RedisJSON module.

Uses standard Redis SET/GET with JSON serialization instead of native Redis JSON.
This is a workaround for environments where RedisJSON module is not available.

Usage:
    from simple_redis_checkpointer import SimpleAsyncRedisSaver

    async with SimpleAsyncRedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
        agent = create_deep_agent(..., checkpointer=checkpointer)
"""

import asyncio
import base64
import json
import logging
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
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class SimpleAsyncRedisSaver(BaseCheckpointSaver):
    """Simple async Redis checkpointer using standard SET/GET commands.

    No RedisJSON or RediSearch modules required.
    """

    CHECKPOINT_PREFIX = "checkpoint"
    CHECKPOINT_WRITE_PREFIX = "checkpoint_write"

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
        self.ttl_seconds = ttl_seconds

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        redis_url: str = "redis://localhost:6379",
        ttl_seconds: Optional[int] = None,
    ) -> AsyncIterator["SimpleAsyncRedisSaver"]:
        """Create checkpointer from connection string."""
        saver = cls(redis_url=redis_url, ttl_seconds=ttl_seconds)
        async with saver:
            yield saver

    async def __aenter__(self) -> "SimpleAsyncRedisSaver":
        if self._redis is None:
            self._redis = redis.Redis.from_url(self._redis_url or "redis://localhost:6379")
        await self._redis.ping()
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

    def _make_checkpoint_key(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str:
        """Generate checkpoint key."""
        return f"{self.CHECKPOINT_PREFIX}:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _make_writes_key(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        idx: int,
    ) -> str:
        """Generate writes key."""
        return f"{self.CHECKPOINT_WRITE_PREFIX}:{thread_id}:{checkpoint_ns}:{checkpoint_id}:{task_id}:{idx}"

    def _make_latest_key(self, thread_id: str, checkpoint_ns: str) -> str:
        """Generate latest checkpoint pointer key."""
        return f"checkpoint_latest:{thread_id}:{checkpoint_ns}"

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        if not checkpoint_id:
            # Get latest checkpoint
            latest_key = self._make_latest_key(thread_id, checkpoint_ns)
            checkpoint_id = await self._redis.get(latest_key)
            if checkpoint_id:
                checkpoint_id = checkpoint_id.decode() if isinstance(checkpoint_id, bytes) else checkpoint_id
            else:
                return None

        # Get checkpoint data
        checkpoint_key = self._make_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        data = await self._redis.get(checkpoint_key)

        if not data:
            return None

        checkpoint_data = json.loads(data)

        # Deserialize channel values
        channel_values = {}
        for k, v in checkpoint_data.get("channel_values", {}).items():
            if isinstance(v, dict) and "type" in v and "data" in v:
                # Decode base64 string back to bytes
                data_bytes = base64.b64decode(v["data"]) if isinstance(v["data"], str) else v["data"]
                channel_values[k] = self.serde.loads_typed((v["type"], data_bytes))
            else:
                channel_values[k] = v

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
        pending_writes = await self._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )

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
        """List checkpoints (simplified - returns latest only)."""
        if config:
            result = await self.aget_tuple(config)
            if result:
                yield result

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint to Redis."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        # Serialize channel values
        serialized_channel_values = {}
        for k, v in checkpoint.get("channel_values", {}).items():
            type_, data = self.serde.dumps_typed(v)
            # Encode bytes to base64 string for JSON compatibility
            data_b64 = base64.b64encode(data).decode('utf-8') if isinstance(data, bytes) else data
            serialized_channel_values[k] = {"type": type_, "data": data_b64}

        # Build checkpoint data
        checkpoint_data = {
            "v": checkpoint.get("v", 1),
            "ts": checkpoint.get("ts", ""),
            "channel_values": serialized_channel_values,
            "channel_versions": checkpoint.get("channel_versions", {}),
            "versions_seen": checkpoint.get("versions_seen", {}),
            "parent_checkpoint_id": parent_checkpoint_id,
            "metadata": metadata,
        }

        # Store checkpoint
        checkpoint_key = self._make_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        pipeline = self._redis.pipeline()
        pipeline.set(checkpoint_key, json.dumps(checkpoint_data, default=str))

        # Update latest pointer
        latest_key = self._make_latest_key(thread_id, checkpoint_ns)
        pipeline.set(latest_key, checkpoint_id)

        # Apply TTL if configured
        if self.ttl_seconds:
            pipeline.expire(checkpoint_key, self.ttl_seconds)
            pipeline.expire(latest_key, self.ttl_seconds)

        await pipeline.execute()

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
        """Store pending writes."""
        if not writes:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        pipeline = self._redis.pipeline()

        for idx, (channel, value) in enumerate(writes):
            type_, data = self.serde.dumps_typed(value)
            data_b64 = base64.b64encode(data).decode('utf-8') if isinstance(data, bytes) else data
            write_data = {
                "task_id": task_id,
                "task_path": task_path,
                "idx": WRITES_IDX_MAP.get(channel, idx),
                "channel": channel,
                "type": type_,
                "data": data_b64,
            }

            write_key = self._make_writes_key(
                thread_id, checkpoint_ns, checkpoint_id, task_id, idx
            )
            pipeline.set(write_key, json.dumps(write_data, default=str))

            if self.ttl_seconds:
                pipeline.expire(write_key, self.ttl_seconds)

        # Track write keys for this checkpoint
        writes_set_key = f"checkpoint_writes_set:{thread_id}:{checkpoint_ns}:{checkpoint_id}"
        for idx, _ in enumerate(writes):
            write_key = self._make_writes_key(
                thread_id, checkpoint_ns, checkpoint_id, task_id, idx
            )
            pipeline.sadd(writes_set_key, write_key)

        if self.ttl_seconds:
            pipeline.expire(writes_set_key, self.ttl_seconds)

        await pipeline.execute()

    async def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[PendingWrite]:
        """Load pending writes for a checkpoint."""
        writes_set_key = f"checkpoint_writes_set:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

        write_keys = await self._redis.smembers(writes_set_key)
        if not write_keys:
            return []

        # Fetch all writes
        pipeline = self._redis.pipeline()
        for key in write_keys:
            pipeline.get(key)

        results = await pipeline.execute()

        pending_writes = []
        for data in results:
            if data:
                write_data = json.loads(data)
                # Decode base64 string back to bytes
                data_bytes = base64.b64decode(write_data["data"]) if isinstance(write_data["data"], str) else write_data["data"]
                value = self.serde.loads_typed((write_data["type"], data_bytes))
                pending_writes.append(
                    (write_data["task_id"], write_data["channel"], value)
                )

        return pending_writes

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


# Test the simple checkpointer
async def test_simple_redis_checkpointer():
    """Test the simple Redis checkpointer."""
    from langgraph.graph import StateGraph
    from typing import TypedDict, Annotated
    from operator import add
    from uuid import uuid4

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
        async with SimpleAsyncRedisSaver.from_conn_string(
            "redis://localhost:6379",
            ttl_seconds=3600,
        ) as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)

            thread_id = f"test-simple-{uuid4()}"
            config = {"configurable": {"thread_id": thread_id}}

            # First invocation
            result1 = await graph.ainvoke({"messages": []}, config=config)
            logger.info(f"First: {len(result1['messages'])} messages")

            # Second invocation - state should persist
            result2 = await graph.ainvoke({"messages": []}, config=config)
            logger.info(f"Second: {len(result2['messages'])} messages")

            if len(result2["messages"]) == 2:
                print("✅ SUCCESS: Simple Redis checkpointer works!")
                return True
            else:
                print(f"❌ FAILED: Expected 2 messages, got {len(result2['messages'])}")
                return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        print("Make sure Redis is running: docker run -d -p 6379:6379 redis:latest")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_simple_redis_checkpointer())
