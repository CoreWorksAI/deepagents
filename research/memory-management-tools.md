# Memory Management Tools for DeepAgents: Research Report

> **Date:** 2026-02-13
> **Context:** Evaluating memory management solutions for integration with the DeepAgents framework

---

## Current State of Memory in DeepAgents

DeepAgents already has a layered memory architecture:

- **StateBackend** -- ephemeral, in-conversation state storage
- **FilesystemBackend** -- disk-backed file operations under a root directory
- **StoreBackend** -- cross-thread persistent storage via LangGraph's `BaseStore`
- **CompositeBackend** -- routes paths to different backends (e.g., `/working/` → ephemeral, `/memories/` → persistent)
- **AgentMemoryMiddleware** (CLI) -- user-level (`~/.deepagents/`) and project-level (`.deepagents/`) markdown-based memory files
- **SummarizationMiddleware** -- auto-summarizes when context exceeds 85% of model capacity

These provide **file-based persistence and context management**, but lack **semantic memory**, **automatic fact extraction**, **knowledge graphs**, and **intelligent memory retrieval** (similarity search, relevance scoring, temporal reasoning).

---

## Tools Evaluated

### 1. Mem0

**What:** A universal memory layer for AI agents with hybrid storage (vector + graph + key-value). Apache 2.0. Backed by Y Combinator, $24M Series A. AWS selected Mem0 as exclusive memory provider for Strands Agent SDK.

**How it works:**
- Two-phase pipeline: (1) LLM extracts facts from conversation, (2) compares against existing memories and applies create/update/delete operations
- Hybrid storage: vector DB for semantic search, Neo4j for entity relationships, key-value for structured facts
- Scoped memory: `user_id`, `session_id`, `agent_id`, `org_id`

**Key API:**
```python
from mem0 import Memory

memory = Memory.from_config({
    "vector_store": {"provider": "qdrant", "config": {"host": "localhost", "port": 6333}},
    "graph_store": {"provider": "neo4j", "config": {"url": "bolt://localhost:7687", ...}},
})

memory.add(messages, user_id="alice")  # Extract and store facts
results = memory.search("study preferences", user_id="alice", limit=3)  # Semantic search
```

**Benchmarks:** 26% higher accuracy than OpenAI Memory on LOCOMO, 91% lower latency and 90% fewer tokens than full-context approaches.

**Pros:**
- Best accuracy/speed/cost balance in benchmarks
- 50+ backend integrations (vector stores, graph DBs, LLMs, embedders)
- Drop-in simplicity (3 lines of code)
- Automatic deduplication and conflict resolution
- MCP server available

**Cons:**
- Every `add()` requires LLM inference (adds latency and cost)
- Reported scaling issues at high volume (latency degradation)
- Self-hosted requires managing vector DB + graph DB + LLM API keys
- Weaker on complex multi-hop reasoning vs Cognee

**Best for:** General-purpose agent memory, user personalization, preference tracking.

**Pricing:** Open source (Apache 2.0). Platform: Free (10K memories), $19/mo Starter, $249/mo Pro, custom Enterprise.

---

### 2. Cognee

**What:** Open-source knowledge engine (Apache 2.0) that transforms raw data into persistent AI memory using ECL (Extract, Cognify, Load) pipelines. Combines vector search with dense knowledge graphs.

**How it works:**
```python
import cognee

await cognee.add("data source")     # Ingest
await cognee.cognify()               # Build knowledge graph + embeddings
results = await cognee.search("query")  # Hybrid retrieval
```

**Benchmarks:** 0.93 human-like correctness on HotPotQA (outperforms Mem0, LightRAG, Graphiti on multi-hop reasoning).

**Pros:**
- Dense multi-layer semantic connections in knowledge graph
- Session isolation with grouped node sets
- 30+ data source types supported
- Best-in-class multi-hop reasoning performance

**Cons:**
- Pre-v1.0 (API still maturing)
- Scaling to terabyte datasets is a known gap
- Smaller ecosystem than Mem0

**Best for:** Enterprise knowledge systems, legal/scientific research, multi-hop reasoning tasks.

---

### 3. Letta (formerly MemGPT)

**What:** Open-source LLM Operating System (Apache 2.0) from UC Berkeley. Agents actively manage their own memory, inspired by OS memory management (RAM/disk hierarchy).

**Architecture:**
- **Core Memory** (~RAM): In the LLM context window, actively edited by the agent via `memory_replace`, `memory_insert`, `memory_rethink`
- **Archival Memory** (~disk): External long-term storage with vector search
- **Recall Memory**: Searchable conversation history

**Key innovation:** Self-editing memory -- agents decide what to remember, update, or archive autonomously.

**Recent (2025-2026):**
- Letta V1 architecture (Oct 2025): new agent loop, supports GPT-5, Claude 4.5 Sonnet
- Letta Code (Dec 2025): #1 model-agnostic open-source agent on Terminal-Bench
- Context Repositories (Feb 2026): git-based versioning for context

**Pros:**
- Most principled approach (OS theory-backed)
- Self-editing memory is genuinely novel
- White-box -- developers can inspect/edit agent memory
- Model-agnostic

**Cons:**
- Architecture complexity for simple use cases
- Memory quality depends on LLM quality
- V1 transition = API changes for existing users

**Best for:** Long-running autonomous agents, personal AI assistants, coding agents, transparent/auditable memory.

---

### 4. Zep (Temporal Knowledge Graph)

**What:** Memory layer built on Graphiti, a temporal knowledge graph engine. Tracks when facts change over time using a bi-temporal model.

**Architecture:** Three-layer graph:
- **Episode Subgraph** -- raw conversational data
- **Semantic Entity Subgraph** -- entities and relationships
- **Community Subgraph** -- entity clusters for broader context

**Benchmarks:** 94.8% accuracy on DMR (outperforms MemGPT's 93.4%), 90% faster retrieval than full-context, P95 latency ~300ms.

**Pros:**
- Temporal reasoning (tracks when facts change)
- No LLM calls during retrieval (low latency)
- Conflict resolution with intelligent invalidation
- Automatic ontology building

**Cons:**
- Requires Neo4j infrastructure
- More complex than vector-only solutions
- Newer in the market

**Best for:** Apps requiring temporal reasoning, tracking changing preferences, enterprise audit trails.

---

### 5. Redis (with Vector Search)

**What:** Unified platform for AI agent infrastructure. #1 AI agent data storage tool (2025 Stack Overflow survey, 43% of developers). Combines vector search, caching, session state in one system.

**Key features:**
- **Vector Sets** (April 2025): native vector similarity data type
- 9.5X higher QPS and 9.7X lower latency than pgvector
- Sub-millisecond retrieval
- Semantic caching (reduces response times by 15X, costs by 90%)

**Supports all 5 memory types:** short-term, long-term, episodic, semantic, procedural.

**Integration with DeepAgents:** LangGraph provides `RedisSaver` and `RedisStore` -- direct integration with DeepAgents' existing LangGraph-based architecture.

**Pros:**
- Fastest option for real-time memory retrieval
- Unified platform (no separate vector DB needed)
- Massive ecosystem, battle-tested at scale
- Memory decay support

**Cons:**
- Memory-bound (all data in RAM by default)
- Vector search features are newer than dedicated vector DBs
- Licensing complexity (BSD-3 / SSPL)

**Best for:** High-performance agent memory, applications already using Redis, unified memory + caching.

---

### 6. LangChain/LangGraph Memory (Already Partially Used)

**What:** DeepAgents already builds on LangGraph. The ecosystem offers checkpointers and stores for persistence.

**Available backends:** InMemorySaver, SqliteSaver, PostgresSaver, RedisSaver, DynamoDBSaver (checkpointers); InMemoryStore, PostgresStore, RedisStore (cross-thread stores).

**Current DeepAgents usage:** Uses `checkpointer` and `store` parameters in `create_deep_agent()`, plus `StoreBackend` for cross-thread persistence.

**Gap:** No automatic summarization, knowledge graph, or intelligent retrieval built into LangGraph stores -- DeepAgents already fills some of this with `SummarizationMiddleware` but lacks semantic search.

---

### 7. LlamaIndex Memory

**What:** Composable Memory Block architecture with built-in fact extraction.

**Key memory blocks:**
- `StaticBlock` -- stores static info (persona, system prompts)
- `FactExtractionBlock` -- LLM-powered fact extraction from chat history
- `VectorBlock` -- vector similarity retrieval

**Pros:** Composable, built-in fact extraction, RAG integration.
**Cons:** Tied to LlamaIndex ecosystem, memory APIs still evolving.

---

### 8. Emerging Tools

| Tool | Key Feature | License |
|------|-------------|---------|
| **Memori** (GibsonAI) | SQL-native memory, one-line integration, no vector DB needed | Apache 2.0 |
| **OpenMemory** (CaviraOSS) | Multi-sector embeddings (episodic, semantic, procedural, emotional, reflective), MCP support | Apache 2.0 |

---

## Comparative Matrix

| Tool | Memory Types | Retrieval | Auto-Extract | Graph Support | DeepAgents Fit | Complexity |
|------|-------------|-----------|-------------|---------------|----------------|------------|
| **Mem0** | User/Session/Agent/Org | Vector + Graph + KV | Yes (LLM) | Yes (Neo4j) | High -- middleware integration | Medium |
| **Cognee** | Semantic/Episodic/Procedural | Vector + Graph | Yes (ECL pipeline) | Yes (built-in) | Medium -- pipeline integration | Medium-High |
| **Letta** | Core/Archival/Recall | Self-directed | Yes (self-editing) | No | Low -- competing framework | High |
| **Zep** | Short/Long/Episodic/Semantic | Temporal graph | Yes | Yes (Graphiti) | Medium -- API integration | Medium-High |
| **Redis** | All 5 types | Vector + Cache | No | No | **Very High** -- LangGraph native | Low-Medium |
| **LangGraph Stores** | Thread/Cross-thread | Key-based | No | No | **Already integrated** | Low |
| **ChromaDB** | Semantic | Vector | No | No | High -- simple vector store | Low |

---

## Recommendations for DeepAgents

### Tier 1: Highest Priority

1. **Mem0 as a Memory Middleware**
   - Implement a `Mem0MemoryMiddleware` that wraps Mem0's `Memory` class
   - On `before_agent`: search relevant memories and inject into context
   - On `after_agent`: extract and store new memories from conversation
   - Scoping: map DeepAgents threads to `session_id`, users to `user_id`, agent names to `agent_id`
   - Natural fit with the existing middleware architecture

2. **Redis as a Backend**
   - Add a `RedisBackend` implementing `BackendProtocol`
   - Use Redis Vector Sets for semantic search within the file/memory operations
   - Leverage existing LangGraph `RedisSaver`/`RedisStore` for checkpointing
   - Provides unified caching + memory + state in one infrastructure component

### Tier 2: Strong Candidates

3. **Cognee for Knowledge-Intensive Agents**
   - Add as an optional middleware for agents requiring multi-hop reasoning
   - Use Cognee's ECL pipeline to build knowledge graphs from agent interactions
   - Best for specialized sub-agents handling research or analysis tasks

4. **Zep/Graphiti for Temporal Memory**
   - When agents need to track changing facts over time
   - Graphiti (open-source core) can be integrated as a memory backend
   - Useful for customer support or advisory agents where preferences evolve

### Tier 3: Consider for Specific Use Cases

5. **Letta's self-editing memory pattern** (concept, not the framework)
   - Adapt the idea of agents managing their own memory via tools
   - DeepAgents already has `write_file`/`edit_file` -- extend with semantic memory tools like `remember`, `recall`, `forget`
   - This would be a lightweight alternative to full Mem0 integration

### Architecture Approach

The recommended integration pattern leverages DeepAgents' existing middleware system:

```
Agent Call
  ↓
Mem0MemoryMiddleware.before_agent()
  → Search memories relevant to current message
  → Inject as system prompt context
  ↓
[Agent processes with memory context]
  ↓
Mem0MemoryMiddleware.after_agent()
  → Extract facts from new messages
  → Store via Mem0 (vector + graph + KV)
  ↓
Existing SummarizationMiddleware
  → Manage context window as usual
```

This approach:
- Requires no changes to the core `create_deep_agent()` function
- Is opt-in (just add the middleware)
- Works with all existing backends and tools
- Separates semantic memory (Mem0) from filesystem memory (existing backends)

---

## Key Takeaways

1. **Mem0 is the strongest general-purpose option** -- best benchmark results, broadest integration ecosystem, and natural middleware fit with DeepAgents
2. **Redis is the strongest infrastructure option** -- fastest retrieval, already has LangGraph integrations, and can serve as a unified backend
3. **Cognee excels at complex reasoning** -- best choice for research/analysis sub-agents that need multi-hop knowledge graph traversal
4. **The middleware pattern is the right integration point** -- DeepAgents' architecture already supports this cleanly
5. **Memory is becoming a first-class primitive** in agent design (not an afterthought) -- investing in this capability is strategically important
