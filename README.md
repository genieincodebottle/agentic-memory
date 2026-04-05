# agentic-memory

**The missing memory layer for AI agents.** Works with any LLM. Zero dependencies.

Available for both **TypeScript/JavaScript** and **Python**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-72%20tests-blue)](./tests/)
[![Python](https://img.shields.io/badge/Python-62%20tests-green)](./python/tests/)

---

## Why this exists

Every AI agent framework gives you tools, chains, and orchestration. None of them solve **memory** properly.

```
Without agentic-memory:

  Session 1: "I'm vegetarian"      -->  stored somewhere
  Session 5: "Find me a recipe"    -->  agent suggests steak
  40-min workflow                   -->  context fills up, loses all progress
  Multi-agent system                -->  agents duplicate each other's work
```

```
With agentic-memory:

  Session 1: "I'm vegetarian"      -->  stored as hard constraint (never decays)
  Session 5: "Find me steak"       -->  conflict detected, agent asks to confirm
  40-min workflow                   -->  checkpoint saves state, resumes after overflow
  Multi-agent system                -->  scoped memory, no cross-contamination
```

---

## Where it fits

<p align="center">
  <img src="https://raw.githubusercontent.com/genieincodebottle/agentic-memory/master/assets/architecture.svg" alt="Architecture diagram" width="720"/>
</p>

---

## Install

```bash
# JavaScript / TypeScript
npm install agentic-memory

# Python
pip install agentic-memory-ai
```

---

## End-to-end example: Build a memory-aware agent

This shows exactly how `agentic-memory` fits into a real agent loop - from storing user context, to retrieving it before LLM calls, to catching dangerous contradictions.

### TypeScript

```typescript
import { AgentMemory } from 'agentic-memory';

const memory = new AgentMemory();

// ── Session 1: User onboarding ──
// Store facts the agent learns about the user
await memory.store({
  content: 'User is vegetarian',
  type: 'preference',
  scope: 'user:42',
  importance: 'hard',       // hard = never forgets (like allergies)
});

await memory.store({
  content: 'User prefers quick 30-minute recipes',
  type: 'preference',
  scope: 'user:42',
  importance: 'soft',       // soft = fades over time if not reinforced
});

// ── Session 2: User asks for help ──
// STEP 1: Retrieve relevant memories BEFORE calling the LLM
const context = await memory.retrieve({
  query: 'recipe suggestions',
  scope: 'user:42',
  signals: ['similarity', 'recency', 'importance'],
  limit: 5,
});
// context[0].entry.content = "User is vegetarian"
// context[1].entry.content = "User prefers quick 30-minute recipes"

// STEP 2: Inject memories into your LLM prompt
const prompt = `
  User context: ${context.map(r => r.entry.content).join('. ')}
  User message: Suggest a recipe for dinner tonight.
`;
// LLM now knows: vegetarian + quick recipes
// Response: "Here's a 25-minute mushroom risotto..."

// ── Session 5: Catch contradictions ──
// User (or another agent) tries something that conflicts
const conflicts = await memory.checkConflicts(
  'Order a steak dinner for the user',
  'user:42',
);

if (conflicts.length > 0) {
  console.log(conflicts[0].action);   // 'clarify'
  console.log(conflicts[0].reason);
  // "Conflicts with a hard constraint: 'User is vegetarian'.
  //  This memory was marked as non-negotiable - please confirm the change."

  // Agent should ASK the user, not silently override
}

// ── Long workflow: Survive context overflow ──
const cp = await memory.checkpoint({
  taskGraph: [
    { id: 't1', description: 'Scraped 500 recipes', status: 'done', dependencies: [], result: recipes },
    { id: 't2', description: 'Filtering by diet', status: 'in_progress', dependencies: ['t1'] },
    { id: 't3', description: 'Rank by prep time', status: 'pending', dependencies: ['t2'] },
  ],
  summary: 'Scraped 500 recipes, filtering for vegetarian, 200 remaining',
  toolOutputs: { scraped: recipes },
  activeMemoryIds: context.map(r => r.entry.id),
});

// ... context window fills up and resets ...

// Resume exactly where you left off
const restored = await memory.rehydrate(cp.id);
// restored.checkpoint.taskGraph[1].status = 'in_progress'
// restored.memories = [vegetarian preference, quick recipe preference]
```

### Python

```python
import asyncio
from agentic_memory import AgentMemory, MemoryType, ImportanceLevel, RetrievalQuery

async def main():
    memory = AgentMemory()

    # Session 1: Store user preferences
    await memory.store(
        content="User is vegetarian",
        type=MemoryType.PREFERENCE,
        scope="user:42",
        importance=ImportanceLevel.HARD,
    )

    # Session 2: Retrieve context before LLM call
    context = await memory.retrieve(RetrievalQuery(
        query="recipe suggestions",
        scope="user:42",
        signals=["similarity", "recency", "importance"],
        limit=5,
    ))

    # Inject into prompt
    memory_context = ". ".join(r.entry.content for r in context)
    prompt = f"User context: {memory_context}\nSuggest a recipe."

    # Catch contradictions
    conflicts = await memory.check_conflicts(
        "Order a steak dinner for the user", "user:42"
    )
    if conflicts:
        print(f"Action: {conflicts[0].action.value}")  # 'clarify'
        print(f"Reason: {conflicts[0].reason}")

asyncio.run(main())
```

**No config, no database, no API keys.** Just `import` and go.

---

## What it solves

### 1. Smarter retrieval (not just similarity search)

RAG gives you "topically related" results. Agents need more than that.

<p align="center">
  <img src="https://raw.githubusercontent.com/genieincodebottle/agentic-memory/master/assets/retrieval-signals.svg" alt="Multi-signal retrieval" width="640"/>
</p>

```typescript
const results = await memory.retrieve({
  query: 'programming preferences',
  taskContext: 'Building a REST API',   // boosts API-related memories
  signals: ['similarity', 'recency', 'importance', 'taskRelevance'],
  limit: 5,
});
```

### 2. Conflict detection (catch contradictions before they ship)

<p align="center">
  <img src="https://raw.githubusercontent.com/genieincodebottle/agentic-memory/master/assets/conflict-detection.svg" alt="Conflict detection flow" width="640"/>
</p>

```typescript
const conflicts = await memory.checkConflicts('Order a steak dinner', 'user:123');
// Returns:
// {
//   confidence: 0.85,
//   action: 'clarify',
//   reason: 'Conflicts with hard constraint: "User is vegetarian"'
// }
```

Uses negation detection, antonym matching, and change-language detection - not just keyword overlap.

### 3. Typed decay (a peanut allergy != a favorite color)

<p align="center">
  <img src="https://raw.githubusercontent.com/genieincodebottle/agentic-memory/master/assets/typed-decay.svg" alt="Typed decay curves" width="640"/>
</p>

```typescript
// This will still be there in 5 years
await memory.store({
  content: 'User has peanut allergy',
  type: 'constraint',
  importance: 'hard',
});

// This expires after 7 days
await memory.store({
  content: 'Currently debugging auth module',
  type: 'task',
  importance: 'ephemeral',
});

// Periodic cleanup
const deletedCount = await memory.cleanup('user:123');
```

### 4. Checkpointing (survive context overflow)

A 40-minute agent workflow shouldn't lose everything when context fills up.

<p align="center">
  <img src="https://raw.githubusercontent.com/genieincodebottle/agentic-memory/master/assets/checkpoint-flow.svg" alt="Checkpoint and rehydrate flow" width="640"/>
</p>

```typescript
// Save state before overflow
const cp = await memory.checkpoint({
  taskGraph: [
    { id: 's1', description: 'Fetch data', status: 'done', result: data, dependencies: [] },
    { id: 's2', description: 'Process', status: 'in_progress', dependencies: ['s1'] },
    { id: 's3', description: 'Report', status: 'pending', dependencies: ['s2'] },
  ],
  summary: 'Fetched 1000 records, processing row 450/1000',
  toolOutputs: { api_response: apiData },
  activeMemoryIds: ['mem_abc', 'mem_def'],
});

// After context reset - pick up where you left off
const { checkpoint, memories } = await memory.rehydrate(cp.id);
```

### 5. Scope isolation (multi-agent without the chaos)

```typescript
// Each agent gets its own namespace
await memory.store({ content: 'Plan: 3 steps', scope: 'agent:planner' });
await memory.store({ content: 'API returned 200', scope: 'agent:executor' });

// No cross-contamination
const plannerOnly = await memory.getAll('agent:planner');

// User memories are separate too
await memory.store({ content: 'Prefers Python', scope: 'user:alice' });
await memory.store({ content: 'Prefers Rust', scope: 'user:bob' });
```

---

## Embedder adapters (production-ready)

The built-in embedder (TF-IDF) works for dev/testing. For production, use the included adapters:

### TypeScript

```typescript
import { AgentMemory } from 'agentic-memory';
import { OpenAIEmbedder } from 'agentic-memory/adapters/openai';
import { VoyageEmbedder } from 'agentic-memory/adapters/voyageai';

// OpenAI (most popular)
const memory = new AgentMemory({
  embedder: new OpenAIEmbedder({
    apiKey: process.env.OPENAI_API_KEY!,
    model: 'text-embedding-3-small',  // default
    dimensions: 512,                   // smaller = faster + cheaper
  }),
});

// Voyage AI (higher quality, cheaper)
const memory2 = new AgentMemory({
  embedder: new VoyageEmbedder({
    apiKey: process.env.VOYAGE_API_KEY!,
    model: 'voyage-3-lite',
  }),
});
```

### Python

```python
from agentic_memory import AgentMemory
from agentic_memory.adapters import OpenAIEmbedder, VoyageEmbedder

# OpenAI
memory = AgentMemory(
    embedder=OpenAIEmbedder(api_key="sk-...", dim=512)
)

# Voyage AI
memory = AgentMemory(
    embedder=VoyageEmbedder(api_key="voyage-...")
)
```

Or bring your own - just implement `embed()`, `embed_batch()`, and `dimensions()`.

---

## Storage backends

```typescript
import { AgentMemory, FileStore } from 'agentic-memory';

// Default: in-memory (dev/testing)
const memory = new AgentMemory();

// File-based (persists across restarts)
const memory = new AgentMemory({
  store: new FileStore('./agent-memory.json'),
});

// Custom backend (Redis, Postgres, etc.)
const memory = new AgentMemory({ store: myCustomBackend });
```

Implement the `StorageBackend` interface for any database:

| Method | Signature |
|--------|-----------|
| `get` | `(id: string) => Promise<MemoryEntry \| null>` |
| `getAll` | `(scope?: string) => Promise<MemoryEntry[]>` |
| `set` | `(entry: MemoryEntry) => Promise<void>` |
| `delete` | `(id: string) => Promise<boolean>` |
| `clear` | `(scope?: string) => Promise<void>` |
| `search` | `(query: RetrievalQuery) => Promise<MemoryEntry[]>` |

---

## API at a glance

```typescript
const memory = new AgentMemory(config?)

// CRUD
await memory.store({ content, type?, scope?, importance?, confidence?, metadata? })
await memory.get(id)
await memory.update(id, { content?, type?, importance?, confidence?, metadata? })
await memory.delete(id)
await memory.getAll(scope?)
await memory.clear(scope?)

// Intelligence
await memory.retrieve({ query, taskContext?, scope?, types?, signals?, limit?, threshold? })
await memory.checkConflicts(content, scope?)
memory.getDecayedConfidence(entry)
await memory.cleanup(scope?, threshold?)

// Checkpointing
await memory.checkpoint({ taskGraph, summary, toolOutputs?, activeMemoryIds? })
await memory.rehydrate(checkpointId)
memory.getLatestCheckpoint()
memory.listCheckpoints()
```

---

## Runnable examples

```bash
# No API key needed - shows checkpoint/rehydrate flow
npx tsx examples/checkpoint-recovery.ts

# Full demo with OpenAI embeddings + LLM
export OPENAI_API_KEY=sk-...
npx tsx examples/with-openai.ts

# Full demo with Claude + OpenAI embeddings
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
npx tsx examples/with-anthropic.ts

# Python
export OPENAI_API_KEY=sk-...
cd python && python examples/with_openai.py
```

---

## Roadmap

- [x] **v0.1** - Core memory, multi-signal retrieval, conflict detection, typed decay, checkpointing
- [x] **v0.1** - Python port with full test parity
- [ ] **v0.2** - Redis and Postgres storage backends
- [ ] **v0.3** - Multi-agent coordination (SharedMemory, TaskRegistry, Plan Store)
- [ ] **v0.4** - Audit/attribution layer, GDPR cascade delete
- [ ] **v1.0** - Framework adapters (LangChain, Vercel AI SDK, Claude Agent SDK)

---

## Contributing

PRs welcome. Run tests before submitting:

```bash
# TypeScript
npm test            # 72 tests

# Python
cd python
pip install -e ".[dev]"
pytest tests/ -v    # 62 tests
```

---

## License

MIT
