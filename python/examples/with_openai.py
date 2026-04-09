"""
Complete working example: agentic-memory + OpenAI

Run:
    export OPENAI_API_KEY=sk-...
    python examples/with_openai.py

What this demo shows:
    1. Store user preferences with real embeddings
    2. Retrieve relevant context before an LLM call
    3. Catch a dangerous contradiction (vegetarian vs egg)
    4. Make an LLM call that uses memory context
"""

import asyncio
import json
import os
from urllib.request import Request, urlopen

from agentic_memory import AgentMemory, MemoryType, ImportanceLevel, RetrievalQuery
from agentic_memory.adapters import OpenAIEmbedder

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Set OPENAI_API_KEY environment variable")
    exit(1)


def chat(prompt: str) -> str:
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    }).encode()

    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    with urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


async def main():
    print("=== agentic-memory + OpenAI Demo ===\n")

    # Setup: real embeddings for accurate similarity
    memory = AgentMemory(
        embedder=OpenAIEmbedder(
            api_key=OPENAI_API_KEY,
            model="text-embedding-3-small",
            dim=512,
        )
    )

    # Step 1: Store user preferences
    print("Step 1: Storing user preferences...")

    await memory.store(
        content="User is vegetarian and avoids all meat products",
        type=MemoryType.PREFERENCE,
        scope="user:demo",
        importance=ImportanceLevel.HARD,
    )
    await memory.store(
        content="User prefers Italian cuisine, especially pasta dishes",
        type=MemoryType.PREFERENCE,
        scope="user:demo",
        importance=ImportanceLevel.SOFT,
    )
    await memory.store(
        content="User is allergic to peanuts",
        type=MemoryType.CONSTRAINT,
        scope="user:demo",
        importance=ImportanceLevel.HARD,
    )
    await memory.store(
        content="User prefers quick recipes under 30 minutes",
        type=MemoryType.PREFERENCE,
        scope="user:demo",
        importance=ImportanceLevel.SOFT,
    )
    print("  Stored 4 memories (2 hard, 2 soft)\n")

    # Step 2: Retrieve relevant context
    print('Step 2: User asks "Suggest a dinner recipe"')
    print("  Retrieving relevant memories...")

    results = await memory.retrieve(RetrievalQuery(
        query="dinner recipe suggestion",
        scope="user:demo",
        signals=["similarity", "recency", "importance"],
        limit=3,
    ))

    print(f"  Found {len(results)} relevant memories:")
    for r in results:
        print(f"    [{r.score:.2f}] {r.entry.content}")

    # Step 3: LLM call with memory context
    print("\nStep 3: Calling LLM with memory context...")

    memory_context = "\n".join(f"- {r.entry.content}" for r in results)
    prompt = f"""You are a helpful cooking assistant. Here is what you know about the user:
{memory_context}

The user says: "Suggest a dinner recipe for tonight."

Give a short, specific suggestion (2-3 sentences)."""

    response = chat(prompt)
    print(f"\n  LLM Response: {response}\n")

    # Step 4: Conflict detection
    print("Step 4: Another agent tries to order egg...")

    conflicts = await memory.check_conflicts(
        "Order an egg dinner for the user", "user:demo"
    )

    if conflicts:
        print(f"  CONFLICT DETECTED!")
        print(f"    Confidence: {conflicts[0].confidence:.2f}")
        print(f"    Action: {conflicts[0].action.value}")
        print(f"    Reason: {conflicts[0].reason}")
    else:
        print("  No conflicts found")

    # Step 5: Check peanut allergy conflict
    print("\nStep 5: Recipe suggestion with peanuts...")

    peanut_conflicts = await memory.check_conflicts(
        "Make a Thai peanut noodle dish for the user", "user:demo"
    )

    if peanut_conflicts:
        print(f"  CONFLICT DETECTED!")
        print(f"    Confidence: {peanut_conflicts[0].confidence:.2f}")
        print(f"    Reason: {peanut_conflicts[0].reason}")

    print("\n=== Demo complete ===")


if __name__ == "__main__":
    asyncio.run(main())
