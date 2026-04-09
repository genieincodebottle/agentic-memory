"""Tests for AgentMemory - mirrors the TypeScript test suite."""

import asyncio
import pytest

from agentic_memory import (
    AgentMemory,
    MemoryType,
    ImportanceLevel,
    RetrievalQuery,
    TaskNode,
    TaskStatus,
)


# ─── Store & Retrieve ───

@pytest.mark.asyncio
async def test_store_and_get():
    memory = AgentMemory()
    entry = await memory.store(
        content="User prefers dark mode",
        type=MemoryType.PREFERENCE,
        scope="user:1",
    )
    assert entry.id
    assert entry.content == "User prefers dark mode"
    assert entry.type == MemoryType.PREFERENCE
    assert entry.version == 1

    retrieved = await memory.get(entry.id)
    assert retrieved is not None
    assert retrieved.content == "User prefers dark mode"


@pytest.mark.asyncio
async def test_store_defaults():
    memory = AgentMemory()
    entry = await memory.store(content="test fact")
    assert entry.type == MemoryType.FACT
    assert entry.importance == ImportanceLevel.SOFT
    assert entry.confidence == 1.0
    assert entry.scope == "default"


@pytest.mark.asyncio
async def test_retrieve_by_query():
    memory = AgentMemory()
    await memory.store(content="User likes Python programming", scope="user:1")
    await memory.store(content="User dislikes Java", scope="user:1")
    await memory.store(content="Meeting at 3pm tomorrow", scope="user:1", type=MemoryType.TASK)

    results = await memory.retrieve(RetrievalQuery(
        query="programming language preferences",
        scope="user:1",
        limit=5,
    ))
    assert len(results) > 0
    assert results[0].score > 0
    assert results[0].signal_scores


# ��── Update & Delete ───

@pytest.mark.asyncio
async def test_update_increments_version():
    memory = AgentMemory()
    entry = await memory.store(content="User likes blue")
    updated = await memory.update(entry.id, content="User likes red")
    assert updated is not None
    assert updated.content == "User likes red"
    assert updated.version == 2


@pytest.mark.asyncio
async def test_update_nonexistent():
    memory = AgentMemory()
    result = await memory.update("nonexistent", content="test")
    assert result is None


@pytest.mark.asyncio
async def test_delete():
    memory = AgentMemory()
    entry = await memory.store(content="temporary")
    assert await memory.delete(entry.id) is True
    assert await memory.get(entry.id) is None


# ─── Scoping ───

@pytest.mark.asyncio
async def test_scope_isolation():
    memory = AgentMemory()
    await memory.store(content="Agent A memory", scope="agent:a")
    await memory.store(content="Agent B memory", scope="agent:b")

    a = await memory.get_all("agent:a")
    b = await memory.get_all("agent:b")
    assert len(a) == 1
    assert a[0].content == "Agent A memory"
    assert len(b) == 1


@pytest.mark.asyncio
async def test_clear_scoped():
    memory = AgentMemory()
    await memory.store(content="keep this", scope="keep")
    await memory.store(content="delete this", scope="remove")
    await memory.clear("remove")

    assert len(await memory.get_all("keep")) == 1
    assert len(await memory.get_all("remove")) == 0


# ─── Conflict Detection ───

@pytest.mark.asyncio
async def test_conflict_negation():
    memory = AgentMemory()
    await memory.store(
        content="User is vegetarian",
        type=MemoryType.PREFERENCE,
        importance=ImportanceLevel.HARD,
        scope="user:1",
    )
    conflicts = await memory.check_conflicts(
        "User is not vegetarian and wants egg", "user:1"
    )
    assert len(conflicts) > 0
    assert conflicts[0].action.value == "clarify"


@pytest.mark.asyncio
async def test_no_conflict_unrelated():
    memory = AgentMemory()
    await memory.store(content="User likes Python", scope="user:1")
    conflicts = await memory.check_conflicts("The weather is nice today", "user:1")
    high = [c for c in conflicts if c.confidence > 0.5]
    assert len(high) == 0


# ─── Decay ───

@pytest.mark.asyncio
async def test_hard_never_decays():
    memory = AgentMemory()
    entry = await memory.store(
        content="User has peanut allergy",
        importance=ImportanceLevel.HARD,
        type=MemoryType.CONSTRAINT,
    )
    assert memory.get_decayed_confidence(entry) == 1.0


@pytest.mark.asyncio
async def test_task_step_decay():
    from datetime import datetime, timezone, timedelta

    memory = AgentMemory()
    entry = await memory.store(
        content="Task in progress",
        importance=ImportanceLevel.EPHEMERAL,
        type=MemoryType.TASK,
    )
    # Simulate 30-day-old entry
    old_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    entry.updated_at = old_date
    assert memory.get_decayed_confidence(entry) == 0  # step: 7-day max


# ─── Checkpointing ───

@pytest.mark.asyncio
async def test_checkpoint_and_rehydrate():
    memory = AgentMemory()
    entry = await memory.store(content="Important context")

    cp = await memory.checkpoint(
        task_graph=[
            TaskNode(id="t1", description="Step 1", status=TaskStatus.DONE, dependencies=[], result="ok"),
            TaskNode(id="t2", description="Step 2", status=TaskStatus.PENDING, dependencies=["t1"]),
        ],
        summary="Completed step 1, about to start step 2",
        tool_outputs={"api_call_1": {"data": [1, 2, 3]}},
        active_memory_ids=[entry.id],
    )
    assert cp.id
    assert len(cp.task_graph) == 2

    restored = await memory.rehydrate(cp.id)
    assert restored is not None
    assert restored["checkpoint"].summary == "Completed step 1, about to start step 2"
    assert len(restored["memories"]) == 1


@pytest.mark.asyncio
async def test_latest_checkpoint():
    memory = AgentMemory()
    await memory.checkpoint(task_graph=[], summary="first")
    await memory.checkpoint(task_graph=[], summary="second")
    latest = memory.get_latest_checkpoint()
    assert latest is not None
    assert latest.summary == "second"


@pytest.mark.asyncio
async def test_max_checkpoints():
    memory = AgentMemory(max_checkpoints=2)
    await memory.checkpoint(task_graph=[], summary="cp1")
    await memory.checkpoint(task_graph=[], summary="cp2")
    await memory.checkpoint(task_graph=[], summary="cp3")
    all_cp = memory.list_checkpoints()
    assert len(all_cp) == 2
    assert all_cp[0].summary == "cp2"
