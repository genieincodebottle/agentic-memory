"""Edge case tests - mirrors the TypeScript edge case suite."""

import asyncio
import math
import pytest
from datetime import datetime, timezone, timedelta

from agentic_memory import (
    AgentMemory,
    MemoryEntry,
    MemoryType,
    ImportanceLevel,
    DecayPolicy,
    DecayConfig,
    DecayTypeConfig,
    RetrievalQuery,
    BuiltinEmbedder,
    DecayEngine,
    cosine_similarity,
    generate_id,
)
from agentic_memory.utils import days_since


def make_entry(**overrides) -> MemoryEntry:
    defaults = dict(
        id=generate_id(),
        content="test content",
        type=MemoryType.FACT,
        scope="default",
        importance=ImportanceLevel.SOFT,
        confidence=1.0,
        metadata={},
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        version=1,
    )
    defaults.update(overrides)
    return MemoryEntry(**defaults)


# ─── Empty & Invalid Input ───

class TestEmptyInput:
    @pytest.mark.asyncio
    async def test_reject_empty_string(self):
        memory = AgentMemory()
        with pytest.raises(ValueError, match="empty"):
            await memory.store(content="")

    @pytest.mark.asyncio
    async def test_reject_whitespace(self):
        memory = AgentMemory()
        with pytest.raises(ValueError, match="empty"):
            await memory.store(content="   \n\t  ")

    @pytest.mark.asyncio
    async def test_reject_empty_on_update(self):
        memory = AgentMemory()
        entry = await memory.store(content="valid content")
        with pytest.raises(ValueError, match="empty"):
            await memory.update(entry.id, content="   ")

    @pytest.mark.asyncio
    async def test_long_content(self):
        memory = AgentMemory()
        long_content = "word " * 10000
        entry = await memory.store(content=long_content)
        assert len(entry.content) == len(long_content)
        assert entry.embedding is not None
        assert len(entry.embedding) == 256

    @pytest.mark.asyncio
    async def test_unicode_content(self):
        memory = AgentMemory()
        entry = await memory.store(content="User prefers 日本語 and emojis with $pecial chars!")
        retrieved = await memory.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == entry.content


# ─── Retrieval Edge Cases ───

class TestRetrieval:
    @pytest.mark.asyncio
    async def test_empty_store(self):
        memory = AgentMemory()
        results = await memory.retrieve(RetrievalQuery(query="anything"))
        assert results == []

    @pytest.mark.asyncio
    async def test_limit_zero(self):
        memory = AgentMemory()
        await memory.store(content="something here")
        results = await memory.retrieve(RetrievalQuery(query="something", limit=0))
        assert results == []

    @pytest.mark.asyncio
    async def test_limit_one(self):
        memory = AgentMemory()
        await memory.store(content="apple fruit")
        await memory.store(content="banana fruit")
        await memory.store(content="cherry fruit")
        results = await memory.retrieve(RetrievalQuery(query="fruit", limit=1))
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_filter_by_type(self):
        memory = AgentMemory()
        await memory.store(content="User likes Python", type=MemoryType.PREFERENCE)
        await memory.store(content="Meeting tomorrow", type=MemoryType.TASK)
        results = await memory.retrieve(RetrievalQuery(
            query="Python meeting",
            types=[MemoryType.PREFERENCE],
        ))
        for r in results:
            assert r.entry.type == MemoryType.PREFERENCE

    @pytest.mark.asyncio
    async def test_importance_signal_only(self):
        memory = AgentMemory()
        await memory.store(content="hard fact", importance=ImportanceLevel.HARD)
        await memory.store(content="soft fact", importance=ImportanceLevel.SOFT)
        await memory.store(content="ephemeral fact", importance=ImportanceLevel.EPHEMERAL)
        results = await memory.retrieve(RetrievalQuery(
            query="fact",
            signals=["importance"],
        ))
        assert len(results) == 3
        assert results[0].entry.importance == ImportanceLevel.HARD

    @pytest.mark.asyncio
    async def test_task_relevance_without_context(self):
        memory = AgentMemory()
        await memory.store(content="some data here")
        results = await memory.retrieve(RetrievalQuery(
            query="data",
            signals=["taskRelevance"],
        ))
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_threshold_filtering(self):
        memory = AgentMemory()
        await memory.store(content="Python programming language")
        await memory.store(content="completely unrelated astronomy topic stars galaxies")
        results = await memory.retrieve(RetrievalQuery(
            query="Python programming",
            threshold=0.5,
            signals=["similarity"],
        ))
        for r in results:
            assert r.score >= 0.5


# ─── Conflict Detection Edge Cases ───

class TestConflict:
    @pytest.mark.asyncio
    async def test_empty_store(self):
        memory = AgentMemory()
        conflicts = await memory.check_conflicts("anything")
        assert conflicts == []

    @pytest.mark.asyncio
    async def test_negation_conflict(self):
        memory = AgentMemory()
        await memory.store(content="User likes dark mode", scope="user:1")
        conflicts = await memory.check_conflicts("User does not like dark mode", "user:1")
        assert len(conflicts) > 0

    @pytest.mark.asyncio
    async def test_antonym_conflict(self):
        memory = AgentMemory()
        await memory.store(content="User prefers light theme", scope="user:1")
        conflicts = await memory.check_conflicts("User prefers dark theme", "user:1")
        assert len(conflicts) > 0

    @pytest.mark.asyncio
    async def test_scope_isolation(self):
        memory = AgentMemory()
        await memory.store(
            content="User is vegetarian",
            scope="user:1",
            importance=ImportanceLevel.HARD,
        )
        conflicts = await memory.check_conflicts("User wants steak", "user:2")
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_ephemeral_override(self):
        memory = AgentMemory()
        await memory.store(
            content="User prefers morning meetings",
            scope="user:1",
            importance=ImportanceLevel.EPHEMERAL,
        )
        conflicts = await memory.check_conflicts("User does not want morning meetings", "user:1")
        if conflicts:
            assert conflicts[0].action.value == "override"


# ─── Decay Edge Cases ───

class TestDecay:
    def test_zero_confidence(self):
        engine = DecayEngine()
        entry = make_entry(confidence=0.0, importance=ImportanceLevel.SOFT, type=MemoryType.PREFERENCE)
        assert engine.compute_decayed_confidence(entry) == 0.0

    def test_future_timestamp(self):
        engine = DecayEngine()
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        entry = make_entry(updated_at=future, importance=ImportanceLevel.SOFT, type=MemoryType.PREFERENCE)
        decayed = engine.compute_decayed_confidence(entry)
        assert decayed >= 0

    def test_very_old_entry(self):
        engine = DecayEngine()
        old = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        entry = make_entry(updated_at=old, importance=ImportanceLevel.SOFT, type=MemoryType.PREFERENCE)
        decayed = engine.compute_decayed_confidence(entry)
        assert 0 <= decayed < 0.01

    def test_constraint_never_decays(self):
        engine = DecayEngine()
        old = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        entry = make_entry(updated_at=old, type=MemoryType.CONSTRAINT, importance=ImportanceLevel.SOFT)
        assert engine.compute_decayed_confidence(entry) == 1.0

    def test_step_cutoff_boundary(self):
        engine = DecayEngine()
        just_before = (datetime.now(timezone.utc) - timedelta(days=6.9)).isoformat()
        just_after = (datetime.now(timezone.utc) - timedelta(days=7.1)).isoformat()

        before = make_entry(updated_at=just_before, type=MemoryType.TASK, importance=ImportanceLevel.SOFT)
        after = make_entry(updated_at=just_after, type=MemoryType.TASK, importance=ImportanceLevel.SOFT)

        assert engine.compute_decayed_confidence(before) == 1.0
        assert engine.compute_decayed_confidence(after) == 0.0

    def test_linear_decay(self):
        engine = DecayEngine(DecayConfig(
            policies={MemoryType.FACT: DecayTypeConfig(policy=DecayPolicy.LINEAR, rate_per_day=0.1)},
        ))
        ten_days = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        entry = make_entry(updated_at=ten_days, type=MemoryType.FACT, importance=ImportanceLevel.SOFT)
        assert engine.compute_decayed_confidence(entry) == pytest.approx(0.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_cleanup_fresh_entries(self):
        memory = AgentMemory()
        await memory.store(content="fresh task", type=MemoryType.TASK)
        deleted = await memory.cleanup()
        assert deleted == 0


# ─── Checkpoint Edge Cases ───

class TestCheckpoint:
    @pytest.mark.asyncio
    async def test_nonexistent_checkpoint(self):
        memory = AgentMemory()
        assert await memory.rehydrate("nonexistent") is None

    @pytest.mark.asyncio
    async def test_rehydrate_with_deleted_memory(self):
        memory = AgentMemory()
        entry = await memory.store(content="will be deleted")
        cp = await memory.checkpoint(
            task_graph=[], summary="test", active_memory_ids=[entry.id],
        )
        await memory.delete(entry.id)
        restored = await memory.rehydrate(cp.id)
        assert restored is not None
        assert len(restored["memories"]) == 0

    @pytest.mark.asyncio
    async def test_empty_checkpoint(self):
        memory = AgentMemory()
        cp = await memory.checkpoint(task_graph=[], summary="")
        assert cp.task_graph == []
        assert cp.summary == ""

    @pytest.mark.asyncio
    async def test_no_latest_checkpoint(self):
        memory = AgentMemory()
        assert memory.get_latest_checkpoint() is None


# ─── Utilities ───

class TestUtils:
    def test_cosine_zero_vectors(self):
        assert cosine_similarity([0, 0, 0], [0, 0, 0]) == 0.0

    def test_cosine_identical(self):
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_opposite(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0, abs=1e-5)

    def test_cosine_different_lengths(self):
        assert cosine_similarity([1, 2], [1, 2, 3]) == 0.0

    def test_cosine_empty(self):
        assert cosine_similarity([], []) == 0.0

    def test_unique_ids(self):
        ids = {generate_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_days_since_now(self):
        assert days_since(datetime.now(timezone.utc).isoformat()) == pytest.approx(0, abs=0.01)


# ─── Embedder ───

class TestEmbedder:
    @pytest.mark.asyncio
    async def test_empty_string(self):
        embedder = BuiltinEmbedder(64)
        vec = await embedder.embed("")
        assert len(vec) == 64
        assert all(v == 0 for v in vec)

    @pytest.mark.asyncio
    async def test_deterministic(self):
        embedder = BuiltinEmbedder(64)
        v1 = await embedder.embed("hello world")
        v2 = await embedder.embed("hello world")
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_different_texts(self):
        embedder = BuiltinEmbedder(64)
        v1 = await embedder.embed("hello world")
        v2 = await embedder.embed("machine learning algorithms")
        assert v1 != v2

    @pytest.mark.asyncio
    async def test_normalized(self):
        embedder = BuiltinEmbedder(64)
        vec = await embedder.embed("test embedding normalization check")
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        embedder = BuiltinEmbedder(32)
        results = await embedder.embed_batch(["ab cd", "ef gh", "ij kl"])
        assert len(results) == 3
        assert len(results[0]) == 32


# ─── Concurrency ───

class TestConcurrency:
    @pytest.mark.asyncio
    async def test_rapid_stores(self):
        memory = AgentMemory()
        entries = await asyncio.gather(
            *[memory.store(content=f"Memory entry number {i}", scope="test") for i in range(50)]
        )
        assert len(entries) == 50
        ids = {e.id for e in entries}
        assert len(ids) == 50
        assert len(await memory.get_all("test")) == 50

    @pytest.mark.asyncio
    async def test_store_then_retrieve(self):
        memory = AgentMemory()
        await memory.store(content="Python is great for data science")
        results = await memory.retrieve(RetrievalQuery(query="Python data science"))
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_double_delete(self):
        memory = AgentMemory()
        entry = await memory.store(content="test")
        assert await memory.delete(entry.id) is True
        assert await memory.delete(entry.id) is False


# ─── Custom Config ───

class TestConfig:
    @pytest.mark.asyncio
    async def test_custom_scope(self):
        memory = AgentMemory(default_scope="my-agent")
        entry = await memory.store(content="test")
        assert entry.scope == "my-agent"

    @pytest.mark.asyncio
    async def test_custom_embedder(self):
        class MockEmbedder:
            async def embed(self, text: str) -> list[float]:
                vec = [0.0] * 8
                vec[len(text) % 8] = 1.0
                return vec
            async def embed_batch(self, texts):
                return [await self.embed(t) for t in texts]
            def dimensions(self):
                return 8

        memory = AgentMemory(embedder=MockEmbedder())
        entry = await memory.store(content="test with custom embedder")
        assert entry.embedding is not None
        assert len(entry.embedding) == 8

    @pytest.mark.asyncio
    async def test_version_increments(self):
        memory = AgentMemory()
        e = await memory.store(content="v1")
        assert e.version == 1
        e2 = await memory.update(e.id, content="v2")
        assert e2.version == 2
        e3 = await memory.update(e.id, content="v3")
        assert e3.version == 3
        e4 = await memory.update(e.id, metadata={"note": "just metadata"})
        assert e4.version == 4
        assert e4.content == "v3"

    @pytest.mark.asyncio
    async def test_no_reembed_on_metadata_update(self):
        embed_count = 0

        class CountingEmbedder:
            async def embed(self, text: str) -> list[float]:
                nonlocal embed_count
                embed_count += 1
                return [0.1] * 8
            async def embed_batch(self, texts):
                return [await self.embed(t) for t in texts]
            def dimensions(self):
                return 8

        memory = AgentMemory(embedder=CountingEmbedder())
        entry = await memory.store(content="test")
        assert embed_count == 1
        await memory.update(entry.id, metadata={"tag": "updated"})
        assert embed_count == 1  # no re-embed
