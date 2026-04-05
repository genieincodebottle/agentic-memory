"""AgentMemory - the main entry point."""

from __future__ import annotations

from typing import Any, Optional

from .types import (
    MemoryEntry,
    MemoryType,
    ImportanceLevel,
    RetrievalQuery,
    RetrievalResult,
    ConflictResult,
    Checkpoint,
    TaskNode,
    DecayConfig,
)
from .utils import generate_id, now
from .store.local import LocalStore
from .retrieval.embedder import BuiltinEmbedder
from .retrieval.multi_signal import MultiSignalRetriever
from .retrieval.conflict import ConflictDetector
from .decay import DecayEngine


class AgentMemory:
    """
    Framework-agnostic memory layer for AI agents.

    Example::

        memory = AgentMemory()

        await memory.store(
            content="User prefers dark mode",
            type=MemoryType.PREFERENCE,
            scope="user:123",
        )

        results = await memory.retrieve(RetrievalQuery(query="UI preferences"))
        conflicts = await memory.check_conflicts("User prefers light mode", "user:123")
    """

    def __init__(
        self,
        *,
        store: Any = None,
        embedder: Any = None,
        decay: DecayConfig | None = None,
        default_scope: str = "default",
        max_checkpoints: int = 10,
    ) -> None:
        self._backend = store if store is not None else LocalStore()
        self._embedder = embedder if embedder is not None else BuiltinEmbedder()
        self._retriever = MultiSignalRetriever(self._backend, self._embedder)
        self._conflict_detector = ConflictDetector(self._backend, self._embedder)
        self._decay_engine = DecayEngine(decay)
        self._checkpoints: dict[str, Checkpoint] = {}
        self._default_scope = default_scope
        self._max_checkpoints = max_checkpoints

    # ─── Store ───

    async def store(
        self,
        content: str,
        *,
        type: MemoryType = MemoryType.FACT,
        scope: Optional[str] = None,
        importance: ImportanceLevel = ImportanceLevel.SOFT,
        confidence: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Store a new memory entry."""
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty")

        # Train before embed so IDF stats include this document
        if isinstance(self._embedder, BuiltinEmbedder):
            self._embedder.train([content])

        embedding = await self._embedder.embed(content)

        entry = MemoryEntry(
            id=generate_id(),
            content=content,
            type=type,
            scope=scope or self._default_scope,
            importance=importance,
            confidence=confidence,
            embedding=embedding,
            metadata=metadata or {},
            created_at=now(),
            updated_at=now(),
            version=1,
        )

        await self._backend.set(entry)
        return entry

    async def update(
        self,
        id: str,
        *,
        content: Optional[str] = None,
        type: Optional[MemoryType] = None,
        importance: Optional[ImportanceLevel] = None,
        confidence: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[MemoryEntry]:
        """Update an existing memory entry."""
        entry = await self._backend.get(id)
        if entry is None:
            return None

        if content is not None:
            if not content.strip():
                raise ValueError("Memory content cannot be empty")
            entry.content = content

        if type is not None:
            entry.type = type
        if importance is not None:
            entry.importance = importance
        if confidence is not None:
            entry.confidence = confidence
        if metadata is not None:
            entry.metadata = metadata

        entry.updated_at = now()
        entry.version += 1

        # Re-embed if content changed
        if content is not None:
            if isinstance(self._embedder, BuiltinEmbedder):
                self._embedder.train([content])
            entry.embedding = await self._embedder.embed(content)

        await self._backend.set(entry)
        return entry

    async def get(self, id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID."""
        return await self._backend.get(id)

    async def delete(self, id: str) -> bool:
        """Delete a memory by ID."""
        return await self._backend.delete(id)

    async def get_all(self, scope: Optional[str] = None) -> list[MemoryEntry]:
        """Get all memories, optionally filtered by scope."""
        return await self._backend.get_all(scope)

    async def clear(self, scope: Optional[str] = None) -> None:
        """Clear all memories, optionally for a specific scope."""
        return await self._backend.clear(scope)

    # ─── Retrieve ───

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Multi-signal retrieval."""
        return await self._retriever.retrieve(query)

    # ─── Conflict Detection ───

    async def check_conflicts(
        self, content: str, scope: Optional[str] = None
    ) -> list[ConflictResult]:
        """Check if content conflicts with stored memories."""
        return await self._conflict_detector.check(
            content, scope or self._default_scope
        )

    # ─── Decay ───

    def get_decayed_confidence(self, entry: MemoryEntry) -> float:
        """Get the current effective confidence after decay."""
        return self._decay_engine.compute_decayed_confidence(entry)

    async def cleanup(
        self, scope: Optional[str] = None, threshold: float = 0.01
    ) -> int:
        """Clean up expired memories. Returns count deleted."""
        entries = await self._backend.get_all(scope)
        expired = self._decay_engine.get_expired(entries, threshold)
        for entry in expired:
            await self._backend.delete(entry.id)
        return len(expired)

    # ─── Checkpointing ───

    async def checkpoint(
        self,
        *,
        task_graph: list[TaskNode],
        summary: str,
        tool_outputs: Optional[dict[str, Any]] = None,
        active_memory_ids: Optional[list[str]] = None,
    ) -> Checkpoint:
        """Create a checkpoint of current task state."""
        cp = Checkpoint(
            id=generate_id(),
            task_graph=task_graph,
            summary=summary,
            tool_outputs=tool_outputs or {},
            active_memory_ids=active_memory_ids or [],
            created_at=now(),
        )

        self._checkpoints[cp.id] = cp

        # Enforce max checkpoints (LRU eviction)
        if len(self._checkpoints) > self._max_checkpoints:
            oldest_key = next(iter(self._checkpoints))
            del self._checkpoints[oldest_key]

        return cp

    async def rehydrate(
        self, checkpoint_id: str
    ) -> Optional[dict[str, Any]]:
        """Rehydrate from a checkpoint. Returns checkpoint + memories."""
        cp = self._checkpoints.get(checkpoint_id)
        if cp is None:
            return None

        memories: list[MemoryEntry] = []
        for mid in cp.active_memory_ids:
            entry = await self._backend.get(mid)
            if entry is not None:
                memories.append(entry)

        return {"checkpoint": cp, "memories": memories}

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get the latest checkpoint."""
        if not self._checkpoints:
            return None
        return list(self._checkpoints.values())[-1]

    def list_checkpoints(self) -> list[Checkpoint]:
        """List all checkpoints."""
        return list(self._checkpoints.values())
