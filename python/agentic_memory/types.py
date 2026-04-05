"""Core types for agentic-memory."""

from __future__ import annotations

import time
import random
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol


# ─── Enums ───

class MemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    TASK = "task"
    EPISODIC = "episodic"
    CONSTRAINT = "constraint"


class ImportanceLevel(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    EPHEMERAL = "ephemeral"


class DecayPolicy(str, Enum):
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"


class ConflictAction(str, Enum):
    CLARIFY = "clarify"
    OVERRIDE = "override"
    KEEP_BOTH = "keep_both"
    IGNORE = "ignore"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


# ─── Data Classes ───

@dataclass
class MemoryEntry:
    id: str
    content: str
    type: MemoryType
    scope: str
    importance: ImportanceLevel
    confidence: float
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    version: int
    embedding: Optional[list[float]] = None


@dataclass
class RetrievalQuery:
    query: str
    task_context: Optional[str] = None
    scope: Optional[str] = None
    types: Optional[list[MemoryType]] = None
    signals: Optional[list[str]] = None
    limit: int = 10
    threshold: float = 0.0


@dataclass
class RetrievalResult:
    entry: MemoryEntry
    score: float
    signal_scores: dict[str, float]


@dataclass
class ConflictResult:
    incoming: str
    stored: MemoryEntry
    confidence: float
    action: ConflictAction
    reason: str


@dataclass
class TaskNode:
    id: str
    description: str
    status: TaskStatus
    dependencies: list[str]
    result: Any = None


@dataclass
class Checkpoint:
    id: str
    task_graph: list[TaskNode]
    summary: str
    tool_outputs: dict[str, Any]
    active_memory_ids: list[str]
    created_at: str


@dataclass
class DecayTypeConfig:
    policy: DecayPolicy
    half_life: Optional[float] = None  # milliseconds
    rate_per_day: Optional[float] = None
    max_age: Optional[float] = None  # milliseconds


@dataclass
class DecayConfig:
    policies: dict[MemoryType, DecayTypeConfig]
    default_policy: DecayPolicy = DecayPolicy.EXPONENTIAL
    default_half_life: float = 30 * 24 * 60 * 60 * 1000  # 30 days in ms


# ─── Protocols (Interfaces) ───

class StorageBackend(Protocol):
    async def get(self, id: str) -> Optional[MemoryEntry]: ...
    async def get_all(self, scope: Optional[str] = None) -> list[MemoryEntry]: ...
    async def set(self, entry: MemoryEntry) -> None: ...
    async def delete(self, id: str) -> bool: ...
    async def clear(self, scope: Optional[str] = None) -> None: ...
    async def search(self, query: RetrievalQuery) -> list[MemoryEntry]: ...


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    def dimensions(self) -> int: ...
