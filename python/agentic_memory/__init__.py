"""agentic-memory: Framework-agnostic memory layer for AI agents."""

from .memory import AgentMemory
from .decay import DecayEngine
from .types import (
    MemoryEntry,
    MemoryType,
    ImportanceLevel,
    DecayPolicy,
    DecayConfig,
    DecayTypeConfig,
    RetrievalQuery,
    RetrievalResult,
    ConflictResult,
    ConflictAction,
    Checkpoint,
    TaskNode,
    TaskStatus,
)
from .store import LocalStore, FileStore
from .retrieval import BuiltinEmbedder, MultiSignalRetriever, ConflictDetector
from .utils import cosine_similarity, generate_id

__version__ = "0.1.0"

__all__ = [
    "AgentMemory",
    "DecayEngine",
    "MemoryEntry",
    "MemoryType",
    "ImportanceLevel",
    "DecayPolicy",
    "DecayConfig",
    "DecayTypeConfig",
    "RetrievalQuery",
    "RetrievalResult",
    "ConflictResult",
    "ConflictAction",
    "Checkpoint",
    "TaskNode",
    "TaskStatus",
    "LocalStore",
    "FileStore",
    "BuiltinEmbedder",
    "MultiSignalRetriever",
    "ConflictDetector",
    "cosine_similarity",
    "generate_id",
]
