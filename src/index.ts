// Core
export { AgentMemory } from './core/memory.js';
export { DecayEngine } from './core/decay.js';

// Types
export type {
  AgentMemoryConfig,
  MemoryEntry,
  MemoryType,
  ImportanceLevel,
  DecayPolicy,
  DecayConfig,
  RetrievalQuery,
  RetrievalResult,
  RetrievalSignal,
  ConflictResult,
  Checkpoint,
  TaskNode,
  StorageBackend,
  Embedder,
} from './core/types.js';

// Storage backends
export { LocalStore } from './store/local.js';
export { FileStore } from './store/file.js';

// Retrieval
export { MultiSignalRetriever } from './retrieval/multi-signal.js';
export { ConflictDetector } from './retrieval/conflict.js';
export { BuiltinEmbedder } from './retrieval/embedder.js';

// Utilities
export { cosineSimilarity, generateId } from './core/utils.js';
