import type {
  AgentMemoryConfig,
  MemoryEntry,
  MemoryType,
  ImportanceLevel,
  RetrievalQuery,
  RetrievalResult,
  ConflictResult,
  StorageBackend,
  Embedder,
  Checkpoint,
  TaskNode,
} from './types.js';
import { generateId, now } from './utils.js';
import { LocalStore } from '../store/local.js';
import { BuiltinEmbedder } from '../retrieval/embedder.js';
import { MultiSignalRetriever } from '../retrieval/multi-signal.js';
import { ConflictDetector } from '../retrieval/conflict.js';
import { DecayEngine } from './decay.js';

/**
 * AgentMemory - the main entry point.
 *
 * Framework-agnostic memory layer for AI agents.
 * Handles storage, multi-signal retrieval, conflict detection,
 * typed decay, and checkpointing.
 *
 * @example
 * ```typescript
 * const memory = new AgentMemory();
 *
 * await memory.store({
 *   content: 'User prefers dark mode',
 *   type: 'preference',
 *   scope: 'user:123',
 * });
 *
 * const results = await memory.retrieve({ query: 'UI preferences' });
 * const conflicts = await memory.checkConflicts('User prefers light mode', 'user:123');
 * ```
 */
export class AgentMemory {
  private backend: StorageBackend;
  private embedder: Embedder;
  private retriever: MultiSignalRetriever;
  private conflictDetector: ConflictDetector;
  private decayEngine: DecayEngine;
  private checkpoints: Map<string, Checkpoint> = new Map();
  private defaultScope: string;
  private maxCheckpoints: number;

  constructor(config: AgentMemoryConfig = {}) {
    // Initialize storage backend
    this.backend = config.store === 'local' || !config.store
      ? new LocalStore()
      : config.store;

    // Initialize embedder
    this.embedder = config.embedder === 'builtin' || !config.embedder
      ? new BuiltinEmbedder()
      : config.embedder;

    // Initialize retriever and conflict detector
    this.retriever = new MultiSignalRetriever(this.backend, this.embedder);
    this.conflictDetector = new ConflictDetector(this.backend, this.embedder);
    this.decayEngine = new DecayEngine(config.decay);

    this.defaultScope = config.defaultScope ?? 'default';
    this.maxCheckpoints = config.checkpoint?.maxCheckpoints ?? 10;
  }

  // ─── Store ───

  /**
   * Store a new memory entry.
   * Automatically generates ID, timestamps, and embedding.
   */
  async store(params: {
    content: string;
    type?: MemoryType;
    scope?: string;
    importance?: ImportanceLevel;
    confidence?: number;
    metadata?: Record<string, unknown>;
  }): Promise<MemoryEntry> {
    if (!params.content || params.content.trim().length === 0) {
      throw new Error('Memory content cannot be empty');
    }

    // Train the builtin embedder BEFORE embedding so IDF stats include this document
    if (this.embedder instanceof BuiltinEmbedder) {
      this.embedder.train([params.content]);
    }

    const embedding = await this.embedder.embed(params.content);

    const entry: MemoryEntry = {
      id: generateId(),
      content: params.content,
      type: params.type ?? 'fact',
      scope: params.scope ?? this.defaultScope,
      importance: params.importance ?? 'soft',
      confidence: params.confidence ?? 1.0,
      embedding,
      metadata: params.metadata ?? {},
      createdAt: now(),
      updatedAt: now(),
      version: 1,
    };

    await this.backend.set(entry);
    return entry;
  }

  /**
   * Update an existing memory entry.
   * Increments version and updates timestamp.
   */
  async update(id: string, updates: Partial<Pick<MemoryEntry, 'content' | 'type' | 'importance' | 'confidence' | 'metadata'>>): Promise<MemoryEntry | null> {
    const entry = await this.backend.get(id);
    if (!entry) return null;

    const updated: MemoryEntry = {
      ...entry,
      ...updates,
      updatedAt: now(),
      version: entry.version + 1,
    };

    // Re-embed if content changed
    if (updates.content && updates.content !== entry.content) {
      if (updates.content.trim().length === 0) {
        throw new Error('Memory content cannot be empty');
      }
      if (this.embedder instanceof BuiltinEmbedder) {
        this.embedder.train([updates.content]);
      }
      updated.embedding = await this.embedder.embed(updates.content);
    }

    await this.backend.set(updated);
    return updated;
  }

  /** Get a memory by ID */
  async get(id: string): Promise<MemoryEntry | null> {
    return this.backend.get(id);
  }

  /** Delete a memory by ID */
  async delete(id: string): Promise<boolean> {
    return this.backend.delete(id);
  }

  /** Get all memories, optionally filtered by scope */
  async getAll(scope?: string): Promise<MemoryEntry[]> {
    return this.backend.getAll(scope);
  }

  /** Clear all memories, optionally for a specific scope only */
  async clear(scope?: string): Promise<void> {
    return this.backend.clear(scope);
  }

  // ─── Retrieve ───

  /**
   * Multi-signal retrieval.
   * Combines similarity, recency, importance, and task-relevance.
   */
  async retrieve(query: RetrievalQuery): Promise<RetrievalResult[]> {
    return this.retriever.retrieve(query);
  }

  // ─── Conflict Detection ───

  /**
   * Check if content conflicts with stored memories.
   * Returns conflicts sorted by confidence.
   *
   * @example
   * ```typescript
   * const conflicts = await memory.checkConflicts('User likes meat', 'user:123');
   * if (conflicts.length > 0 && conflicts[0].action === 'clarify') {
   *   // Ask user to confirm preference change
   * }
   * ```
   */
  async checkConflicts(content: string, scope?: string): Promise<ConflictResult[]> {
    return this.conflictDetector.check(content, scope ?? this.defaultScope);
  }

  // ─── Decay ───

  /**
   * Get the current effective confidence of a memory after decay.
   */
  getDecayedConfidence(entry: MemoryEntry): number {
    return this.decayEngine.computeDecayedConfidence(entry);
  }

  /**
   * Clean up expired memories (decayed below threshold).
   * Returns the number of deleted entries.
   */
  async cleanup(scope?: string, threshold = 0.01): Promise<number> {
    const entries = await this.backend.getAll(scope);
    const expired = this.decayEngine.getExpired(entries, threshold);
    for (const entry of expired) {
      await this.backend.delete(entry.id);
    }
    return expired.length;
  }

  // ─── Checkpointing ───

  /**
   * Create a checkpoint of current task state.
   * Use this before context overflow to preserve state.
   */
  async checkpoint(params: {
    taskGraph: TaskNode[];
    summary: string;
    toolOutputs?: Record<string, unknown>;
    activeMemoryIds?: string[];
  }): Promise<Checkpoint> {
    const cp: Checkpoint = {
      id: generateId(),
      taskGraph: params.taskGraph,
      summary: params.summary,
      toolOutputs: params.toolOutputs ?? {},
      activeMemoryIds: params.activeMemoryIds ?? [],
      createdAt: now(),
    };

    this.checkpoints.set(cp.id, cp);

    // Enforce max checkpoints (LRU eviction)
    if (this.checkpoints.size > this.maxCheckpoints) {
      const oldest = this.checkpoints.keys().next().value;
      if (oldest) this.checkpoints.delete(oldest);
    }

    return cp;
  }

  /**
   * Rehydrate from a checkpoint.
   * Returns the checkpoint data + relevant memories.
   */
  async rehydrate(checkpointId: string): Promise<{
    checkpoint: Checkpoint;
    memories: MemoryEntry[];
  } | null> {
    const cp = this.checkpoints.get(checkpointId);
    if (!cp) return null;

    // Fetch the memories that were active at checkpoint time
    const memories: MemoryEntry[] = [];
    for (const id of cp.activeMemoryIds) {
      const entry = await this.backend.get(id);
      if (entry) memories.push(entry);
    }

    return { checkpoint: cp, memories };
  }

  /** Get the latest checkpoint */
  getLatestCheckpoint(): Checkpoint | null {
    const all = Array.from(this.checkpoints.values());
    if (all.length === 0) return null;
    return all[all.length - 1];
  }

  /** List all checkpoints */
  listCheckpoints(): Checkpoint[] {
    return Array.from(this.checkpoints.values());
  }
}
