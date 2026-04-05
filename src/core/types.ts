// ─── Memory Entry Types ───

export type MemoryType = 'preference' | 'fact' | 'task' | 'episodic' | 'constraint';

export type ImportanceLevel = 'hard' | 'soft' | 'ephemeral';

export type DecayPolicy = 'none' | 'linear' | 'exponential' | 'step';

export interface MemoryEntry {
  /** Unique identifier */
  id: string;
  /** The content of the memory */
  content: string;
  /** Semantic type - determines decay and conflict behavior */
  type: MemoryType;
  /** Namespace for isolation (e.g., 'user:123', 'agent:planner') */
  scope: string;
  /** How important this memory is - 'hard' never decays */
  importance: ImportanceLevel;
  /** Numeric confidence score 0-1 */
  confidence: number;
  /** Embedding vector (populated by retriever) */
  embedding?: number[];
  /** Arbitrary metadata */
  metadata: Record<string, unknown>;
  /** ISO timestamp of creation */
  createdAt: string;
  /** ISO timestamp of last update */
  updatedAt: string;
  /** Version counter for optimistic concurrency */
  version: number;
}

// ─── Retrieval Types ───

export type RetrievalSignal = 'similarity' | 'recency' | 'importance' | 'taskRelevance';

export interface RetrievalQuery {
  /** The search query text */
  query: string;
  /** Current task context for task-relevance scoring */
  taskContext?: string;
  /** Scope to search within */
  scope?: string;
  /** Memory types to include */
  types?: MemoryType[];
  /** Signals to combine for ranking */
  signals?: RetrievalSignal[];
  /** Max results to return */
  limit?: number;
  /** Minimum combined score threshold (0-1) */
  threshold?: number;
}

export interface RetrievalResult {
  entry: MemoryEntry;
  /** Combined score from all signals */
  score: number;
  /** Breakdown of individual signal scores */
  signalScores: Partial<Record<RetrievalSignal, number>>;
}

// ─── Conflict Types ───

export interface ConflictResult {
  /** The new content that conflicts */
  incoming: string;
  /** The stored entry it conflicts with */
  stored: MemoryEntry;
  /** Confidence that this is a real conflict (0-1) */
  confidence: number;
  /** Suggested action */
  action: 'clarify' | 'override' | 'keep_both' | 'ignore';
  /** Reason for the suggestion */
  reason: string;
}

// ─── Checkpoint Types ───

export interface Checkpoint {
  id: string;
  /** Structured task state */
  taskGraph: TaskNode[];
  /** Compressed summary of conversation so far */
  summary: string;
  /** Key tool outputs to preserve */
  toolOutputs: Record<string, unknown>;
  /** Memories that were in active use */
  activeMemoryIds: string[];
  /** ISO timestamp */
  createdAt: string;
}

export interface TaskNode {
  id: string;
  description: string;
  status: 'pending' | 'in_progress' | 'done' | 'failed' | 'skipped';
  result?: unknown;
  dependencies: string[];
}

// ─── Store Backend Interface ───

export interface StorageBackend {
  get(id: string): Promise<MemoryEntry | null>;
  getAll(scope?: string): Promise<MemoryEntry[]>;
  set(entry: MemoryEntry): Promise<void>;
  delete(id: string): Promise<boolean>;
  clear(scope?: string): Promise<void>;
  search(query: RetrievalQuery): Promise<MemoryEntry[]>;
}

// ─── Embedder Interface ───

export interface Embedder {
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
  dimensions(): number;
}

// ─── Decay Config ───

export interface DecayConfig {
  /** Policy per memory type */
  policies: Partial<Record<MemoryType, {
    policy: DecayPolicy;
    /** Half-life in milliseconds (for exponential) */
    halfLife?: number;
    /** Linear decay rate per day (0-1) */
    ratePerDay?: number;
    /** Step function: drop to 0 after this many ms */
    maxAge?: number;
  }>>;
  /** Default policy for unlisted types */
  defaultPolicy: DecayPolicy;
  /** Default half-life in ms */
  defaultHalfLife: number;
}

// ─── Config ───

export interface AgentMemoryConfig {
  /** Storage backend - 'local' uses in-memory Map */
  store?: 'local' | StorageBackend;
  /** Embedder for similarity search */
  embedder?: 'builtin' | Embedder;
  /** Decay configuration */
  decay?: Partial<DecayConfig>;
  /** Auto-checkpoint config */
  checkpoint?: {
    /** Context usage ratio to trigger checkpoint (0-1) */
    threshold?: number;
    /** Max checkpoints to retain */
    maxCheckpoints?: number;
  };
  /** Default scope for entries without explicit scope */
  defaultScope?: string;
}
