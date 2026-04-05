import type {
  MemoryEntry,
  RetrievalQuery,
  RetrievalResult,
  RetrievalSignal,
  Embedder,
  StorageBackend,
} from '../core/types.js';
import { cosineSimilarity, daysSince, normalize } from '../core/utils.js';

/** Default signal weights */
const DEFAULT_WEIGHTS: Record<RetrievalSignal, number> = {
  similarity: 0.4,
  recency: 0.25,
  importance: 0.2,
  taskRelevance: 0.15,
};

/** Importance level to numeric score */
const IMPORTANCE_SCORES: Record<string, number> = {
  hard: 1.0,
  soft: 0.5,
  ephemeral: 0.2,
};

/**
 * Multi-signal retriever.
 * Combines similarity, recency, importance, and task-relevance
 * into a single ranked result set.
 */
export class MultiSignalRetriever {
  private store: StorageBackend;
  private embedder: Embedder;
  private weights: Record<RetrievalSignal, number>;

  constructor(
    store: StorageBackend,
    embedder: Embedder,
    weights?: Partial<Record<RetrievalSignal, number>>
  ) {
    this.store = store;
    this.embedder = embedder;
    this.weights = { ...DEFAULT_WEIGHTS, ...weights };
  }

  async retrieve(query: RetrievalQuery): Promise<RetrievalResult[]> {
    const signals = query.signals ?? ['similarity', 'recency', 'importance'];
    const limit = query.limit ?? 10;
    const threshold = query.threshold ?? 0;

    // Get candidate entries from store
    const candidates = await this.store.search(query);
    if (candidates.length === 0) return [];

    // Compute query embedding if similarity is requested
    let queryEmbedding: number[] | null = null;
    if (signals.includes('similarity')) {
      queryEmbedding = await this.embedder.embed(query.query);
    }

    // Compute task context embedding if task relevance is requested
    let taskEmbedding: number[] | null = null;
    if (signals.includes('taskRelevance') && query.taskContext) {
      taskEmbedding = await this.embedder.embed(query.taskContext);
    }

    // Score each candidate
    const scored: RetrievalResult[] = [];
    for (const entry of candidates) {
      const signalScores: Partial<Record<RetrievalSignal, number>> = {};
      let totalScore = 0;
      let totalWeight = 0;

      // Cache entry embedding - avoid re-embedding for both similarity and taskRelevance
      let entryEmbedding: number[] | null = null;
      const needsEmbedding = (signals.includes('similarity') && queryEmbedding)
        || (signals.includes('taskRelevance') && taskEmbedding);
      if (needsEmbedding) {
        entryEmbedding = entry.embedding ?? await this.embedder.embed(entry.content);
      }

      // Similarity score
      if (signals.includes('similarity') && queryEmbedding && entryEmbedding) {
        signalScores.similarity = Math.max(0, cosineSimilarity(queryEmbedding, entryEmbedding));
        totalScore += signalScores.similarity * this.weights.similarity;
        totalWeight += this.weights.similarity;
      }

      // Recency score - exponential decay, half-life of 7 days
      if (signals.includes('recency')) {
        const age = Math.max(0, daysSince(entry.updatedAt)); // clamp negative (future timestamps)
        signalScores.recency = Math.exp(-0.693 * age / 7); // ln(2)/7 * age
        totalScore += signalScores.recency * this.weights.recency;
        totalWeight += this.weights.recency;
      }

      // Importance score
      if (signals.includes('importance')) {
        signalScores.importance = IMPORTANCE_SCORES[entry.importance] ?? 0.5;
        totalScore += signalScores.importance * this.weights.importance;
        totalWeight += this.weights.importance;
      }

      // Task relevance score
      if (signals.includes('taskRelevance') && taskEmbedding && entryEmbedding) {
        signalScores.taskRelevance = Math.max(0, cosineSimilarity(taskEmbedding, entryEmbedding));
        totalScore += signalScores.taskRelevance * this.weights.taskRelevance;
        totalWeight += this.weights.taskRelevance;
      }

      // Normalize by total weight of active signals
      const normalizedScore = totalWeight > 0 ? totalScore / totalWeight : 0;

      if (normalizedScore >= threshold) {
        scored.push({ entry, score: normalizedScore, signalScores });
      }
    }

    // Sort by score descending, return top N
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
  }
}
