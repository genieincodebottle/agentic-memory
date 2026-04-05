import type { Embedder } from '../core/types.js';

/**
 * Built-in lightweight embedder using bag-of-words TF-IDF.
 * No external dependencies, no API calls.
 * Good enough for conflict detection and basic similarity.
 * For production, plug in OpenAI/Cohere/local embeddings.
 */
export class BuiltinEmbedder implements Embedder {
  private readonly dim: number;
  private vocabulary: Map<string, number> = new Map();
  private idfCache: Map<string, number> = new Map();
  private documentCount = 0;
  private documentFreq: Map<string, number> = new Map();

  constructor(dimensions = 256) {
    this.dim = dimensions;
  }

  dimensions(): number {
    return this.dim;
  }

  async embed(text: string): Promise<number[]> {
    const tokens = this.tokenize(text);
    return this.tokensToVector(tokens);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map(t => this.embed(t)));
  }

  /**
   * Feed documents to build vocabulary and IDF stats.
   * Call this when adding memories to improve embedding quality.
   */
  train(documents: string[]): void {
    for (const doc of documents) {
      this.documentCount++;
      const uniqueTokens = new Set(this.tokenize(doc));
      for (const token of uniqueTokens) {
        this.documentFreq.set(token, (this.documentFreq.get(token) ?? 0) + 1);
      }
    }
    // Rebuild IDF cache
    this.idfCache.clear();
    for (const [token, freq] of this.documentFreq) {
      this.idfCache.set(token, Math.log((this.documentCount + 1) / (freq + 1)) + 1);
    }
  }

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(t => t.length > 1);
  }

  private tokensToVector(tokens: string[]): number[] {
    const vector = new Array(this.dim).fill(0);

    // Term frequency
    const tf: Map<string, number> = new Map();
    for (const token of tokens) {
      tf.set(token, (tf.get(token) ?? 0) + 1);
    }

    // Hash each token to a dimension and accumulate TF-IDF weight
    for (const [token, freq] of tf) {
      const idf = this.idfCache.get(token) ?? 1;
      const weight = (freq / tokens.length) * idf;
      // Deterministic hash to map token to dimension
      const hash = this.hashToken(token);
      const idx = Math.abs(hash) % this.dim;
      // Use sign of secondary hash for +/- (random projection)
      const sign = this.hashToken(token + '_sign') > 0 ? 1 : -1;
      vector[idx] += sign * weight;
    }

    // L2 normalize
    const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
    if (norm > 0) {
      for (let i = 0; i < vector.length; i++) {
        vector[i] /= norm;
      }
    }

    return vector;
  }

  private hashToken(token: string): number {
    let hash = 0;
    for (let i = 0; i < token.length; i++) {
      const char = token.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash |= 0; // Convert to 32-bit int
    }
    return hash;
  }
}
