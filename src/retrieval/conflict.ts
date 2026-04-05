import type {
  MemoryEntry,
  ConflictResult,
  Embedder,
  StorageBackend,
} from '../core/types.js';
import { cosineSimilarity } from '../core/utils.js';

/** Words/phrases that signal negation or opposition */
const NEGATION_PATTERNS = [
  /\bnot\b/i, /\bnever\b/i, /\bno\b/i, /\bdon't\b/i, /\bdoesn't\b/i,
  /\bwon't\b/i, /\bcan't\b/i, /\bhate\b/i, /\bdislike\b/i, /\bavoid\b/i,
  /\bstop\b/i, /\bquit\b/i, /\bremove\b/i, /\bdelete\b/i,
  /\binstead of\b/i, /\brather than\b/i, /\bno longer\b/i,
];

/** Words that signal preference changes */
const CHANGE_PATTERNS = [
  /\bnow\b/i, /\bactually\b/i, /\bchanged?\b/i, /\bswitch/i,
  /\bprefer\b/i, /\bwant\b/i, /\bused to\b/i, /\banymore\b/i,
];

/**
 * Conflict detector.
 * Compares new content against stored memories to find contradictions.
 * Uses semantic similarity + negation detection, not just keyword matching.
 */
export class ConflictDetector {
  private store: StorageBackend;
  private embedder: Embedder;
  /** Similarity threshold to consider two entries as "same topic" */
  private topicThreshold: number;

  constructor(
    store: StorageBackend,
    embedder: Embedder,
    topicThreshold = 0.6
  ) {
    this.store = store;
    this.embedder = embedder;
    this.topicThreshold = topicThreshold;
  }

  /**
   * Check if new content conflicts with any stored memories.
   * Returns conflicts sorted by confidence (highest first).
   */
  async check(content: string, scope?: string): Promise<ConflictResult[]> {
    const entries = await this.store.getAll(scope);
    if (entries.length === 0) return [];

    const contentEmbedding = await this.embedder.embed(content);
    const conflicts: ConflictResult[] = [];

    for (const entry of entries) {
      const entryEmbedding = entry.embedding ?? await this.embedder.embed(entry.content);
      const similarity = cosineSimilarity(contentEmbedding, entryEmbedding);

      // Only check entries on the same topic
      if (similarity < this.topicThreshold) continue;

      // Check for contradiction signals
      const contradictionScore = this.scoreContradiction(content, entry.content);
      if (contradictionScore <= 0) continue;

      // Combined confidence: high topic similarity + contradiction signals
      const confidence = Math.min(1, similarity * 0.5 + contradictionScore * 0.5);

      conflicts.push({
        incoming: content,
        stored: entry,
        confidence,
        action: this.suggestAction(entry, confidence),
        reason: this.generateReason(content, entry, contradictionScore),
      });
    }

    conflicts.sort((a, b) => b.confidence - a.confidence);
    return conflicts;
  }

  /**
   * Score how contradictory two pieces of text are.
   * Returns 0 (no contradiction) to 1 (strong contradiction).
   */
  private scoreContradiction(incoming: string, stored: string): number {
    let score = 0;

    // Check for negation in incoming that opposes stored
    const incomingHasNegation = NEGATION_PATTERNS.some(p => p.test(incoming));
    const storedHasNegation = NEGATION_PATTERNS.some(p => p.test(stored));

    // One is negated, the other isn't - likely contradiction
    if (incomingHasNegation !== storedHasNegation) {
      score += 0.5;
    }

    // Check for explicit change language
    const hasChangeLanguage = CHANGE_PATTERNS.some(p => p.test(incoming));
    if (hasChangeLanguage) {
      score += 0.3;
    }

    // Check for antonym pairs in the texts
    const antonymScore = this.checkAntonyms(incoming, stored);
    score += antonymScore * 0.4;

    return Math.min(1, score);
  }

  /** Check for common antonym pairs */
  private checkAntonyms(a: string, b: string): number {
    const antonyms: [RegExp, RegExp][] = [
      [/\blike\b/i, /\bdislike\b/i],
      [/\blove\b/i, /\bhate\b/i],
      [/\byes\b/i, /\bno\b/i],
      [/\btrue\b/i, /\bfalse\b/i],
      [/\bvegetarian\b/i, /\bmeat\b/i],
      [/\bvegan\b/i, /\bmeat\b|\bdairy\b/i],
      [/\bmorning\b/i, /\bevening\b|\bnight\b/i],
      [/\blight\b/i, /\bdark\b/i],
      [/\bhot\b/i, /\bcold\b/i],
      [/\bfast\b/i, /\bslow\b/i],
      [/\benable\b/i, /\bdisable\b/i],
      [/\ballow\b/i, /\bblock\b|\bdeny\b/i],
      [/\baccept\b/i, /\breject\b/i],
    ];

    let matches = 0;
    for (const [word1, word2] of antonyms) {
      if ((word1.test(a) && word2.test(b)) || (word2.test(a) && word1.test(b))) {
        matches++;
      }
    }
    return Math.min(1, matches * 0.5);
  }

  private suggestAction(stored: MemoryEntry, confidence: number): ConflictResult['action'] {
    // Hard constraints should always trigger clarification
    if (stored.importance === 'hard') return 'clarify';
    // High confidence conflicts with soft importance - suggest override
    if (confidence > 0.7 && stored.importance === 'soft') return 'override';
    // Ephemeral memories can be overridden easily
    if (stored.importance === 'ephemeral') return 'override';
    // Default: ask for clarification
    return 'clarify';
  }

  private generateReason(
    incoming: string,
    stored: MemoryEntry,
    contradictionScore: number
  ): string {
    if (stored.importance === 'hard') {
      return `Conflicts with a hard constraint: "${stored.content}". This memory was marked as non-negotiable - please confirm the change.`;
    }
    if (contradictionScore > 0.7) {
      return `Directly contradicts stored memory: "${stored.content}". The new statement appears to reverse a previous preference.`;
    }
    return `Potentially conflicts with: "${stored.content}". The statements may be inconsistent.`;
  }
}
