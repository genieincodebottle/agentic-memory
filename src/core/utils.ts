/** Generate a unique ID */
export function generateId(): string {
  return `mem_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 9)}`;
}

/** Get current ISO timestamp */
export function now(): string {
  return new Date().toISOString();
}

/** Cosine similarity between two vectors */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dotProduct / denom;
}

/** Normalize a score to 0-1 range */
export function normalize(value: number, min: number, max: number): number {
  if (max === min) return 0.5;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

/** Time difference in days from ISO timestamp to now */
export function daysSince(isoTimestamp: string): number {
  const diff = Date.now() - new Date(isoTimestamp).getTime();
  return diff / (1000 * 60 * 60 * 24);
}
