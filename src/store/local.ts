import type { StorageBackend, MemoryEntry, RetrievalQuery } from '../core/types.js';

/**
 * In-memory storage backend. Zero dependencies.
 * Perfect for development, testing, and single-process agents.
 * Data is lost when the process exits.
 */
export class LocalStore implements StorageBackend {
  private entries: Map<string, MemoryEntry> = new Map();

  async get(id: string): Promise<MemoryEntry | null> {
    return this.entries.get(id) ?? null;
  }

  async getAll(scope?: string): Promise<MemoryEntry[]> {
    const all = Array.from(this.entries.values());
    if (!scope) return all;
    return all.filter(e => e.scope === scope);
  }

  async set(entry: MemoryEntry): Promise<void> {
    this.entries.set(entry.id, { ...entry });
  }

  async delete(id: string): Promise<boolean> {
    return this.entries.delete(id);
  }

  async clear(scope?: string): Promise<void> {
    if (!scope) {
      this.entries.clear();
      return;
    }
    for (const [id, entry] of this.entries) {
      if (entry.scope === scope) {
        this.entries.delete(id);
      }
    }
  }

  async search(query: RetrievalQuery): Promise<MemoryEntry[]> {
    let results = Array.from(this.entries.values());

    if (query.scope) {
      results = results.filter(e => e.scope === query.scope);
    }

    if (query.types && query.types.length > 0) {
      results = results.filter(e => query.types!.includes(e.type));
    }

    return results;
  }

  /** Get the number of stored entries */
  get size(): number {
    return this.entries.size;
  }
}
