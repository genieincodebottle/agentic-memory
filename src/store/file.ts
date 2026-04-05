import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import type { StorageBackend, MemoryEntry, RetrievalQuery } from '../core/types.js';

/**
 * File-based storage backend. Persists to a JSON file on disk.
 * Good for single-process agents that need persistence across restarts.
 * Not suitable for concurrent multi-agent access - use Redis/Postgres for that.
 */
export class FileStore implements StorageBackend {
  private entries: Map<string, MemoryEntry> = new Map();
  private filePath: string;
  private dirty = false;

  constructor(filePath: string) {
    this.filePath = filePath;
    this.load();
  }

  private load(): void {
    if (existsSync(this.filePath)) {
      try {
        const raw = readFileSync(this.filePath, 'utf-8');
        const data: MemoryEntry[] = JSON.parse(raw);
        for (const entry of data) {
          this.entries.set(entry.id, entry);
        }
      } catch {
        // Corrupted file - start fresh
        this.entries.clear();
      }
    }
  }

  private persist(): void {
    if (!this.dirty) return;
    const dir = dirname(this.filePath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
    const data = Array.from(this.entries.values());
    writeFileSync(this.filePath, JSON.stringify(data, null, 2), 'utf-8');
    this.dirty = false;
  }

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
    this.dirty = true;
    this.persist();
  }

  async delete(id: string): Promise<boolean> {
    const deleted = this.entries.delete(id);
    if (deleted) {
      this.dirty = true;
      this.persist();
    }
    return deleted;
  }

  async clear(scope?: string): Promise<void> {
    if (!scope) {
      this.entries.clear();
    } else {
      for (const [id, entry] of this.entries) {
        if (entry.scope === scope) {
          this.entries.delete(id);
        }
      }
    }
    this.dirty = true;
    this.persist();
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

  get size(): number {
    return this.entries.size;
  }
}
