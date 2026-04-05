import { describe, it, expect, afterEach } from 'vitest';
import { AgentMemory } from '../src/core/memory.js';
import { FileStore } from '../src/store/file.js';
import { existsSync, unlinkSync } from 'fs';
import { join } from 'path';

const TEST_FILE = join(__dirname, '.test-memory.json');

describe('FileStore', () => {
  afterEach(() => {
    if (existsSync(TEST_FILE)) {
      unlinkSync(TEST_FILE);
    }
  });

  it('should persist and reload memories from file', async () => {
    // Store with first instance
    const store1 = new FileStore(TEST_FILE);
    const memory1 = new AgentMemory({ store: store1 });
    await memory1.store({ content: 'Persisted memory', scope: 'test' });

    // Load with second instance
    const store2 = new FileStore(TEST_FILE);
    const memory2 = new AgentMemory({ store: store2 });
    const all = await memory2.getAll('test');

    expect(all.length).toBe(1);
    expect(all[0].content).toBe('Persisted memory');
  });

  it('should handle empty/corrupted file gracefully', async () => {
    const store = new FileStore(join(__dirname, '.nonexistent.json'));
    const memory = new AgentMemory({ store });
    const all = await memory.getAll();
    expect(all.length).toBe(0);
  });
});
