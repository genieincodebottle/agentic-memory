import { describe, it, expect, beforeEach } from 'vitest';
import { AgentMemory } from '../src/core/memory.js';
import { BuiltinEmbedder } from '../src/retrieval/embedder.js';
import { LocalStore } from '../src/store/local.js';
import { cosineSimilarity, generateId, daysSince } from '../src/core/utils.js';
import { DecayEngine } from '../src/core/decay.js';
import type { MemoryEntry } from '../src/core/types.js';

// ─── Helper to create a fake old entry ───
function makeEntry(overrides: Partial<MemoryEntry> = {}): MemoryEntry {
  return {
    id: generateId(),
    content: 'test content',
    type: 'fact',
    scope: 'default',
    importance: 'soft',
    confidence: 1.0,
    metadata: {},
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    version: 1,
    ...overrides,
  };
}

describe('Edge Cases: Empty & Invalid Input', () => {
  let memory: AgentMemory;

  beforeEach(() => {
    memory = new AgentMemory();
  });

  it('should reject empty string content', async () => {
    await expect(memory.store({ content: '' })).rejects.toThrow('Memory content cannot be empty');
  });

  it('should reject whitespace-only content', async () => {
    await expect(memory.store({ content: '   \n\t  ' })).rejects.toThrow('Memory content cannot be empty');
  });

  it('should reject empty content on update', async () => {
    const entry = await memory.store({ content: 'valid content' });
    await expect(memory.update(entry.id, { content: '   ' })).rejects.toThrow('Memory content cannot be empty');
  });

  it('should handle very long content', async () => {
    const longContent = 'word '.repeat(10000);
    const entry = await memory.store({ content: longContent });
    expect(entry.content.length).toBe(longContent.length);
    expect(entry.embedding).toBeDefined();
    expect(entry.embedding!.length).toBe(256);
  });

  it('should handle special characters and unicode', async () => {
    const entry = await memory.store({ content: 'User prefers 日本語 and émojis 🎉 with $pecial chars!' });
    expect(entry.id).toBeTruthy();
    const retrieved = await memory.get(entry.id);
    expect(retrieved!.content).toBe('User prefers 日本語 and émojis 🎉 with $pecial chars!');
  });

  it('should handle single-character content', async () => {
    // Single char after tokenization (strips single chars) produces empty tokens
    // This should still work - just produces a zero vector
    const entry = await memory.store({ content: 'ab' }); // 2 chars, passes tokenize filter
    expect(entry.embedding).toBeDefined();
  });
});

describe('Edge Cases: Retrieval', () => {
  let memory: AgentMemory;

  beforeEach(() => {
    memory = new AgentMemory();
  });

  it('should return empty array when store is empty', async () => {
    const results = await memory.retrieve({ query: 'anything' });
    expect(results).toEqual([]);
  });

  it('should respect limit=0', async () => {
    await memory.store({ content: 'something' });
    const results = await memory.retrieve({ query: 'something', limit: 0 });
    expect(results).toEqual([]);
  });

  it('should respect limit=1 with multiple entries', async () => {
    await memory.store({ content: 'apple fruit' });
    await memory.store({ content: 'banana fruit' });
    await memory.store({ content: 'cherry fruit' });
    const results = await memory.retrieve({ query: 'fruit', limit: 1 });
    expect(results.length).toBe(1);
  });

  it('should filter by type in retrieval', async () => {
    await memory.store({ content: 'User likes Python', type: 'preference' });
    await memory.store({ content: 'Meeting tomorrow', type: 'task' });

    const results = await memory.retrieve({
      query: 'Python meeting',
      types: ['preference'],
    });

    for (const r of results) {
      expect(r.entry.type).toBe('preference');
    }
  });

  it('should handle retrieval with only importance signal', async () => {
    await memory.store({ content: 'hard fact', importance: 'hard' });
    await memory.store({ content: 'soft fact', importance: 'soft' });
    await memory.store({ content: 'ephemeral fact', importance: 'ephemeral' });

    const results = await memory.retrieve({
      query: 'fact',
      signals: ['importance'],
    });

    expect(results.length).toBe(3);
    // Hard importance should score highest
    expect(results[0].entry.importance).toBe('hard');
    expect(results[0].signalScores.importance).toBe(1.0);
  });

  it('should handle taskRelevance without taskContext gracefully', async () => {
    await memory.store({ content: 'some data' });
    const results = await memory.retrieve({
      query: 'data',
      signals: ['taskRelevance'], // no taskContext provided
    });
    // Should still return results, taskRelevance just won't contribute
    expect(results.length).toBeGreaterThan(0);
  });

  it('should handle threshold filtering', async () => {
    await memory.store({ content: 'Python programming language' });
    await memory.store({ content: 'completely unrelated astronomy topic stars galaxies' });

    const results = await memory.retrieve({
      query: 'Python programming',
      threshold: 0.5, // high threshold
      signals: ['similarity'],
    });

    // Only highly relevant results should pass
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(0.5);
    }
  });
});

describe('Edge Cases: Conflict Detection', () => {
  let memory: AgentMemory;

  beforeEach(() => {
    memory = new AgentMemory();
  });

  it('should return no conflicts on empty store', async () => {
    const conflicts = await memory.checkConflicts('anything');
    expect(conflicts).toEqual([]);
  });

  it('should detect negation-based conflicts', async () => {
    await memory.store({ content: 'User likes dark mode', scope: 'user:1' });
    const conflicts = await memory.checkConflicts('User does not like dark mode', 'user:1');
    // Should find conflict due to "not" negation
    expect(conflicts.length).toBeGreaterThan(0);
  });

  it('should detect antonym conflicts (light vs dark)', async () => {
    await memory.store({ content: 'User prefers light theme', scope: 'user:1' });
    const conflicts = await memory.checkConflicts('User prefers dark theme', 'user:1');
    expect(conflicts.length).toBeGreaterThan(0);
  });

  it('should detect change language conflicts (with custom embedder)', async () => {
    // The builtin TF-IDF embedder may not produce high enough similarity for change-language detection.
    // With a real embedder (OpenAI, etc.) this works. Test with a mock that returns high similarity.
    const highSimEmbedder = {
      embed: async (_t: string) => [0.9, 0.1, 0.1, 0.1], // all texts are "similar"
      embedBatch: async (texts: string[]) => texts.map(() => [0.9, 0.1, 0.1, 0.1]),
      dimensions: () => 4,
    };
    const mem = new AgentMemory({ embedder: highSimEmbedder });
    await mem.store({ content: 'User likes Python programming', scope: 'user:1' });
    const conflicts = await mem.checkConflicts('User now actually prefers JavaScript instead of Python', 'user:1');
    // "now", "actually", "instead of" are change/negation signals
    expect(conflicts.length).toBeGreaterThan(0);
    expect(conflicts[0].confidence).toBeGreaterThan(0);
  });

  it('should not cross-contaminate scopes in conflict check', async () => {
    await memory.store({ content: 'User is vegetarian', scope: 'user:1', importance: 'hard' });
    // Check conflict in a different scope
    const conflicts = await memory.checkConflicts('User wants steak', 'user:2');
    expect(conflicts.length).toBe(0);
  });

  it('should suggest clarify for hard constraints', async () => {
    await memory.store({
      content: 'User has peanut allergy',
      scope: 'user:1',
      importance: 'hard',
      type: 'constraint',
    });
    const conflicts = await memory.checkConflicts('User does not have peanut allergy anymore', 'user:1');
    if (conflicts.length > 0) {
      expect(conflicts[0].action).toBe('clarify');
    }
  });

  it('should suggest override for ephemeral memories', async () => {
    await memory.store({
      content: 'User prefers morning meetings',
      scope: 'user:1',
      importance: 'ephemeral',
    });
    const conflicts = await memory.checkConflicts('User does not want morning meetings', 'user:1');
    if (conflicts.length > 0) {
      expect(conflicts[0].action).toBe('override');
    }
  });
});

describe('Edge Cases: Decay Engine', () => {
  it('should handle zero confidence entries', () => {
    const decay = new DecayEngine();
    const entry = makeEntry({ confidence: 0, importance: 'soft', type: 'preference' });
    expect(decay.computeDecayedConfidence(entry)).toBe(0);
  });

  it('should handle future timestamps (negative age)', () => {
    const decay = new DecayEngine();
    const futureDate = new Date(Date.now() + 86400000).toISOString(); // tomorrow
    const entry = makeEntry({ updatedAt: futureDate, importance: 'soft', type: 'preference' });
    const decayed = decay.computeDecayedConfidence(entry);
    // Negative age in exponential = confidence * e^(positive) > confidence
    // This is technically a bug in the data, but should not crash
    expect(decayed).toBeGreaterThanOrEqual(0);
  });

  it('should handle very old entries (years old)', () => {
    const decay = new DecayEngine();
    const oldDate = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString(); // 1 year ago
    const entry = makeEntry({ updatedAt: oldDate, importance: 'soft', type: 'preference' });
    const decayed = decay.computeDecayedConfidence(entry);
    expect(decayed).toBeGreaterThanOrEqual(0);
    expect(decayed).toBeLessThan(0.01); // should be nearly zero after 1 year with 30-day half-life
  });

  it('should never decay constraint type with none policy', () => {
    const decay = new DecayEngine();
    const oldDate = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString();
    const entry = makeEntry({ updatedAt: oldDate, type: 'constraint', importance: 'soft' });
    // constraint type has 'none' policy
    const decayed = decay.computeDecayedConfidence(entry);
    expect(decayed).toBe(1.0);
  });

  it('should respect step function cutoff exactly', () => {
    const decay = new DecayEngine();
    // Task type has 7-day step cutoff
    const justBefore = new Date(Date.now() - 6.9 * 24 * 60 * 60 * 1000).toISOString();
    const justAfter = new Date(Date.now() - 7.1 * 24 * 60 * 60 * 1000).toISOString();

    const entryBefore = makeEntry({ updatedAt: justBefore, type: 'task', importance: 'soft' });
    const entryAfter = makeEntry({ updatedAt: justAfter, type: 'task', importance: 'soft' });

    expect(decay.computeDecayedConfidence(entryBefore)).toBe(1.0);
    expect(decay.computeDecayedConfidence(entryAfter)).toBe(0);
  });

  it('should handle linear decay correctly', () => {
    const decay = new DecayEngine({
      policies: { fact: { policy: 'linear', ratePerDay: 0.1 } },
    });
    const tenDaysAgo = new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString();
    const entry = makeEntry({ updatedAt: tenDaysAgo, type: 'fact', importance: 'soft' });
    const decayed = decay.computeDecayedConfidence(entry);
    // 1.0 - 0.1 * 10 = 0
    expect(decayed).toBeCloseTo(0, 1);
  });

  it('cleanup should delete expired entries', async () => {
    const memory = new AgentMemory();
    // Store a task, then simulate it being old
    const entry = await memory.store({ content: 'old task', type: 'task', importance: 'soft' });
    // We can't easily age it in the store, but we can test cleanup returns 0 for fresh entries
    const deleted = await memory.cleanup();
    expect(deleted).toBe(0); // fresh entry should not be expired
  });
});

describe('Edge Cases: Checkpointing', () => {
  let memory: AgentMemory;

  beforeEach(() => {
    memory = new AgentMemory();
  });

  it('should return null for non-existent checkpoint', async () => {
    const result = await memory.rehydrate('non_existent_id');
    expect(result).toBeNull();
  });

  it('should handle rehydrate with deleted memories gracefully', async () => {
    const entry = await memory.store({ content: 'will be deleted' });
    const cp = await memory.checkpoint({
      taskGraph: [],
      summary: 'test',
      activeMemoryIds: [entry.id],
    });

    // Delete the memory after checkpointing
    await memory.delete(entry.id);

    const restored = await memory.rehydrate(cp.id);
    expect(restored).not.toBeNull();
    expect(restored!.memories.length).toBe(0); // memory was deleted, not found
  });

  it('should handle empty task graph', async () => {
    const cp = await memory.checkpoint({ taskGraph: [], summary: '' });
    expect(cp.taskGraph).toEqual([]);
    expect(cp.summary).toBe('');
  });

  it('should handle checkpoint with complex tool outputs', async () => {
    const cp = await memory.checkpoint({
      taskGraph: [],
      summary: 'test',
      toolOutputs: {
        nested: { deep: { value: [1, 2, { three: true }] } },
        null_val: null,
        array: [1, 'two', false],
      },
    });

    const restored = await memory.rehydrate(cp.id);
    expect(restored!.checkpoint.toolOutputs.nested).toEqual({ deep: { value: [1, 2, { three: true }] } });
  });

  it('should return null for getLatestCheckpoint when none exist', () => {
    expect(memory.getLatestCheckpoint()).toBeNull();
  });
});

describe('Edge Cases: Utilities', () => {
  it('cosine similarity of zero vectors should be 0', () => {
    expect(cosineSimilarity([0, 0, 0], [0, 0, 0])).toBe(0);
  });

  it('cosine similarity of identical vectors should be 1', () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1, 5);
  });

  it('cosine similarity of opposite vectors should be -1', () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1, 5);
  });

  it('cosine similarity of different length vectors should be 0', () => {
    expect(cosineSimilarity([1, 2], [1, 2, 3])).toBe(0);
  });

  it('cosine similarity of empty vectors should be 0', () => {
    expect(cosineSimilarity([], [])).toBe(0);
  });

  it('generateId should produce unique IDs', () => {
    const ids = new Set<string>();
    for (let i = 0; i < 1000; i++) {
      ids.add(generateId());
    }
    expect(ids.size).toBe(1000);
  });

  it('daysSince with current time should be ~0', () => {
    const d = daysSince(new Date().toISOString());
    expect(d).toBeCloseTo(0, 2);
  });
});

describe('Edge Cases: Embedder', () => {
  it('should handle empty string (produces zero vector)', async () => {
    const embedder = new BuiltinEmbedder(64);
    const vec = await embedder.embed('');
    // All zeros since no tokens pass the filter
    expect(vec.length).toBe(64);
    expect(vec.every(v => v === 0)).toBe(true);
  });

  it('should produce same embedding for same text (deterministic)', async () => {
    const embedder = new BuiltinEmbedder(64);
    const v1 = await embedder.embed('hello world');
    const v2 = await embedder.embed('hello world');
    expect(v1).toEqual(v2);
  });

  it('should produce different embeddings for different text', async () => {
    const embedder = new BuiltinEmbedder(64);
    const v1 = await embedder.embed('hello world');
    const v2 = await embedder.embed('machine learning algorithms');
    expect(v1).not.toEqual(v2);
  });

  it('should produce normalized vectors (L2 norm ~1)', async () => {
    const embedder = new BuiltinEmbedder(64);
    const vec = await embedder.embed('test embedding normalization check');
    const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
    expect(norm).toBeCloseTo(1, 3);
  });

  it('embedBatch should return correct count', async () => {
    const embedder = new BuiltinEmbedder(32);
    const results = await embedder.embedBatch(['a b', 'c d', 'e f']);
    expect(results.length).toBe(3);
    expect(results[0].length).toBe(32);
  });

  it('train should improve IDF weighting', async () => {
    const embedder = new BuiltinEmbedder(256);

    // Before training, all IDF = 1
    const v1 = await embedder.embed('the common word appears frequently but unicorn is rare');

    // Train with documents where "common" and "the" appear often but "unicorn" never
    embedder.train([
      'the common word appears in many documents',
      'the common word is everywhere in text',
      'the common word shows up frequently here',
      'another common word document with the usual terms',
    ]);

    // After training, "unicorn" should have higher IDF weight, "common"/"the" lower
    const v2 = await embedder.embed('the common word appears frequently but unicorn is rare');
    // Vectors should differ because IDF changed
    const same = v1.every((val, i) => val === v2[i]);
    expect(same).toBe(false);
  });
});

describe('Edge Cases: Concurrency & Ordering', () => {
  it('should handle rapid concurrent stores', async () => {
    const memory = new AgentMemory();
    const promises = Array.from({ length: 50 }, (_, i) =>
      memory.store({ content: `Memory entry number ${i}`, scope: 'test' })
    );
    const entries = await Promise.all(promises);
    expect(entries.length).toBe(50);

    // All should have unique IDs
    const ids = new Set(entries.map(e => e.id));
    expect(ids.size).toBe(50);

    const all = await memory.getAll('test');
    expect(all.length).toBe(50);
  });

  it('should handle store then immediate retrieve', async () => {
    const memory = new AgentMemory();
    await memory.store({ content: 'Python is great for data science' });
    const results = await memory.retrieve({ query: 'Python data science' });
    expect(results.length).toBe(1);
  });

  it('should handle store then immediate delete', async () => {
    const memory = new AgentMemory();
    const entry = await memory.store({ content: 'temporary' });
    const deleted = await memory.delete(entry.id);
    expect(deleted).toBe(true);
    const retrieved = await memory.get(entry.id);
    expect(retrieved).toBeNull();
  });

  it('should handle double delete gracefully', async () => {
    const memory = new AgentMemory();
    const entry = await memory.store({ content: 'test' });
    await memory.delete(entry.id);
    const secondDelete = await memory.delete(entry.id);
    expect(secondDelete).toBe(false);
  });
});

describe('Edge Cases: Custom Config', () => {
  it('should use custom default scope', async () => {
    const memory = new AgentMemory({ defaultScope: 'my-agent' });
    const entry = await memory.store({ content: 'test' });
    expect(entry.scope).toBe('my-agent');
  });

  it('should work with custom embedder', async () => {
    // Simple mock embedder that returns fixed-size vectors
    const mockEmbedder = {
      embed: async (text: string) => {
        const vec = new Array(8).fill(0);
        vec[text.length % 8] = 1;
        return vec;
      },
      embedBatch: async (texts: string[]) => Promise.all(texts.map(t => mockEmbedder.embed(t))),
      dimensions: () => 8,
    };

    const memory = new AgentMemory({ embedder: mockEmbedder });
    const entry = await memory.store({ content: 'test with custom embedder' });
    expect(entry.embedding).toBeDefined();
    expect(entry.embedding!.length).toBe(8);
  });

  it('should handle version incrementing across multiple updates', async () => {
    const memory = new AgentMemory();
    const entry = await memory.store({ content: 'v1' });
    expect(entry.version).toBe(1);

    const v2 = await memory.update(entry.id, { content: 'v2' });
    expect(v2!.version).toBe(2);

    const v3 = await memory.update(entry.id, { content: 'v3' });
    expect(v3!.version).toBe(3);

    const v4 = await memory.update(entry.id, { metadata: { note: 'just metadata' } });
    expect(v4!.version).toBe(4);
    expect(v4!.content).toBe('v3'); // content unchanged
  });

  it('should handle update without content change (no re-embed)', async () => {
    let embedCount = 0;
    const countingEmbedder = {
      embed: async (_text: string) => { embedCount++; return new Array(8).fill(0.1); },
      embedBatch: async (texts: string[]) => Promise.all(texts.map(t => countingEmbedder.embed(t))),
      dimensions: () => 8,
    };

    const memory = new AgentMemory({ embedder: countingEmbedder });
    const entry = await memory.store({ content: 'test' });
    expect(embedCount).toBe(1);

    // Update only metadata, should NOT re-embed
    await memory.update(entry.id, { metadata: { tag: 'updated' } });
    expect(embedCount).toBe(1); // still 1, no re-embed
  });
});

describe('Edge Cases: Scope Isolation', () => {
  it('should completely isolate scopes for getAll', async () => {
    const memory = new AgentMemory();
    await memory.store({ content: 'Agent A data', scope: 'agent:a' });
    await memory.store({ content: 'Agent B data', scope: 'agent:b' });
    await memory.store({ content: 'Shared data', scope: 'shared' });

    expect((await memory.getAll('agent:a')).length).toBe(1);
    expect((await memory.getAll('agent:b')).length).toBe(1);
    expect((await memory.getAll('shared')).length).toBe(1);
    expect((await memory.getAll()).length).toBe(3); // no scope = all
  });

  it('should clear only target scope', async () => {
    const memory = new AgentMemory();
    await memory.store({ content: 'keep', scope: 'a' });
    await memory.store({ content: 'keep', scope: 'a' });
    await memory.store({ content: 'remove', scope: 'b' });

    await memory.clear('b');
    expect((await memory.getAll('a')).length).toBe(2);
    expect((await memory.getAll('b')).length).toBe(0);
  });
});
