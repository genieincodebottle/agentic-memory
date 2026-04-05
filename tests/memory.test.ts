import { describe, it, expect, beforeEach } from 'vitest';
import { AgentMemory } from '../src/core/memory.js';

describe('AgentMemory', () => {
  let memory: AgentMemory;

  beforeEach(() => {
    memory = new AgentMemory();
  });

  describe('store and retrieve', () => {
    it('should store and get a memory entry', async () => {
      const entry = await memory.store({
        content: 'User prefers dark mode',
        type: 'preference',
        scope: 'user:1',
      });

      expect(entry.id).toBeTruthy();
      expect(entry.content).toBe('User prefers dark mode');
      expect(entry.type).toBe('preference');
      expect(entry.version).toBe(1);

      const retrieved = await memory.get(entry.id);
      expect(retrieved).not.toBeNull();
      expect(retrieved!.content).toBe('User prefers dark mode');
    });

    it('should store with default values', async () => {
      const entry = await memory.store({ content: 'test fact' });
      expect(entry.type).toBe('fact');
      expect(entry.importance).toBe('soft');
      expect(entry.confidence).toBe(1.0);
      expect(entry.scope).toBe('default');
    });

    it('should retrieve memories by query', async () => {
      await memory.store({ content: 'User likes Python programming', scope: 'user:1' });
      await memory.store({ content: 'User dislikes Java', scope: 'user:1' });
      await memory.store({ content: 'Meeting at 3pm tomorrow', scope: 'user:1', type: 'task' });

      const results = await memory.retrieve({
        query: 'programming language preferences',
        scope: 'user:1',
        limit: 5,
      });

      expect(results.length).toBeGreaterThan(0);
      expect(results[0].score).toBeGreaterThan(0);
      expect(results[0].signalScores).toBeDefined();
    });
  });

  describe('update and delete', () => {
    it('should update an entry and increment version', async () => {
      const entry = await memory.store({ content: 'User likes blue' });
      const updated = await memory.update(entry.id, { content: 'User likes red' });

      expect(updated).not.toBeNull();
      expect(updated!.content).toBe('User likes red');
      expect(updated!.version).toBe(2);
    });

    it('should return null when updating non-existent entry', async () => {
      const result = await memory.update('non_existent', { content: 'test' });
      expect(result).toBeNull();
    });

    it('should delete an entry', async () => {
      const entry = await memory.store({ content: 'temporary' });
      const deleted = await memory.delete(entry.id);
      expect(deleted).toBe(true);

      const retrieved = await memory.get(entry.id);
      expect(retrieved).toBeNull();
    });
  });

  describe('scoping', () => {
    it('should isolate memories by scope', async () => {
      await memory.store({ content: 'Agent A memory', scope: 'agent:a' });
      await memory.store({ content: 'Agent B memory', scope: 'agent:b' });

      const agentA = await memory.getAll('agent:a');
      const agentB = await memory.getAll('agent:b');

      expect(agentA.length).toBe(1);
      expect(agentA[0].content).toBe('Agent A memory');
      expect(agentB.length).toBe(1);
      expect(agentB[0].content).toBe('Agent B memory');
    });

    it('should clear only scoped memories', async () => {
      await memory.store({ content: 'keep this', scope: 'keep' });
      await memory.store({ content: 'delete this', scope: 'remove' });

      await memory.clear('remove');

      const kept = await memory.getAll('keep');
      const removed = await memory.getAll('remove');

      expect(kept.length).toBe(1);
      expect(removed.length).toBe(0);
    });
  });

  describe('conflict detection', () => {
    it('should detect contradiction with negation', async () => {
      await memory.store({
        content: 'User is vegetarian',
        type: 'preference',
        importance: 'hard',
        scope: 'user:1',
      });

      const conflicts = await memory.checkConflicts(
        'User is not vegetarian and wants steak',
        'user:1'
      );

      expect(conflicts.length).toBeGreaterThan(0);
      expect(conflicts[0].action).toBe('clarify');
      expect(conflicts[0].confidence).toBeGreaterThan(0);
    });

    it('should return no conflicts for unrelated content', async () => {
      await memory.store({
        content: 'User likes Python',
        scope: 'user:1',
      });

      const conflicts = await memory.checkConflicts(
        'The weather is nice today',
        'user:1'
      );

      // Unrelated topics should have low/no conflicts
      const highConfidence = conflicts.filter(c => c.confidence > 0.5);
      expect(highConfidence.length).toBe(0);
    });
  });

  describe('decay', () => {
    it('should not decay hard importance memories', async () => {
      const entry = await memory.store({
        content: 'User has peanut allergy',
        importance: 'hard',
        type: 'constraint',
      });

      const decayed = memory.getDecayedConfidence(entry);
      expect(decayed).toBe(1.0);
    });

    it('should decay ephemeral memories faster', async () => {
      const ephemeral = await memory.store({
        content: 'Task in progress',
        importance: 'ephemeral',
        type: 'task',
      });

      // Simulate age by modifying updatedAt
      const oldEntry = { ...ephemeral, updatedAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString() };
      const decayed = memory.getDecayedConfidence(oldEntry);
      expect(decayed).toBe(0); // Step function: 7-day max for tasks
    });
  });

  describe('checkpointing', () => {
    it('should create and rehydrate a checkpoint', async () => {
      const entry = await memory.store({ content: 'Important context' });

      const cp = await memory.checkpoint({
        taskGraph: [
          { id: 'task1', description: 'Step 1', status: 'done', dependencies: [], result: 'ok' },
          { id: 'task2', description: 'Step 2', status: 'pending', dependencies: ['task1'] },
        ],
        summary: 'Completed step 1, about to start step 2',
        toolOutputs: { 'api_call_1': { data: [1, 2, 3] } },
        activeMemoryIds: [entry.id],
      });

      expect(cp.id).toBeTruthy();
      expect(cp.taskGraph.length).toBe(2);

      const restored = await memory.rehydrate(cp.id);
      expect(restored).not.toBeNull();
      expect(restored!.checkpoint.summary).toBe('Completed step 1, about to start step 2');
      expect(restored!.memories.length).toBe(1);
      expect(restored!.memories[0].content).toBe('Important context');
    });

    it('should get latest checkpoint', async () => {
      await memory.checkpoint({
        taskGraph: [],
        summary: 'First checkpoint',
      });
      await memory.checkpoint({
        taskGraph: [],
        summary: 'Second checkpoint',
      });

      const latest = memory.getLatestCheckpoint();
      expect(latest).not.toBeNull();
      expect(latest!.summary).toBe('Second checkpoint');
    });

    it('should enforce max checkpoints', async () => {
      const mem = new AgentMemory({ checkpoint: { maxCheckpoints: 2 } });

      await mem.checkpoint({ taskGraph: [], summary: 'cp1' });
      await mem.checkpoint({ taskGraph: [], summary: 'cp2' });
      await mem.checkpoint({ taskGraph: [], summary: 'cp3' });

      const all = mem.listCheckpoints();
      expect(all.length).toBe(2);
      expect(all[0].summary).toBe('cp2');
    });
  });
});
