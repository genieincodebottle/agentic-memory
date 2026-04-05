/**
 * Checkpoint & recovery demo - survive context overflow
 *
 * Run:
 *   npx tsx examples/checkpoint-recovery.ts
 *
 * No API keys needed - uses builtin embedder.
 * Shows how an agent saves state mid-workflow and resumes.
 */

import { AgentMemory } from '../src/index.js';

async function main() {
  console.log('=== Checkpoint Recovery Demo ===\n');

  const memory = new AgentMemory();

  // ── Simulate a long-running data pipeline agent ──
  console.log('Agent starts a data pipeline workflow...\n');

  // Store some context the agent has gathered
  const m1 = await memory.store({
    content: 'Source database has 50,000 customer records',
    type: 'fact',
    scope: 'pipeline:job-42',
  });

  const m2 = await memory.store({
    content: 'Customer data must be anonymized before export (GDPR requirement)',
    type: 'constraint',
    scope: 'pipeline:job-42',
    importance: 'hard',
  });

  const m3 = await memory.store({
    content: 'Export format is Parquet, partitioned by region',
    type: 'fact',
    scope: 'pipeline:job-42',
  });

  // Agent is processing step 2 when context fills up
  console.log('  Step 1: Connected to source DB      [done]');
  console.log('  Step 2: Extracting records           [in progress - row 23,456/50,000]');
  console.log('  Step 3: Anonymize PII fields         [pending]');
  console.log('  Step 4: Convert to Parquet           [pending]');
  console.log('  Step 5: Upload to S3                 [pending]');

  console.log('\n  Context window at 95% - saving checkpoint...');

  const checkpoint = await memory.checkpoint({
    taskGraph: [
      { id: 'connect', description: 'Connect to source DB', status: 'done', dependencies: [], result: { host: 'db.prod', connected: true } },
      { id: 'extract', description: 'Extract customer records', status: 'in_progress', dependencies: ['connect'], result: { processed: 23456, total: 50000, lastId: 'cust_23456' } },
      { id: 'anonymize', description: 'Anonymize PII fields', status: 'pending', dependencies: ['extract'] },
      { id: 'convert', description: 'Convert to Parquet', status: 'pending', dependencies: ['anonymize'] },
      { id: 'upload', description: 'Upload to S3', status: 'pending', dependencies: ['convert'] },
    ],
    summary: 'Data pipeline job-42: extracted 23,456 of 50,000 records from db.prod. Last processed customer ID: cust_23456. GDPR anonymization pending.',
    toolOutputs: {
      dbConnection: { host: 'db.prod', status: 'connected' },
      extractionCursor: { lastId: 'cust_23456', batchSize: 1000 },
    },
    activeMemoryIds: [m1.id, m2.id, m3.id],
  });

  console.log(`  Checkpoint saved: ${checkpoint.id}\n`);

  // ── Simulate context reset ──
  console.log('  ~~~~ CONTEXT WINDOW RESETS ~~~~\n');

  // ── New context starts - agent rehydrates ──
  console.log('  New context: rehydrating from checkpoint...\n');

  const restored = await memory.rehydrate(checkpoint.id);

  if (!restored) {
    console.error('Failed to rehydrate!');
    return;
  }

  const { checkpoint: cp, memories } = restored;

  console.log(`  Summary: ${cp.summary}\n`);

  console.log('  Task graph:');
  for (const task of cp.taskGraph) {
    const icon = task.status === 'done' ? 'done' : task.status === 'in_progress' ? 'resuming' : 'pending';
    console.log(`    [${icon}] ${task.description}`);
    if (task.result) {
      console.log(`           result: ${JSON.stringify(task.result)}`);
    }
  }

  console.log(`\n  Restored ${memories.length} active memories:`);
  for (const mem of memories) {
    console.log(`    [${mem.importance}] ${mem.content}`);
  }

  console.log(`\n  Tool state recovered:`);
  console.log(`    DB: ${JSON.stringify(cp.toolOutputs.dbConnection)}`);
  console.log(`    Cursor: ${JSON.stringify(cp.toolOutputs.extractionCursor)}`);

  // Agent knows exactly where to resume
  const extractTask = cp.taskGraph.find(t => t.id === 'extract')!;
  console.log(`\n  Resuming extraction from customer ${(extractTask.result as any).lastId}...`);
  console.log('  (would continue processing rows 23,457 - 50,000)\n');

  // The GDPR constraint is still in memory
  console.log('  GDPR constraint still active:');
  const gdpr = memories.find(m => m.importance === 'hard');
  if (gdpr) {
    console.log(`    "${gdpr.content}"`);
  }

  console.log('\n=== Demo complete ===');
}

main().catch(console.error);
