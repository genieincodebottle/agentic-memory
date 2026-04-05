/**
 * Complete working example: agentic-memory + OpenAI
 *
 * Run:
 *   export OPENAI_API_KEY=sk-...
 *   npx tsx examples/with-openai.ts
 *
 * What this demo shows:
 *   1. Store user preferences with real embeddings
 *   2. Retrieve relevant context before an LLM call
 *   3. Catch a dangerous contradiction (vegetarian vs steak)
 *   4. Make an LLM call that uses memory context
 */

import { AgentMemory } from '../src/index.js';
import { OpenAIEmbedder } from '../src/adapters/openai.js';

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error('Set OPENAI_API_KEY environment variable');
  process.exit(1);
}

async function chat(prompt: string): Promise<string> {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 200,
    }),
  });
  const data = await res.json() as any;
  return data.choices[0].message.content;
}

async function main() {
  console.log('=== agentic-memory + OpenAI Demo ===\n');

  // ── Setup: real embeddings for accurate similarity ──
  const memory = new AgentMemory({
    embedder: new OpenAIEmbedder({
      apiKey: OPENAI_API_KEY,
      model: 'text-embedding-3-small',
      dimensions: 512, // smaller = faster + cheaper
    }),
  });

  // ── Step 1: Store user preferences ──
  console.log('Step 1: Storing user preferences...');

  await memory.store({
    content: 'User is vegetarian and avoids all meat products',
    type: 'preference',
    scope: 'user:demo',
    importance: 'hard',
  });

  await memory.store({
    content: 'User prefers Italian cuisine, especially pasta dishes',
    type: 'preference',
    scope: 'user:demo',
    importance: 'soft',
  });

  await memory.store({
    content: 'User is allergic to peanuts',
    type: 'constraint',
    scope: 'user:demo',
    importance: 'hard',
  });

  await memory.store({
    content: 'User prefers quick recipes under 30 minutes',
    type: 'preference',
    scope: 'user:demo',
    importance: 'soft',
  });

  console.log('  Stored 4 memories (2 hard, 2 soft)\n');

  // ── Step 2: Retrieve relevant context ──
  console.log('Step 2: User asks "Suggest a dinner recipe"');
  console.log('  Retrieving relevant memories...');

  const results = await memory.retrieve({
    query: 'dinner recipe suggestion',
    scope: 'user:demo',
    signals: ['similarity', 'recency', 'importance'],
    limit: 3,
  });

  console.log(`  Found ${results.length} relevant memories:`);
  for (const r of results) {
    console.log(`    [${r.score.toFixed(2)}] ${r.entry.content}`);
  }

  // ── Step 3: LLM call with memory context ──
  console.log('\nStep 3: Calling LLM with memory context...');

  const memoryContext = results.map(r => `- ${r.entry.content}`).join('\n');
  const prompt = `You are a helpful cooking assistant. Here is what you know about the user:
${memoryContext}

The user says: "Suggest a dinner recipe for tonight."

Give a short, specific suggestion (2-3 sentences).`;

  const response = await chat(prompt);
  console.log(`\n  LLM Response: ${response}\n`);

  // ── Step 4: Conflict detection ──
  console.log('Step 4: Another agent tries to order steak...');

  const conflicts = await memory.checkConflicts(
    'Order a ribeye steak dinner for the user',
    'user:demo',
  );

  if (conflicts.length > 0) {
    console.log(`  CONFLICT DETECTED!`);
    console.log(`    Confidence: ${conflicts[0].confidence.toFixed(2)}`);
    console.log(`    Action: ${conflicts[0].action}`);
    console.log(`    Reason: ${conflicts[0].reason}`);
  } else {
    console.log('  No conflicts found');
  }

  // ── Step 5: Check peanut allergy conflict ──
  console.log('\nStep 5: Recipe suggestion with peanuts...');

  const peanutConflicts = await memory.checkConflicts(
    'Make a Thai peanut noodle dish for the user',
    'user:demo',
  );

  if (peanutConflicts.length > 0) {
    console.log(`  CONFLICT DETECTED!`);
    console.log(`    Confidence: ${peanutConflicts[0].confidence.toFixed(2)}`);
    console.log(`    Reason: ${peanutConflicts[0].reason}`);
  }

  console.log('\n=== Demo complete ===');
}

main().catch(console.error);
