/**
 * Complete working example: agentic-memory + Claude (Anthropic)
 *
 * Uses OpenAI embeddings (Anthropic doesn't have an embedding API)
 * + Claude for the chat completion.
 *
 * Run:
 *   export OPENAI_API_KEY=sk-...
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   npx tsx examples/with-anthropic.ts
 */

import { AgentMemory } from '../src/index.js';
import { OpenAIEmbedder } from '../src/adapters/openai.js';

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;

if (!OPENAI_API_KEY || !ANTHROPIC_API_KEY) {
  console.error('Set both OPENAI_API_KEY and ANTHROPIC_API_KEY');
  process.exit(1);
}

async function askClaude(prompt: string): Promise<string> {
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': ANTHROPIC_API_KEY!,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 200,
      messages: [{ role: 'user', content: prompt }],
    }),
  });
  const data = await res.json() as any;
  return data.content[0].text;
}

async function main() {
  console.log('=== agentic-memory + Claude Demo ===\n');

  const memory = new AgentMemory({
    embedder: new OpenAIEmbedder({
      apiKey: OPENAI_API_KEY!,
      dimensions: 512,
    }),
  });

  // ── Store context from previous sessions ──
  console.log('Storing user context from previous sessions...');

  await memory.store({
    content: 'User is a senior backend engineer who works with Go and PostgreSQL',
    type: 'fact',
    scope: 'user:dev',
    importance: 'soft',
  });

  await memory.store({
    content: 'User strongly dislikes ORMs and prefers writing raw SQL',
    type: 'preference',
    scope: 'user:dev',
    importance: 'hard',
  });

  await memory.store({
    content: 'User is building a high-throughput event processing system',
    type: 'task',
    scope: 'user:dev',
    importance: 'soft',
  });

  // ── Retrieve + ask Claude ──
  console.log('\nUser asks: "How should I handle database connections?"');

  const context = await memory.retrieve({
    query: 'database connection management',
    scope: 'user:dev',
    signals: ['similarity', 'recency', 'importance', 'taskRelevance'],
    taskContext: 'high-throughput event processing database layer',
    limit: 3,
  });

  console.log(`\nRetrieved ${context.length} memories:`);
  for (const r of context) {
    console.log(`  [${r.score.toFixed(2)}] ${r.entry.content}`);
  }

  const memContext = context.map(r => `- ${r.entry.content}`).join('\n');
  const prompt = `What you know about the user:
${memContext}

User asks: "How should I handle database connections in my system?"

Give a specific, concise answer (3-4 sentences) tailored to their stack and preferences.`;

  console.log('\nAsking Claude...');
  const response = await askClaude(prompt);
  console.log(`\nClaude: ${response}`);

  // ── Conflict: suggesting an ORM ──
  console.log('\n--- Conflict Check ---');
  console.log('Another agent suggests: "Use Prisma ORM for the database layer"');

  const conflicts = await memory.checkConflicts(
    'Use Prisma ORM for the database layer',
    'user:dev',
  );

  if (conflicts.length > 0) {
    console.log(`  BLOCKED - ${conflicts[0].action}: ${conflicts[0].reason}`);
  }

  console.log('\n=== Demo complete ===');
}

main().catch(console.error);
