/**
 * OpenAI embedder adapter.
 * Requires: npm install openai
 *
 * Usage:
 *   import { AgentMemory } from 'agentic-memory';
 *   import { OpenAIEmbedder } from 'agentic-memory/adapters/openai';
 *
 *   const memory = new AgentMemory({
 *     embedder: new OpenAIEmbedder({ apiKey: process.env.OPENAI_API_KEY }),
 *   });
 */

import type { Embedder } from '../core/types.js';

export interface OpenAIEmbedderConfig {
  apiKey: string;
  model?: string;
  dimensions?: number;
  baseURL?: string;
}

export class OpenAIEmbedder implements Embedder {
  private apiKey: string;
  private model: string;
  private dim: number;
  private baseURL: string;

  constructor(config: OpenAIEmbedderConfig) {
    this.apiKey = config.apiKey;
    this.model = config.model ?? 'text-embedding-3-small';
    this.dim = config.dimensions ?? 1536;
    this.baseURL = config.baseURL ?? 'https://api.openai.com/v1';
  }

  dimensions(): number {
    return this.dim;
  }

  async embed(text: string): Promise<number[]> {
    const res = await fetch(`${this.baseURL}/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        input: text,
        dimensions: this.dim,
      }),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`OpenAI embedding failed (${res.status}): ${err}`);
    }

    const data = await res.json() as { data: { embedding: number[] }[] };
    return data.data[0].embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    const res = await fetch(`${this.baseURL}/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        input: texts,
        dimensions: this.dim,
      }),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`OpenAI embedding failed (${res.status}): ${err}`);
    }

    const data = await res.json() as { data: { embedding: number[]; index: number }[] };
    // OpenAI may return out of order, sort by index
    return data.data
      .sort((a, b) => a.index - b.index)
      .map(d => d.embedding);
  }
}
