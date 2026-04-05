/**
 * Voyage AI embedder adapter (high-quality, cheaper than OpenAI).
 * Requires: VOYAGE_API_KEY
 *
 * Usage:
 *   import { AgentMemory } from 'agentic-memory';
 *   import { VoyageEmbedder } from 'agentic-memory/adapters/voyageai';
 *
 *   const memory = new AgentMemory({
 *     embedder: new VoyageEmbedder({ apiKey: process.env.VOYAGE_API_KEY }),
 *   });
 */

import type { Embedder } from '../core/types.js';

export interface VoyageEmbedderConfig {
  apiKey: string;
  model?: string;
}

export class VoyageEmbedder implements Embedder {
  private apiKey: string;
  private model: string;

  constructor(config: VoyageEmbedderConfig) {
    this.apiKey = config.apiKey;
    this.model = config.model ?? 'voyage-3-lite';
  }

  dimensions(): number {
    return this.model.includes('lite') ? 512 : 1024;
  }

  async embed(text: string): Promise<number[]> {
    const result = await this._call([text]);
    return result[0];
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    return this._call(texts);
  }

  private async _call(input: string[]): Promise<number[][]> {
    const res = await fetch('https://api.voyageai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ model: this.model, input }),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Voyage AI embedding failed (${res.status}): ${err}`);
    }

    const data = await res.json() as { data: { embedding: number[] }[] };
    return data.data.map(d => d.embedding);
  }
}
