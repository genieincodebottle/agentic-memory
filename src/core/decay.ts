import type { MemoryEntry, DecayConfig, MemoryType } from './types.js';
import { daysSince } from './utils.js';

/** Default decay configuration - hard constraints never decay */
const DEFAULT_DECAY_CONFIG: DecayConfig = {
  policies: {
    constraint: { policy: 'none' },
    preference: { policy: 'exponential', halfLife: 30 * 24 * 60 * 60 * 1000 }, // 30 days
    fact: { policy: 'exponential', halfLife: 90 * 24 * 60 * 60 * 1000 }, // 90 days
    task: { policy: 'step', maxAge: 7 * 24 * 60 * 60 * 1000 }, // 7 days then gone
    episodic: { policy: 'exponential', halfLife: 14 * 24 * 60 * 60 * 1000 }, // 14 days
  },
  defaultPolicy: 'exponential',
  defaultHalfLife: 30 * 24 * 60 * 60 * 1000,
};

/**
 * Typed decay engine.
 * Different memory types decay at different rates.
 * Hard constraints (allergies, legal requirements) never decay.
 * Ephemeral task state decays fast.
 */
export class DecayEngine {
  private config: DecayConfig;

  constructor(config?: Partial<DecayConfig>) {
    this.config = {
      ...DEFAULT_DECAY_CONFIG,
      ...config,
      policies: { ...DEFAULT_DECAY_CONFIG.policies, ...config?.policies },
    };
  }

  /**
   * Compute the current effective confidence of a memory entry
   * after applying temporal decay.
   * Returns a value between 0 and original confidence.
   */
  computeDecayedConfidence(entry: MemoryEntry): number {
    // Hard importance entries never decay
    if (entry.importance === 'hard') return entry.confidence;

    const typeConfig = this.config.policies[entry.type];
    const policy = typeConfig?.policy ?? this.config.defaultPolicy;

    const ageMs = Date.now() - new Date(entry.updatedAt).getTime();

    switch (policy) {
      case 'none':
        return entry.confidence;

      case 'linear': {
        const ratePerDay = typeConfig?.ratePerDay ?? 0.01;
        const ageDays = ageMs / (1000 * 60 * 60 * 24);
        return Math.max(0, entry.confidence - ratePerDay * ageDays);
      }

      case 'exponential': {
        const halfLife = typeConfig?.halfLife ?? this.config.defaultHalfLife;
        const decayFactor = Math.exp(-0.693 * ageMs / halfLife);
        return entry.confidence * decayFactor;
      }

      case 'step': {
        const maxAge = typeConfig?.maxAge ?? 7 * 24 * 60 * 60 * 1000;
        return ageMs > maxAge ? 0 : entry.confidence;
      }

      default:
        return entry.confidence;
    }
  }

  /**
   * Filter out memories that have decayed below a threshold.
   * Useful for periodic cleanup.
   */
  filterDecayed(entries: MemoryEntry[], threshold = 0.05): MemoryEntry[] {
    return entries.filter(e => this.computeDecayedConfidence(e) >= threshold);
  }

  /**
   * Get entries that should be cleaned up (decayed to near-zero).
   */
  getExpired(entries: MemoryEntry[], threshold = 0.01): MemoryEntry[] {
    return entries.filter(e => this.computeDecayedConfidence(e) < threshold);
  }
}
