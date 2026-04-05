"""Typed decay engine - different memory types decay at different rates."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from .types import (
    MemoryEntry,
    MemoryType,
    ImportanceLevel,
    DecayPolicy,
    DecayConfig,
    DecayTypeConfig,
)

DAY_MS = 24 * 60 * 60 * 1000

DEFAULT_DECAY_CONFIG = DecayConfig(
    policies={
        MemoryType.CONSTRAINT: DecayTypeConfig(policy=DecayPolicy.NONE),
        MemoryType.PREFERENCE: DecayTypeConfig(policy=DecayPolicy.EXPONENTIAL, half_life=30 * DAY_MS),
        MemoryType.FACT: DecayTypeConfig(policy=DecayPolicy.EXPONENTIAL, half_life=90 * DAY_MS),
        MemoryType.TASK: DecayTypeConfig(policy=DecayPolicy.STEP, max_age=7 * DAY_MS),
        MemoryType.EPISODIC: DecayTypeConfig(policy=DecayPolicy.EXPONENTIAL, half_life=14 * DAY_MS),
    },
    default_policy=DecayPolicy.EXPONENTIAL,
    default_half_life=30 * DAY_MS,
)


class DecayEngine:
    """
    Typed decay engine.
    Different memory types decay at different rates.
    Hard constraints (allergies, legal requirements) never decay.
    """

    def __init__(self, config: DecayConfig | None = None) -> None:
        if config is None:
            self._config = DEFAULT_DECAY_CONFIG
        else:
            # Merge with defaults
            merged_policies = {**DEFAULT_DECAY_CONFIG.policies, **config.policies}
            self._config = DecayConfig(
                policies=merged_policies,
                default_policy=config.default_policy,
                default_half_life=config.default_half_life,
            )

    def compute_decayed_confidence(self, entry: MemoryEntry) -> float:
        """Compute effective confidence after temporal decay."""
        if entry.importance == ImportanceLevel.HARD:
            return entry.confidence

        type_config = self._config.policies.get(entry.type)
        policy = type_config.policy if type_config else self._config.default_policy

        dt = datetime.fromisoformat(entry.updated_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_ms = (datetime.now(timezone.utc) - dt).total_seconds() * 1000

        if policy == DecayPolicy.NONE:
            return entry.confidence

        if policy == DecayPolicy.LINEAR:
            rate = type_config.rate_per_day if type_config and type_config.rate_per_day else 0.01
            age_days = age_ms / DAY_MS
            return max(0.0, entry.confidence - rate * age_days)

        if policy == DecayPolicy.EXPONENTIAL:
            half_life = (
                type_config.half_life
                if type_config and type_config.half_life
                else self._config.default_half_life
            )
            decay_factor = math.exp(-0.693 * age_ms / half_life)
            return entry.confidence * decay_factor

        if policy == DecayPolicy.STEP:
            max_age = type_config.max_age if type_config and type_config.max_age else 7 * DAY_MS
            return 0.0 if age_ms > max_age else entry.confidence

        return entry.confidence

    def filter_decayed(self, entries: list[MemoryEntry], threshold: float = 0.05) -> list[MemoryEntry]:
        """Filter out memories that have decayed below threshold."""
        return [e for e in entries if self.compute_decayed_confidence(e) >= threshold]

    def get_expired(self, entries: list[MemoryEntry], threshold: float = 0.01) -> list[MemoryEntry]:
        """Get entries that should be cleaned up."""
        return [e for e in entries if self.compute_decayed_confidence(e) < threshold]
