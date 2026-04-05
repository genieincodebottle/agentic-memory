"""Utility functions for agentic-memory."""

import time
import math
import random
from datetime import datetime, timezone


def generate_id() -> str:
    """Generate a unique memory ID."""
    ts = int(time.time() * 1000)
    rand = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=7))
    return f"mem_{ts:x}_{rand}"


def now() -> str:
    """Get current ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    if len(a) == 0:
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    denom = norm_a * norm_b

    return 0.0 if denom == 0 else dot_product / denom


def days_since(iso_timestamp: str) -> float:
    """Time difference in days from ISO timestamp to now."""
    dt = datetime.fromisoformat(iso_timestamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    diff = datetime.now(timezone.utc) - dt
    return diff.total_seconds() / 86400
