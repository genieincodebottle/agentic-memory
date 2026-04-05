"""Conflict detector for finding contradictions in stored memories."""

from __future__ import annotations

import re
from typing import Any, Optional

from ..types import MemoryEntry, ConflictResult, ConflictAction, ImportanceLevel
from ..utils import cosine_similarity

NEGATION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bdon't\b", r"\bdoesn't\b",
        r"\bwon't\b", r"\bcan't\b", r"\bhate\b", r"\bdislike\b", r"\bavoid\b",
        r"\bstop\b", r"\bquit\b", r"\bremove\b", r"\bdelete\b",
        r"\binstead of\b", r"\brather than\b", r"\bno longer\b",
    ]
]

CHANGE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bnow\b", r"\bactually\b", r"\bchanged?\b", r"\bswitch",
        r"\bprefer\b", r"\bwant\b", r"\bused to\b", r"\banymore\b",
    ]
]

ANTONYM_PAIRS: list[tuple[re.Pattern, re.Pattern]] = [
    (re.compile(a, re.IGNORECASE), re.compile(b, re.IGNORECASE))
    for a, b in [
        (r"\blike\b", r"\bdislike\b"),
        (r"\blove\b", r"\bhate\b"),
        (r"\byes\b", r"\bno\b"),
        (r"\btrue\b", r"\bfalse\b"),
        (r"\bvegetarian\b", r"\bmeat\b"),
        (r"\bvegan\b", r"\bmeat\b|\bdairy\b"),
        (r"\bmorning\b", r"\bevening\b|\bnight\b"),
        (r"\blight\b", r"\bdark\b"),
        (r"\bhot\b", r"\bcold\b"),
        (r"\bfast\b", r"\bslow\b"),
        (r"\benable\b", r"\bdisable\b"),
        (r"\ballow\b", r"\bblock\b|\bdeny\b"),
        (r"\baccept\b", r"\breject\b"),
    ]
]


class ConflictDetector:
    """
    Conflict detector.
    Compares new content against stored memories to find contradictions.
    Uses semantic similarity + negation detection.
    """

    def __init__(
        self,
        store: Any,
        embedder: Any,
        topic_threshold: float = 0.6,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._topic_threshold = topic_threshold

    async def check(self, content: str, scope: Optional[str] = None) -> list[ConflictResult]:
        entries = await self._store.get_all(scope)
        if not entries:
            return []

        content_embedding = await self._embedder.embed(content)
        conflicts: list[ConflictResult] = []

        for entry in entries:
            entry_embedding = entry.embedding or await self._embedder.embed(entry.content)
            similarity = cosine_similarity(content_embedding, entry_embedding)

            if similarity < self._topic_threshold:
                continue

            contradiction_score = self._score_contradiction(content, entry.content)
            if contradiction_score <= 0:
                continue

            confidence = min(1.0, similarity * 0.5 + contradiction_score * 0.5)

            conflicts.append(ConflictResult(
                incoming=content,
                stored=entry,
                confidence=confidence,
                action=self._suggest_action(entry, confidence),
                reason=self._generate_reason(content, entry, contradiction_score),
            ))

        conflicts.sort(key=lambda c: c.confidence, reverse=True)
        return conflicts

    def _score_contradiction(self, incoming: str, stored: str) -> float:
        score = 0.0

        incoming_negated = any(p.search(incoming) for p in NEGATION_PATTERNS)
        stored_negated = any(p.search(stored) for p in NEGATION_PATTERNS)

        if incoming_negated != stored_negated:
            score += 0.5

        if any(p.search(incoming) for p in CHANGE_PATTERNS):
            score += 0.3

        score += self._check_antonyms(incoming, stored) * 0.4

        return min(1.0, score)

    @staticmethod
    def _check_antonyms(a: str, b: str) -> float:
        matches = 0
        for word1, word2 in ANTONYM_PAIRS:
            if (word1.search(a) and word2.search(b)) or (word2.search(a) and word1.search(b)):
                matches += 1
        return min(1.0, matches * 0.5)

    @staticmethod
    def _suggest_action(stored: MemoryEntry, confidence: float) -> ConflictAction:
        if stored.importance == ImportanceLevel.HARD:
            return ConflictAction.CLARIFY
        if confidence > 0.7 and stored.importance == ImportanceLevel.SOFT:
            return ConflictAction.OVERRIDE
        if stored.importance == ImportanceLevel.EPHEMERAL:
            return ConflictAction.OVERRIDE
        return ConflictAction.CLARIFY

    @staticmethod
    def _generate_reason(incoming: str, stored: MemoryEntry, contradiction_score: float) -> str:
        if stored.importance == ImportanceLevel.HARD:
            return (
                f'Conflicts with a hard constraint: "{stored.content}". '
                "This memory was marked as non-negotiable - please confirm the change."
            )
        if contradiction_score > 0.7:
            return (
                f'Directly contradicts stored memory: "{stored.content}". '
                "The new statement appears to reverse a previous preference."
            )
        return (
            f'Potentially conflicts with: "{stored.content}". '
            "The statements may be inconsistent."
        )
