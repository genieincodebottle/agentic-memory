"""Multi-signal retriever combining similarity, recency, importance, and task relevance."""

from __future__ import annotations

import math
from typing import Any

from ..types import (
    MemoryEntry,
    RetrievalQuery,
    RetrievalResult,
    ImportanceLevel,
)
from ..utils import cosine_similarity, days_since

DEFAULT_WEIGHTS: dict[str, float] = {
    "similarity": 0.4,
    "recency": 0.25,
    "importance": 0.2,
    "taskRelevance": 0.15,
}

IMPORTANCE_SCORES: dict[str, float] = {
    "hard": 1.0,
    "soft": 0.5,
    "ephemeral": 0.2,
}


class MultiSignalRetriever:
    """
    Multi-signal retriever.
    Combines similarity, recency, importance, and task-relevance
    into a single ranked result set.
    """

    def __init__(
        self,
        store: Any,  # StorageBackend
        embedder: Any,  # Embedder
        weights: dict[str, float] | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        signals = query.signals or ["similarity", "recency", "importance"]
        limit = query.limit
        threshold = query.threshold

        candidates = await self._store.search(query)
        if not candidates:
            return []

        # Compute query embedding if similarity is requested
        query_embedding: list[float] | None = None
        if "similarity" in signals:
            query_embedding = await self._embedder.embed(query.query)

        # Compute task context embedding if task relevance is requested
        task_embedding: list[float] | None = None
        if "taskRelevance" in signals and query.task_context:
            task_embedding = await self._embedder.embed(query.task_context)

        scored: list[RetrievalResult] = []

        for entry in candidates:
            signal_scores: dict[str, float] = {}
            total_score = 0.0
            total_weight = 0.0

            # Cache entry embedding
            entry_embedding: list[float] | None = None
            needs_embedding = (
                ("similarity" in signals and query_embedding is not None)
                or ("taskRelevance" in signals and task_embedding is not None)
            )
            if needs_embedding:
                entry_embedding = entry.embedding or await self._embedder.embed(entry.content)

            # Similarity score
            if "similarity" in signals and query_embedding is not None and entry_embedding is not None:
                sim = max(0.0, cosine_similarity(query_embedding, entry_embedding))
                signal_scores["similarity"] = sim
                total_score += sim * self._weights["similarity"]
                total_weight += self._weights["similarity"]

            # Recency score
            if "recency" in signals:
                age = max(0.0, days_since(entry.updated_at))
                recency = math.exp(-0.693 * age / 7)
                signal_scores["recency"] = recency
                total_score += recency * self._weights["recency"]
                total_weight += self._weights["recency"]

            # Importance score
            if "importance" in signals:
                imp = IMPORTANCE_SCORES.get(entry.importance.value, 0.5)
                signal_scores["importance"] = imp
                total_score += imp * self._weights["importance"]
                total_weight += self._weights["importance"]

            # Task relevance score
            if "taskRelevance" in signals and task_embedding is not None and entry_embedding is not None:
                rel = max(0.0, cosine_similarity(task_embedding, entry_embedding))
                signal_scores["taskRelevance"] = rel
                total_score += rel * self._weights["taskRelevance"]
                total_weight += self._weights["taskRelevance"]

            normalized_score = total_score / total_weight if total_weight > 0 else 0.0

            if normalized_score >= threshold:
                scored.append(RetrievalResult(
                    entry=entry,
                    score=normalized_score,
                    signal_scores=signal_scores,
                ))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:limit]
