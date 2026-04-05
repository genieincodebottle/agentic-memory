"""Built-in lightweight embedder using bag-of-words TF-IDF."""

from __future__ import annotations

import math
import re
from collections import Counter


class BuiltinEmbedder:
    """
    Built-in lightweight embedder using bag-of-words TF-IDF.
    No external dependencies, no API calls.
    Good enough for conflict detection and basic similarity.
    For production, plug in OpenAI/Cohere/local embeddings.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim
        self._document_count = 0
        self._document_freq: Counter[str] = Counter()
        self._idf_cache: dict[str, float] = {}

    def dimensions(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        return self._tokens_to_vector(tokens)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    def train(self, documents: list[str]) -> None:
        """Feed documents to build vocabulary and IDF stats."""
        for doc in documents:
            self._document_count += 1
            unique_tokens = set(self._tokenize(doc))
            for token in unique_tokens:
                self._document_freq[token] += 1

        # Rebuild IDF cache
        self._idf_cache.clear()
        for token, freq in self._document_freq.items():
            self._idf_cache[token] = math.log((self._document_count + 1) / (freq + 1)) + 1

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def _tokens_to_vector(self, tokens: list[str]) -> list[float]:
        vector = [0.0] * self._dim
        if not tokens:
            return vector

        tf: Counter[str] = Counter(tokens)

        for token, freq in tf.items():
            idf = self._idf_cache.get(token, 1.0)
            weight = (freq / len(tokens)) * idf
            h = self._hash_token(token)
            idx = abs(h) % self._dim
            sign = 1 if self._hash_token(token + "_sign") > 0 else -1
            vector[idx] += sign * weight

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    @staticmethod
    def _hash_token(token: str) -> int:
        h = 0
        for ch in token:
            h = ((h << 5) - h) + ord(ch)
            h &= 0xFFFFFFFF  # 32-bit
        # Convert to signed 32-bit
        if h >= 0x80000000:
            h -= 0x100000000
        return h
