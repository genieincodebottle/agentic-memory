"""
Voyage AI embedder adapter (high-quality, cheaper than OpenAI).
Requires: VOYAGE_API_KEY

Usage:
    from agentic_memory import AgentMemory
    from agentic_memory.adapters import VoyageEmbedder

    memory = AgentMemory(
        embedder=VoyageEmbedder(api_key="voyage-...")
    )
"""

from __future__ import annotations

import json
from urllib.request import Request, urlopen


class VoyageEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3-lite",
    ) -> None:
        self._api_key = api_key
        self._model = model

    def dimensions(self) -> int:
        return 512 if "lite" in self._model else 1024

    async def embed(self, text: str) -> list[float]:
        result = self._call([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._call(texts)

    def _call(self, input_texts: list[str]) -> list[list[float]]:
        payload = json.dumps({
            "model": self._model,
            "input": input_texts,
        }).encode("utf-8")

        req = Request(
            "https://api.voyageai.com/v1/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        with urlopen(req) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Voyage AI embedding failed ({resp.status})")
            data = json.loads(resp.read())

        return [d["embedding"] for d in data["data"]]
