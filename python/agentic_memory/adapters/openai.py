"""
OpenAI embedder adapter.
Requires: pip install httpx (or aiohttp)

Usage:
    from agentic_memory import AgentMemory
    from agentic_memory.adapters import OpenAIEmbedder

    memory = AgentMemory(
        embedder=OpenAIEmbedder(api_key="sk-...")
    )
"""

from __future__ import annotations

import json
from urllib.request import Request, urlopen
from typing import Optional


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dim = dim
        self._base_url = base_url

    def dimensions(self) -> int:
        return self._dim

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
            "dimensions": self._dim,
        }).encode("utf-8")

        req = Request(
            f"{self._base_url}/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        with urlopen(req) as resp:
            if resp.status != 200:
                raise RuntimeError(f"OpenAI embedding failed ({resp.status})")
            data = json.loads(resp.read())

        # Sort by index (OpenAI may return out of order)
        items = sorted(data["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in items]
