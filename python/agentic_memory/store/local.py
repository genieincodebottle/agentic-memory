"""In-memory storage backend. Zero dependencies."""

from __future__ import annotations

from typing import Optional

from ..types import MemoryEntry, RetrievalQuery


class LocalStore:
    """
    In-memory storage backend.
    Perfect for development, testing, and single-process agents.
    Data is lost when the process exits.
    """

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    async def get(self, id: str) -> Optional[MemoryEntry]:
        entry = self._entries.get(id)
        return entry if entry is None else MemoryEntry(**vars(entry))

    async def get_all(self, scope: Optional[str] = None) -> list[MemoryEntry]:
        entries = list(self._entries.values())
        if scope is not None:
            entries = [e for e in entries if e.scope == scope]
        return entries

    async def set(self, entry: MemoryEntry) -> None:
        self._entries[entry.id] = MemoryEntry(**vars(entry))

    async def delete(self, id: str) -> bool:
        if id in self._entries:
            del self._entries[id]
            return True
        return False

    async def clear(self, scope: Optional[str] = None) -> None:
        if scope is None:
            self._entries.clear()
        else:
            to_delete = [k for k, v in self._entries.items() if v.scope == scope]
            for k in to_delete:
                del self._entries[k]

    async def search(self, query: RetrievalQuery) -> list[MemoryEntry]:
        results = list(self._entries.values())
        if query.scope is not None:
            results = [e for e in results if e.scope == query.scope]
        if query.types:
            results = [e for e in results if e.type in query.types]
        return results

    @property
    def size(self) -> int:
        return len(self._entries)
