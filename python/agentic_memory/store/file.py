"""File-based storage backend. Persists to JSON on disk."""

from __future__ import annotations

import json
import os
from typing import Optional

from ..types import MemoryEntry, MemoryType, ImportanceLevel, RetrievalQuery


class FileStore:
    """
    File-based storage backend. Persists to a JSON file on disk.
    Good for single-process agents that need persistence across restarts.
    Not suitable for concurrent multi-agent access.
    """

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._entries: dict[str, MemoryEntry] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._file_path):
            return
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                item["type"] = MemoryType(item["type"])
                item["importance"] = ImportanceLevel(item["importance"])
                entry = MemoryEntry(**item)
                self._entries[entry.id] = entry
        except (json.JSONDecodeError, KeyError, TypeError):
            self._entries.clear()

    def _persist(self) -> None:
        if not self._dirty:
            return
        dir_path = os.path.dirname(self._file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        data = []
        for entry in self._entries.values():
            d = vars(entry).copy()
            d["type"] = d["type"].value
            d["importance"] = d["importance"].value
            data.append(d)

        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._dirty = False

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
        self._dirty = True
        self._persist()

    async def delete(self, id: str) -> bool:
        if id in self._entries:
            del self._entries[id]
            self._dirty = True
            self._persist()
            return True
        return False

    async def clear(self, scope: Optional[str] = None) -> None:
        if scope is None:
            self._entries.clear()
        else:
            to_delete = [k for k, v in self._entries.items() if v.scope == scope]
            for k in to_delete:
                del self._entries[k]
        self._dirty = True
        self._persist()

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
