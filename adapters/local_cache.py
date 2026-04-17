from __future__ import annotations

import time
from typing import Optional

from services.cache_service import CacheService


class LocalCacheAdapter(CacheService):
    """Simple in-memory cache for local mode."""

    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}

    def get(self, key: str) -> Optional[str]:
        value = self._store.get(key)
        if not value:
            return None
        payload, expires_at = value
        if expires_at and time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return payload

    def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds > 0 else 0.0
        self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
