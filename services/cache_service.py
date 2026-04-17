from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class CacheService(ABC):
    """Abstract cache interface for local memory/sqlite and cloud redis backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Return cached value for key, or None if missing."""

    @abstractmethod
    def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        """Store value with optional TTL."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete key if present."""
