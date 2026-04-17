from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class StorageService(ABC):
    """Abstract storage interface for local/cloud document and artifact storage."""

    @abstractmethod
    def put_file(self, key: str, data: bytes, content_type: Optional[str] = None) -> str:
        """Store bytes at key and return provider-specific URI."""

    @abstractmethod
    def get_file(self, key: str) -> bytes:
        """Read bytes from key."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether key exists."""

    @abstractmethod
    def get_signed_url(self, key: str, expires_seconds: int = 900) -> str:
        """Return a temporary URL for download/upload when supported."""
