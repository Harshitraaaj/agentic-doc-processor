from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class VectorStoreService(ABC):
    """Abstract vector store interface for FAISS/OpenSearch backends."""

    @abstractmethod
    def index(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Index text documents with metadata."""

    @abstractmethod
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Return top-k semantic matches."""

    @abstractmethod
    def refresh(self) -> None:
        """Refresh internal index/cache state."""
