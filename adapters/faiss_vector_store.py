from __future__ import annotations

from typing import Any, Dict, List

from services.vector_store_service import VectorStoreService
from utils.faiss_manager import get_faiss_index


class FAISSVectorStoreAdapter(VectorStoreService):
    """Adapter over existing FAISS manager implementation."""

    def __init__(self):
        self._faiss_index = None

    def _get_index(self):
        if self._faiss_index is None:
            self._faiss_index = get_faiss_index()
        return self._faiss_index

    def index(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        idx = self._get_index()
        idx.add_documents(texts=texts, metadata=metadata)
        idx.save()

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        idx = self._get_index()
        return idx.search(query=query, k=k)

    def refresh(self) -> None:
        idx = self._get_index()
        idx._load_or_create_index()
