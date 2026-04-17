"""
FAISS-based semantic search module for context retrieval
"""
import os
import re
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from utils.config import settings
from utils.logger import logger


class FAISSIndex:
    """
    FAISS vector store for semantic search
    
    Used for retrieving similar document examples during extraction.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_name: str = "document_examples"
    ):
        """
        Initialize FAISS index
        
        Args:
            model_name: SentenceTransformer model name
            index_name: Name for index file
        """
        self.model_name = model_name
        self.index_name = index_name
        self.index_path = settings.FAISS_INDEX_DIR / f"{index_name}.faiss"
        self.metadata_path = settings.FAISS_INDEX_DIR / f"{index_name}_metadata.pkl"
        self.skip_hf = os.getenv("SKIP_HF", "false").strip().lower() in {"1", "true", "yes", "on"}
        
        # Initialize embedding model
        if self.skip_hf:
            self.encoder = None
            self.dimension = 384
            logger.info("SKIP_HF enabled: using local hash embeddings for FAISS (no HuggingFace downloads)")
        else:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is required when SKIP_HF is false. "
                    "Install sentence-transformers or set [runtime].skip_hf = true"
                )
            try:
                self.encoder = SentenceTransformer(model_name)
                self.dimension = self.encoder.get_sentence_embedding_dimension()
                logger.info(f"FAISS encoder loaded: {model_name}, dimension={self.dimension}")
            except Exception as e:
                logger.error(f"Failed to load encoder: {e}")
                raise
        
        # Initialize or load index
        self.index = None
        self.metadata = []
        self._load_or_create_index()
    
    def _load_or_create_index(self) -> None:
        """Load existing index or create new one"""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(
                    f"FAISS index loaded: {self.index.ntotal} vectors",
                    index_name=self.index_name
                )
            except Exception as e:
                logger.warning(f"Failed to load index: {e}, creating new")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self) -> None:
        """Create new FAISS index"""
        # Use IndexFlatL2 for exact search (good for small datasets)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        logger.info(f"Created new FAISS index: dimension={self.dimension}")
    
    def add_documents(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Add documents to index
        
        Args:
            texts: List of text strings to embed
            metadata: List of metadata dicts (same length as texts)
        """
        if len(texts) != len(metadata):
            raise ValueError("texts and metadata must have same length")
        
        if not texts:
            logger.warning("No texts to add")
            return
        
        try:
            # Generate embeddings
            logger.info(f"Encoding {len(texts)} documents")
            embeddings = self._encode_texts(texts)
            
            # Add to index
            self.index.add(embeddings.astype('float32'))
            self.metadata.extend(metadata)
            
            logger.info(f"Added {len(texts)} documents to FAISS index")
        
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def search(
        self,
        query: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Query text
            k: Number of results to return
        
        Returns:
            List of results with metadata and scores
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []
        
        try:
            # Encode query
            query_embedding = self._encode_texts([query]).astype('float32')
            
            # Search
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):
                    results.append({
                        "metadata": self.metadata[idx],
                        "distance": float(dist),
                        "rank": i + 1
                    })
            
            logger.debug(f"FAISS search returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if self.encoder is not None:
            return self.encoder.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        vectors: List[np.ndarray] = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            tokens = re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())
            if not tokens:
                vectors.append(vec)
                continue
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                idx = int(digest, 16) % self.dimension
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)

        return np.vstack(vectors)
    
    def save(self) -> None:
        """Save index and metadata to disk"""
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"FAISS index saved: {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def clear(self) -> None:
        """Clear index and metadata"""
        self._create_new_index()
        logger.info("FAISS index cleared")


# Global FAISS instance (lazy initialization)
_faiss_index: Optional[FAISSIndex] = None


def get_faiss_index() -> FAISSIndex:
    """Get or create global FAISS index"""
    global _faiss_index
    if _faiss_index is None:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        _faiss_index = FAISSIndex(model_name=model_name)
    return _faiss_index
