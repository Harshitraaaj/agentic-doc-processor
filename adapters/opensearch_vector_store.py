from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import urlparse

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer

from services.vector_store_service import VectorStoreService
from utils.config import settings


class OpenSearchVectorStoreAdapter(VectorStoreService):
    """OpenSearch vector store adapter using SentenceTransformers embeddings."""

    def __init__(self):
        endpoint = (settings.AWS_OPENSEARCH_ENDPOINT or "").strip()
        if not endpoint:
            raise ValueError("OpenSearch endpoint not configured. Set [aws].opensearch_endpoint in config.ini")

        parsed = urlparse(endpoint if endpoint.startswith("http") else f"https://{endpoint}")
        host = parsed.hostname or endpoint.replace("https://", "").replace("http://", "")
        port = parsed.port or 443
        use_ssl = (parsed.scheme or "https").lower() == "https"

        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is None:
            raise ValueError("AWS credentials unavailable for OpenSearch client")

        auth = AWSV4SignerAuth(credentials, settings.AWS_REGION, "es")
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=auth,
            use_ssl=use_ssl,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,
        )

        self.index_name = "document_examples"
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = int(self.encoder.get_sentence_embedding_dimension())
        self._ensure_index()

    def _ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            return

        body = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                }
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "metadata": {"type": "object", "enabled": True},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.dimension,
                        "method": {
                            "engine": "nmslib",
                            "space_type": "cosinesimil",
                            "name": "hnsw",
                        },
                    },
                }
            },
        }
        self.client.indices.create(index=self.index_name, body=body)

    def index(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        if len(texts) != len(metadata):
            raise ValueError("texts and metadata must have same length")
        if not texts:
            return

        embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        for text, meta, vector in zip(texts, metadata, embeddings):
            doc = {
                "text": text,
                "metadata": meta,
                "embedding": vector.astype(float).tolist(),
            }
            self.client.index(index=self.index_name, body=doc)

        self.refresh()

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if k <= 0:
            return []

        query_vector = self.encoder.encode([query], convert_to_numpy=True)[0].astype(float).tolist()
        body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": k,
                    }
                }
            },
        }

        result = self.client.search(index=self.index_name, body=body)
        hits = result.get("hits", {}).get("hits", [])
        return [
            {
                "metadata": hit.get("_source", {}).get("metadata", {}),
                "distance": float(1.0 - float(hit.get("_score", 0.0))),
                "rank": rank + 1,
            }
            for rank, hit in enumerate(hits)
        ]

    def refresh(self) -> None:
        self.client.indices.refresh(index=self.index_name)
