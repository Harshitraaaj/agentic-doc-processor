from __future__ import annotations

from adapters.faiss_vector_store import FAISSVectorStoreAdapter
from adapters.local_cache import LocalCacheAdapter
from adapters.local_ocr import LocalOCRAdapter
from adapters.local_storage import LocalStorageAdapter
from adapters.s3_storage import S3StorageAdapter
from adapters.textract_ocr import TextractOCRAdapter
from services.cache_service import CacheService
from services.ocr_service import OCRService
from services.storage_service import StorageService
from services.vector_store_service import VectorStoreService
from utils.config import settings
from utils.logger import logger


class ServiceRegistry:
    """Factory for selecting local/cloud providers from config.ini [stack] switches."""

    @staticmethod
    def get_storage() -> StorageService:
        provider = settings.STACK_STORAGE_PROVIDER.lower().strip()
        if provider in {"s3", "aws_s3"}:
            return S3StorageAdapter()
        return LocalStorageAdapter()

    @staticmethod
    def get_ocr() -> OCRService:
        provider = settings.STACK_OCR_PROVIDER.lower().strip()
        if provider in {"textract", "aws_textract"}:
            try:
                return TextractOCRAdapter()
            except Exception as e:
                logger.warning(
                    "ServiceRegistry: Textract unavailable, falling back to LocalOCRAdapter",
                    error=str(e),
                )
                return LocalOCRAdapter()
        return LocalOCRAdapter()

    @staticmethod
    def get_cache() -> CacheService:
        provider = settings.STACK_CACHE_PROVIDER.lower().strip()
        profile = settings.STACK_PROFILE.lower().strip()

        if profile == "cloud":
            provider = "elasticache"

        if provider in {"redis", "elasticache"}:
            try:
                from adapters.redis_cache import RedisCacheAdapter

                return RedisCacheAdapter()
            except Exception as e:
                logger.warning(
                    "ServiceRegistry: Redis unavailable, falling back to LocalCacheAdapter",
                    error=str(e),
                )
                return LocalCacheAdapter()
        return LocalCacheAdapter()

    @staticmethod
    def get_vector_store() -> VectorStoreService:
        provider = settings.STACK_VECTOR_PROVIDER.lower().strip()
        if provider in {"opensearch", "aws_opensearch"}:
            try:
                from adapters.opensearch_vector_store import OpenSearchVectorStoreAdapter

                return OpenSearchVectorStoreAdapter()
            except Exception as e:
                logger.warning(
                    "ServiceRegistry: OpenSearch unavailable, falling back to FAISSVectorStoreAdapter",
                    error=str(e),
                )
                return FAISSVectorStoreAdapter()
        return FAISSVectorStoreAdapter()
