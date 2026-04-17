from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

import redis

from services.cache_service import CacheService
from utils.config import settings


class RedisCacheAdapter(CacheService):
    """Redis-backed cache adapter for cloud profile."""

    def __init__(self):
        endpoint = (settings.AWS_REDIS_ENDPOINT or "").strip()
        if not endpoint:
            raise ValueError("Redis endpoint not configured. Set [aws].redis_endpoint in config.ini")

        parsed = urlparse(endpoint if "://" in endpoint else f"redis://{endpoint}")
        host = parsed.hostname or endpoint
        port = parsed.port or 6379
        scheme = (parsed.scheme or "redis").lower()
        use_ssl = scheme == "rediss"

        password = os.environ.get("REDIS_AUTH_TOKEN") or None
        username = os.environ.get("REDIS_USERNAME") or None

        self.client = redis.Redis(
            host=host,
            port=port,
            username=username,
            password=password,
            ssl=use_ssl,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )

    def get(self, key: str) -> Optional[str]:
        return self.client.get(key)

    def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        self.client.set(name=key, value=value, ex=ttl_seconds)

    def delete(self, key: str) -> None:
        self.client.delete(key)
