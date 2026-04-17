from __future__ import annotations

from pathlib import Path
from typing import Optional

from services.storage_service import StorageService
from utils.config import settings


class LocalStorageAdapter(StorageService):
    """Filesystem-backed storage adapter for local development."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or (settings.DATA_DIR / "uploads")
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        safe_key = key.strip().lstrip("/\\")
        path = self.root / safe_key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def put_file(self, key: str, data: bytes, content_type: Optional[str] = None) -> str:
        path = self._resolve(key)
        path.write_bytes(data)
        return str(path)

    def get_file(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def get_signed_url(self, key: str, expires_seconds: int = 900) -> str:
        return str(self._resolve(key))
