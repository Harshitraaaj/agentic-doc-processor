from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class OCRService(ABC):
    """Abstract OCR interface for local (Tesseract) and cloud (Textract) providers."""

    @abstractmethod
    def extract_text(self, file_path: Path) -> str:
        """Extract plain text from file path."""
