from __future__ import annotations

from pathlib import Path

from ocr.processor import ocr_processor
from services.ocr_service import OCRService


class LocalOCRAdapter(OCRService):
    """Wrapper around existing local OCR processor."""

    def extract_text(self, file_path: Path) -> str:
        return ocr_processor.extract_text_from_file(file_path)
