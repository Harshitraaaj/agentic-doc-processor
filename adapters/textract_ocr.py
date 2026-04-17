from __future__ import annotations

from pathlib import Path
from typing import Optional

import boto3

from services.ocr_service import OCRService
from utils.config import settings


class TextractOCRAdapter(OCRService):
    """AWS Textract OCR adapter for scanned PDFs/images."""

    def __init__(self, region: Optional[str] = None):
        self.region = region or settings.AWS_REGION
        self.client = boto3.client("textract", region_name=self.region)

    def extract_text(self, file_path: Path) -> str:
        data = file_path.read_bytes()
        response = self.client.detect_document_text(Document={"Bytes": data})
        lines = [
            block.get("Text", "")
            for block in response.get("Blocks", [])
            if block.get("BlockType") == "LINE"
        ]
        return "\n".join(line for line in lines if line).strip()
