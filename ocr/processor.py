"""
OCR Processor for extracting text from images and scanned PDFs
"""
import os
from pathlib import Path
from typing import Optional, List
from enum import Enum

import pytesseract
from PIL import Image
try:
    from pdf2image import convert_from_path
    POPPLER_AVAILABLE = True
except ImportError:
    POPPLER_AVAILABLE = False
    convert_from_path = None

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    PdfReader = None

from utils.config import settings
from utils.logger import logger


class FileType(str, Enum):
    """Supported file types"""
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    UNKNOWN = "unknown"


class OCRProcessor:
    """
    OCR processor using Tesseract
    
    Supports:
    - Images (PNG, JPG, JPEG, TIFF, BMP)
    - Scanned PDFs
    - Plain text files (pass-through)
    """
    
    SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}
    SUPPORTED_TEXT_EXTENSIONS = {'.txt'}
    SUPPORTED_PDF_EXTENSIONS = {'.pdf'}
    
    def __init__(self):
        self._setup_tesseract()
    
    def _setup_tesseract(self) -> None:
        """Configure Tesseract OCR"""
        if settings.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
            logger.info(f"Tesseract configured at: {settings.TESSERACT_CMD}")
        else:
            logger.warning(
                "Tesseract path not configured. "
                "Ensure tesseract is in PATH or set TESSERACT_CMD in config."
            )
    
    def detect_file_type(self, file_path: Path) -> FileType:
        """
        Detect file type from extension
        
        Args:
            file_path: Path to file
        
        Returns:
            FileType enum
        """
        extension = file_path.suffix.lower()
        
        if extension in self.SUPPORTED_IMAGE_EXTENSIONS:
            return FileType.IMAGE
        elif extension in self.SUPPORTED_PDF_EXTENSIONS:
            return FileType.PDF
        elif extension in self.SUPPORTED_TEXT_EXTENSIONS:
            return FileType.TEXT
        else:
            logger.warning(f"Unknown file type: {extension}")
            return FileType.UNKNOWN
    
    def extract_text_from_image(
        self,
        image_path: Path,
        language: str = None
    ) -> str:
        """
        Extract text from image using Tesseract OCR
        
        Args:
            image_path: Path to image file
            language: Tesseract language code(s), comma-separated
        
        Returns:
            Extracted text
        
        Raises:
            RuntimeError if OCR fails
        """
        language = language or settings.OCR_LANGUAGES
        
        try:
            logger.info(f"Performing OCR on image: {image_path.name}")
            
            # Open image
            image = Image.open(image_path)
            
            # Perform OCR
            text = pytesseract.image_to_string(
                image,
                lang=language,
                config='--psm 3'  # Fully automatic page segmentation
            )
            
            logger.info(
                f"OCR completed: extracted {len(text)} characters",
                file=image_path.name
            )
            
            return text.strip()
        
        except pytesseract.TesseractNotFoundError:
            error_msg = (
                "Tesseract not found. "
                "Please install Tesseract OCR and configure TESSERACT_CMD"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            raise RuntimeError(f"OCR failed: {e}")
    
    def extract_text_from_pdf(
        self,
        pdf_path: Path,
        language: str = None,
        dpi: int = 300
    ) -> str:
        """
        Extract text from PDF - tries text extraction first, then OCR if needed
        
        Args:
            pdf_path: Path to PDF file
            language: Tesseract language code(s)
            dpi: DPI for PDF to image conversion
        
        Returns:
            Extracted text from all pages
        
        Raises:
            RuntimeError if extraction fails
        """
        language = language or settings.OCR_LANGUAGES
        
        # Try PyPDF2 text extraction first (doesn't need poppler)
        if PYPDF2_AVAILABLE:
            try:
                logger.info(f"Attempting text extraction from PDF: {pdf_path.name}")
                reader = PdfReader(str(pdf_path))
                all_text = []
                
                for i, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        all_text.append(text.strip())
                
                if all_text:
                    full_text = "\n\n".join(all_text)
                    logger.info(
                        f"PDF text extraction successful: {len(reader.pages)} pages, {len(full_text)} characters",
                        file=pdf_path.name
                    )
                    return full_text
                else:
                    logger.info(f"PDF has no extractable text, will try OCR")
            except Exception as e:
                logger.warning(f"PyPDF2 text extraction failed: {e}, will try OCR")
        
        # Fall back to OCR if text extraction failed or unavailable
        if not POPPLER_AVAILABLE:
            raise RuntimeError(
                "PDF has no extractable text and OCR requires poppler. "
                "Options:\n"
                "1. Use a PDF with selectable text (recommended)\n"
                "2. Convert to .txt file\n"
                "3. Install poppler for OCR: https://github.com/oschwartz10612/poppler-windows/releases/\n"
                "4. Use pip install pdf2image after installing poppler"
            )
        
        try:
            logger.info(f"Converting PDF to images for OCR: {pdf_path.name}")
            
            # Convert PDF pages to images
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                fmt='png'
            )
            
            logger.info(f"PDF has {len(images)} pages")
            
            # Perform OCR on each page
            all_text = []
            for i, image in enumerate(images, start=1):
                logger.info(f"Processing page {i}/{len(images)}")
                
                page_text = pytesseract.image_to_string(
                    image,
                    lang=language,
                    config='--psm 3'
                )
                
                all_text.append(page_text.strip())
            
            # Combine all pages
            full_text = "\n\n".join(all_text)
            
            logger.info(
                f"PDF OCR completed: {len(images)} pages, {len(full_text)} characters",
                file=pdf_path.name
            )
            
            return full_text
        
        except Exception as e:
            error_msg = str(e)
            if "poppler" in error_msg.lower() or "Unable to get page count" in error_msg:
                raise RuntimeError(
                    "PDF OCR requires poppler. "
                    "Install from: https://github.com/oschwartz10612/poppler-windows/releases/ "
                    "Then add to PATH or use text files (.txt) instead."
                )
            logger.error(f"PDF OCR failed for {pdf_path}: {e}")
            raise RuntimeError(f"PDF OCR failed: {e}")
    
    def extract_text_from_file(self, file_path: Path) -> str:
        """
        Extract text from any supported file type
        
        Args:
            file_path: Path to file
        
        Returns:
            Extracted text
        
        Raises:
            ValueError if file type not supported
            RuntimeError if extraction fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = self.detect_file_type(file_path)
        
        logger.info(
            f"Processing file",
            file=file_path.name,
            type=file_type.value
        )
        
        # Handle text files first (most common and easiest)
        if file_type == FileType.TEXT:
            logger.info(f"Reading text file: {file_path.name}")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                logger.info(f"Read {len(text)} characters from text file")
                return text
            except Exception as e:
                raise RuntimeError(f"Failed to read text file: {e}")
        
        elif file_type == FileType.IMAGE:
            return self.extract_text_from_image(file_path)
        
        elif file_type == FileType.PDF:
            return self.extract_text_from_pdf(file_path)
        
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Please use .txt, .pdf, or image files (.png, .jpg)")
    
    def get_ocr_info(self, image_path: Path) -> dict:
        """
        Get detailed OCR information including confidence scores
        
        Args:
            image_path: Path to image
        
        Returns:
            Dictionary with OCR data including confidence scores
        """
        try:
            image = Image.open(image_path)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                image,
                lang=settings.OCR_LANGUAGES,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confidences = [
                conf for conf in data['conf']
                if conf != -1  # -1 means no text detected
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": pytesseract.image_to_string(image, lang=settings.OCR_LANGUAGES),
                "word_count": len([w for w in data['text'] if w.strip()]),
                "avg_confidence": avg_confidence,
                "page_count": 1
            }
        
        except Exception as e:
            logger.error(f"Failed to get OCR info: {e}")
            return {
                "text": "",
                "word_count": 0,
                "avg_confidence": 0.0,
                "page_count": 0
            }


# Global OCR processor instance
ocr_processor = OCRProcessor()
