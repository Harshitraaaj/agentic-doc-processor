"""
LangChain-based Document Loader
Supports multiple document formats: PDF, TXT, CSV, DOCX, PPTX, Images, and more
"""
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
    UnstructuredImageLoader,
)

try:
    import pytesseract
    from PIL import Image as PILImage
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from utils.config import settings
from utils.logger import logger


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    TEXT = "text"
    CSV = "csv"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    IMAGE = "image"
    HTML = "html"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class LangChainDocumentLoader:
    """
    Unified document loader using LangChain loaders
    
    Supports:
    - PDF documents (with text extraction)
    - Text files (.txt, .md)
    - CSV files (structured data)
    - Word documents (.docx)
    - PowerPoint presentations (.pptx)
    - Excel spreadsheets (.xlsx, .xls)
    - Images (.png, .jpg, etc.) with OCR
    - HTML files
    - And more via UnstructuredFileLoader
    """
    
    # File extension to document type mapping
    EXTENSION_MAP = {
        '.pdf': DocumentType.PDF,
        '.txt': DocumentType.TEXT,
        '.text': DocumentType.TEXT,
        '.md': DocumentType.MARKDOWN,
        '.markdown': DocumentType.MARKDOWN,
        '.csv': DocumentType.CSV,
        '.docx': DocumentType.DOCX,
        '.doc': DocumentType.DOCX,
        '.pptx': DocumentType.PPTX,
        '.ppt': DocumentType.PPTX,
        '.xlsx': DocumentType.XLSX,
        '.xls': DocumentType.XLSX,
        '.png': DocumentType.IMAGE,
        '.jpg': DocumentType.IMAGE,
        '.jpeg': DocumentType.IMAGE,
        '.tiff': DocumentType.IMAGE,
        '.tif': DocumentType.IMAGE,
        '.bmp': DocumentType.IMAGE,
        '.gif': DocumentType.IMAGE,
        '.html': DocumentType.HTML,
        '.htm': DocumentType.HTML,
    }
    
    def __init__(self):
        """Initialize document loader"""
        self.tesseract_configured = self._setup_tesseract()
        logger.info("LangChain Document Loader initialized")
    
    def _setup_tesseract(self) -> bool:
        """Configure Tesseract OCR for image processing"""
        if not TESSERACT_AVAILABLE:
            logger.warning("Tesseract not available. Image OCR will be limited.")
            return False
        
        if settings.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
            logger.info(f"Tesseract configured at: {settings.TESSERACT_CMD}")
            
            # Verify it works
            try:
                version = pytesseract.get_tesseract_version()
                logger.info(f"Tesseract version: {version}")
                return True
            except Exception as e:
                logger.error(f"Tesseract verification failed: {e}")
                return False
        else:
            logger.warning("TESSERACT_CMD not set in config")
            return False
    
    def detect_document_type(self, file_path: Path) -> DocumentType:
        """
        Detect document type from file extension
        
        Args:
            file_path: Path to file
        
        Returns:
            DocumentType enum
        """
        extension = file_path.suffix.lower()
        doc_type = self.EXTENSION_MAP.get(extension, DocumentType.UNKNOWN)
        
        if doc_type == DocumentType.UNKNOWN:
            logger.warning(f"Unknown file type: {extension}, will try generic loader")
        
        return doc_type
    
    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load PDF document using PyPDFLoader
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            List of Document objects (one per page)
        """
        try:
            logger.info(f"Loading PDF with PyPDFLoader: {file_path.name}")
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            total_chars = sum(len(doc.page_content) for doc in documents)
            logger.info(
                f"PDF loaded: {len(documents)} pages, {total_chars} characters",
                file=file_path.name
            )
            
            return documents
        
        except Exception as e:
            logger.error(f"PDF loading failed: {e}", file=file_path.name)
            # Fallback to UnstructuredFileLoader
            return self.load_generic(file_path)
    
    def load_text(self, file_path: Path, encoding: str = 'utf-8') -> List[Document]:
        """
        Load text file using TextLoader
        
        Args:
            file_path: Path to text file
            encoding: Text encoding
        
        Returns:
            List with single Document object
        """
        try:
            logger.info(f"Loading text file: {file_path.name}")
            loader = TextLoader(str(file_path), encoding=encoding)
            documents = loader.load()
            
            total_chars = sum(len(doc.page_content) for doc in documents)
            logger.info(
                f"Text file loaded: {total_chars} characters",
                file=file_path.name
            )
            
            return documents
        
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed, trying latin-1")
            try:
                loader = TextLoader(str(file_path), encoding='latin-1')
                return loader.load()
            except Exception as e:
                logger.error(f"Text loading failed: {e}")
                raise
        
        except Exception as e:
            logger.error(f"Text loading failed: {e}", file=file_path.name)
            raise
    
    def load_csv(self, file_path: Path) -> List[Document]:
        """
        Load CSV file using CSVLoader
        
        Args:
            file_path: Path to CSV file
        
        Returns:
            List of Document objects (one per row)
        """
        try:
            logger.info(f"Loading CSV file: {file_path.name}")
            loader = CSVLoader(str(file_path))
            documents = loader.load()
            
            logger.info(
                f"CSV loaded: {len(documents)} rows",
                file=file_path.name
            )
            
            return documents
        
        except Exception as e:
            logger.error(f"CSV loading failed: {e}", file=file_path.name)
            raise
    
    def load_docx(self, file_path: Path) -> List[Document]:
        """
        Load Word document using UnstructuredWordDocumentLoader
        
        Args:
            file_path: Path to DOCX file
        
        Returns:
            List of Document objects
        """
        try:
            logger.info(f"Loading Word document: {file_path.name}")
            loader = UnstructuredWordDocumentLoader(str(file_path))
            documents = loader.load()
            
            total_chars = sum(len(doc.page_content) for doc in documents)
            logger.info(
                f"Word document loaded: {total_chars} characters",
                file=file_path.name
            )
            
            return documents
        
        except Exception as e:
            logger.error(f"Word document loading failed: {e}", file=file_path.name)
            # Fallback to generic loader
            return self.load_generic(file_path)
    
    def load_pptx(self, file_path: Path) -> List[Document]:
        """
        Load PowerPoint presentation using UnstructuredPowerPointLoader
        
        Args:
            file_path: Path to PPTX file
        
        Returns:
            List of Document objects (one per slide)
        """
        try:
            logger.info(f"Loading PowerPoint: {file_path.name}")
            loader = UnstructuredPowerPointLoader(str(file_path))
            documents = loader.load()
            
            total_chars = sum(len(doc.page_content) for doc in documents)
            logger.info(
                f"PowerPoint loaded: {len(documents)} slides, {total_chars} characters",
                file=file_path.name
            )
            
            return documents
        
        except Exception as e:
            logger.error(f"PowerPoint loading failed: {e}", file=file_path.name)
            return self.load_generic(file_path)
    
    def load_xlsx(self, file_path: Path) -> List[Document]:
        """
        Load Excel spreadsheet using UnstructuredExcelLoader
        
        Args:
            file_path: Path to XLSX file
        
        Returns:
            List of Document objects (one per sheet)
        """
        try:
            logger.info(f"Loading Excel file: {file_path.name}")
            loader = UnstructuredExcelLoader(str(file_path))
            documents = loader.load()
            
            logger.info(
                f"Excel loaded: {len(documents)} sheets",
                file=file_path.name
            )
            
            return documents
        
        except Exception as e:
            logger.error(f"Excel loading failed: {e}", file=file_path.name)
            raise
    
    def load_image(self, file_path: Path) -> List[Document]:
        """
        Load image with OCR - GUARANTEED to return content or clear error
        
        Args:
            file_path: Path to image file
        
        Returns:
            List with single Document object
        """
        logger.info(f"Loading image with OCR: {file_path.name}")
        
        # Method 1: Direct Tesseract (most reliable)
        if TESSERACT_AVAILABLE:
            try:
                from PIL import Image as PILImage
                import pytesseract
                
                logger.info(f"Attempting direct Tesseract OCR on {file_path.name}")
                img = PILImage.open(file_path)
                text = pytesseract.image_to_string(img)
                
                logger.info(f"Tesseract extracted {len(text)} characters from {file_path.name}")
                
                if text.strip():
                    return [Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "file_type": "image",
                            "ocr_method": "tesseract_direct"
                        }
                    )]
                else:
                    logger.warning(f"Tesseract returned empty result for {file_path.name}")
                    # Return document with placeholder to avoid workflow failure
                    return [Document(
                        page_content="[Image contains no readable text]",
                        metadata={
                            "source": str(file_path),
                            "file_type": "image",
                            "ocr_method": "tesseract_direct",
                            "note": "No text detected in image"
                        }
                    )]
                    
            except Exception as e:
                logger.error(f"Direct Tesseract failed: {e}")
        else:
            logger.warning("Tesseract not available")
        
        # Method 2: Try UnstructuredImageLoader
        try:
            logger.info(f"Trying UnstructuredImageLoader for {file_path.name}")
            loader = UnstructuredImageLoader(
                str(file_path),
                mode="single"
            )
            documents = loader.load()
            
            total_chars = sum(len(doc.page_content) for doc in documents if doc.page_content)
            logger.info(f"UnstructuredImageLoader extracted {total_chars} characters")
            
            if documents and any(doc.page_content.strip() for doc in documents):
                return documents
            else:
                logger.warning("UnstructuredImageLoader returned empty content")
                
        except Exception as e:
            logger.error(f"UnstructuredImageLoader failed: {e}")
        
        # Fallback: Return placeholder document to avoid complete failure
        logger.warning(f"All OCR methods failed for {file_path.name}, returning placeholder")
        return [Document(
            page_content="[Unable to extract text from image - image may be blank or OCR failed]",
            metadata={
                "source": str(file_path),
                "file_type": "image",
                "ocr_method": "failed",
                "note": "All OCR methods failed. Check if Tesseract is installed and image contains text."
            }
        )]
    
    def load_generic(self, file_path: Path) -> List[Document]:
        """
        Load any file using UnstructuredFileLoader (fallback)
        
        Args:
            file_path: Path to file
        
        Returns:
            List of Document objects
        """
        try:
            logger.info(f"Loading with generic UnstructuredFileLoader: {file_path.name}")
            loader = UnstructuredFileLoader(str(file_path))
            documents = loader.load()
            
            total_chars = sum(len(doc.page_content) for doc in documents)
            logger.info(
                f"Generic loader completed: {total_chars} characters",
                file=file_path.name
            )
            
            return documents
        
        except Exception as e:
            logger.error(f"Generic loading failed: {e}", file=file_path.name)
            raise RuntimeError(f"Failed to load document: {e}")
    
    def load_document(
        self,
        file_path: Path,
        document_type: Optional[DocumentType] = None
    ) -> List[Document]:
        """
        Load document with automatic type detection
        
        Args:
            file_path: Path to document file
            document_type: Optional explicit document type (auto-detected if None)
        
        Returns:
            List of LangChain Document objects
        
        Raises:
            RuntimeError if loading fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect document type if not provided
        if document_type is None:
            document_type = self.detect_document_type(file_path)
        
        logger.info(f"Loading document: {file_path.name} (type: {document_type.value})")
        
        # Route to appropriate loader
        loader_map = {
            DocumentType.PDF: self.load_pdf,
            DocumentType.TEXT: self.load_text,
            DocumentType.MARKDOWN: self.load_text,
            DocumentType.CSV: self.load_csv,
            DocumentType.DOCX: self.load_docx,
            DocumentType.PPTX: self.load_pptx,
            DocumentType.XLSX: self.load_xlsx,
            DocumentType.IMAGE: self.load_image,
        }
        
        loader_func = loader_map.get(document_type, self.load_generic)
        
        try:
            documents = loader_func(file_path)
            
            if not documents:
                raise RuntimeError("No content extracted from document")
            
            return documents
        
        except Exception as e:
            logger.error(f"Document loading failed: {e}", file=file_path.name)
            raise
    
    def extract_text_from_documents(self, documents: List[Document]) -> str:
        """
        Combine multiple Document objects into single text string
        
        Args:
            documents: List of LangChain Document objects
        
        Returns:
            Combined text content
        """
        if not documents:
            return ""
        
        # Combine page contents with separators
        text_parts = []
        for i, doc in enumerate(documents, start=1):
            content = doc.page_content.strip()
            if content:
                # Add page/section separator if multiple documents
                if len(documents) > 1:
                    text_parts.append(f"--- Page/Section {i} ---\n{content}")
                else:
                    text_parts.append(content)
        
        combined_text = "\n\n".join(text_parts)
        
        logger.info(
            f"Combined {len(documents)} document(s) into {len(combined_text)} characters"
        )
        
        return combined_text
    
    def load_and_extract_text(
        self,
        file_path: Path,
        document_type: Optional[DocumentType] = None
    ) -> str:
        """
        Load document and extract text in one step
        
        Args:
            file_path: Path to document file
            document_type: Optional explicit document type
        
        Returns:
            Extracted text content
        """
        documents = self.load_document(file_path, document_type)
        return self.extract_text_from_documents(documents)


# Global instance
document_loader = LangChainDocumentLoader()
