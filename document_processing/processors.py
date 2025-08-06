"""LangChain document processing and loaders integration."""

import datetime
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union

from models.errors import ContentValidationError
from error_handling.handlers import ErrorHandler
from models.chunking import (
    Chunk, ChunkMetadata, ChunkingConfig, ChunkIndex, ChunkedDocument,
    ChunkingStrategy, BoundaryQuality, BaseChunkingStrategy
)

# Import LangChain document loaders
try:
    from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredFileLoader
    from langchain.schema import Document
    import fitz  # PyMuPDF
    LANGCHAIN_LOADERS_AVAILABLE = True
    logging.info("LangChain document loaders successfully imported")
except ImportError as e:
    LANGCHAIN_LOADERS_AVAILABLE = False
    logging.warning(f"LangChain document loaders not available: {e}")

# Import chunking dependencies
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import tiktoken
    CHUNKING_DEPENDENCIES_AVAILABLE = True
    logging.info("Chunking dependencies successfully imported")
except ImportError as e:
    CHUNKING_DEPENDENCIES_AVAILABLE = False
    logging.warning(f"Chunking dependencies not available: {e}")


@dataclass
class ProcessedDocument:
    """Schema for processed document containing content, metadata, and structure."""
    content: str
    metadata: Dict[str, Any]
    pages: List[Dict[str, Any]]
    structure: Dict[str, Any]
    processing_method: str
    original_filename: str


@dataclass
class DocumentPage:
    """Schema for individual document pages."""
    page_number: int
    content: str
    metadata: Dict[str, Any]


class TemporaryFileManager:
    """Secure temporary file management with automatic cleanup."""
    
    def __init__(self):
        """Initialize the temporary file manager."""
        self.temp_files: Dict[str, tempfile.NamedTemporaryFile] = {}
        self.temp_dirs: Dict[str, tempfile.TemporaryDirectory] = {}
        
    def create_temp_file(self, suffix: str = None, prefix: str = "langchain_") -> Tuple[str, str]:
        """
        Create a secure temporary file.
        
        Args:
            suffix: File suffix (e.g., '.pdf')
            prefix: File prefix
            
        Returns:
            Tuple of (file_id, file_path)
        """
        file_id = uuid.uuid4().hex
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            prefix=prefix,
            mode='wb'
        )
        
        self.temp_files[file_id] = temp_file
        logging.info(f"Created temporary file: {file_id} -> {temp_file.name}")
        return file_id, temp_file.name
    
    def write_temp_file(self, file_id: str, content: bytes) -> bool:
        """
        Write content to a temporary file.
        
        Args:
            file_id: Temporary file ID
            content: File content bytes
            
        Returns:
            Success status
        """
        try:
            if file_id in self.temp_files:
                temp_file = self.temp_files[file_id]
                temp_file.write(content)
                temp_file.flush()
                temp_file.close()
                return True
            return False
        except Exception as e:
            logging.error(f"Error writing temporary file {file_id}: {e}")
            return False
    
    def get_temp_file_path(self, file_id: str) -> Optional[str]:
        """Get the path of a temporary file."""
        if file_id in self.temp_files:
            return self.temp_files[file_id].name
        return None
    
    def cleanup_temp_file(self, file_id: str) -> bool:
        """
        Clean up a specific temporary file.
        
        Args:
            file_id: Temporary file ID
            
        Returns:
            Success status
        """
        try:
            if file_id in self.temp_files:
                temp_file = self.temp_files[file_id]
                if hasattr(temp_file, 'name') and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                del self.temp_files[file_id]
                logging.info(f"Cleaned up temporary file: {file_id}")
                return True
            return False
        except Exception as e:
            logging.error(f"Error cleaning up temporary file {file_id}: {e}")
            return False
    
    def cleanup_all(self):
        """Clean up all temporary files."""
        for file_id in list(self.temp_files.keys()):
            self.cleanup_temp_file(file_id)
        
        for dir_id in list(self.temp_dirs.keys()):
            try:
                self.temp_dirs[dir_id].cleanup()
                del self.temp_dirs[dir_id]
            except Exception as e:
                logging.error(f"Error cleaning up temporary directory {dir_id}: {e}")


class PDFProcessor:
    """Advanced PDF processing using PyMuPDFLoader with metadata extraction and structure preservation."""
    
    def __init__(self, temp_file_manager: TemporaryFileManager):
        """Initialize PDF processor with temporary file manager."""
        self.temp_file_manager = temp_file_manager
        self.error_handler = ErrorHandler()
    
    async def process_pdf(
        self,
        file_content: bytes,
        filename: str,
        session_id: Optional[str] = None
    ) -> ProcessedDocument:
        """
        Process PDF file with advanced content extraction and metadata.
        
        Args:
            file_content: PDF file content bytes
            filename: Original filename
            session_id: Current session ID for error context
            
        Returns:
            ProcessedDocument with extracted content and metadata
        """
        temp_file_id = None
        try:
            # Create temporary file
            temp_file_id, temp_file_path = self.temp_file_manager.create_temp_file(
                suffix='.pdf',
                prefix=f'pdf_process_{filename}_'
            )
            
            # Write PDF content to temporary file
            if not self.temp_file_manager.write_temp_file(temp_file_id, file_content):
                raise ContentValidationError("Failed to write PDF to temporary file")
            
            # Process PDF using PyMuPDFLoader
            loader = PyMuPDFLoader(temp_file_path)
            documents = loader.load()
            
            # Extract metadata using PyMuPDF directly
            pdf_metadata = self._extract_pdf_metadata(temp_file_path)
            
            # Process pages and content
            pages = self._process_pdf_pages(temp_file_path)
            
            # Combine all content
            full_content = "\n\n".join([doc.page_content for doc in documents])
            
            # Create document structure
            structure = {
                "total_pages": len(pages),
                "has_images": any(page["metadata"].get("images", []) for page in pages),
                "has_links": any(page["metadata"].get("links", []) for page in pages),
                "content_length": len(full_content),
                "loader_used": "PyMuPDFLoader"
            }
            
            # Enhanced metadata combining LangChain and PyMuPDF data
            enhanced_metadata = {
                **pdf_metadata,
                "processing_timestamp": datetime.datetime.now().isoformat(),
                "session_id": session_id,
                "original_filename": filename,
                "file_size": len(file_content),
                "langchain_metadata": documents[0].metadata if documents else {}
            }
            
            return ProcessedDocument(
                content=full_content,
                metadata=enhanced_metadata,
                pages=pages,
                structure=structure,
                processing_method="PyMuPDFLoader",
                original_filename=filename
            )
            
        except Exception as e:
            logging.error(f"PDF processing error for {filename}: {e}")
            
            # Use error handler for structured error processing
            error_result = await self.error_handler.handle_error(
                exception=e,
                session_id=session_id,
                additional_context={"filename": filename, "processing_type": "pdf"}
            )
            
            # For PDF processing errors, we can try fallback to base64
            raise ContentValidationError(f"PDF processing failed: {error_result.user_message}")
            
        finally:
            # Always clean up temporary file
            if temp_file_id:
                self.temp_file_manager.cleanup_temp_file(temp_file_id)
    
    def _extract_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract comprehensive PDF metadata using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "page_count": doc.page_count,
                "pdf_version": self._get_pdf_version(doc),
                "is_encrypted": doc.is_encrypted,
                "needs_pass": doc.needs_pass,
                "is_pdf": doc.is_pdf,
                "is_closed": doc.is_closed,
                "page_dimensions": [
                    {"page": i + 1, "width": page.rect.width, "height": page.rect.height}
                    for i, page in enumerate(doc)
                ]
            }
        except Exception as e:
            logging.error(f"Error extracting PDF metadata: {e}")
            return {
                "title": "",
                "author": "",
                "page_count": 0,
                "pdf_version": "unknown",
                "error": str(e)
            }
        finally:
            if 'doc' in locals():
                doc.close()
    
    def _get_pdf_version(self, doc) -> str:
        """Get PDF version in a way that works across PyMuPDF versions."""
        try:
            # Try the new method first (PyMuPDF 1.23+)
            if hasattr(doc, 'pdf_version'):
                version_tuple = doc.pdf_version()
                return f"{version_tuple[0]}.{version_tuple[1]}"
            # Try alternative methods
            elif hasattr(doc, 'metadata') and doc.metadata.get('format'):
                return doc.metadata.get('format', 'unknown')
            else:
                return "1.4"  # Default fallback
        except Exception as e:
            logging.warning(f"Could not determine PDF version: {e}")
            return "unknown"
    
    def _process_pdf_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process individual PDF pages for detailed content and metadata."""
        pages = []
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text content
                page_content = page.get_text()
                
                # Extract page metadata
                page_metadata = {
                    "page_width": page.rect.width,
                    "page_height": page.rect.height,
                    "rotation": page.rotation,
                    "images": [img[0] for img in page.get_images()],
                    "links": [link.get("uri", "") for link in page.get_links() if link.get("uri")],
                    "text_blocks": len(page.get_text_blocks()),
                    "char_count": len(page_content)
                }
                
                pages.append({
                    "page_number": page_num + 1,
                    "content": page_content,
                    "metadata": page_metadata
                })
                
        except Exception as e:
            logging.error(f"Error processing PDF pages: {e}")
            # Return empty pages list on error
            return []
        finally:
            if 'doc' in locals():
                doc.close()
                
        return pages


class TextProcessor:
    """Enhanced text processing using LangChain TextLoader with metadata extraction."""
    
    SUPPORTED_MIME_TYPES = {
        'text/plain': '.txt',
        'text/markdown': '.md',
        'text/csv': '.csv',
        'application/json': '.json',
        'text/xml': '.xml',
        'application/xml': '.xml'
    }
    
    def __init__(self, temp_file_manager: TemporaryFileManager):
        """Initialize text processor with temporary file manager."""
        self.temp_file_manager = temp_file_manager
        self.error_handler = ErrorHandler()
    
    async def process_text(
        self,
        file_content: bytes,
        filename: str,
        session_id: Optional[str] = None
    ) -> ProcessedDocument:
        """Process text file with enhanced content extraction and metadata."""
        temp_file_id = None
        try:
            # Detect file extension and determine appropriate suffix
            file_ext = os.path.splitext(filename)[1].lower()
            if not file_ext:
                file_ext = '.txt'  # Default fallback
                
            # Create temporary file
            temp_file_id, temp_file_path = self.temp_file_manager.create_temp_file(
                suffix=file_ext,
                prefix=f'text_process_{filename}_'
            )
            
            # Write text content to temporary file
            if not self.temp_file_manager.write_temp_file(temp_file_id, file_content):
                raise ContentValidationError("Failed to write text to temporary file")
            
            # Process text using TextLoader with encoding detection
            loader = TextLoader(temp_file_path, autodetect_encoding=True)
            documents = loader.load()
            
            # Extract text metadata
            text_metadata = self._extract_text_metadata(file_content, filename)
            
            # Process content structure
            full_content = documents[0].page_content if documents else ""
            structure = self._analyze_text_structure(full_content, file_ext)
            
            # Create single page representation for consistency with PDF processor
            pages = [{
                "page_number": 1,
                "content": full_content,
                "metadata": {
                    "line_count": len(full_content.splitlines()),
                    "character_count": len(full_content),
                    "word_count": len(full_content.split()),
                    "encoding": text_metadata.get("encoding", "unknown")
                }
            }]
            
            # Enhanced metadata combining detected info
            enhanced_metadata = {
                **text_metadata,
                "processing_timestamp": datetime.datetime.now().isoformat(),
                "session_id": session_id,
                "original_filename": filename,
                "file_size": len(file_content),
                "langchain_metadata": documents[0].metadata if documents else {}
            }
            
            return ProcessedDocument(
                content=full_content,
                metadata=enhanced_metadata,
                pages=pages,
                structure=structure,
                processing_method="TextLoader",
                original_filename=filename
            )
            
        except Exception as e:
            logging.error(f"Text processing error for {filename}: {e}")
            
            # Use error handler for structured error processing
            error_result = await self.error_handler.handle_error(
                exception=e,
                session_id=session_id,
                additional_context={"filename": filename, "processing_type": "text"}
            )
            
            raise ContentValidationError(f"Text processing failed: {error_result.user_message}")
            
        finally:
            # Always clean up temporary file
            if temp_file_id:
                self.temp_file_manager.cleanup_temp_file(temp_file_id)
    
    def _extract_text_metadata(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract comprehensive text metadata."""
        try:
            # Detect encoding
            detected_encoding = self._detect_encoding(file_content)
            
            # Decode content for analysis
            try:
                content = file_content.decode(detected_encoding)
            except UnicodeDecodeError:
                content = file_content.decode('utf-8', errors='replace')
                detected_encoding = 'utf-8-fallback'
            
            # Basic text analysis
            lines = content.splitlines()
            words = content.split()
            
            # File extension analysis
            file_ext = os.path.splitext(filename)[1].lower()
            
            metadata = {
                "encoding": detected_encoding,
                "line_count": len(lines),
                "character_count": len(content),
                "word_count": len(words),
                "file_extension": file_ext,
                "has_empty_lines": any(not line.strip() for line in lines),
                "max_line_length": max(len(line) for line in lines) if lines else 0,
                "average_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0
            }
            
            # Format-specific analysis
            if file_ext in ['.json']:
                metadata.update(self._analyze_json_structure(content))
            elif file_ext in ['.csv']:
                metadata.update(self._analyze_csv_structure(content))
            elif file_ext in ['.md']:
                metadata.update(self._analyze_markdown_structure(content))
            elif file_ext in ['.xml']:
                metadata.update(self._analyze_xml_structure(content))
            
            return metadata
            
        except Exception as e:
            logging.warning(f"Text metadata extraction failed: {e}")
            return {
                "encoding": "unknown",
                "line_count": 0,
                "character_count": len(file_content),
                "word_count": 0,
                "file_extension": os.path.splitext(filename)[1].lower(),
                "extraction_error": str(e)
            }
    
    def _detect_encoding(self, file_content: bytes) -> str:
        """Detect file encoding with fallback options."""
        # Try UTF-8 first (most common)
        try:
            file_content.decode('utf-8')
            return 'utf-8'
        except UnicodeDecodeError:
            pass
            
        # Try common encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
        for encoding in encodings:
            try:
                file_content.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
                
        return 'utf-8'  # Fallback to UTF-8 with error handling
    
    def _analyze_text_structure(self, content: str, file_ext: str) -> Dict[str, Any]:
        """Analyze text structure for enhanced context."""
        lines = content.splitlines()
        
        structure = {
            "total_lines": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "content_length": len(content),
            "loader_used": "TextLoader",
            "file_type": file_ext
        }
        
        # Add format-specific structure info
        if file_ext in ['.json']:
            structure["is_valid_json"] = self._is_valid_json(content)
        elif file_ext in ['.csv']:
            structure["estimated_columns"] = self._estimate_csv_columns(content)
        elif file_ext in ['.md']:
            structure["heading_count"] = content.count('#')
        elif file_ext in ['.xml']:
            structure["is_valid_xml"] = self._is_valid_xml(content)
            
        return structure
    
    def _analyze_json_structure(self, content: str) -> Dict[str, Any]:
        """Analyze JSON file structure."""
        try:
            data = json.loads(content)
            return {
                "json_type": type(data).__name__,
                "is_valid_json": True,
                "json_keys": list(data.keys()) if isinstance(data, dict) else [],
                "json_length": len(data) if isinstance(data, (list, dict)) else 1
            }
        except json.JSONDecodeError:
            return {"is_valid_json": False, "json_error": "Invalid JSON format"}
    
    def _analyze_csv_structure(self, content: str) -> Dict[str, Any]:
        """Analyze CSV file structure."""
        lines = content.splitlines()
        if not lines:
            return {"csv_rows": 0, "csv_columns": 0}
            
        # Estimate columns from first line
        first_line = lines[0]
        comma_count = first_line.count(',')
        semicolon_count = first_line.count(';')
        tab_count = first_line.count('\t')
        
        delimiter = ',' if comma_count >= semicolon_count and comma_count >= tab_count else (';' if semicolon_count >= tab_count else '\t')
        estimated_columns = first_line.count(delimiter) + 1
        
        return {
            "csv_rows": len(lines),
            "csv_columns": estimated_columns,
            "delimiter": delimiter,
            "has_header": True  # Assume first row is header
        }
    
    def _analyze_markdown_structure(self, content: str) -> Dict[str, Any]:
        """Analyze Markdown file structure."""
        lines = content.splitlines()
        
        h1_count = len([line for line in lines if line.startswith('# ')])
        h2_count = len([line for line in lines if line.startswith('## ')])
        h3_count = len([line for line in lines if line.startswith('### ')])
        
        return {
            "markdown_headings": {
                "h1": h1_count,
                "h2": h2_count,
                "h3": h3_count
            },
            "code_blocks": content.count('```'),
            "links": content.count('[') and content.count(']('),
            "images": content.count('![')
        }
    
    def _analyze_xml_structure(self, content: str) -> Dict[str, Any]:
        """Analyze XML file structure."""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(content)
            return {
                "is_valid_xml": True,
                "root_tag": root.tag,
                "element_count": len(list(root.iter())),
                "namespace_count": len(set(elem.tag.split('}')[0] + '}' for elem in root.iter() if '}' in elem.tag))
            }
        except ET.ParseError:
            return {"is_valid_xml": False, "xml_error": "Invalid XML format"}
    
    def _is_valid_json(self, content: str) -> bool:
        """Check if content is valid JSON."""
        try:
            json.loads(content)
            return True
        except json.JSONDecodeError:
            return False
    
    def _estimate_csv_columns(self, content: str) -> int:
        """Estimate number of CSV columns."""
        lines = content.splitlines()
        if not lines:
            return 0
        return lines[0].count(',') + 1
    
    def _is_valid_xml(self, content: str) -> bool:
        """Check if content is valid XML."""
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(content)
            return True
        except ET.ParseError:
            return False


class UnstructuredProcessor:
    """Complex document processing using LangChain UnstructuredFileLoader."""
    
    SUPPORTED_MIME_TYPES = {
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
        'text/html': '.html',
        'application/msword': '.doc'
    }
    
    def __init__(self, temp_file_manager: TemporaryFileManager):
        """Initialize unstructured processor with temporary file manager."""
        self.temp_file_manager = temp_file_manager
        self.error_handler = ErrorHandler()
    
    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        session_id: Optional[str] = None
    ) -> ProcessedDocument:
        """Process complex document with structure extraction and metadata."""
        temp_file_id = None
        try:
            # Detect file extension
            file_ext = os.path.splitext(filename)[1].lower()
            if not file_ext or file_ext not in ['.docx', '.pptx', '.html', '.doc']:
                file_ext = '.docx'  # Default fallback
                
            # Create temporary file
            temp_file_id, temp_file_path = self.temp_file_manager.create_temp_file(
                suffix=file_ext,
                prefix=f'unstructured_process_{filename}_'
            )
            
            # Write document content to temporary file
            if not self.temp_file_manager.write_temp_file(temp_file_id, file_content):
                raise ContentValidationError("Failed to write document to temporary file")
            
            # Process document using UnstructuredFileLoader
            loader = UnstructuredFileLoader(temp_file_path)
            documents = loader.load()
            
            # Extract document metadata
            doc_metadata = self._extract_document_metadata(temp_file_path, filename, file_ext)
            
            # Process document structure
            pages = self._process_document_pages(documents)
            full_content = "\n\n".join([doc.page_content for doc in documents])
            
            # Analyze document structure
            structure = {
                "total_elements": len(documents),
                "content_length": len(full_content),
                "file_type": file_ext,
                "loader_used": "UnstructuredFileLoader",
                "has_structured_content": len(documents) > 1
            }
            
            # Enhanced metadata combining detected info
            enhanced_metadata = {
                **doc_metadata,
                "processing_timestamp": datetime.datetime.now().isoformat(),
                "session_id": session_id,
                "original_filename": filename,
                "file_size": len(file_content),
                "langchain_metadata": documents[0].metadata if documents else {}
            }
            
            return ProcessedDocument(
                content=full_content,
                metadata=enhanced_metadata,
                pages=pages,
                structure=structure,
                processing_method="UnstructuredFileLoader",
                original_filename=filename
            )
            
        except Exception as e:
            logging.error(f"Unstructured document processing error for {filename}: {e}")
            
            # Use error handler for structured error processing
            error_result = await self.error_handler.handle_error(
                exception=e,
                session_id=session_id,
                additional_context={"filename": filename, "processing_type": "unstructured"}
            )
            
            raise ContentValidationError(f"Document processing failed: {error_result.user_message}")
            
        finally:
            # Always clean up temporary file
            if temp_file_id:
                self.temp_file_manager.cleanup_temp_file(temp_file_id)
    
    def _extract_document_metadata(self, file_path: str, filename: str, file_ext: str) -> Dict[str, Any]:
        """Extract comprehensive document metadata."""
        metadata = {
            "file_extension": file_ext,
            "document_type": self._determine_document_type(file_ext),
            "extraction_method": "unstructured"
        }
        
        try:
            # Get file stats
            file_stats = os.stat(file_path)
            metadata.update({
                "file_size_bytes": file_stats.st_size,
                "file_modified": datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            })
            
            # Format-specific metadata extraction
            if file_ext in ['.docx']:
                metadata.update(self._extract_docx_metadata(file_path))
            elif file_ext in ['.pptx']:
                metadata.update(self._extract_pptx_metadata(file_path))
            elif file_ext in ['.html']:
                metadata.update(self._extract_html_metadata(file_path))
                
        except Exception as e:
            logging.warning(f"Document metadata extraction failed: {e}")
            metadata["extraction_error"] = str(e)
            
        return metadata
    
    def _process_document_pages(self, documents: List) -> List[Dict[str, Any]]:
        """Process document elements into page-like structure."""
        pages = []
        
        for i, doc in enumerate(documents):
            page_data = {
                "page_number": i + 1,
                "content": doc.page_content,
                "metadata": {
                    "element_type": doc.metadata.get("category", "unknown"),
                    "character_count": len(doc.page_content),
                    "word_count": len(doc.page_content.split()),
                    "langchain_metadata": doc.metadata
                }
            }
            pages.append(page_data)
            
        return pages
    
    def _determine_document_type(self, file_ext: str) -> str:
        """Determine document type from extension."""
        type_map = {
            '.docx': 'Word Document',
            '.doc': 'Legacy Word Document',
            '.pptx': 'PowerPoint Presentation',
            '.html': 'HTML Document'
        }
        return type_map.get(file_ext, 'Unknown Document')
    
    def _extract_docx_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract DOCX-specific metadata."""
        try:
            # Basic DOCX analysis without heavy dependencies
            return {
                "document_format": "Office Open XML",
                "supports_styles": True,
                "supports_images": True,
                "supports_tables": True
            }
        except Exception as e:
            return {"docx_error": str(e)}
    
    def _extract_pptx_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract PPTX-specific metadata."""
        try:
            return {
                "presentation_format": "Office Open XML",
                "supports_slides": True,
                "supports_animations": True,
                "supports_media": True
            }
        except Exception as e:
            return {"pptx_error": str(e)}
    
    def _extract_html_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract HTML-specific metadata."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            return {
                "html_format": "HTML",
                "has_head_section": "<head>" in content.lower(),
                "has_body_section": "<body>" in content.lower(),
                "estimated_tags": content.count('<'),
                "has_scripts": "<script>" in content.lower(),
                "has_styles": "<style>" in content.lower() or "css" in content.lower()
            }
        except Exception as e:
            return {"html_error": str(e)}


class DocumentProcessorFactory:
    """Factory for selecting appropriate document loaders based on file type."""
    
    def __init__(self, app_config=None):
        """Initialize the document processor factory."""
        self.temp_file_manager = TemporaryFileManager()
        self.pdf_processor = PDFProcessor(self.temp_file_manager)
        self.text_processor = TextProcessor(self.temp_file_manager)
        self.unstructured_processor = UnstructuredProcessor(self.temp_file_manager)
        self.error_handler = ErrorHandler()
        
        # Initialize chunk processor if app config provided
        self.chunk_processor = None
        if app_config:
            try:
                self.chunk_processor = ChunkProcessor(app_config)
                logging.info("ChunkProcessor initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize ChunkProcessor: {e}")
                self.chunk_processor = None
    
    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        mime_type: str,
        session_id: Optional[str] = None
    ) -> Union[ProcessedDocument, ChunkedDocument, None]:
        """
        Process document using appropriate loader based on MIME type.
        
        Args:
            file_content: File content bytes
            filename: Original filename
            mime_type: MIME type of the file
            session_id: Current session ID
            
        Returns:
            ProcessedDocument or ChunkedDocument if successful, None if fallback to base64 should be used
        """
        if not LANGCHAIN_LOADERS_AVAILABLE:
            logging.warning("LangChain loaders not available, falling back to base64")
            return None
        
        try:
            # Route to appropriate processor based on MIME type
            processed_document = None
            if mime_type == 'application/pdf':
                processed_document = await self.pdf_processor.process_pdf(file_content, filename, session_id)
            elif mime_type in self.text_processor.SUPPORTED_MIME_TYPES:
                processed_document = await self.text_processor.process_text(file_content, filename, session_id)
            elif mime_type in self.unstructured_processor.SUPPORTED_MIME_TYPES:
                processed_document = await self.unstructured_processor.process_document(file_content, filename, session_id)
            else:
                # For unsupported files, return None to use base64 fallback
                logging.info(f"No LangChain processor for {mime_type}, using base64 fallback")
                return None
            
            # Apply chunking if enabled and chunk processor is available
            if processed_document and self.chunk_processor:
                try:
                    # Add MIME type to metadata for chunking strategy selection
                    processed_document.metadata["mime_type"] = mime_type
                    chunked_result = await self.chunk_processor.process_document(processed_document, session_id)
                    return chunked_result
                except Exception as chunk_error:
                    logging.warning(f"Chunking failed for {filename}: {chunk_error}, returning original document")
                    return processed_document
            
            return processed_document
                
        except Exception as e:
            logging.error(f"Document processing failed for {filename}: {e}")
            
            # Process error with context
            error_result = await self.error_handler.handle_error(
                exception=e,
                session_id=session_id,
                additional_context={
                    "filename": filename,
                    "mime_type": mime_type,
                    "processing_type": "document_loader"
                }
            )
            
            # Return None to fallback to base64 processing
            logging.info(f"Falling back to base64 processing for {filename} due to error: {error_result.user_message}")
            return None
    
    def cleanup(self):
        """Clean up all temporary files."""
        self.temp_file_manager.cleanup_all()


# ===== CHUNKING IMPLEMENTATIONS =====

class RecursiveCharacterChunkingStrategy(BaseChunkingStrategy):
    """Chunking strategy using LangChain's RecursiveCharacterTextSplitter."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize with configuration."""
        super().__init__(config)
        
        if not CHUNKING_DEPENDENCIES_AVAILABLE:
            raise ImportError("Chunking dependencies not available")
        
        # Initialize the LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=config.separators or ["\n\n", "\n", " ", ""],
            keep_separator=config.keep_separator
        )
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk text using RecursiveCharacterTextSplitter.
        
        Args:
            text: Text to chunk
            metadata: Document metadata for context
            
        Returns:
            List of chunks with metadata
        """
        if not text.strip():
            return []
        
        # Use LangChain splitter to get text chunks
        text_chunks = self.text_splitter.split_text(text)
        
        if not text_chunks:
            return []
        
        chunks = []
        current_position = 0
        
        for i, chunk_text in enumerate(text_chunks):
            # Find the actual position of this chunk in the original text
            start_index = text.find(chunk_text, current_position)
            if start_index == -1:
                start_index = current_position
            
            end_index = start_index + len(chunk_text)
            
            # Create chunk metadata
            chunk_metadata = self._create_chunk_metadata(
                content=text,
                start_index=start_index,
                end_index=end_index,
                chunk_number=i + 1,
                total_chunks=len(text_chunks),
                document_metadata=metadata
            )
            
            # Update processing metadata
            chunk_metadata.processing_metadata.update({
                "strategy_used": self.get_strategy_name(),
                "processing_timestamp": datetime.datetime.now().isoformat(),
                "langchain_chunk_size": len(chunk_text),
                "boundary_quality": self._assess_boundary_quality(chunk_text, text, start_index, end_index)
            })
            
            chunks.append(Chunk(content=chunk_text, metadata=chunk_metadata))
            current_position = max(current_position, start_index + 1)
        
        # Link adjacent chunks
        self._link_adjacent_chunks(chunks)
        
        return chunks
    
    def get_strategy_name(self) -> str:
        """Get the name of this chunking strategy."""
        return "recursive_character"
    
    def _assess_boundary_quality(self, chunk_text: str, full_text: str, start_index: int, end_index: int) -> BoundaryQuality:
        """Assess the quality of chunk boundaries."""
        # Simple heuristics for boundary quality
        
        # Check if chunk ends with sentence boundary
        if chunk_text.rstrip().endswith(('.', '!', '?')):
            return BoundaryQuality.EXCELLENT
        
        # Check if chunk ends with paragraph boundary
        if chunk_text.endswith('\n\n'):
            return BoundaryQuality.GOOD
        
        # Check if chunk ends at word boundary
        if end_index < len(full_text) and full_text[end_index] == ' ':
            return BoundaryQuality.FAIR
        
        return BoundaryQuality.POOR


class PDFChunkingStrategy(BaseChunkingStrategy):
    """PDF-aware chunking strategy that respects page boundaries."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize with configuration."""
        super().__init__(config)
        self.recursive_strategy = RecursiveCharacterChunkingStrategy(config)
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk text with PDF page awareness.
        
        Args:
            text: Text to chunk
            metadata: Document metadata with page information
            
        Returns:
            List of chunks with PDF-aware boundaries
        """
        # For now, use recursive character strategy
        # TODO: Implement page-aware chunking when page boundaries are available
        chunks = self.recursive_strategy.chunk_text(text, metadata)
        
        # Update strategy name in metadata
        for chunk in chunks:
            chunk.metadata.processing_metadata["strategy_used"] = self.get_strategy_name()
        
        return chunks
    
    def get_strategy_name(self) -> str:
        """Get the name of this chunking strategy."""
        return "pdf_aware"


class TextChunkingStrategy(BaseChunkingStrategy):
    """Text-specific chunking strategy that respects paragraph boundaries."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize with configuration."""
        super().__init__(config)
        
        # Use paragraph-aware separators
        paragraph_config = ChunkingConfig(
            strategy=config.strategy,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_size=config.min_chunk_size,
            max_chunk_size=config.max_chunk_size,
            separators=["\n\n", "\n", ". ", " ", ""],  # Paragraph and sentence aware
            keep_separator=True
        )
        
        self.recursive_strategy = RecursiveCharacterChunkingStrategy(paragraph_config)
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk text with paragraph awareness.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunks with paragraph-aware boundaries
        """
        chunks = self.recursive_strategy.chunk_text(text, metadata)
        
        # Update strategy name in metadata
        for chunk in chunks:
            chunk.metadata.processing_metadata["strategy_used"] = self.get_strategy_name()
        
        return chunks
    
    def get_strategy_name(self) -> str:
        """Get the name of this chunking strategy."""
        return "text_paragraph"


class ChunkingStrategyFactory:
    """Factory for selecting appropriate chunking strategies."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize with configuration."""
        self.config = config
    
    def get_strategy(self, mime_type: str, document_metadata: Dict[str, Any]) -> BaseChunkingStrategy:
        """
        Get appropriate chunking strategy based on document type.
        
        Args:
            mime_type: Document MIME type
            document_metadata: Document metadata
            
        Returns:
            Appropriate chunking strategy
        """
        # Strategy selection based on MIME type
        if mime_type == "application/pdf":
            return PDFChunkingStrategy(self.config)
        elif mime_type.startswith("text/"):
            return TextChunkingStrategy(self.config)
        else:
            # Default to recursive character strategy
            return RecursiveCharacterChunkingStrategy(self.config)


class ChunkOptimizer:
    """Optimizes chunk sizes based on content complexity."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize with configuration."""
        self.config = config
        
        # Initialize tiktoken encoder for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 encoding
        except Exception as e:
            logging.warning(f"Failed to initialize tiktoken encoder: {e}")
            self.tokenizer = None
    
    def optimize_chunk_size(self, text: str, metadata: Dict[str, Any]) -> int:
        """
        Optimize chunk size based on content complexity.
        
        Args:
            text: Text to analyze
            metadata: Content metadata
            
        Returns:
            Optimized chunk size
        """
        base_size = self.config.chunk_size
        
        # Get complexity score from metadata or calculate it
        complexity = metadata.get("complexity_score", self._calculate_complexity(text))
        
        # Adjust chunk size based on complexity
        if complexity > 0.8:
            # High complexity - use smaller chunks
            adjusted_size = int(base_size * 0.7)
        elif complexity > 0.6:
            # Medium-high complexity - slightly smaller chunks
            adjusted_size = int(base_size * 0.85)
        elif complexity < 0.3:
            # Low complexity - can use larger chunks
            adjusted_size = int(base_size * 1.2)
        else:
            # Normal complexity - use base size
            adjusted_size = base_size
        
        # Ensure within configured bounds
        adjusted_size = max(self.config.min_chunk_size, 
                          min(adjusted_size, self.config.max_chunk_size))
        
        return adjusted_size
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logging.warning(f"Token counting failed: {e}")
        
        # Fallback to simple word count approximation
        return len(text.split()) * 1.3  # Rough token approximation
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Simple complexity metrics
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentences = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = len(words) / max(sentences, 1)
        
        # Normalize to 0-1 scale
        word_complexity = min(1.0, avg_word_length / 8.0)
        sentence_complexity = min(1.0, avg_sentence_length / 25.0)
        
        return (word_complexity + sentence_complexity) / 2.0


class OverlapManager:
    """Manages overlap between chunks for context preservation."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize with configuration."""
        self.config = config
        self.overlap_size = config.chunk_overlap
    
    def calculate_overlap_boundaries(self, text: str, chunk_boundaries: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Calculate optimized overlap boundaries.
        
        Args:
            text: Original text
            chunk_boundaries: List of (start, end) tuples
            
        Returns:
            Optimized chunk boundaries with overlap
        """
        if not chunk_boundaries:
            return []
        
        optimized_boundaries = []
        
        for i, (start, end) in enumerate(chunk_boundaries):
            new_start = start
            new_end = end
            
            # Adjust overlap with previous chunk
            if i > 0 and self.overlap_size > 0:
                # Try to find good overlap point (sentence or word boundary)
                desired_start = max(0, start - self.overlap_size)
                new_start = self._find_good_boundary(text, desired_start, start, reverse=True)
            
            # Adjust overlap with next chunk
            if i < len(chunk_boundaries) - 1 and self.overlap_size > 0:
                next_start = chunk_boundaries[i + 1][0]
                desired_end = min(len(text), end + self.overlap_size)
                # Ensure we don't overlap too much with next chunk
                desired_end = min(desired_end, next_start + self.overlap_size // 2)
                new_end = self._find_good_boundary(text, end, desired_end)
            
            optimized_boundaries.append((new_start, new_end))
        
        return optimized_boundaries
    
    def validate_overlap_quality(self, text: str, chunk1: str, chunk2: str) -> float:
        """
        Validate the quality of overlap between two chunks.
        
        Args:
            text: Original text
            chunk1: First chunk content
            chunk2: Second chunk content
            
        Returns:
            Quality score between 0 and 1
        """
        if not chunk1 or not chunk2:
            return 0.0
        
        # Find overlap between chunks
        overlap = self._find_overlap(chunk1, chunk2)
        
        if not overlap:
            return 0.0
        
        # Calculate quality metrics
        overlap_length = len(overlap)
        total_boundary_length = (len(chunk1) + len(chunk2)) / 2
        
        # Length score
        length_score = min(1.0, overlap_length / (self.overlap_size + 1))
        
        # Semantic score (simple heuristic)
        semantic_score = self._calculate_semantic_score(overlap)
        
        return (length_score + semantic_score) / 2.0
    
    def _find_good_boundary(self, text: str, start: int, end: int, reverse: bool = False) -> int:
        """Find a good boundary point (sentence, paragraph, or word)."""
        if start >= end or start < 0 or end > len(text):
            return start if not reverse else end
        
        search_text = text[start:end]
        
        if reverse:
            # Search backwards for good boundary
            for boundary in ['\n\n', '\n', '. ', ' ']:
                idx = search_text.rfind(boundary)
                if idx != -1:
                    return start + idx + len(boundary)
        else:
            # Search forwards for good boundary
            for boundary in ['. ', '\n\n', '\n', ' ']:
                idx = search_text.find(boundary)
                if idx != -1:
                    return start + idx
        
        return start if not reverse else end
    
    def _find_overlap(self, chunk1: str, chunk2: str) -> str:
        """Find overlapping content between two chunks."""
        # Simple suffix-prefix matching
        max_overlap = min(len(chunk1), len(chunk2), self.overlap_size * 2)
        
        for i in range(max_overlap, 0, -1):
            if chunk1[-i:] == chunk2[:i]:
                return chunk1[-i:]
        
        return ""
    
    def _calculate_semantic_score(self, overlap_text: str) -> float:
        """Calculate semantic quality of overlap text."""
        if not overlap_text:
            return 0.0
        
        # Simple heuristics for semantic completeness
        score = 0.0
        
        # Complete sentences get higher score
        if overlap_text.strip().endswith(('.', '!', '?')):
            score += 0.4
        
        # Complete words get higher score
        if overlap_text.strip().endswith(' ') or len(overlap_text.split()) > 1:
            score += 0.3
        
        # Longer overlaps that make sense get higher score
        if len(overlap_text.split()) >= 3:
            score += 0.3
        
        return min(1.0, score)


class ChunkProcessor:
    """Main processor for document chunking operations."""
    
    def __init__(self, config: Any):
        """Initialize with application configuration."""
        self.config = config
        
        # Create chunking configuration
        self.chunking_config = config.get_chunking_config()
        
        # Initialize components
        self.strategy_factory = ChunkingStrategyFactory(self.chunking_config)
        self.chunk_optimizer = ChunkOptimizer(self.chunking_config)
        self.overlap_manager = OverlapManager(self.chunking_config)
        self.error_handler = ErrorHandler()
    
    async def process_document(
        self, 
        document: ProcessedDocument, 
        session_id: Optional[str] = None
    ) -> Union[ProcessedDocument, ChunkedDocument]:
        """
        Process a document with intelligent chunking.
        
        Args:
            document: ProcessedDocument to chunk
            session_id: Current session ID
            
        Returns:
            ChunkedDocument if chunking enabled and successful, otherwise ProcessedDocument
        """
        # Check if chunking is enabled
        if not self.config.enable_chunking:
            logging.info("Chunking disabled, returning original document")
            return document
        
        # Check if chunking dependencies are available
        if not CHUNKING_DEPENDENCIES_AVAILABLE:
            logging.warning("Chunking dependencies not available, returning original document")
            return document
        
        try:
            # Extract document content and metadata
            content = document.content
            if not content.strip():
                logging.info("Empty document content, returning original document")
                return document
            
            # Determine appropriate chunking strategy
            mime_type = document.metadata.get("mime_type", "text/plain")
            strategy = self.strategy_factory.get_strategy(mime_type, document.metadata)
            
            # Optimize chunk size based on content
            if self.chunking_config.enable_semantic_boundaries:
                optimized_size = self.chunk_optimizer.optimize_chunk_size(content, document.metadata)
                # Update strategy configuration
                strategy.config.chunk_size = optimized_size
                # Recreate text splitter with optimized size
                if hasattr(strategy, 'text_splitter'):
                    strategy.text_splitter.chunk_size = optimized_size
            
            # Chunk the document
            chunks = strategy.chunk_text(content, {
                "document_id": document.original_filename,
                "mime_type": mime_type,
                **document.metadata
            })
            
            if not chunks:
                logging.info("No chunks created, returning original document")
                return document
            
            # Create chunk index
            chunk_index = ChunkIndex()
            for chunk in chunks:
                chunk_index.add_chunk(chunk, document.original_filename)
            
            # Create chunking metadata
            chunking_metadata = {
                "strategy_used": strategy.get_strategy_name(),
                "total_chunks": len(chunks),
                "average_chunk_size": sum(chunk.size for chunk in chunks) / len(chunks),
                "total_tokens": sum(self.chunk_optimizer.count_tokens(chunk.content) for chunk in chunks),
                "processing_timestamp": datetime.datetime.now().isoformat(),
                "chunking_config": {
                    "chunk_size": strategy.config.chunk_size,
                    "chunk_overlap": strategy.config.chunk_overlap,
                    "strategy": strategy.config.strategy.value
                }
            }
            
            # Create chunked document
            chunked_document = ChunkedDocument(
                original_document=document,
                chunks=chunks,
                chunk_index=chunk_index,
                chunking_metadata=chunking_metadata
            )
            
            logging.info(f"Successfully chunked document into {len(chunks)} chunks")
            return chunked_document
            
        except Exception as e:
            logging.error(f"Chunking failed for {document.original_filename}: {e}")
            
            # Use error handler for structured error processing
            try:
                await self.error_handler.handle_error(
                    exception=e,
                    session_id=session_id,
                    additional_context={
                        "filename": document.original_filename,
                        "processing_type": "chunking"
                    }
                )
            except Exception as eh:
                logging.error(f"Error handler failed: {eh}")
            
            # Fallback to original document
            logging.info("Falling back to original document without chunking")
            return document