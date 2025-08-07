"""
Integration tests for PDF processing with the /stream endpoint.

This test module validates the end-to-end PDF processing workflow including
document factory integration and fallback mechanisms.
"""

import asyncio
import os

# Ensure the main module can be imported
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, "/Users/eduardo/Documents/projects/ai-foundation/sse-chatgpt")


class TestPDFIntegration:
    """Test PDF processing integration with the main application."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_document_processor_factory_creation(self):
        """Test that DocumentProcessorFactory can be created."""
        try:
            from main import DocumentProcessorFactory

            factory = DocumentProcessorFactory()

            assert factory is not None
            assert hasattr(factory, "process_document")
            assert hasattr(factory, "pdf_processor")
            assert hasattr(factory, "temp_file_manager")
            assert hasattr(factory, "error_handler")
            assert hasattr(factory, "cleanup")

        except ImportError as e:
            pytest.fail(f"DocumentProcessorFactory should be available: {e}")

    @pytest.mark.asyncio
    async def test_document_processor_fallback(self):
        """Test document processor fallback to None for unsupported types."""
        try:
            from main import DocumentProcessorFactory

            factory = DocumentProcessorFactory()

            # Test with truly unsupported MIME type
            result = await factory.process_document(
                file_content=b"test content", filename="test.xyz", mime_type="application/unknown", session_id="test_session"
            )

            # Should return None for fallback to base64
            assert result is None

        except ImportError as e:
            pytest.fail(f"DocumentProcessorFactory should be available: {e}")

    def test_global_document_processor_factory(self):
        """Test that global document processor factory is initialized."""
        try:
            from main import document_processor_factory

            assert document_processor_factory is not None
            assert hasattr(document_processor_factory, "process_document")

        except ImportError as e:
            pytest.fail(f"Global document_processor_factory should be available: {e}")

    def test_temporary_file_manager_integration(self):
        """Test TemporaryFileManager integration."""
        try:
            from main import TemporaryFileManager

            manager = TemporaryFileManager()

            # Test file creation
            file_id, file_path = manager.create_temp_file(suffix=".pdf")
            assert file_id is not None
            assert file_path is not None
            assert file_path.endswith(".pdf")

            # Test file writing
            test_content = b"test pdf content"
            success = manager.write_temp_file(file_id, test_content)
            assert success is True

            # Test file exists
            assert os.path.exists(file_path)

            # Test file cleanup
            cleanup_success = manager.cleanup_temp_file(file_id)
            assert cleanup_success is True
            assert not os.path.exists(file_path)

        except ImportError as e:
            pytest.fail(f"TemporaryFileManager should be available: {e}")

    def test_processed_document_schema(self):
        """Test ProcessedDocument schema validation."""
        try:
            from main import ProcessedDocument

            # Create a sample processed document
            doc = ProcessedDocument(
                content="Sample PDF content",
                metadata={"title": "Test Document", "author": "Test Author", "page_count": 2, "creation_date": "2024-01-01"},
                pages=[
                    {"page_number": 1, "content": "Page 1 content", "metadata": {"page_width": 612, "page_height": 792}},
                    {"page_number": 2, "content": "Page 2 content", "metadata": {"page_width": 612, "page_height": 792}},
                ],
                structure={
                    "total_pages": 2,
                    "has_images": False,
                    "has_links": False,
                    "content_length": 28,
                    "loader_used": "PyMuPDFLoader",
                },
                processing_method="PyMuPDFLoader",
                original_filename="test.pdf",
            )

            # Validate schema
            assert doc.content == "Sample PDF content"
            assert doc.metadata["title"] == "Test Document"
            assert doc.metadata["author"] == "Test Author"
            assert doc.metadata["page_count"] == 2
            assert len(doc.pages) == 2
            assert doc.pages[0]["page_number"] == 1
            assert doc.structure["total_pages"] == 2
            assert doc.processing_method == "PyMuPDFLoader"
            assert doc.original_filename == "test.pdf"

        except ImportError as e:
            pytest.fail(f"ProcessedDocument should be available: {e}")

    @pytest.mark.asyncio
    async def test_pdf_processor_with_mock_content(self):
        """Test PDF processor with mock PDF content."""
        try:
            from main import PDFProcessor, TemporaryFileManager

            temp_manager = TemporaryFileManager()
            processor = PDFProcessor(temp_manager)

            # Mock minimal PDF content that PyMuPDF can at least open
            # This is a very basic but valid PDF structure
            mock_pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj

xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000074 00000 n 
0000000120 00000 n 
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
173
%%EOF"""

            # This should now process successfully (though with minimal content)
            result = await processor.process_pdf(file_content=mock_pdf_content, filename="test.pdf", session_id="test_session")

            # Verify the result structure
            assert result is not None
            assert result.processing_method == "PyMuPDFLoader"
            assert result.original_filename == "test.pdf"
            assert result.metadata is not None
            assert result.structure is not None
            assert result.pages is not None

        except ImportError as e:
            pytest.fail(f"PDF processing components should be available: {e}")

    def test_langchain_imports_available(self):
        """Test that LangChain imports are available."""
        try:
            from main import LANGCHAIN_LOADERS_AVAILABLE

            # Should be True if dependencies are properly installed
            assert LANGCHAIN_LOADERS_AVAILABLE is True

        except ImportError as e:
            pytest.fail(f"LangChain availability flag should be accessible: {e}")

    def test_document_page_schema(self):
        """Test DocumentPage schema validation."""
        try:
            from main import DocumentPage

            # Create a sample document page
            page = DocumentPage(
                page_number=1,
                content="This is page 1 content",
                metadata={
                    "page_width": 612.0,
                    "page_height": 792.0,
                    "rotation": 0,
                    "images": [],
                    "links": [],
                    "text_blocks": 5,
                    "char_count": 23,
                },
            )

            # Validate schema
            assert page.page_number == 1
            assert page.content == "This is page 1 content"
            assert page.metadata["page_width"] == 612.0
            assert page.metadata["page_height"] == 792.0
            assert page.metadata["char_count"] == 23

        except ImportError as e:
            pytest.fail(f"DocumentPage should be available: {e}")


class TestErrorHandlingIntegration:
    """Test error handling integration with PDF processing."""

    @pytest.mark.asyncio
    async def test_pdf_processing_error_handling(self):
        """Test that PDF processing errors are handled gracefully."""
        try:
            from main import DocumentProcessorFactory

            factory = DocumentProcessorFactory()

            # Test with corrupted PDF content
            corrupted_content = b"not a pdf file"

            # Should return None (fallback) rather than raising unhandled exception
            result = await factory.process_document(
                file_content=corrupted_content,
                filename="corrupted.pdf",
                mime_type="application/pdf",
                session_id="test_session",
            )

            # Should return None to trigger base64 fallback
            assert result is None

        except ImportError as e:
            pytest.fail(f"DocumentProcessorFactory should be available: {e}")

    def test_error_handler_integration(self):
        """Test that error handler is properly integrated."""
        try:
            from main import ErrorHandler, PDFProcessor, TemporaryFileManager

            temp_manager = TemporaryFileManager()
            processor = PDFProcessor(temp_manager)

            # Verify error handler is integrated
            assert hasattr(processor, "error_handler")
            assert isinstance(processor.error_handler, ErrorHandler)

        except ImportError as e:
            pytest.fail(f"Error handling integration should be available: {e}")


class TestCleanupMechanisms:
    """Test cleanup mechanisms for temporary files."""

    def test_temp_file_cleanup_on_success(self):
        """Test that temporary files are cleaned up on successful processing."""
        try:
            from main import TemporaryFileManager

            manager = TemporaryFileManager()

            # Create multiple temp files
            file_ids = []
            file_paths = []

            for i in range(3):
                file_id, file_path = manager.create_temp_file(suffix=f".test{i}")
                file_ids.append(file_id)
                file_paths.append(file_path)
                manager.write_temp_file(file_id, f"test content {i}".encode())

            # Verify files exist
            for file_path in file_paths:
                assert os.path.exists(file_path)

            # Clean up all files
            manager.cleanup_all()

            # Verify files are cleaned up
            for file_path in file_paths:
                assert not os.path.exists(file_path)

        except ImportError as e:
            pytest.fail(f"TemporaryFileManager should be available: {e}")

    def test_temp_file_cleanup_on_error(self):
        """Test that temporary files are cleaned up even when errors occur."""
        try:
            from main import TemporaryFileManager

            manager = TemporaryFileManager()

            # Create temp file
            file_id, file_path = manager.create_temp_file(suffix=".test")
            manager.write_temp_file(file_id, b"test content")

            # Verify file exists
            assert os.path.exists(file_path)

            # Simulate error scenario - file should still be cleanable
            try:
                # Simulate some processing error
                raise ValueError("Simulated processing error")
            except ValueError:
                # Cleanup should still work
                success = manager.cleanup_temp_file(file_id)
                assert success is True
                assert not os.path.exists(file_path)

        except ImportError as e:
            pytest.fail(f"TemporaryFileManager should be available: {e}")


def test_integration_fixtures_and_imports():
    """Test that all integration components can be imported."""
    try:
        from main import (
            LANGCHAIN_LOADERS_AVAILABLE,
            DocumentPage,
            DocumentProcessorFactory,
            PDFProcessor,
            ProcessedDocument,
            TemporaryFileManager,
            document_processor_factory,
        )

        # Verify all components are available
        assert DocumentProcessorFactory is not None
        assert PDFProcessor is not None
        assert TemporaryFileManager is not None
        assert ProcessedDocument is not None
        assert DocumentPage is not None
        assert document_processor_factory is not None
        assert isinstance(LANGCHAIN_LOADERS_AVAILABLE, bool)

    except ImportError as e:
        pytest.fail(f"Integration components should be available: {e}")
