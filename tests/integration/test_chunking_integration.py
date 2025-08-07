"""Integration tests for chunking with document processing pipeline."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.config import AppConfig
from document_processing.processors import ChunkedDocument, DocumentProcessorFactory, ProcessedDocument
from models.chunking import ChunkingConfig, ChunkingStrategy


class TestChunkingIntegration:
    """Test chunking integration with document processing pipeline."""

    @pytest.fixture
    def app_config(self):
        """Create application configuration for testing."""
        config = AppConfig()
        config.enable_chunking = True
        config.default_chunk_size = 500
        config.default_chunk_overlap = 100
        return config

    @pytest.fixture
    def app_config_disabled(self):
        """Create application configuration with chunking disabled."""
        config = AppConfig()
        config.enable_chunking = False
        return config

    @pytest.fixture
    def sample_text_content(self):
        """Create sample text content for testing."""
        return b"""This is a comprehensive document that contains multiple paragraphs and should be chunked appropriately by the intelligent chunking system.

The first paragraph introduces the topic and provides context for the reader. It contains important information that should be preserved during the chunking process.

The second paragraph continues the discussion and adds more details. This paragraph has enough content to demonstrate the chunking behavior and overlap functionality.

The third paragraph provides additional context and examples. It helps to test the boundary detection and quality assessment features of the chunking system.

The final paragraph concludes the document and summarizes the key points. This ensures we have sufficient content for multiple chunks to be generated during testing."""

    @pytest.mark.asyncio
    async def test_document_processing_with_chunking_enabled(self, app_config, sample_text_content):
        """Test document processing with chunking enabled."""
        factory = DocumentProcessorFactory(app_config=app_config)

        result = await factory.process_document(
            file_content=sample_text_content, filename="test_document.txt", mime_type="text/plain", session_id="test_session"
        )

        # Should return ChunkedDocument
        assert isinstance(result, ChunkedDocument)
        assert result.chunk_count > 1
        assert result.original_document is not None
        assert len(result.chunks) > 1
        assert result.chunk_index is not None

        # Verify chunks have proper metadata
        for chunk in result.chunks:
            assert chunk.chunk_id is not None
            assert chunk.content is not None
            assert len(chunk.content) > 0
            assert chunk.metadata.processing_metadata["strategy_used"] == "text_paragraph"

    @pytest.mark.asyncio
    async def test_document_processing_with_chunking_disabled(self, app_config_disabled, sample_text_content):
        """Test document processing with chunking disabled."""
        factory = DocumentProcessorFactory(app_config=app_config_disabled)

        result = await factory.process_document(
            file_content=sample_text_content, filename="test_document.txt", mime_type="text/plain", session_id="test_session"
        )

        # Should return ProcessedDocument
        assert isinstance(result, ProcessedDocument)
        assert not isinstance(result, ChunkedDocument)
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_pdf_document_chunking(self, app_config):
        """Test PDF document chunking strategy selection."""
        factory = DocumentProcessorFactory(app_config=app_config)

        # Mock PDF content processing
        with patch.object(factory.pdf_processor, "process_pdf") as mock_pdf:
            mock_processed = ProcessedDocument(
                content="PDF content for chunking test. " * 50,
                metadata={"file_type": "pdf", "pages": 2},
                pages=[{"page_number": 1, "content": "Page 1"}],
                structure={"total_pages": 1},
                processing_method="PyMuPDFLoader",
                original_filename="test.pdf",
            )
            mock_pdf.return_value = mock_processed

            result = await factory.process_document(
                file_content=b"fake pdf content", filename="test.pdf", mime_type="application/pdf", session_id="test_session"
            )

            # Should return ChunkedDocument with PDF strategy
            assert isinstance(result, ChunkedDocument)
            assert result.chunking_metadata["strategy_used"] == "pdf_aware"

    @pytest.mark.asyncio
    async def test_chunking_fallback_on_error(self, app_config, sample_text_content):
        """Test that chunking falls back to original document on error."""
        factory = DocumentProcessorFactory(app_config=app_config)

        # Mock chunk processor to raise exception
        with patch.object(factory.chunk_processor, "process_document") as mock_chunk:
            mock_chunk.side_effect = Exception("Chunking failed")

            result = await factory.process_document(
                file_content=sample_text_content,
                filename="test_document.txt",
                mime_type="text/plain",
                session_id="test_session",
            )

            # Should return ProcessedDocument (fallback)
            assert isinstance(result, ProcessedDocument)
            assert not isinstance(result, ChunkedDocument)

    @pytest.mark.asyncio
    async def test_chunking_preserves_original_document(self, app_config, sample_text_content):
        """Test that chunking preserves reference to original document."""
        factory = DocumentProcessorFactory(app_config=app_config)

        result = await factory.process_document(
            file_content=sample_text_content, filename="test_document.txt", mime_type="text/plain", session_id="test_session"
        )

        assert isinstance(result, ChunkedDocument)
        assert result.original_document is not None
        assert isinstance(result.original_document, ProcessedDocument)
        assert result.original_document.original_filename == "test_document.txt"
        assert result.original_document.processing_method == "TextLoader"

    @pytest.mark.asyncio
    async def test_chunk_metadata_completeness(self, app_config, sample_text_content):
        """Test that chunk metadata is complete and accurate."""
        factory = DocumentProcessorFactory(app_config=app_config)

        result = await factory.process_document(
            file_content=sample_text_content, filename="test_document.txt", mime_type="text/plain", session_id="test_session"
        )

        assert isinstance(result, ChunkedDocument)

        # Verify chunking metadata
        assert "strategy_used" in result.chunking_metadata
        assert "total_chunks" in result.chunking_metadata
        assert "average_chunk_size" in result.chunking_metadata
        assert "total_tokens" in result.chunking_metadata
        assert "processing_timestamp" in result.chunking_metadata
        assert "chunking_config" in result.chunking_metadata

        # Verify individual chunk metadata
        for i, chunk in enumerate(result.chunks):
            metadata = chunk.metadata

            # Position metadata
            assert "start_index" in metadata.position
            assert "end_index" in metadata.position
            assert "chunk_number" in metadata.position
            assert metadata.position["chunk_number"] == i + 1
            assert metadata.position["total_chunks"] == len(result.chunks)

            # Content analysis
            assert "word_count" in metadata.content_analysis
            assert "char_count" in metadata.content_analysis
            assert metadata.content_analysis["char_count"] == len(chunk.content)

            # Processing metadata
            assert "strategy_used" in metadata.processing_metadata
            assert "processing_timestamp" in metadata.processing_metadata

    @pytest.mark.asyncio
    async def test_chunk_index_functionality(self, app_config, sample_text_content):
        """Test chunk index functionality."""
        factory = DocumentProcessorFactory(app_config=app_config)

        result = await factory.process_document(
            file_content=sample_text_content, filename="test_document.txt", mime_type="text/plain", session_id="test_session"
        )

        assert isinstance(result, ChunkedDocument)

        # Test index operations
        first_chunk = result.chunks[0]
        retrieved_chunk = result.get_chunk_by_id(first_chunk.chunk_id)
        assert retrieved_chunk is not None
        assert retrieved_chunk.chunk_id == first_chunk.chunk_id

        # Test adjacent chunk relationships
        if len(result.chunks) > 1:
            adjacent = result.chunk_index.get_adjacent_chunks(first_chunk.chunk_id)
            assert adjacent["previous"] is None  # First chunk has no previous
            assert adjacent["next"] is not None  # First chunk should have next
            assert adjacent["next"].chunk_id == result.chunks[1].chunk_id

    def test_chunking_config_integration(self, app_config):
        """Test that app configuration is properly passed to chunking components."""
        factory = DocumentProcessorFactory(app_config=app_config)

        assert factory.chunk_processor is not None
        assert factory.chunk_processor.config == app_config

        # Test chunking configuration
        chunking_config = factory.chunk_processor.chunking_config
        assert chunking_config.chunk_size == app_config.default_chunk_size
        assert chunking_config.chunk_overlap == app_config.default_chunk_overlap
        assert chunking_config.min_chunk_size == app_config.min_chunk_size
        assert chunking_config.max_chunk_size == app_config.max_chunk_size

    @pytest.mark.asyncio
    async def test_mime_type_strategy_selection(self, app_config):
        """Test that different MIME types select appropriate strategies."""
        factory = DocumentProcessorFactory(app_config=app_config)

        # Test text file
        with patch.object(factory.text_processor, "process_text") as mock_text:
            mock_processed = ProcessedDocument(
                content="Text content",
                metadata={},
                pages=[],
                structure={},
                processing_method="TextLoader",
                original_filename="test.txt",
            )
            mock_text.return_value = mock_processed

            result = await factory.process_document(
                file_content=b"text content", filename="test.txt", mime_type="text/plain", session_id="test"
            )

            assert isinstance(result, ChunkedDocument)
            assert result.chunking_metadata["strategy_used"] == "text_paragraph"

        # Test unknown MIME type (should use recursive character)
        with patch.object(factory.text_processor, "process_text") as mock_text:
            mock_processed.metadata["mime_type"] = "application/unknown"
            mock_text.return_value = mock_processed

            result = await factory.process_document(
                file_content=b"unknown content", filename="test.unknown", mime_type="application/unknown", session_id="test"
            )

            if isinstance(result, ChunkedDocument):
                # Should fall back to recursive character strategy
                assert result.chunking_metadata["strategy_used"] == "recursive_character"
