"""Tests for chunking strategies and processors."""

import pytest
from unittest.mock import Mock, patch
from models.chunking import (
    ChunkingConfig, ChunkingStrategy, Chunk, ChunkMetadata, ChunkIndex, ChunkedDocument
)


class TestRecursiveCharacterChunkingStrategy:
    """Test RecursiveCharacterChunkingStrategy functionality."""
    
    @pytest.fixture
    def basic_config(self):
        """Create basic chunking configuration."""
        return ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=30,
            max_chunk_size=200
        )
    
    @pytest.fixture
    def sample_text(self):
        """Create sample text for testing."""
        return """This is the first paragraph with multiple sentences. It contains various types of content that should be chunked appropriately.

This is the second paragraph. It also has multiple sentences and should be processed correctly by the chunking algorithm.

This is the third paragraph. It provides additional content for testing the chunking boundaries and overlap functionality.

Finally, this is the last paragraph. It ensures we have enough content to test multiple chunks and their relationships."""
    
    def test_recursive_character_strategy_creation(self, basic_config):
        """Test creation of RecursiveCharacterChunkingStrategy."""
        from document_processing.processors import RecursiveCharacterChunkingStrategy
        
        strategy = RecursiveCharacterChunkingStrategy(basic_config)
        assert strategy.config == basic_config
        assert strategy.get_strategy_name() == "recursive_character"
    
    def test_chunk_text_basic(self, basic_config, sample_text):
        """Test basic text chunking functionality."""
        from document_processing.processors import RecursiveCharacterChunkingStrategy
        
        strategy = RecursiveCharacterChunkingStrategy(basic_config)
        metadata = {"document_id": "test_doc_1"}
        
        chunks = strategy.chunk_text(sample_text, metadata)
        
        # Verify we got chunks
        assert len(chunks) > 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        
        # Verify chunk sizes are within limits
        for chunk in chunks:
            assert len(chunk.content) >= basic_config.min_chunk_size
            assert len(chunk.content) <= basic_config.max_chunk_size
    
    def test_chunk_text_overlap(self, basic_config, sample_text):
        """Test that chunks have appropriate overlap."""
        from document_processing.processors import RecursiveCharacterChunkingStrategy
        
        strategy = RecursiveCharacterChunkingStrategy(basic_config)
        chunks = strategy.chunk_text(sample_text, {"document_id": "test_overlap"})
        
        if len(chunks) > 1:
            # Check that adjacent chunks have some overlap
            first_chunk_end = chunks[0].metadata.position["end_index"]
            second_chunk_start = chunks[1].metadata.position["start_index"]
            
            # There should be some overlap (second starts before first ends + chunk_size)
            expected_second_start = first_chunk_end - basic_config.chunk_overlap
            assert abs(second_chunk_start - expected_second_start) <= 50  # Allow some flexibility
    
    def test_chunk_metadata_completeness(self, basic_config, sample_text):
        """Test that chunk metadata is complete and accurate."""
        from document_processing.processors import RecursiveCharacterChunkingStrategy
        
        strategy = RecursiveCharacterChunkingStrategy(basic_config)
        chunks = strategy.chunk_text(sample_text, {"document_id": "metadata_test"})
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata
            
            # Check required metadata fields
            assert metadata.chunk_id is not None and metadata.chunk_id != ""
            assert "start_index" in metadata.position
            assert "end_index" in metadata.position
            assert "chunk_number" in metadata.position
            assert "total_chunks" in metadata.position
            
            # Check metadata values
            assert metadata.position["chunk_number"] == i + 1
            assert metadata.position["total_chunks"] == len(chunks)
            assert metadata.position["start_index"] >= 0
            assert metadata.position["end_index"] > metadata.position["start_index"]
            
            # Check content analysis
            assert "word_count" in metadata.content_analysis
            assert "char_count" in metadata.content_analysis
            assert metadata.content_analysis["word_count"] > 0
            assert metadata.content_analysis["char_count"] == len(chunk.content)
    
    def test_chunk_relationships_linking(self, basic_config, sample_text):
        """Test that chunks are properly linked with relationships."""
        from document_processing.processors import RecursiveCharacterChunkingStrategy
        
        strategy = RecursiveCharacterChunkingStrategy(basic_config)
        chunks = strategy.chunk_text(sample_text, {"document_id": "relationship_test"})
        
        if len(chunks) > 1:
            # First chunk should have no previous, but has next
            assert chunks[0].metadata.relationships["previous_chunk_id"] is None
            assert chunks[0].metadata.relationships["next_chunk_id"] == chunks[1].chunk_id
            
            # Last chunk should have previous, but no next
            assert chunks[-1].metadata.relationships["previous_chunk_id"] == chunks[-2].chunk_id
            assert chunks[-1].metadata.relationships["next_chunk_id"] is None
            
            # Middle chunks should have both
            if len(chunks) > 2:
                middle_chunk = chunks[1]
                assert middle_chunk.metadata.relationships["previous_chunk_id"] == chunks[0].chunk_id
                assert middle_chunk.metadata.relationships["next_chunk_id"] == chunks[2].chunk_id
    
    def test_empty_text_handling(self, basic_config):
        """Test handling of empty or minimal text."""
        from document_processing.processors import RecursiveCharacterChunkingStrategy
        
        strategy = RecursiveCharacterChunkingStrategy(basic_config)
        
        # Empty text
        chunks = strategy.chunk_text("", {"document_id": "empty_test"})
        assert len(chunks) == 0
        
        # Very short text (below min_chunk_size)
        short_text = "Short"
        chunks = strategy.chunk_text(short_text, {"document_id": "short_test"})
        assert len(chunks) <= 1  # Should create at most one chunk
    
    def test_single_chunk_text(self, basic_config):
        """Test text that should result in a single chunk."""
        from document_processing.processors import RecursiveCharacterChunkingStrategy
        
        strategy = RecursiveCharacterChunkingStrategy(basic_config)
        
        # Text that fits in one chunk
        single_chunk_text = "This is a medium-length text that should fit within a single chunk boundary."
        chunks = strategy.chunk_text(single_chunk_text, {"document_id": "single_test"})
        
        assert len(chunks) == 1
        assert chunks[0].content == single_chunk_text
        assert chunks[0].metadata.position["chunk_number"] == 1
        assert chunks[0].metadata.position["total_chunks"] == 1
        assert chunks[0].metadata.relationships["previous_chunk_id"] is None
        assert chunks[0].metadata.relationships["next_chunk_id"] is None


class TestChunkProcessor:
    """Test ChunkProcessor functionality."""
    
    @pytest.fixture
    def mock_processed_document(self):
        """Create mock ProcessedDocument for testing."""
        from document_processing.processors import ProcessedDocument
        
        return ProcessedDocument(
            content="This is test content for chunk processing. " * 20,  # Long enough to chunk
            metadata={"original_filename": "test.txt", "file_size": 1000},
            pages=[{"page_number": 1, "content": "Test content", "metadata": {}}],
            structure={"total_pages": 1, "content_length": 500},
            processing_method="TextLoader",
            original_filename="test.txt"
        )
    
    def test_chunk_processor_creation(self):
        """Test creation of ChunkProcessor."""
        from document_processing.processors import ChunkProcessor
        from core.config import AppConfig
        
        config = AppConfig()
        processor = ChunkProcessor(config)
        
        assert processor.config == config
        assert processor.strategy_factory is not None
    
    @pytest.mark.asyncio
    async def test_process_document_chunking_disabled(self, mock_processed_document):
        """Test processing with chunking disabled."""
        from document_processing.processors import ChunkProcessor
        from core.config import AppConfig
        
        config = AppConfig()
        config.enable_chunking = False
        processor = ChunkProcessor(config)
        
        result = await processor.process_document(mock_processed_document, "test_session")
        
        # Should return original document without chunking
        assert result == mock_processed_document
    
    @pytest.mark.asyncio
    async def test_process_document_chunking_enabled(self, mock_processed_document):
        """Test processing with chunking enabled."""
        from document_processing.processors import ChunkProcessor
        from core.config import AppConfig
        
        config = AppConfig()
        config.enable_chunking = True
        processor = ChunkProcessor(config)
        
        result = await processor.process_document(mock_processed_document, "test_session")
        
        # Should return ChunkedDocument
        from document_processing.processors import ChunkedDocument
        assert isinstance(result, ChunkedDocument)
        assert result.original_document == mock_processed_document
        assert len(result.chunks) > 0
        assert result.chunk_index is not None
    
    @pytest.mark.asyncio
    async def test_process_document_error_handling(self, mock_processed_document):
        """Test error handling during chunking."""
        from document_processing.processors import ChunkProcessor
        from core.config import AppConfig
        
        config = AppConfig()
        config.enable_chunking = True
        processor = ChunkProcessor(config)
        
        # Mock strategy to raise exception
        with patch.object(processor.strategy_factory, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.chunk_text.side_effect = Exception("Chunking failed")
            mock_get_strategy.return_value = mock_strategy
            
            # Should fall back to original document
            result = await processor.process_document(mock_processed_document, "test_session")
            assert result == mock_processed_document


class TestChunkingStrategyFactory:
    """Test ChunkingStrategyFactory functionality."""
    
    def test_strategy_factory_creation(self):
        """Test creation of ChunkingStrategyFactory."""
        from document_processing.processors import ChunkingStrategyFactory
        from models.chunking import ChunkingConfig
        
        config = ChunkingConfig()
        factory = ChunkingStrategyFactory(config)
        
        assert factory.config == config
    
    def test_get_strategy_recursive_character(self):
        """Test getting recursive character strategy for non-text MIME type."""
        from document_processing.processors import ChunkingStrategyFactory, RecursiveCharacterChunkingStrategy
        from models.chunking import ChunkingConfig, ChunkingStrategy
        
        config = ChunkingConfig(strategy=ChunkingStrategy.RECURSIVE_CHARACTER)
        factory = ChunkingStrategyFactory(config)
        
        # Use a non-text MIME type to get recursive character strategy
        strategy = factory.get_strategy("application/json", {"file_extension": ".json"})
        
        assert isinstance(strategy, RecursiveCharacterChunkingStrategy)
        assert strategy.get_strategy_name() == "recursive_character"
    
    def test_get_strategy_text_aware(self):
        """Test getting text-aware strategy."""
        from document_processing.processors import ChunkingStrategyFactory, TextChunkingStrategy
        from models.chunking import ChunkingConfig, ChunkingStrategy
        
        config = ChunkingConfig(strategy=ChunkingStrategy.TEXT_PARAGRAPH)
        factory = ChunkingStrategyFactory(config)
        
        strategy = factory.get_strategy("text/plain", {"file_extension": ".txt"})
        
        assert isinstance(strategy, TextChunkingStrategy)
        assert strategy.get_strategy_name() == "text_paragraph"
    
    def test_get_strategy_pdf_aware(self):
        """Test getting PDF-aware strategy."""
        from document_processing.processors import ChunkingStrategyFactory
        from models.chunking import ChunkingConfig, ChunkingStrategy
        
        config = ChunkingConfig(strategy=ChunkingStrategy.PDF_AWARE)
        factory = ChunkingStrategyFactory(config)
        
        strategy = factory.get_strategy("application/pdf", {"file_extension": ".pdf"})
        
        # Should get PDF-specific strategy or fall back to recursive character
        assert strategy is not None
        assert hasattr(strategy, 'chunk_text')
        assert hasattr(strategy, 'get_strategy_name')
    
    def test_get_strategy_unknown_mime_type(self):
        """Test getting strategy for unknown MIME type."""
        from document_processing.processors import ChunkingStrategyFactory, RecursiveCharacterChunkingStrategy
        from models.chunking import ChunkingConfig
        
        config = ChunkingConfig()
        factory = ChunkingStrategyFactory(config)
        
        strategy = factory.get_strategy("unknown/type", {})
        
        # Should fall back to recursive character strategy
        assert isinstance(strategy, RecursiveCharacterChunkingStrategy)


class TestChunkOptimizer:
    """Test ChunkOptimizer functionality."""
    
    def test_chunk_optimizer_creation(self):
        """Test creation of ChunkOptimizer."""
        from document_processing.processors import ChunkOptimizer
        from models.chunking import ChunkingConfig
        
        config = ChunkingConfig()
        optimizer = ChunkOptimizer(config)
        
        assert optimizer.config == config
    
    def test_optimize_chunk_size_basic(self):
        """Test basic chunk size optimization."""
        from document_processing.processors import ChunkOptimizer
        from models.chunking import ChunkingConfig
        
        config = ChunkingConfig(chunk_size=1000)
        optimizer = ChunkOptimizer(config)
        
        # Simple text should use standard size
        simple_text = "This is simple text. " * 10
        optimal_size = optimizer.optimize_chunk_size(simple_text, {"complexity_score": 0.3})
        
        assert 500 <= optimal_size <= 1500  # Should be within reasonable range
    
    def test_optimize_chunk_size_complex_content(self):
        """Test chunk size optimization for complex content."""
        from document_processing.processors import ChunkOptimizer
        from models.chunking import ChunkingConfig
        
        config = ChunkingConfig(chunk_size=1000)
        optimizer = ChunkOptimizer(config)
        
        # Complex text might need smaller chunks
        complex_text = "The sophisticated methodological paradigms necessitate comprehensive evaluation. " * 10
        optimal_size = optimizer.optimize_chunk_size(complex_text, {"complexity_score": 0.9})
        
        assert 300 <= optimal_size <= 1200  # Should adjust based on complexity
    
    def test_count_tokens(self):
        """Test token counting functionality."""
        from document_processing.processors import ChunkOptimizer
        from models.chunking import ChunkingConfig
        
        config = ChunkingConfig()
        optimizer = ChunkOptimizer(config)
        
        text = "This is a test sentence with multiple words."
        token_count = optimizer.count_tokens(text)
        
        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count <= len(text.split()) * 2  # Rough upper bound


class TestOverlapManager:
    """Test OverlapManager functionality."""
    
    def test_overlap_manager_creation(self):
        """Test creation of OverlapManager."""
        from document_processing.processors import OverlapManager
        from models.chunking import ChunkingConfig
        
        config = ChunkingConfig(chunk_overlap=100)
        manager = OverlapManager(config)
        
        assert manager.config == config
        assert manager.overlap_size == 100
    
    def test_calculate_overlap_boundaries(self):
        """Test calculation of overlap boundaries."""
        from document_processing.processors import OverlapManager
        from models.chunking import ChunkingConfig
        
        config = ChunkingConfig(chunk_size=200, chunk_overlap=50)
        manager = OverlapManager(config)
        
        text = "This is test text. " * 20  # Long enough for multiple chunks
        chunks = [(0, 200), (150, 350), (300, 500)]  # Overlapping chunks
        
        optimized_boundaries = manager.calculate_overlap_boundaries(text, chunks)
        
        assert len(optimized_boundaries) == len(chunks)
        for boundary in optimized_boundaries:
            assert isinstance(boundary, tuple)
            assert len(boundary) == 2
            assert boundary[0] < boundary[1]
    
    def test_validate_overlap_quality(self):
        """Test overlap quality validation."""
        from document_processing.processors import OverlapManager
        from models.chunking import ChunkingConfig
        
        config = ChunkingConfig()
        manager = OverlapManager(config)
        
        text = "This is a sentence. This is another sentence."
        chunk1 = "This is a sentence."
        chunk2 = "sentence. This is another"
        
        quality_score = manager.validate_overlap_quality(text, chunk1, chunk2)
        
        assert 0.0 <= quality_score <= 1.0
        assert isinstance(quality_score, float)