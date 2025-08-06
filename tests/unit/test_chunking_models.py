"""Tests for chunking models and data structures."""

import pytest
from models.chunking import (
    Chunk, ChunkMetadata, ChunkingConfig, ChunkIndex, ChunkedDocument,
    ChunkingStrategy, BoundaryQuality, BaseChunkingStrategy
)


class TestChunkMetadata:
    """Test ChunkMetadata functionality."""
    
    def test_chunk_metadata_creation(self):
        """Test basic chunk metadata creation."""
        metadata = ChunkMetadata(
            chunk_id="test_chunk_1",
            position={"start_index": 0, "end_index": 100},
            relationships={"parent_document_id": "doc_1"},
            content_analysis={"word_count": 20, "char_count": 100},
            processing_metadata={"strategy_used": "test"},
            quality_metrics={"coherence_score": 0.8}
        )
        
        assert metadata.chunk_id == "test_chunk_1"
        assert metadata.position["start_index"] == 0
        assert metadata.position["end_index"] == 100
    
    def test_chunk_metadata_auto_id_generation(self):
        """Test automatic chunk ID generation."""
        metadata = ChunkMetadata(
            chunk_id="",  # Empty ID should trigger auto-generation
            position={},
            relationships={},
            content_analysis={},
            processing_metadata={},
            quality_metrics={}
        )
        
        assert metadata.chunk_id.startswith("chunk_")
        assert len(metadata.chunk_id) == 14  # "chunk_" + 8 hex chars


class TestChunk:
    """Test Chunk functionality."""
    
    def test_chunk_creation(self):
        """Test basic chunk creation."""
        metadata = ChunkMetadata(
            chunk_id="test_chunk",
            position={"start_index": 0, "end_index": 20},
            relationships={},
            content_analysis={"word_count": 4, "char_count": 20},
            processing_metadata={},
            quality_metrics={}
        )
        
        chunk = Chunk(content="This is test content", metadata=metadata)
        
        assert chunk.content == "This is test content"
        assert chunk.chunk_id == "test_chunk"
        assert chunk.size == 20
        assert chunk.word_count == 4
    
    def test_chunk_properties(self):
        """Test chunk property calculations."""
        metadata = ChunkMetadata(
            chunk_id="prop_test",
            position={},
            relationships={},
            content_analysis={},
            processing_metadata={},
            quality_metrics={}
        )
        
        chunk = Chunk(content="Hello world test", metadata=metadata)
        
        assert chunk.size == 16  # Character count
        assert chunk.word_count == 3  # Word count


class TestChunkingConfig:
    """Test ChunkingConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        
        assert config.strategy == ChunkingStrategy.RECURSIVE_CHARACTER
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 4000
        assert config.enable_semantic_boundaries == True
        assert config.enable_overlap_optimization == True
        assert config.enable_metadata_extraction == True
    
    def test_config_validation_positive_chunk_size(self):
        """Test that chunk_size must be positive."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=0)
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=-100)
    
    def test_config_validation_negative_overlap(self):
        """Test that chunk_overlap cannot be negative."""
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            ChunkingConfig(chunk_overlap=-50)
    
    def test_config_validation_overlap_vs_chunk_size(self):
        """Test that overlap must be less than chunk size."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)
        
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            ChunkingConfig(chunk_size=100, chunk_overlap=150)
    
    def test_config_validation_min_chunk_size(self):
        """Test that min_chunk_size cannot be negative."""
        with pytest.raises(ValueError, match="min_chunk_size cannot be negative"):
            ChunkingConfig(min_chunk_size=-10)
    
    def test_config_validation_max_chunk_size(self):
        """Test that max_chunk_size must be >= chunk_size."""
        with pytest.raises(ValueError, match="max_chunk_size must be >= chunk_size"):
            ChunkingConfig(chunk_size=1000, max_chunk_size=500)
    
    def test_valid_config(self):
        """Test creation of valid configuration."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=100,
            min_chunk_size=50,
            max_chunk_size=2000
        )
        
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.min_chunk_size == 50
        assert config.max_chunk_size == 2000


class TestChunkIndex:
    """Test ChunkIndex functionality."""
    
    def create_test_chunk(self, chunk_id: str, content: str, start_index: int = 0) -> Chunk:
        """Create a test chunk for index testing."""
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            position={"start_index": start_index, "end_index": start_index + len(content)},
            relationships={},
            content_analysis={"word_count": len(content.split())},
            processing_metadata={},
            quality_metrics={}
        )
        return Chunk(content=content, metadata=metadata)
    
    def test_chunk_index_creation(self):
        """Test basic chunk index creation."""
        index = ChunkIndex()
        
        assert len(index.chunks_by_id) == 0
        assert len(index.chunks_by_position) == 0
        assert len(index.chunks_by_document) == 0
    
    def test_add_chunk_to_index(self):
        """Test adding chunks to index."""
        index = ChunkIndex()
        chunk = self.create_test_chunk("chunk_1", "Test content", 0)
        
        index.add_chunk(chunk, "doc_1")
        
        assert len(index.chunks_by_id) == 1
        assert "chunk_1" in index.chunks_by_id
        assert len(index.chunks_by_position) == 1
        assert 0 in index.chunks_by_position
        assert "chunk_1" in index.chunks_by_position[0]
        assert len(index.chunks_by_document) == 1
        assert "doc_1" in index.chunks_by_document
        assert "chunk_1" in index.chunks_by_document["doc_1"]
    
    def test_get_chunk_by_id(self):
        """Test retrieving chunk by ID."""
        index = ChunkIndex()
        chunk = self.create_test_chunk("chunk_test", "Test content")
        
        index.add_chunk(chunk)
        retrieved_chunk = index.get_chunk("chunk_test")
        
        assert retrieved_chunk is not None
        assert retrieved_chunk.chunk_id == "chunk_test"
        assert retrieved_chunk.content == "Test content"
    
    def test_get_nonexistent_chunk(self):
        """Test retrieving non-existent chunk."""
        index = ChunkIndex()
        retrieved_chunk = index.get_chunk("nonexistent")
        
        assert retrieved_chunk is None
    
    def test_get_chunks_by_document(self):
        """Test retrieving chunks by document ID."""
        index = ChunkIndex()
        chunk1 = self.create_test_chunk("chunk_1", "Content 1", 0)
        chunk2 = self.create_test_chunk("chunk_2", "Content 2", 20)
        
        index.add_chunk(chunk1, "doc_1")
        index.add_chunk(chunk2, "doc_1")
        
        doc_chunks = index.get_chunks_by_document("doc_1")
        
        assert len(doc_chunks) == 2
        chunk_ids = [chunk.chunk_id for chunk in doc_chunks]
        assert "chunk_1" in chunk_ids
        assert "chunk_2" in chunk_ids
    
    def test_get_chunks_empty_document(self):
        """Test retrieving chunks for empty document."""
        index = ChunkIndex()
        doc_chunks = index.get_chunks_by_document("nonexistent_doc")
        
        assert len(doc_chunks) == 0
    
    def test_get_adjacent_chunks(self):
        """Test retrieving adjacent chunks."""
        index = ChunkIndex()
        
        # Create linked chunks
        chunk1 = self.create_test_chunk("chunk_1", "Content 1")
        chunk2 = self.create_test_chunk("chunk_2", "Content 2")
        chunk3 = self.create_test_chunk("chunk_3", "Content 3")
        
        # Link them manually
        chunk1.metadata.relationships["next_chunk_id"] = "chunk_2"
        chunk2.metadata.relationships["previous_chunk_id"] = "chunk_1"
        chunk2.metadata.relationships["next_chunk_id"] = "chunk_3"
        chunk3.metadata.relationships["previous_chunk_id"] = "chunk_2"
        
        index.add_chunk(chunk1, "doc_1")
        index.add_chunk(chunk2, "doc_1")
        index.add_chunk(chunk3, "doc_1")
        
        # Test middle chunk
        adjacent = index.get_adjacent_chunks("chunk_2")
        assert adjacent["previous"] is not None
        assert adjacent["previous"].chunk_id == "chunk_1"
        assert adjacent["next"] is not None
        assert adjacent["next"].chunk_id == "chunk_3"
        
        # Test first chunk
        adjacent = index.get_adjacent_chunks("chunk_1")
        assert adjacent["previous"] is None
        assert adjacent["next"] is not None
        assert adjacent["next"].chunk_id == "chunk_2"
        
        # Test last chunk
        adjacent = index.get_adjacent_chunks("chunk_3")
        assert adjacent["previous"] is not None
        assert adjacent["previous"].chunk_id == "chunk_2"
        assert adjacent["next"] is None


class TestChunkedDocument:
    """Test ChunkedDocument functionality."""
    
    def create_test_chunked_document(self) -> ChunkedDocument:
        """Create a test chunked document."""
        # Create test chunks
        metadata1 = ChunkMetadata(
            chunk_id="chunk_1",
            position={"start_index": 0, "end_index": 10},
            relationships={},
            content_analysis={},
            processing_metadata={},
            quality_metrics={}
        )
        metadata2 = ChunkMetadata(
            chunk_id="chunk_2",
            position={"start_index": 8, "end_index": 18},
            relationships={},
            content_analysis={},
            processing_metadata={},
            quality_metrics={}
        )
        
        chunk1 = Chunk(content="Content 1", metadata=metadata1)
        chunk2 = Chunk(content="Content 2", metadata=metadata2)
        
        # Create index
        index = ChunkIndex()
        index.add_chunk(chunk1, "doc_1")
        index.add_chunk(chunk2, "doc_1")
        
        return ChunkedDocument(
            original_document=None,  # Mock original document
            chunks=[chunk1, chunk2],
            chunk_index=index,
            chunking_metadata={"strategy": "test", "total_chunks": 2}
        )
    
    def test_chunked_document_properties(self):
        """Test chunked document properties."""
        doc = self.create_test_chunked_document()
        
        assert doc.chunk_count == 2
        assert doc.total_content_length == 18  # 9 + 9 characters
    
    def test_get_chunk_by_id(self):
        """Test getting chunk by ID."""
        doc = self.create_test_chunked_document()
        
        chunk = doc.get_chunk_by_id("chunk_1")
        assert chunk is not None
        assert chunk.content == "Content 1"
        
        chunk = doc.get_chunk_by_id("nonexistent")
        assert chunk is None
    
    def test_get_chunks_in_range(self):
        """Test getting chunks in position range."""
        doc = self.create_test_chunked_document()
        
        # Test range that overlaps both chunks
        chunks = doc.get_chunks_in_range(5, 15)
        assert len(chunks) == 2
        
        # Test range that overlaps only first chunk
        chunks = doc.get_chunks_in_range(0, 5)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "chunk_1"
        
        # Test range that overlaps only second chunk
        chunks = doc.get_chunks_in_range(15, 20)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "chunk_2"
        
        # Test range with no overlap
        chunks = doc.get_chunks_in_range(25, 30)
        assert len(chunks) == 0


class MockChunkingStrategy(BaseChunkingStrategy):
    """Mock chunking strategy for testing abstract base class."""
    
    def chunk_text(self, text: str, metadata: dict) -> list[Chunk]:
        """Mock implementation."""
        # Simple implementation for testing
        chunks = []
        words = text.split()
        chunk_size = 5  # 5 words per chunk
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i+chunk_size]
            content = " ".join(chunk_words)
            
            chunk_metadata = self._create_chunk_metadata(
                content=text,
                start_index=text.find(chunk_words[0]) if chunk_words else 0,
                end_index=text.find(chunk_words[-1]) + len(chunk_words[-1]) if chunk_words else 0,
                chunk_number=i // chunk_size + 1,
                total_chunks=(len(words) + chunk_size - 1) // chunk_size,
                document_metadata=metadata
            )
            
            chunks.append(Chunk(content=content, metadata=chunk_metadata))
        
        self._link_adjacent_chunks(chunks)
        return chunks
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "mock_strategy"


class TestBaseChunkingStrategy:
    """Test BaseChunkingStrategy functionality."""
    
    def test_create_chunk_metadata(self):
        """Test chunk metadata creation."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        strategy = MockChunkingStrategy(config)
        
        text = "This is a test document with some content for testing chunk metadata creation."
        metadata = strategy._create_chunk_metadata(
            content=text,
            start_index=0,
            end_index=20,
            chunk_number=1,
            total_chunks=3,
            document_metadata={"document_id": "test_doc"}
        )
        
        assert metadata.chunk_id.startswith("chunk_")
        assert metadata.position["start_index"] == 0
        assert metadata.position["end_index"] == 20
        assert metadata.position["chunk_number"] == 1
        assert metadata.position["total_chunks"] == 3
        assert metadata.relationships["parent_document_id"] == "test_doc"
        assert metadata.content_analysis["word_count"] > 0
        assert metadata.content_analysis["char_count"] == 20
        assert 0.0 <= metadata.content_analysis["complexity_score"] <= 1.0
    
    def test_calculate_complexity_score(self):
        """Test complexity score calculation."""
        config = ChunkingConfig()
        strategy = MockChunkingStrategy(config)
        
        # Simple text
        simple_score = strategy._calculate_complexity_score("cat dog run")
        # Complex text
        complex_score = strategy._calculate_complexity_score(
            "The sophisticated methodological approaches utilized in contemporary "
            "interdisciplinary research paradigms necessitate comprehensive evaluation."
        )
        
        assert 0.0 <= simple_score <= 1.0
        assert 0.0 <= complex_score <= 1.0
        # Complex text should generally have higher score
        assert complex_score >= simple_score
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        config = ChunkingConfig()
        strategy = MockChunkingStrategy(config)
        
        text = "machine learning algorithms are powerful tools for data analysis and pattern recognition"
        keywords = strategy._extract_keywords(text, max_keywords=3)
        
        assert len(keywords) <= 3
        assert isinstance(keywords, list)
        # Should not contain stop words
        assert "are" not in keywords
        assert "for" not in keywords
        assert "and" not in keywords
    
    def test_link_adjacent_chunks(self):
        """Test linking adjacent chunks."""
        config = ChunkingConfig()
        strategy = MockChunkingStrategy(config)
        
        text = "This is a longer text that will be split into multiple chunks for testing the adjacent chunk linking functionality."
        chunks = strategy.chunk_text(text, {"document_id": "test_doc"})
        
        assert len(chunks) > 1
        
        # Check first chunk
        assert chunks[0].metadata.relationships["previous_chunk_id"] is None
        assert chunks[0].metadata.relationships["next_chunk_id"] == chunks[1].chunk_id
        
        # Check middle chunk (if exists)
        if len(chunks) > 2:
            assert chunks[1].metadata.relationships["previous_chunk_id"] == chunks[0].chunk_id
            assert chunks[1].metadata.relationships["next_chunk_id"] == chunks[2].chunk_id
        
        # Check last chunk
        assert chunks[-1].metadata.relationships["previous_chunk_id"] == chunks[-2].chunk_id
        assert chunks[-1].metadata.relationships["next_chunk_id"] is None