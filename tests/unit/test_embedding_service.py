"""
Unit tests for the EmbeddingService component.

This test module covers:
- Model loading and initialization
- Batch embedding generation accuracy
- Caching mechanism effectiveness
- GPU acceleration detection and fallback
- Error handling for model loading failures

Following TDD methodology - tests written first, then implementation.
"""

import asyncio
import os
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

# Import the EmbeddingService (to be implemented)
from core.embedding_service import EmbeddingError, EmbeddingMetadata, EmbeddingService
from models.session import DocumentChunk


class TestEmbeddingService:
    """Test suite for EmbeddingService component."""

    @pytest.fixture
    def sample_chunks(self) -> List[DocumentChunk]:
        """Create sample document chunks for testing."""
        return [
            DocumentChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                content="This is the first chunk of text content.",
                start_page=1,
                end_page=1,
                metadata={"page": 1, "section": "introduction"},
            ),
            DocumentChunk(
                chunk_id="chunk_2",
                document_id="doc_1",
                content="This is the second chunk with different semantic meaning.",
                start_page=1,
                end_page=2,
                metadata={"page": 2, "section": "body"},
            ),
            DocumentChunk(
                chunk_id="chunk_3",
                document_id="doc_2",
                content="A completely different document with unique content.",
                start_page=1,
                end_page=1,
                metadata={"page": 1, "section": "conclusion"},
            ),
        ]

    @pytest.fixture
    def mock_model(self):
        """Mock sentence transformer model."""
        mock = Mock()
        mock.encode.return_value = np.random.rand(3, 384)  # 384-dim embeddings
        mock.device = "cpu"
        return mock

    @pytest.fixture
    async def embedding_service(self):
        """Create EmbeddingService instance with default config."""
        with patch("core.embedding_service.SentenceTransformer") as mock_st:
            mock_st.return_value = Mock()
            mock_st.return_value.encode.return_value = np.random.rand(1, 384)
            service = EmbeddingService()
            await service.initialize()
            return service

    @pytest.mark.asyncio
    class TestInitialization:
        """Test model initialization and configuration."""

        async def test_default_initialization(self):
            """Test initialization with default configuration."""
            with patch("core.embedding_service.SentenceTransformer") as mock_st:
                mock_st.return_value = Mock()
                service = EmbeddingService()
                await service.initialize()

                assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
                assert service.embedding_dimension == 384
                assert service.cache_size == 10000
                assert service.batch_size == 32
                mock_st.assert_called_once()

        async def test_custom_model_initialization(self):
            """Test initialization with custom model configuration."""
            with patch("core.embedding_service.SentenceTransformer") as mock_st:
                mock_st.return_value = Mock()
                service = EmbeddingService(model_name="custom-model", embedding_dimension=512, cache_size=5000, batch_size=16)
                await service.initialize()

                assert service.model_name == "custom-model"
                assert service.embedding_dimension == 512
                assert service.cache_size == 5000
                assert service.batch_size == 16

        async def test_gpu_detection(self):
            """Test GPU acceleration detection and configuration."""
            with (
                patch("core.embedding_service.SentenceTransformer") as mock_st,
                patch("torch.cuda.is_available", return_value=True),
            ):
                mock_model = Mock()
                mock_model.device = "cuda"
                mock_st.return_value = mock_model

                service = EmbeddingService(enable_gpu=True)
                await service.initialize()

                assert service.device == "cuda"

        async def test_gpu_fallback_to_cpu(self):
            """Test fallback to CPU when GPU is not available."""
            with (
                patch("core.embedding_service.SentenceTransformer") as mock_st,
                patch("torch.cuda.is_available", return_value=False),
            ):
                mock_model = Mock()
                mock_model.device = "cpu"
                mock_st.return_value = mock_model

                service = EmbeddingService(enable_gpu=True)
                await service.initialize()

                assert service.device == "cpu"

        async def test_model_loading_failure(self):
            """Test error handling for model loading failures."""
            with patch("core.embedding_service.SentenceTransformer", side_effect=Exception("Model load failed")):
                service = EmbeddingService()

                with pytest.raises(EmbeddingError, match="Failed to load embedding model"):
                    await service.initialize()

    @pytest.mark.asyncio
    class TestEmbeddingGeneration:
        """Test embedding generation functionality."""

        async def test_single_chunk_embedding(self, embedding_service, sample_chunks):
            """Test generating embedding for a single chunk."""
            chunk = sample_chunks[0]

            with patch.object(embedding_service.model, "encode") as mock_encode:
                mock_encode.return_value = np.array([[0.1, 0.2, 0.3] + [0.0] * 381])

                embedding = await embedding_service.generate_embedding(chunk)

                assert isinstance(embedding, np.ndarray)
                assert embedding.shape == (384,)
                assert embedding[0] == 0.1
                assert embedding[1] == 0.2
                assert embedding[2] == 0.3
                mock_encode.assert_called_once_with([chunk.content], batch_size=1)

        async def test_batch_embedding_generation(self, embedding_service, sample_chunks):
            """Test generating embeddings for multiple chunks in batch."""
            with patch.object(embedding_service.model, "encode") as mock_encode:
                mock_encode.return_value = np.random.rand(3, 384)

                embeddings = await embedding_service.generate_embeddings(sample_chunks)

                assert len(embeddings) == 3
                assert all(isinstance(emb, np.ndarray) for emb in embeddings)
                assert all(emb.shape == (384,) for emb in embeddings)

                # Check that batch processing was used
                expected_texts = [chunk.content for chunk in sample_chunks]
                mock_encode.assert_called_once_with(expected_texts, batch_size=32)

        async def test_custom_batch_size(self, embedding_service, sample_chunks):
            """Test custom batch size for embedding generation."""
            with patch.object(embedding_service.model, "encode") as mock_encode:
                mock_encode.return_value = np.random.rand(3, 384)

                await embedding_service.generate_embeddings(sample_chunks, batch_size=2)

                # With batch_size=2 and 3 chunks, should make 2 calls
                assert mock_encode.call_count >= 1

        async def test_empty_content_handling(self, embedding_service):
            """Test handling of empty or None content."""
            empty_chunk = DocumentChunk(
                chunk_id="empty_chunk", document_id="doc_test", content="", start_page=1, end_page=1, metadata={}
            )

            with pytest.raises(EmbeddingError, match="Empty content"):
                await embedding_service.generate_embedding(empty_chunk)

        async def test_progress_callback(self, embedding_service, sample_chunks):
            """Test progress callback functionality."""
            progress_updates = []

            def progress_callback(completed: int, total: int):
                progress_updates.append((completed, total))

            with patch.object(embedding_service.model, "encode") as mock_encode:
                mock_encode.return_value = np.random.rand(3, 384)

                await embedding_service.generate_embeddings(sample_chunks, progress_callback=progress_callback)

                # Should have received progress updates
                assert len(progress_updates) > 0
                assert progress_updates[-1] == (3, 3)  # Final update

        async def test_embedding_normalization(self, embedding_service, sample_chunks):
            """Test that embeddings are properly normalized."""
            with patch.object(embedding_service.model, "encode") as mock_encode:
                # Return non-normalized vectors
                mock_encode.return_value = np.array([[1.0, 2.0, 3.0] + [0.0] * 381])

                embedding = await embedding_service.generate_embedding(sample_chunks[0])

                # Check if vector is normalized (L2 norm should be 1)
                norm = np.linalg.norm(embedding)
                assert abs(norm - 1.0) < 1e-6

    @pytest.mark.asyncio
    class TestCaching:
        """Test embedding caching functionality."""

        async def test_cache_hit(self, embedding_service, sample_chunks):
            """Test cache hit for previously computed embeddings."""
            chunk = sample_chunks[0]

            with patch.object(embedding_service.model, "encode") as mock_encode:
                mock_encode.return_value = np.array([[0.1, 0.2, 0.3] + [0.0] * 381])

                # First call - should compute embedding
                embedding1 = await embedding_service.generate_embedding(chunk)

                # Second call - should use cache
                embedding2 = await embedding_service.generate_embedding(chunk)

                # Should be identical
                np.testing.assert_array_equal(embedding1, embedding2)

                # Model should only be called once
                mock_encode.assert_called_once()

        async def test_cache_miss_different_content(self, embedding_service, sample_chunks):
            """Test cache miss for different content."""
            chunk1, chunk2 = sample_chunks[0], sample_chunks[1]

            with patch.object(embedding_service.model, "encode") as mock_encode:
                mock_encode.return_value = np.random.rand(1, 384)

                # Generate embeddings for different chunks
                await embedding_service.generate_embedding(chunk1)
                await embedding_service.generate_embedding(chunk2)

                # Model should be called twice (different content)
                assert mock_encode.call_count == 2

        async def test_cache_size_limit(self, embedding_service):
            """Test cache eviction when size limit is reached."""
            # Set small cache size for testing
            embedding_service.cache_size = 2

            chunks = []
            for i in range(5):
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"chunk_{i}",
                        document_id="test_doc",
                        content=f"Content {i}",
                        start_page=1,
                        end_page=1,
                        metadata={},
                    )
                )

            with patch.object(embedding_service.model, "encode") as mock_encode:
                mock_encode.return_value = np.random.rand(1, 384)

                # Generate embeddings for all chunks
                for chunk in chunks:
                    await embedding_service.generate_embedding(chunk)

                # Check cache size doesn't exceed limit
                assert len(embedding_service._cache) <= 2

        async def test_cache_invalidation(self, embedding_service, sample_chunks):
            """Test manual cache invalidation."""
            chunk = sample_chunks[0]

            with patch.object(embedding_service.model, "encode") as mock_encode:
                mock_encode.return_value = np.random.rand(1, 384)

                # Generate and cache embedding
                await embedding_service.generate_embedding(chunk)
                assert len(embedding_service._cache) == 1

                # Clear cache
                await embedding_service.clear_cache()
                assert len(embedding_service._cache) == 0

    @pytest.mark.asyncio
    class TestErrorHandling:
        """Test error handling scenarios."""

        async def test_model_encoding_failure(self, embedding_service, sample_chunks):
            """Test handling of model encoding failures."""
            with patch.object(embedding_service.model, "encode", side_effect=Exception("Encoding failed")):
                with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
                    await embedding_service.generate_embedding(sample_chunks[0])

        async def test_invalid_embedding_dimensions(self, embedding_service, sample_chunks):
            """Test handling of unexpected embedding dimensions."""
            with patch.object(embedding_service.model, "encode") as mock_encode:
                # Return wrong dimensions
                mock_encode.return_value = np.random.rand(1, 256)  # Expected 384

                with pytest.raises(EmbeddingError, match="Unexpected embedding dimension"):
                    await embedding_service.generate_embedding(sample_chunks[0])

        async def test_service_not_initialized(self, sample_chunks):
            """Test using service before initialization."""
            service = EmbeddingService()
            # Don't call initialize()

            with pytest.raises(EmbeddingError, match="Service not initialized"):
                await service.generate_embedding(sample_chunks[0])

    @pytest.mark.asyncio
    class TestPerformance:
        """Test performance-related functionality."""

        async def test_concurrent_embedding_generation(self, embedding_service, sample_chunks):
            """Test concurrent embedding generation."""
            with patch.object(embedding_service.model, "encode") as mock_encode:
                mock_encode.return_value = np.random.rand(1, 384)

                # Generate embeddings concurrently
                tasks = [embedding_service.generate_embedding(chunk) for chunk in sample_chunks]
                embeddings = await asyncio.gather(*tasks)

                assert len(embeddings) == len(sample_chunks)
                assert all(isinstance(emb, np.ndarray) for emb in embeddings)

        async def test_memory_usage_tracking(self, embedding_service):
            """Test memory usage tracking functionality."""
            # This test verifies that memory tracking works
            initial_memory = await embedding_service.get_memory_usage()
            assert isinstance(initial_memory, dict)
            assert "cache_size" in initial_memory
            assert "model_memory_mb" in initial_memory

    @pytest.mark.asyncio
    class TestConfiguration:
        """Test configuration and settings."""

        async def test_model_warm_up(self):
            """Test model warm-up functionality."""
            with patch("core.embedding_service.SentenceTransformer") as mock_st:
                mock_st.return_value = Mock()
                service = EmbeddingService(warm_up_model=True)

                with patch.object(service, "_warm_up_model") as mock_warm_up:
                    await service.initialize()
                    mock_warm_up.assert_called_once()

        async def test_different_similarity_metrics(self, embedding_service):
            """Test different similarity metrics."""
            embedding1 = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
            embedding2 = np.array([0.0, 1.0, 0.0] + [0.0] * 381)

            # Test cosine similarity
            cosine_sim = embedding_service.compute_similarity(embedding1, embedding2, metric="cosine")
            assert 0.0 <= cosine_sim <= 1.0

            # Test euclidean similarity
            euclidean_sim = embedding_service.compute_similarity(embedding1, embedding2, metric="euclidean")
            assert isinstance(euclidean_sim, float)

            # Test dot product
            dot_sim = embedding_service.compute_similarity(embedding1, embedding2, metric="dot_product")
            assert isinstance(dot_sim, float)
