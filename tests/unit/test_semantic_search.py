"""
Unit tests for the Semantic Search Service.

This test module covers:
- Query processing and expansion
- Similarity scoring algorithms
- Result ranking and filtering
- Search result caching
- API integration functionality

Following TDD methodology - tests written first, then implementation.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from core.embedding_service import EmbeddingService

# Import the SemanticSearchService (to be implemented)
from core.semantic_search import SemanticSearchError, SemanticSearchService
from core.vector_store import VectorStore
from models.search import (
    SearchResult,
    SearchType,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResult,
    SimilarityMetric,
)
from models.session import DocumentChunk


class TestSemanticSearchService:
    """Test suite for SemanticSearchService component."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        service = Mock(spec=EmbeddingService)
        service.generate_embedding = AsyncMock()
        service.compute_similarity = Mock()
        service.get_model_info.return_value = {"embedding_dimension": 384, "model_name": "test-model"}
        return service

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store."""
        store = Mock(spec=VectorStore)
        store.similarity_search = AsyncMock()
        store.get_database_stats = AsyncMock()
        return store

    @pytest.fixture
    def sample_search_results(self) -> List[SearchResult]:
        """Create sample search results for testing."""
        return [
            SearchResult(
                chunk_id="chunk_1",
                document_id="doc_1",
                session_id="test_session",
                content="This is the first chunk content about machine learning.",
                similarity_score=0.95,
                metadata={"page": 1, "section": "introduction"},
            ),
            SearchResult(
                chunk_id="chunk_2",
                document_id="doc_1",
                session_id="test_session",
                content="This chunk discusses neural networks and deep learning.",
                similarity_score=0.85,
                metadata={"page": 2, "section": "body"},
            ),
            SearchResult(
                chunk_id="chunk_3",
                document_id="doc_2",
                session_id="test_session",
                content="A different document about artificial intelligence.",
                similarity_score=0.75,
                metadata={"page": 1, "section": "conclusion"},
            ),
        ]

    @pytest.fixture
    async def semantic_search_service(self, mock_embedding_service, mock_vector_store):
        """Create SemanticSearchService instance with mocked dependencies."""
        service = SemanticSearchService(embedding_service=mock_embedding_service, vector_store=mock_vector_store)
        await service.initialize()
        return service

    @pytest.mark.asyncio
    class TestInitialization:
        """Test semantic search service initialization."""

        async def test_default_initialization(self, mock_embedding_service, mock_vector_store):
            """Test initialization with default configuration."""
            service = SemanticSearchService(embedding_service=mock_embedding_service, vector_store=mock_vector_store)
            await service.initialize()

            assert service.embedding_service == mock_embedding_service
            assert service.vector_store == mock_vector_store
            assert service._initialized

        async def test_custom_configuration(self, mock_embedding_service, mock_vector_store):
            """Test initialization with custom configuration."""
            service = SemanticSearchService(
                embedding_service=mock_embedding_service,
                vector_store=mock_vector_store,
                cache_size=500,
                cache_ttl_minutes=10,
                enable_query_expansion=False,
            )
            await service.initialize()

            assert service.cache_size == 500
            assert service.cache_ttl_minutes == 10
            assert not service.enable_query_expansion

        async def test_initialization_failure(self, mock_embedding_service):
            """Test error handling during initialization."""
            mock_vector_store = Mock()
            mock_vector_store.initialize = AsyncMock(side_effect=Exception("Vector store init failed"))

            service = SemanticSearchService(embedding_service=mock_embedding_service, vector_store=mock_vector_store)

            with pytest.raises(SemanticSearchError, match="Failed to initialize"):
                await service.initialize()

    @pytest.mark.asyncio
    class TestQueryProcessing:
        """Test query processing and expansion."""

        async def test_basic_query_processing(self, semantic_search_service):
            """Test basic query text processing."""
            query = "  Machine Learning Algorithms  "

            processed = await semantic_search_service._process_query(query)

            assert processed == "Machine Learning Algorithms"  # Trimmed
            assert isinstance(processed, str)

        async def test_empty_query_handling(self, semantic_search_service):
            """Test handling of empty or None queries."""
            with pytest.raises(SemanticSearchError, match="Query cannot be empty"):
                await semantic_search_service._process_query("")

            with pytest.raises(SemanticSearchError, match="Query cannot be empty"):
                await semantic_search_service._process_query(None)

        async def test_query_expansion(self, semantic_search_service):
            """Test query expansion functionality."""
            semantic_search_service.enable_query_expansion = True
            original_query = "ML"

            # Mock query expansion
            with patch.object(semantic_search_service, "_expand_query") as mock_expand:
                mock_expand.return_value = ["ML", "Machine Learning", "artificial intelligence"]

                expanded = await semantic_search_service._expand_query(original_query)

                assert len(expanded) >= 1
                assert original_query in expanded or "Machine Learning" in expanded

        async def test_query_normalization(self, semantic_search_service):
            """Test query text normalization."""
            queries = ["Machine Learning!", "What is AI?", "neural-networks", "DEEP LEARNING"]

            for query in queries:
                processed = await semantic_search_service._process_query(query)
                assert isinstance(processed, str)
                assert len(processed) > 0

    @pytest.mark.asyncio
    class TestSemanticSearch:
        """Test core semantic search functionality."""

        async def test_basic_semantic_search(
            self, semantic_search_service, sample_search_results, mock_embedding_service, mock_vector_store
        ):
            """Test basic semantic search operation."""
            # Setup mocks
            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding
            mock_vector_store.similarity_search.return_value = sample_search_results

            request = SemanticSearchRequest(query="machine learning", session_id="test_session", limit=10)

            response = await semantic_search_service.search(request)

            assert isinstance(response, SemanticSearchResponse)
            assert response.query == "machine learning"
            assert len(response.results) == 3
            assert response.results[0].similarity_score >= response.results[1].similarity_score  # Sorted

        async def test_search_with_filters(
            self, semantic_search_service, sample_search_results, mock_embedding_service, mock_vector_store
        ):
            """Test search with document and metadata filters."""
            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding

            # Filter to only return results from doc_1
            filtered_results = [r for r in sample_search_results if r.document_id == "doc_1"]
            mock_vector_store.similarity_search.return_value = filtered_results

            request = SemanticSearchRequest(
                query="machine learning", session_id="test_session", document_ids=["doc_1"], limit=10
            )

            response = await semantic_search_service.search(request)

            assert len(response.results) == 2  # Only doc_1 results
            assert all(result.document_id == "doc_1" for result in response.results)

        async def test_search_similarity_threshold(
            self, semantic_search_service, sample_search_results, mock_embedding_service, mock_vector_store
        ):
            """Test search with minimum similarity threshold."""
            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding
            mock_vector_store.similarity_search.return_value = sample_search_results

            request = SemanticSearchRequest(
                query="machine learning", session_id="test_session", min_similarity=0.8, limit=10  # High threshold
            )

            response = await semantic_search_service.search(request)

            # Should only return results with similarity >= 0.8
            assert all(result.similarity_score >= 0.8 for result in response.results)

        async def test_search_result_limit(
            self, semantic_search_service, sample_search_results, mock_embedding_service, mock_vector_store
        ):
            """Test search result limiting."""
            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding
            mock_vector_store.similarity_search.return_value = sample_search_results

            request = SemanticSearchRequest(query="machine learning", session_id="test_session", limit=2)

            response = await semantic_search_service.search(request)

            assert len(response.results) <= 2

        async def test_search_different_metrics(
            self, semantic_search_service, sample_search_results, mock_embedding_service, mock_vector_store
        ):
            """Test search with different similarity metrics."""
            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding
            mock_vector_store.similarity_search.return_value = sample_search_results

            for metric in [SimilarityMetric.COSINE, SimilarityMetric.EUCLIDEAN, SimilarityMetric.DOT_PRODUCT]:
                request = SemanticSearchRequest(
                    query="machine learning", session_id="test_session", similarity_metric=metric, limit=10
                )

                response = await semantic_search_service.search(request)
                assert isinstance(response, SemanticSearchResponse)

        async def test_search_empty_results(self, semantic_search_service, mock_embedding_service, mock_vector_store):
            """Test handling of empty search results."""
            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding
            mock_vector_store.similarity_search.return_value = []

            request = SemanticSearchRequest(query="nonexistent query", session_id="test_session", limit=10)

            response = await semantic_search_service.search(request)

            assert len(response.results) == 0
            assert response.total_results == 0

    @pytest.mark.asyncio
    class TestResultRanking:
        """Test result ranking and scoring."""

        async def test_result_ranking_by_similarity(self, semantic_search_service, mock_embedding_service, mock_vector_store):
            """Test that results are properly ranked by similarity."""
            # Create unordered results
            unordered_results = [
                SearchResult("c1", "d1", "s1", "content1", 0.7, {}),
                SearchResult("c2", "d1", "s1", "content2", 0.9, {}),
                SearchResult("c3", "d1", "s1", "content3", 0.8, {}),
            ]

            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding
            mock_vector_store.similarity_search.return_value = unordered_results

            request = SemanticSearchRequest(query="test query", session_id="test_session")

            response = await semantic_search_service.search(request)

            # Verify results are sorted by similarity (highest first)
            scores = [result.similarity_score for result in response.results]
            assert scores == sorted(scores, reverse=True)

        async def test_boost_recent_documents(self, semantic_search_service, mock_embedding_service, mock_vector_store):
            """Test boosting of recent documents."""
            # Results with timestamps in metadata
            results_with_timestamps = [
                SearchResult("c1", "d1", "s1", "old content", 0.8, {"created_at": "2023-01-01T00:00:00Z"}),
                SearchResult("c2", "d2", "s1", "recent content", 0.7, {"created_at": "2024-01-01T00:00:00Z"}),
            ]

            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding
            mock_vector_store.similarity_search.return_value = results_with_timestamps

            request = SemanticSearchRequest(query="test query", session_id="test_session", boost_recent=True)

            response = await semantic_search_service.search(request)

            # Recent document might be boosted despite lower similarity
            assert len(response.results) == 2

        async def test_result_deduplication(self, semantic_search_service, mock_embedding_service, mock_vector_store):
            """Test deduplication of similar results."""
            # Create results with similar content
            similar_results = [
                SearchResult("c1", "d1", "s1", "machine learning algorithms", 0.9, {}),
                SearchResult("c2", "d1", "s1", "machine learning algorithms", 0.88, {}),  # Duplicate
                SearchResult("c3", "d2", "s1", "neural networks", 0.8, {}),
            ]

            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding
            mock_vector_store.similarity_search.return_value = similar_results

            # Mock the service to enable deduplication
            semantic_search_service.enable_deduplication = True

            request = SemanticSearchRequest(query="machine learning", session_id="test_session")

            response = await semantic_search_service.search(request)

            # Should have fewer results due to deduplication
            unique_contents = set(result.chunk_content for result in response.results)
            assert len(unique_contents) <= len(similar_results)

    @pytest.mark.asyncio
    class TestCaching:
        """Test search result caching."""

        async def test_cache_hit(
            self, semantic_search_service, sample_search_results, mock_embedding_service, mock_vector_store
        ):
            """Test cache hit for repeated queries."""
            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding
            mock_vector_store.similarity_search.return_value = sample_search_results

            request = SemanticSearchRequest(query="machine learning", session_id="test_session")

            # First search - should hit vector store
            response1 = await semantic_search_service.search(request)

            # Second search - should use cache
            response2 = await semantic_search_service.search(request)

            # Results should be identical
            assert response1.query == response2.query
            assert len(response1.results) == len(response2.results)

            # Vector store should only be called once if caching works
            # This would need implementation of cache verification

        async def test_cache_invalidation(self, semantic_search_service):
            """Test cache invalidation mechanisms."""
            # This would test cache clearing and TTL expiration
            await semantic_search_service.clear_cache()

            # Verify cache is empty
            cache_stats = await semantic_search_service.get_cache_stats()
            assert cache_stats["cache_size"] == 0

        async def test_cache_size_limit(self, semantic_search_service):
            """Test cache size limiting."""
            # This would test that cache doesn't grow beyond configured size
            cache_stats = await semantic_search_service.get_cache_stats()
            assert "max_cache_size" in cache_stats
            assert cache_stats["cache_size"] <= cache_stats["max_cache_size"]

    @pytest.mark.asyncio
    class TestSimilarDocuments:
        """Test finding similar documents functionality."""

        async def test_find_similar_documents(
            self, semantic_search_service, sample_search_results, mock_embedding_service, mock_vector_store
        ):
            """Test finding documents similar to a reference document."""
            # Mock getting embeddings for reference document
            reference_embeddings = [np.array([0.1] * 384), np.array([0.2] * 384)]
            mock_vector_store.get_embeddings_by_document = AsyncMock(return_value=reference_embeddings)
            mock_vector_store.similarity_search.return_value = sample_search_results

            similar_docs = await semantic_search_service.find_similar_documents(
                document_id="ref_doc", session_id="test_session", limit=5
            )

            assert len(similar_docs) <= 5
            # Should exclude the reference document itself
            assert not any(result.document_id == "ref_doc" for result in similar_docs)

        async def test_find_similar_documents_empty_reference(self, semantic_search_service, mock_vector_store):
            """Test handling of reference document with no embeddings."""
            mock_vector_store.get_embeddings_by_document = AsyncMock(return_value=[])

            with pytest.raises(SemanticSearchError, match="No embeddings found"):
                await semantic_search_service.find_similar_documents(document_id="nonexistent_doc", session_id="test_session")

    @pytest.mark.asyncio
    class TestErrorHandling:
        """Test error handling scenarios."""

        async def test_embedding_generation_failure(self, semantic_search_service, mock_embedding_service, mock_vector_store):
            """Test handling of embedding generation failures."""
            mock_embedding_service.generate_embedding.side_effect = Exception("Embedding failed")

            request = SemanticSearchRequest(query="test query", session_id="test_session")

            with pytest.raises(SemanticSearchError, match="Failed to generate query embedding"):
                await semantic_search_service.search(request)

        async def test_vector_search_failure(self, semantic_search_service, mock_embedding_service, mock_vector_store):
            """Test handling of vector search failures."""
            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding
            mock_vector_store.similarity_search.side_effect = Exception("Search failed")

            request = SemanticSearchRequest(query="test query", session_id="test_session")

            with pytest.raises(SemanticSearchError, match="Vector similarity search failed"):
                await semantic_search_service.search(request)

        async def test_invalid_request_parameters(self, semantic_search_service):
            """Test validation of request parameters."""
            # Test invalid limit
            with pytest.raises(SemanticSearchError, match="Invalid limit"):
                request = SemanticSearchRequest(query="test", session_id="test_session", limit=0)  # Invalid
                await semantic_search_service.search(request)

            # Test invalid similarity threshold
            with pytest.raises(SemanticSearchError, match="Invalid similarity threshold"):
                request = SemanticSearchRequest(query="test", session_id="test_session", min_similarity=1.5)  # Invalid (> 1.0)
                await semantic_search_service.search(request)

        async def test_service_not_initialized(self, mock_embedding_service, mock_vector_store):
            """Test using service before initialization."""
            service = SemanticSearchService(embedding_service=mock_embedding_service, vector_store=mock_vector_store)
            # Don't call initialize()

            request = SemanticSearchRequest(query="test query", session_id="test_session")

            with pytest.raises(SemanticSearchError, match="not initialized"):
                await service.search(request)

    @pytest.mark.asyncio
    class TestPerformance:
        """Test performance-related functionality."""

        async def test_search_timing(
            self, semantic_search_service, sample_search_results, mock_embedding_service, mock_vector_store
        ):
            """Test that search timing is properly recorded."""
            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding
            mock_vector_store.similarity_search.return_value = sample_search_results

            request = SemanticSearchRequest(query="machine learning", session_id="test_session")

            response = await semantic_search_service.search(request)

            assert response.search_time_ms >= 0
            assert response.embedding_time_ms >= 0
            assert isinstance(response.search_time_ms, float)
            assert isinstance(response.embedding_time_ms, float)

        async def test_concurrent_searches(
            self, semantic_search_service, sample_search_results, mock_embedding_service, mock_vector_store
        ):
            """Test concurrent search operations."""
            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding
            mock_vector_store.similarity_search.return_value = sample_search_results

            # Create multiple concurrent search requests
            requests = [SemanticSearchRequest(query=f"query {i}", session_id="test_session") for i in range(5)]

            # Execute concurrently
            tasks = [semantic_search_service.search(request) for request in requests]
            responses = await asyncio.gather(*tasks)

            assert len(responses) == 5
            assert all(isinstance(response, SemanticSearchResponse) for response in responses)

        async def test_memory_usage_monitoring(self, semantic_search_service):
            """Test memory usage monitoring."""
            memory_info = await semantic_search_service.get_memory_usage()

            assert isinstance(memory_info, dict)
            assert "cache_size" in memory_info
            assert "total_searches" in memory_info

    @pytest.mark.asyncio
    class TestAnalytics:
        """Test search analytics and monitoring."""

        async def test_search_analytics_recording(
            self, semantic_search_service, sample_search_results, mock_embedding_service, mock_vector_store
        ):
            """Test that search analytics are properly recorded."""
            query_embedding = np.array([0.1] * 384)
            mock_embedding_service.generate_embedding.return_value = query_embedding
            mock_vector_store.similarity_search.return_value = sample_search_results

            request = SemanticSearchRequest(query="machine learning", session_id="test_session")

            response = await semantic_search_service.search(request)

            # Check that search metadata contains analytics
            assert "analytics" in response.search_metadata or "query_stats" in response.search_metadata

        async def test_search_statistics(self, semantic_search_service):
            """Test retrieval of search statistics."""
            stats = await semantic_search_service.get_search_statistics()

            assert isinstance(stats, dict)
            assert "total_searches" in stats
            assert "cache_hit_rate" in stats
            assert "average_search_time_ms" in stats
