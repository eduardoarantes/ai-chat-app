"""
Unit tests for the Vector Store service.

This test module covers:
- ChromaDB connection and collection management
- Vector storage and retrieval accuracy
- Similarity search algorithm validation
- Database persistence and recovery
- Bulk operation performance

Following TDD methodology - tests written first, then implementation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any, Optional
import asyncio
import tempfile
import shutil
from datetime import datetime

# Import the VectorStore (to be implemented)
from core.vector_store import VectorStore, VectorStoreError
from models.search import ChunkEmbedding, EmbeddingMetadata, SearchResult, DatabaseStats


class TestVectorStore:
    """Test suite for VectorStore component."""

    @pytest.fixture
    def sample_embeddings(self) -> List[ChunkEmbedding]:
        """Create sample chunk embeddings for testing."""
        embeddings = []
        for i in range(3):
            metadata = EmbeddingMetadata(
                model_name="test-model",
                dimension=384,
                generation_time=0.1
            )
            embeddings.append(ChunkEmbedding(
                chunk_id=f"chunk_{i}",
                document_id=f"doc_{i // 2}",  # 2 chunks per doc
                session_id="test_session",
                content_hash=f"hash_{i}",
                embedding_vector=[0.1 * i + j * 0.01 for j in range(384)],
                metadata=metadata
            ))
        return embeddings

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.list_collections.return_value = []
        return mock_client, mock_collection

    @pytest.fixture
    async def vector_store(self, temp_db_path, mock_chroma_client):
        """Create VectorStore instance with mocked ChromaDB."""
        mock_client, mock_collection = mock_chroma_client
        
        with patch('core.vector_store.chromadb.PersistentClient') as mock_persistent_client:
            mock_persistent_client.return_value = mock_client
            
            store = VectorStore(persistence_path=temp_db_path)
            await store.initialize()
            return store

    @pytest.mark.asyncio
    class TestInitialization:
        """Test vector store initialization and configuration."""

        async def test_default_initialization(self, temp_db_path):
            """Test initialization with default configuration."""
            with patch('core.vector_store.chromadb.PersistentClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                store = VectorStore(persistence_path=temp_db_path)
                await store.initialize()
                
                assert store.persistence_path == temp_db_path
                assert store._initialized
                mock_client_class.assert_called_once_with(path=temp_db_path)

        async def test_custom_configuration(self, temp_db_path):
            """Test initialization with custom configuration."""
            with patch('core.vector_store.chromadb.PersistentClient') as mock_client_class:
                mock_client_class.return_value = Mock()
                
                store = VectorStore(
                    persistence_path=temp_db_path,
                    collection_prefix="custom",
                    enable_backup=False
                )
                await store.initialize()
                
                assert store.collection_prefix == "custom"
                assert not store.enable_backup

        async def test_initialization_failure(self, temp_db_path):
            """Test error handling during initialization."""
            with patch('core.vector_store.chromadb.PersistentClient', side_effect=Exception("DB init failed")):
                store = VectorStore(persistence_path=temp_db_path)
                
                with pytest.raises(VectorStoreError, match="Failed to initialize"):
                    await store.initialize()

        async def test_double_initialization(self, vector_store):
            """Test that double initialization is handled gracefully."""
            # Should not raise an error
            await vector_store.initialize()
            assert vector_store._initialized

    @pytest.mark.asyncio
    class TestCollectionManagement:
        """Test collection creation and management."""

        async def test_get_collection_name(self, vector_store):
            """Test collection naming strategy."""
            collection_name = vector_store._get_collection_name("test_session")
            assert collection_name.startswith(vector_store.collection_prefix)
            assert "test_session" in collection_name

        async def test_create_collection_success(self, vector_store, mock_chroma_client):
            """Test successful collection creation."""
            mock_client, mock_collection = mock_chroma_client
            
            collection = await vector_store._get_or_create_collection("test_session")
            
            assert collection is not None
            mock_client.get_or_create_collection.assert_called()

        async def test_create_collection_with_metadata(self, vector_store, mock_chroma_client):
            """Test collection creation with custom metadata."""
            mock_client, mock_collection = mock_chroma_client
            
            await vector_store._get_or_create_collection(
                "test_session", 
                metadata={"created_by": "test"}
            )
            
            # Verify metadata was passed
            call_args = mock_client.get_or_create_collection.call_args
            assert call_args is not None

        async def test_list_collections(self, vector_store, mock_chroma_client):
            """Test listing all collections."""
            mock_client, mock_collection = mock_chroma_client
            mock_client.list_collections.return_value = [
                Mock(name="documents_session1"),
                Mock(name="documents_session2")
            ]
            
            collections = await vector_store.list_collections()
            
            assert len(collections) == 2
            mock_client.list_collections.assert_called_once()

    @pytest.mark.asyncio
    class TestVectorOperations:
        """Test vector storage and retrieval operations."""

        async def test_store_single_embedding(self, vector_store, sample_embeddings, mock_chroma_client):
            """Test storing a single embedding."""
            mock_client, mock_collection = mock_chroma_client
            embedding = sample_embeddings[0]
            
            result = await vector_store.store_embedding(embedding)
            
            assert result is True
            mock_collection.add.assert_called_once()
            
            # Verify the call arguments
            call_args = mock_collection.add.call_args
            assert call_args[1]['ids'] == [embedding.chunk_id]
            assert len(call_args[1]['embeddings']) == 1
            assert len(call_args[1]['metadatas']) == 1

        async def test_store_embeddings_batch(self, vector_store, sample_embeddings, mock_chroma_client):
            """Test storing multiple embeddings in batch."""
            mock_client, mock_collection = mock_chroma_client
            
            result = await vector_store.store_embeddings(sample_embeddings, "test_session")
            
            assert result is True
            mock_collection.add.assert_called_once()
            
            # Verify batch storage
            call_args = mock_collection.add.call_args
            assert len(call_args[1]['ids']) == 3
            assert len(call_args[1]['embeddings']) == 3
            assert len(call_args[1]['metadatas']) == 3

        async def test_store_embeddings_with_progress(self, vector_store, sample_embeddings, mock_chroma_client):
            """Test batch storage with progress callback."""
            mock_client, mock_collection = mock_chroma_client
            progress_updates = []
            
            def progress_callback(completed: int, total: int):
                progress_updates.append((completed, total))
            
            await vector_store.store_embeddings(
                sample_embeddings, 
                "test_session",
                progress_callback=progress_callback
            )
            
            assert len(progress_updates) > 0
            assert progress_updates[-1] == (3, 3)

        async def test_store_empty_embeddings_list(self, vector_store, mock_chroma_client):
            """Test storing empty embeddings list."""
            mock_client, mock_collection = mock_chroma_client
            
            result = await vector_store.store_embeddings([], "test_session")
            
            assert result is True
            mock_collection.add.assert_not_called()

        async def test_retrieve_embedding_by_id(self, vector_store, mock_chroma_client):
            """Test retrieving embedding by chunk ID."""
            mock_client, mock_collection = mock_chroma_client
            mock_collection.get.return_value = {
                'ids': ['chunk_1'],
                'embeddings': [[0.1] * 384],
                'metadatas': [{
                    'document_id': 'doc_1',
                    'session_id': 'test_session',
                    'content_hash': 'hash_1'
                }]
            }
            
            embedding = await vector_store.get_embedding("chunk_1", "test_session")
            
            assert embedding is not None
            assert embedding.chunk_id == "chunk_1"
            assert len(embedding.embedding_vector) == 384
            mock_collection.get.assert_called_once()

        async def test_retrieve_nonexistent_embedding(self, vector_store, mock_chroma_client):
            """Test retrieving non-existent embedding."""
            mock_client, mock_collection = mock_chroma_client
            mock_collection.get.return_value = {'ids': [], 'embeddings': [], 'metadatas': []}
            
            embedding = await vector_store.get_embedding("nonexistent", "test_session")
            
            assert embedding is None

    @pytest.mark.asyncio
    class TestSimilaritySearch:
        """Test similarity search functionality."""

        async def test_similarity_search_basic(self, vector_store, mock_chroma_client):
            """Test basic similarity search."""
            mock_client, mock_collection = mock_chroma_client
            query_vector = [0.1] * 384
            
            mock_collection.query.return_value = {
                'ids': [['chunk_1', 'chunk_2']],
                'distances': [[0.1, 0.2]],
                'embeddings': [[[0.1] * 384, [0.2] * 384]],
                'metadatas': [[
                    {'document_id': 'doc_1', 'session_id': 'test_session'},
                    {'document_id': 'doc_2', 'session_id': 'test_session'}
                ]]
            }
            
            results = await vector_store.similarity_search(
                query_vector, "test_session", limit=10
            )
            
            assert len(results) == 2
            assert all(isinstance(result, SearchResult) for result in results)
            assert results[0].similarity_score > results[1].similarity_score  # Should be sorted
            mock_collection.query.assert_called_once()

        async def test_similarity_search_with_filters(self, vector_store, mock_chroma_client):
            """Test similarity search with metadata filters."""
            mock_client, mock_collection = mock_chroma_client
            query_vector = [0.1] * 384
            filters = {"document_id": "doc_1"}
            
            mock_collection.query.return_value = {
                'ids': [['chunk_1']], 
                'distances': [[0.1]],
                'embeddings': [[[0.1] * 384]],
                'metadatas': [[{'document_id': 'doc_1', 'session_id': 'test_session'}]]
            }
            
            results = await vector_store.similarity_search(
                query_vector, "test_session", filters=filters
            )
            
            assert len(results) == 1
            # Verify filters were applied
            call_args = mock_collection.query.call_args
            assert 'where' in call_args[1]

        async def test_similarity_search_threshold(self, vector_store, mock_chroma_client):
            """Test similarity search with minimum threshold."""
            mock_client, mock_collection = mock_chroma_client
            query_vector = [0.1] * 384
            
            # Return results with varying distances (converted to similarities)
            mock_collection.query.return_value = {
                'ids': [['chunk_1', 'chunk_2', 'chunk_3']],
                'distances': [[0.1, 0.5, 0.8]],  # High distance = low similarity
                'embeddings': [[[0.1] * 384, [0.2] * 384, [0.3] * 384]],
                'metadatas': [[
                    {'document_id': 'doc_1', 'session_id': 'test_session'},
                    {'document_id': 'doc_2', 'session_id': 'test_session'},
                    {'document_id': 'doc_3', 'session_id': 'test_session'}
                ]]
            }
            
            results = await vector_store.similarity_search(
                query_vector, "test_session", min_similarity=0.6
            )
            
            # Should filter out low similarity results
            assert all(result.similarity_score >= 0.6 for result in results)

        async def test_similarity_search_empty_results(self, vector_store, mock_chroma_client):
            """Test similarity search with no matching results."""
            mock_client, mock_collection = mock_chroma_client
            query_vector = [0.1] * 384
            
            mock_collection.query.return_value = {
                'ids': [[]], 'distances': [[]], 'embeddings': [[]], 'metadatas': [[]]
            }
            
            results = await vector_store.similarity_search(query_vector, "test_session")
            
            assert len(results) == 0

    @pytest.mark.asyncio
    class TestUpdateAndDelete:
        """Test update and delete operations."""

        async def test_update_embedding(self, vector_store, sample_embeddings, mock_chroma_client):
            """Test updating an existing embedding."""
            mock_client, mock_collection = mock_chroma_client
            embedding = sample_embeddings[0]
            new_vector = [0.2] * 384
            
            result = await vector_store.update_embedding(
                embedding.chunk_id, "test_session", new_vector
            )
            
            assert result is True
            mock_collection.update.assert_called_once()

        async def test_delete_embedding(self, vector_store, mock_chroma_client):
            """Test deleting a single embedding."""
            mock_client, mock_collection = mock_chroma_client
            
            result = await vector_store.delete_embedding("chunk_1", "test_session")
            
            assert result is True
            mock_collection.delete.assert_called_once_with(ids=["chunk_1"])

        async def test_delete_embeddings_batch(self, vector_store, mock_chroma_client):
            """Test deleting multiple embeddings."""
            mock_client, mock_collection = mock_chroma_client
            chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
            
            result = await vector_store.delete_embeddings(chunk_ids, "test_session")
            
            assert result == 3
            mock_collection.delete.assert_called_once_with(ids=chunk_ids)

        async def test_delete_document_embeddings(self, vector_store, mock_chroma_client):
            """Test deleting all embeddings for a document."""
            mock_client, mock_collection = mock_chroma_client
            
            # Mock get operation to find embeddings for document
            mock_collection.get.return_value = {
                'ids': ['chunk_1', 'chunk_2'],
                'embeddings': [[0.1] * 384, [0.2] * 384],
                'metadatas': [
                    {'document_id': 'doc_1', 'session_id': 'test_session'},
                    {'document_id': 'doc_1', 'session_id': 'test_session'}
                ]
            }
            
            count = await vector_store.delete_document_embeddings("doc_1", "test_session")
            
            assert count == 2
            mock_collection.delete.assert_called_once()

    @pytest.mark.asyncio
    class TestDatabaseManagement:
        """Test database management operations."""

        async def test_get_database_stats(self, vector_store, mock_chroma_client):
            """Test retrieving database statistics."""
            mock_client, mock_collection = mock_chroma_client
            mock_collection.count.return_value = 100
            mock_client.list_collections.return_value = [Mock(name="col1"), Mock(name="col2")]
            
            stats = await vector_store.get_database_stats()
            
            assert isinstance(stats, DatabaseStats)
            assert stats.total_collections == 2
            mock_client.list_collections.assert_called_once()

        async def test_health_check(self, vector_store, mock_chroma_client):
            """Test database health check."""
            mock_client, mock_collection = mock_chroma_client
            
            health = await vector_store.health_check()
            
            assert isinstance(health, dict)
            assert "status" in health
            assert health["status"] in ["healthy", "degraded", "unhealthy"]

        async def test_backup_operations(self, vector_store, temp_db_path):
            """Test backup creation and restoration."""
            # This would test actual backup functionality
            # For now, just test the interface exists
            backup_path = await vector_store.create_backup()
            assert isinstance(backup_path, str)
            
            backups = await vector_store.list_backups()
            assert isinstance(backups, list)

    @pytest.mark.asyncio
    class TestErrorHandling:
        """Test error handling scenarios."""

        async def test_storage_operation_failure(self, vector_store, sample_embeddings, mock_chroma_client):
            """Test handling of storage operation failures."""
            mock_client, mock_collection = mock_chroma_client
            mock_collection.add.side_effect = Exception("Storage failed")
            
            with pytest.raises(VectorStoreError, match="Failed to store"):
                await vector_store.store_embedding(sample_embeddings[0])

        async def test_search_operation_failure(self, vector_store, mock_chroma_client):
            """Test handling of search operation failures."""
            mock_client, mock_collection = mock_chroma_client
            mock_collection.query.side_effect = Exception("Search failed")
            
            query_vector = [0.1] * 384
            with pytest.raises(VectorStoreError, match="Similarity search failed"):
                await vector_store.similarity_search(query_vector, "test_session")

        async def test_invalid_embedding_dimensions(self, vector_store, mock_chroma_client):
            """Test handling of invalid embedding dimensions."""
            mock_client, mock_collection = mock_chroma_client
            
            invalid_embedding = ChunkEmbedding(
                chunk_id="invalid",
                document_id="doc_1",
                session_id="test_session",
                content_hash="hash_invalid",
                embedding_vector=[0.1] * 256,  # Wrong dimension
                metadata=EmbeddingMetadata(model_name="test", dimension=256)
            )
            
            # Should validate dimensions
            with pytest.raises(VectorStoreError, match="dimension"):
                await vector_store.store_embedding(invalid_embedding)

        async def test_service_not_initialized(self, temp_db_path):
            """Test using service before initialization."""
            store = VectorStore(persistence_path=temp_db_path)
            # Don't call initialize()
            
            query_vector = [0.1] * 384
            with pytest.raises(VectorStoreError, match="not initialized"):
                await store.similarity_search(query_vector, "test_session")

    @pytest.mark.asyncio
    class TestPerformance:
        """Test performance-related functionality."""

        async def test_bulk_operations_performance(self, vector_store, mock_chroma_client):
            """Test performance of bulk operations."""
            mock_client, mock_collection = mock_chroma_client
            
            # Create large batch of embeddings
            large_batch = []
            for i in range(1000):
                metadata = EmbeddingMetadata(model_name="test", dimension=384)
                large_batch.append(ChunkEmbedding(
                    chunk_id=f"chunk_{i}",
                    document_id=f"doc_{i // 10}",
                    session_id="test_session",
                    content_hash=f"hash_{i}",
                    embedding_vector=[0.1 * i] * 384,
                    metadata=metadata
                ))
            
            # Should handle large batches without errors
            result = await vector_store.store_embeddings(large_batch, "test_session", batch_size=100)
            assert result is True

        async def test_concurrent_operations(self, vector_store, sample_embeddings, mock_chroma_client):
            """Test concurrent vector operations."""
            mock_client, mock_collection = mock_chroma_client
            
            # Perform concurrent storage operations
            tasks = [
                vector_store.store_embedding(embedding)
                for embedding in sample_embeddings
            ]
            results = await asyncio.gather(*tasks)
            
            assert all(result is True for result in results)

        async def test_memory_usage_monitoring(self, vector_store):
            """Test memory usage monitoring."""
            # This would test memory usage tracking
            # For now, just verify the interface exists
            memory_info = await vector_store.get_memory_usage()
            assert isinstance(memory_info, dict)