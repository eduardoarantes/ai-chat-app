"""
VectorStore service for ChromaDB integration and vector operations.

This service handles:
- ChromaDB client initialization and collection management
- Vector storage, retrieval, and CRUD operations
- Similarity search with configurable metrics and filters
- Database persistence, backup, and health monitoring
- Performance optimization for bulk operations
"""

import asyncio
import logging
import time
import os
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any, Tuple
from pathlib import Path
import json

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
import numpy as np

from models.search import ChunkEmbedding, EmbeddingMetadata, SearchResult, DatabaseStats
from core.config import get_config

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""
    pass


class VectorStore:
    """Service for vector storage and retrieval using ChromaDB."""

    def __init__(
        self,
        persistence_path: str = "./data/vector_db",
        collection_prefix: str = "documents",
        enable_backup: bool = True,
        backup_interval_hours: int = 24,
        max_backup_files: int = 7,
        expected_dimension: int = 384
    ):
        """Initialize the vector store.
        
        Args:
            persistence_path: Path for persistent storage
            collection_prefix: Prefix for collection names
            enable_backup: Whether to enable automatic backups
            backup_interval_hours: Hours between automatic backups
            max_backup_files: Maximum number of backup files to keep
            expected_dimension: Expected embedding vector dimension
        """
        self.persistence_path = persistence_path
        self.collection_prefix = collection_prefix
        self.enable_backup = enable_backup
        self.backup_interval_hours = backup_interval_hours
        self.max_backup_files = max_backup_files
        self.expected_dimension = expected_dimension
        
        # State variables
        self.client: Optional[ClientAPI] = None
        self._collections: Dict[str, Collection] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        
        # Performance tracking
        self._operation_count = 0
        self._last_backup = None

    async def initialize(self) -> None:
        """Initialize the vector store with ChromaDB client."""
        async with self._lock:
            if self._initialized:
                return
                
            try:
                logger.info(f"Initializing vector store at: {self.persistence_path}")
                start_time = time.time()
                
                # Ensure persistence directory exists
                Path(self.persistence_path).mkdir(parents=True, exist_ok=True)
                
                # Initialize ChromaDB client with persistence
                self.client = chromadb.PersistentClient(path=self.persistence_path)
                
                # Test connection
                collections = self.client.list_collections()
                logger.info(f"Connected to ChromaDB. Found {len(collections)} existing collections.")
                
                init_time = time.time() - start_time
                logger.info(f"Vector store initialized successfully in {init_time:.2f}s")
                
                self._initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                raise VectorStoreError(f"Failed to initialize vector store: {e}")

    def _check_initialized(self) -> None:
        """Check if the service is initialized."""
        if not self._initialized or self.client is None:
            raise VectorStoreError("Vector store not initialized. Call initialize() first.")

    def _get_collection_name(self, session_id: str) -> str:
        """Generate collection name for a session."""
        # Use session-based collections for better isolation
        safe_session_id = session_id.replace("-", "_")[:32]  # ChromaDB name limits
        return f"{self.collection_prefix}_{safe_session_id}"

    async def _get_or_create_collection(
        self, 
        session_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Collection:
        """Get or create a collection for the session."""
        collection_name = self._get_collection_name(session_id)
        
        # Check cache first
        if collection_name in self._collections:
            return self._collections[collection_name]
        
        try:
            # Create collection metadata
            collection_metadata = {
                "session_id": session_id,
                "created_at": datetime.utcnow().isoformat(),
                "expected_dimension": self.expected_dimension
            }
            if metadata:
                collection_metadata.update(metadata)
            
            # Get or create collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=collection_metadata
            )
            
            # Cache the collection
            self._collections[collection_name] = collection
            logger.debug(f"Got/created collection: {collection_name}")
            
            return collection
            
        except Exception as e:
            logger.error(f"Failed to get/create collection {collection_name}: {e}")
            raise VectorStoreError(f"Failed to get/create collection: {e}")

    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections in the database."""
        self._check_initialized()
        
        try:
            collections = self.client.list_collections()
            return [
                {
                    "name": col.name,
                    "id": col.id,
                    "metadata": col.metadata or {}
                }
                for col in collections
            ]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise VectorStoreError(f"Failed to list collections: {e}")

    async def store_embedding(self, embedding: ChunkEmbedding) -> bool:
        """Store a single embedding in the vector database.
        
        Args:
            embedding: ChunkEmbedding to store
            
        Returns:
            True if successful
            
        Raises:
            VectorStoreError: If storage fails
        """
        self._check_initialized()
        
        # Validate embedding dimension
        if len(embedding.embedding_vector) != self.expected_dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.expected_dimension}, "
                f"got {len(embedding.embedding_vector)}"
            )
        
        try:
            collection = await self._get_or_create_collection(embedding.session_id)
            
            # Prepare data for ChromaDB
            metadata = {
                "document_id": embedding.document_id,
                "session_id": embedding.session_id,
                "content_hash": embedding.content_hash,
                "created_at": embedding.metadata.created_at.isoformat(),
                "model_name": embedding.metadata.model_name,
                "generation_time": embedding.metadata.generation_time
            }
            
            # Store in ChromaDB
            collection.add(
                ids=[embedding.chunk_id],
                embeddings=[embedding.embedding_vector],
                metadatas=[metadata]
            )
            
            self._operation_count += 1
            logger.debug(f"Stored embedding for chunk: {embedding.chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embedding {embedding.chunk_id}: {e}")
            raise VectorStoreError(f"Failed to store embedding: {e}")

    async def store_embeddings(
        self,
        embeddings: List[ChunkEmbedding],
        session_id: str,
        batch_size: int = 100,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """Store multiple embeddings in batches.
        
        Args:
            embeddings: List of ChunkEmbedding objects to store
            session_id: Session ID for the embeddings
            batch_size: Batch size for bulk operations
            progress_callback: Optional progress callback
            
        Returns:
            True if successful
            
        Raises:
            VectorStoreError: If storage fails
        """
        self._check_initialized()
        
        if not embeddings:
            return True
        
        try:
            collection = await self._get_or_create_collection(session_id)
            
            # Process in batches
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                
                # Validate dimensions
                for embedding in batch:
                    if len(embedding.embedding_vector) != self.expected_dimension:
                        raise VectorStoreError(
                            f"Embedding dimension mismatch in batch: expected {self.expected_dimension}, "
                            f"got {len(embedding.embedding_vector)}"
                        )
                
                # Prepare batch data
                ids = [emb.chunk_id for emb in batch]
                vectors = [emb.embedding_vector for emb in batch]
                metadatas = []
                
                for emb in batch:
                    metadata = {
                        "document_id": emb.document_id,
                        "session_id": emb.session_id,
                        "content_hash": emb.content_hash,
                        "created_at": emb.metadata.created_at.isoformat(),
                        "model_name": emb.metadata.model_name,
                        "generation_time": emb.metadata.generation_time
                    }
                    metadatas.append(metadata)
                
                # Store batch in ChromaDB
                collection.add(
                    ids=ids,
                    embeddings=vectors,
                    metadatas=metadatas
                )
                
                self._operation_count += len(batch)
                
                # Progress callback
                if progress_callback:
                    progress_callback(min(i + batch_size, len(embeddings)), len(embeddings))
            
            logger.info(f"Stored {len(embeddings)} embeddings for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embeddings batch: {e}")
            raise VectorStoreError(f"Failed to store embeddings: {e}")

    async def get_embedding(self, chunk_id: str, session_id: str) -> Optional[ChunkEmbedding]:
        """Retrieve a single embedding by chunk ID.
        
        Args:
            chunk_id: Chunk ID to retrieve
            session_id: Session ID
            
        Returns:
            ChunkEmbedding if found, None otherwise
        """
        self._check_initialized()
        
        try:
            collection = await self._get_or_create_collection(session_id)
            
            # Query for specific chunk
            result = collection.get(ids=[chunk_id])
            
            if not result['ids'] or len(result['ids']) == 0:
                return None
            
            # Reconstruct ChunkEmbedding
            metadata_dict = result['metadatas'][0]
            embedding_metadata = EmbeddingMetadata(
                model_name=metadata_dict.get("model_name", "unknown"),
                dimension=len(result['embeddings'][0]),
                created_at=datetime.fromisoformat(metadata_dict.get("created_at", datetime.utcnow().isoformat())),
                generation_time=metadata_dict.get("generation_time", 0.0)
            )
            
            return ChunkEmbedding(
                chunk_id=result['ids'][0],
                document_id=metadata_dict["document_id"],
                session_id=metadata_dict["session_id"],
                content_hash=metadata_dict["content_hash"],
                embedding_vector=result['embeddings'][0],
                metadata=embedding_metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve embedding {chunk_id}: {e}")
            return None

    async def similarity_search(
        self,
        query_vector: List[float],
        session_id: str,
        limit: int = 10,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform similarity search.
        
        Args:
            query_vector: Query embedding vector
            session_id: Session ID to search within
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            filters: Optional metadata filters
            
        Returns:
            List of SearchResult objects sorted by similarity
            
        Raises:
            VectorStoreError: If search fails
        """
        self._check_initialized()
        
        if len(query_vector) != self.expected_dimension:
            raise VectorStoreError(
                f"Query vector dimension mismatch: expected {self.expected_dimension}, "
                f"got {len(query_vector)}"
            )
        
        try:
            collection = await self._get_or_create_collection(session_id)
            
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_vector],
                "n_results": limit,
                "include": ["distances", "embeddings", "metadatas"]
            }
            
            # Add filters if provided
            if filters:
                query_params["where"] = filters
            
            # Execute similarity search
            results = collection.query(**query_params)
            
            # Process results
            search_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                ids = results['ids'][0]
                distances = results['distances'][0]
                metadatas = results['metadatas'][0]
                
                for i, (chunk_id, distance, metadata) in enumerate(zip(ids, distances, metadatas)):
                    # Convert distance to similarity (ChromaDB uses cosine distance)
                    # For cosine distance: similarity = 1 - distance
                    similarity = max(0.0, 1.0 - distance)
                    
                    # Apply similarity threshold
                    if similarity < min_similarity:
                        continue
                    
                    # Create SearchResult
                    search_result = SearchResult(
                        chunk_id=chunk_id,
                        document_id=metadata["document_id"],
                        session_id=metadata["session_id"],
                        content="",  # Content would need to be retrieved separately
                        similarity_score=similarity,
                        metadata=metadata
                    )
                    search_results.append(search_result)
            
            # Sort by similarity (highest first)
            search_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.debug(f"Similarity search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise VectorStoreError(f"Similarity search failed: {e}")

    async def update_embedding(
        self, 
        chunk_id: str, 
        session_id: str, 
        new_vector: List[float],
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing embedding.
        
        Args:
            chunk_id: Chunk ID to update
            session_id: Session ID
            new_vector: New embedding vector
            metadata_updates: Optional metadata updates
            
        Returns:
            True if successful
        """
        self._check_initialized()
        
        try:
            collection = await self._get_or_create_collection(session_id)
            
            # Prepare update data
            update_data = {
                "ids": [chunk_id],
                "embeddings": [new_vector]
            }
            
            if metadata_updates:
                update_data["metadatas"] = [metadata_updates]
            
            # Update in ChromaDB
            collection.update(**update_data)
            
            logger.debug(f"Updated embedding for chunk: {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update embedding {chunk_id}: {e}")
            return False

    async def delete_embedding(self, chunk_id: str, session_id: str) -> bool:
        """Delete a single embedding.
        
        Args:
            chunk_id: Chunk ID to delete
            session_id: Session ID
            
        Returns:
            True if successful
        """
        self._check_initialized()
        
        try:
            collection = await self._get_or_create_collection(session_id)
            collection.delete(ids=[chunk_id])
            
            logger.debug(f"Deleted embedding for chunk: {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embedding {chunk_id}: {e}")
            return False

    async def delete_embeddings(self, chunk_ids: List[str], session_id: str) -> int:
        """Delete multiple embeddings.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            session_id: Session ID
            
        Returns:
            Number of embeddings deleted
        """
        self._check_initialized()
        
        if not chunk_ids:
            return 0
        
        try:
            collection = await self._get_or_create_collection(session_id)
            collection.delete(ids=chunk_ids)
            
            logger.debug(f"Deleted {len(chunk_ids)} embeddings")
            return len(chunk_ids)
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            return 0

    async def delete_document_embeddings(self, document_id: str, session_id: str) -> int:
        """Delete all embeddings for a document.
        
        Args:
            document_id: Document ID
            session_id: Session ID
            
        Returns:
            Number of embeddings deleted
        """
        self._check_initialized()
        
        try:
            collection = await self._get_or_create_collection(session_id)
            
            # First, find all embeddings for the document
            result = collection.get(where={"document_id": document_id})
            
            if not result['ids'] or len(result['ids']) == 0:
                return 0
            
            # Delete found embeddings
            chunk_ids = result['ids']
            collection.delete(ids=chunk_ids)
            
            logger.info(f"Deleted {len(chunk_ids)} embeddings for document {document_id}")
            return len(chunk_ids)
            
        except Exception as e:
            logger.error(f"Failed to delete document embeddings for {document_id}: {e}")
            return 0

    async def get_embeddings_by_document(
        self, 
        document_id: str, 
        session_id: str
    ) -> List[ChunkEmbedding]:
        """Get all embeddings for a document.
        
        Args:
            document_id: Document ID
            session_id: Session ID
            
        Returns:
            List of ChunkEmbedding objects for the document
        """
        self._check_initialized()
        
        try:
            collection = await self._get_or_create_collection(session_id)
            
            # Query for document embeddings
            result = collection.get(where={"document_id": document_id})
            
            if not result['ids'] or len(result['ids']) == 0:
                return []
            
            # Convert to ChunkEmbedding objects
            embeddings = []
            for i, chunk_id in enumerate(result['ids']):
                metadata_dict = result['metadatas'][i]
                embedding_metadata = EmbeddingMetadata(
                    model_name=metadata_dict.get("model_name", "unknown"),
                    dimension=len(result['embeddings'][i]),
                    created_at=datetime.fromisoformat(metadata_dict.get("created_at", datetime.utcnow().isoformat())),
                    generation_time=metadata_dict.get("generation_time", 0.0)
                )
                
                chunk_embedding = ChunkEmbedding(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    session_id=session_id,
                    content_hash=metadata_dict["content_hash"],
                    embedding_vector=result['embeddings'][i],
                    metadata=embedding_metadata
                )
                embeddings.append(chunk_embedding)
            
            logger.debug(f"Retrieved {len(embeddings)} embeddings for document {document_id}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get embeddings for document {document_id}: {e}")
            return []

    async def get_database_stats(self) -> DatabaseStats:
        """Get comprehensive database statistics.
        
        Returns:
            DatabaseStats object with database information
        """
        self._check_initialized()
        
        try:
            collections = self.client.list_collections()
            total_embeddings = 0
            total_documents = set()
            total_sessions = set()
            model_distribution = {}
            oldest_embedding = None
            newest_embedding = None
            
            # Calculate storage size
            storage_size_mb = 0.0
            if os.path.exists(self.persistence_path):
                for root, dirs, files in os.walk(self.persistence_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            storage_size_mb += os.path.getsize(file_path)
                storage_size_mb = storage_size_mb / (1024 * 1024)  # Convert to MB
            
            # Gather stats from each collection
            for collection_info in collections:
                try:
                    collection = self.client.get_collection(collection_info.name)
                    count = collection.count()
                    total_embeddings += count
                    
                    if count > 0:
                        # Get sample metadata for analysis
                        sample = collection.get(limit=min(count, 100))
                        for metadata in sample.get('metadatas', []):
                            if 'document_id' in metadata:
                                total_documents.add(metadata['document_id'])
                            if 'session_id' in metadata:
                                total_sessions.add(metadata['session_id'])
                            if 'model_name' in metadata:
                                model = metadata['model_name']
                                model_distribution[model] = model_distribution.get(model, 0) + 1
                            
                            # Track oldest/newest embeddings
                            if 'created_at' in metadata:
                                try:
                                    created_at = datetime.fromisoformat(metadata['created_at'])
                                    if oldest_embedding is None or created_at < oldest_embedding:
                                        oldest_embedding = created_at
                                    if newest_embedding is None or created_at > newest_embedding:
                                        newest_embedding = created_at
                                except:
                                    pass  # Skip invalid dates
                            
                except Exception as e:
                    logger.warning(f"Failed to get stats for collection {collection_info.name}: {e}")
                    continue
            
            return DatabaseStats(
                total_embeddings=total_embeddings,
                total_documents=len(total_documents),
                total_sessions=len(total_sessions),
                database_size_mb=round(storage_size_mb, 2),
                oldest_embedding=oldest_embedding,
                newest_embedding=newest_embedding,
                model_distribution=model_distribution
            )
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            raise VectorStoreError(f"Failed to get database stats: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive database health check.
        
        Returns:
            Dictionary containing health information
        """
        self._check_initialized()
        
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "collections": {},
            "storage": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # Test basic operations
            collections = self.client.list_collections()
            health["collections"]["total"] = len(collections)
            health["collections"]["accessible"] = 0
            
            # Test each collection
            for collection_info in collections:
                try:
                    collection = self.client.get_collection(collection_info.name)
                    count = collection.count()
                    health["collections"]["accessible"] += 1
                    health["collections"][collection_info.name] = {
                        "count": count,
                        "metadata": collection_info.metadata or {}
                    }
                except Exception as e:
                    health["errors"].append(f"Collection {collection_info.name}: {str(e)}")
            
            # Check storage
            if os.path.exists(self.persistence_path):
                total_size = sum(
                    os.path.getsize(os.path.join(root, file))
                    for root, dirs, files in os.walk(self.persistence_path)
                    for file in files
                    if os.path.exists(os.path.join(root, file))
                )
                health["storage"]["size_mb"] = round(total_size / (1024 * 1024), 2)
                health["storage"]["path"] = self.persistence_path
                health["storage"]["accessible"] = os.access(self.persistence_path, os.R_OK | os.W_OK)
            
            # Performance metrics
            health["performance"]["total_operations"] = self._operation_count
            health["performance"]["last_backup"] = (
                self._last_backup.isoformat() if self._last_backup else None
            )
            
            # Determine overall status
            if health["errors"]:
                if len(health["errors"]) > len(collections) / 2:
                    health["status"] = "unhealthy"
                else:
                    health["status"] = "degraded"
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["errors"].append(f"Health check failed: {str(e)}")
        
        return health

    async def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a backup of the database.
        
        Args:
            backup_name: Optional custom backup name
            
        Returns:
            Path to the created backup
        """
        self._check_initialized()
        
        try:
            # Generate backup name
            if not backup_name:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                backup_name = f"vector_db_backup_{timestamp}"
            
            backup_path = f"{self.persistence_path}_backups/{backup_name}"
            
            # Create backup directory
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Copy database files
            if os.path.exists(self.persistence_path):
                shutil.copytree(self.persistence_path, backup_path, dirs_exist_ok=True)
            
            # Create backup metadata
            metadata = {
                "created_at": datetime.utcnow().isoformat(),
                "source_path": self.persistence_path,
                "backup_name": backup_name,
                "collections": len(self.client.list_collections())
            }
            
            metadata_path = f"{backup_path}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self._last_backup = datetime.utcnow()
            logger.info(f"Created backup: {backup_path}")
            
            # Cleanup old backups
            await self._cleanup_old_backups()
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise VectorStoreError(f"Failed to create backup: {e}")

    async def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups.
        
        Returns:
            List of backup information dictionaries
        """
        backups = []
        backup_dir = f"{self.persistence_path}_backups"
        
        if not os.path.exists(backup_dir):
            return backups
        
        try:
            for item in os.listdir(backup_dir):
                if item.endswith("_metadata.json"):
                    metadata_path = os.path.join(backup_dir, item)
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        backup_path = os.path.join(backup_dir, metadata["backup_name"])
                        if os.path.exists(backup_path):
                            metadata["path"] = backup_path
                            metadata["size_mb"] = self._get_directory_size(backup_path)
                            backups.append(metadata)
                    except Exception as e:
                        logger.warning(f"Failed to read backup metadata {metadata_path}: {e}")
            
            # Sort by creation date (newest first)
            backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
        
        return backups

    def _get_directory_size(self, path: str) -> float:
        """Get directory size in MB."""
        total_size = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return round(total_size / (1024 * 1024), 2)

    async def _cleanup_old_backups(self) -> None:
        """Clean up old backup files beyond the maximum limit."""
        if self.max_backup_files <= 0:
            return
        
        try:
            backups = await self.list_backups()
            if len(backups) > self.max_backup_files:
                # Remove oldest backups
                backups_to_remove = backups[self.max_backup_files:]
                for backup in backups_to_remove:
                    backup_path = backup["path"]
                    metadata_path = f"{backup_path}_metadata.json"
                    
                    # Remove backup directory and metadata
                    if os.path.exists(backup_path):
                        shutil.rmtree(backup_path)
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                    
                    logger.info(f"Removed old backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        try:
            # Basic memory usage estimation
            collections_count = len(self._collections)
            storage_size = 0.0
            
            if os.path.exists(self.persistence_path):
                storage_size = self._get_directory_size(self.persistence_path)
            
            return {
                "collections_cached": collections_count,
                "storage_size_mb": storage_size,
                "operation_count": self._operation_count,
                "persistence_path": self.persistence_path
            }
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {"error": str(e)}