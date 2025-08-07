"""
EmbeddingService for generating and managing vector embeddings.

This service handles:
- Sentence Transformers model loading and management
- Batch embedding generation with GPU acceleration
- LRU caching for embedding optimization
- Progress tracking for large document processing
- Error handling and fallback mechanisms
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import List, Dict, Optional, Callable, Any, Tuple
import numpy as np
import torch
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize

from models.session import DocumentChunk
from models.search import EmbeddingMetadata, ChunkEmbedding, SimilarityMetric
from core.config import get_config

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass


class EmbeddingService:
    """Service for generating and managing vector embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        cache_size: int = 10000,
        batch_size: int = 32,
        enable_gpu: bool = True,
        warm_up_model: bool = False,
        normalize_embeddings: bool = True
    ):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence transformer model
            embedding_dimension: Expected embedding dimension
            cache_size: Maximum number of embeddings to cache
            batch_size: Batch size for embedding generation
            enable_gpu: Whether to use GPU acceleration if available
            warm_up_model: Whether to warm up the model during initialization
            normalize_embeddings: Whether to normalize embeddings to unit vectors
        """
        self.model_name = model_name
        self.embedding_dimension = embedding_dimension
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.enable_gpu = enable_gpu
        self.warm_up_model = warm_up_model
        self.normalize_embeddings = normalize_embeddings
        
        # State variables
        self.model: Optional[SentenceTransformer] = None
        self.device: str = "cpu"
        self._cache: OrderedDict[str, Tuple[np.ndarray, EmbeddingMetadata]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._initialized = False
        
        # Performance tracking
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_generations = 0

    async def initialize(self) -> None:
        """Initialize the embedding service with model loading."""
        async with self._lock:
            if self._initialized:
                return
                
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                start_time = time.time()
                
                # Load model
                self.model = SentenceTransformer(self.model_name)
                
                # Configure device
                if self.enable_gpu and torch.cuda.is_available():
                    self.device = "cuda"
                    self.model = self.model.to(self.device)
                    logger.info(f"Using GPU acceleration: {torch.cuda.get_device_name()}")
                else:
                    self.device = "cpu"
                    logger.info("Using CPU for embeddings")
                
                # Verify model dimension
                test_embedding = self.model.encode(["test"], convert_to_numpy=True)
                actual_dimension = test_embedding.shape[1]
                if actual_dimension != self.embedding_dimension:
                    logger.warning(
                        f"Model dimension {actual_dimension} differs from expected {self.embedding_dimension}. "
                        f"Updating expected dimension."
                    )
                    self.embedding_dimension = actual_dimension
                
                # Warm up model if requested
                if self.warm_up_model:
                    await self._warm_up_model()
                
                load_time = time.time() - start_time
                logger.info(f"Model loaded successfully in {load_time:.2f}s")
                
                self._initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize embedding service: {e}")
                raise EmbeddingError(f"Failed to load embedding model: {e}")

    async def _warm_up_model(self) -> None:
        """Warm up the model with sample texts."""
        logger.info("Warming up embedding model...")
        warm_up_texts = [
            "This is a sample document for warming up the embedding model.",
            "Another example text to ensure the model is properly loaded.",
            "Final warm-up text to complete the initialization process."
        ]
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.model.encode, warm_up_texts
            )
            logger.info("Model warm-up completed")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def _check_initialized(self) -> None:
        """Check if the service is initialized."""
        if not self._initialized or self.model is None:
            raise EmbeddingError("Service not initialized. Call initialize() first.")

    def _generate_cache_key(self, chunk: DocumentChunk) -> str:
        """Generate a cache key for a document chunk."""
        content_hash = hashlib.sha256(chunk.content.encode('utf-8')).hexdigest()
        return f"{self.model_name}:{content_hash}"

    async def generate_embedding(self, chunk: DocumentChunk) -> np.ndarray:
        """Generate embedding for a single document chunk.
        
        Args:
            chunk: Document chunk to generate embedding for
            
        Returns:
            Numpy array representing the embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        self._check_initialized()
        
        # Validate input
        if not chunk.content or chunk.content.strip() == "":
            raise EmbeddingError("Empty content cannot be embedded")
        
        # Check cache
        cache_key = self._generate_cache_key(chunk)
        async with self._lock:
            if cache_key in self._cache:
                embedding, metadata = self._cache[cache_key]
                metadata.last_accessed = time.time()
                # Move to end (LRU)
                self._cache.move_to_end(cache_key)
                self._cache_hits += 1
                return embedding.copy()
        
        # Generate new embedding
        try:
            start_time = time.time()
            
            # Run embedding generation in thread pool to avoid blocking
            embedding_matrix = await asyncio.get_event_loop().run_in_executor(
                None, self.model.encode, [chunk.content], 1
            )
            
            generation_time = time.time() - start_time
            
            # Validate dimensions
            if embedding_matrix.shape[1] != self.embedding_dimension:
                raise EmbeddingError(
                    f"Unexpected embedding dimension: expected {self.embedding_dimension}, "
                    f"got {embedding_matrix.shape[1]}"
                )
            
            # Extract single embedding
            embedding = embedding_matrix[0]
            
            # Normalize if requested
            if self.normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
            
            # Create metadata
            metadata = EmbeddingMetadata(
                model_name=self.model_name,
                dimension=self.embedding_dimension,
                generation_time=generation_time,
                cache_hit=False,
                device=self.device
            )
            
            # Cache the result
            async with self._lock:
                # Ensure cache size limit
                while len(self._cache) >= self.cache_size:
                    self._cache.popitem(last=False)  # Remove oldest
                
                self._cache[cache_key] = (embedding.copy(), metadata)
                self._cache_misses += 1
                self._total_generations += 1
            
            logger.debug(f"Generated embedding for chunk {chunk.chunk_id} in {generation_time:.3f}s")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for chunk {chunk.chunk_id}: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")

    async def generate_embeddings(
        self,
        chunks: List[DocumentChunk],
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple chunks in batches.
        
        Args:
            chunks: List of document chunks to process
            batch_size: Batch size for processing (uses default if None)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If batch processing fails
        """
        self._check_initialized()
        
        if not chunks:
            return []
        
        batch_size = batch_size or self.batch_size
        embeddings = []
        
        try:
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Check cache for batch items
                batch_embeddings = []
                uncached_chunks = []
                uncached_indices = []
                
                for j, chunk in enumerate(batch):
                    cache_key = self._generate_cache_key(chunk)
                    async with self._lock:
                        if cache_key in self._cache:
                            embedding, metadata = self._cache[cache_key]
                            metadata.last_accessed = time.time()
                            self._cache.move_to_end(cache_key)
                            batch_embeddings.append((j, embedding.copy()))
                            self._cache_hits += 1
                        else:
                            uncached_chunks.append(chunk)
                            uncached_indices.append(j)
                
                # Generate embeddings for uncached chunks
                if uncached_chunks:
                    texts = [chunk.content for chunk in uncached_chunks]
                    start_time = time.time()
                    
                    embedding_matrix = await asyncio.get_event_loop().run_in_executor(
                        None, self.model.encode, texts, min(batch_size, len(texts))
                    )
                    
                    generation_time = time.time() - start_time
                    
                    # Validate dimensions
                    if embedding_matrix.shape[1] != self.embedding_dimension:
                        raise EmbeddingError(
                            f"Unexpected embedding dimension: expected {self.embedding_dimension}, "
                            f"got {embedding_matrix.shape[1]}"
                        )
                    
                    # Normalize if requested
                    if self.normalize_embeddings:
                        embedding_matrix = normalize(embedding_matrix, norm='l2')
                    
                    # Cache new embeddings and add to batch
                    async with self._lock:
                        for k, (chunk, embedding) in enumerate(zip(uncached_chunks, embedding_matrix)):
                            # Cache the embedding
                            cache_key = self._generate_cache_key(chunk)
                            metadata = EmbeddingMetadata(
                                model_name=self.model_name,
                                dimension=self.embedding_dimension,
                                generation_time=generation_time / len(uncached_chunks),
                                cache_hit=False,
                                device=self.device
                            )
                            
                            # Ensure cache size limit
                            while len(self._cache) >= self.cache_size:
                                self._cache.popitem(last=False)
                            
                            self._cache[cache_key] = (embedding.copy(), metadata)
                            
                            # Add to batch results
                            batch_embeddings.append((uncached_indices[k], embedding))
                        
                        self._cache_misses += len(uncached_chunks)
                        self._total_generations += len(uncached_chunks)
                
                # Sort batch embeddings by original index and add to results
                batch_embeddings.sort(key=lambda x: x[0])
                embeddings.extend([emb for _, emb in batch_embeddings])
                
                # Call progress callback
                if progress_callback:
                    progress_callback(min(i + batch_size, len(chunks)), len(chunks))
            
            logger.info(f"Generated {len(embeddings)} embeddings for {len(chunks)} chunks")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}")

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric to use
            
        Returns:
            Similarity score
        """
        try:
            if metric == "cosine":
                # Cosine similarity (0 to 1 range)
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                return max(0.0, similarity)  # Ensure non-negative
            elif metric == "euclidean":
                # Convert euclidean distance to similarity (0 to 1 range)
                distance = euclidean_distances([embedding1], [embedding2])[0][0]
                # Convert distance to similarity using exp(-distance)
                return float(np.exp(-distance))
            elif metric == "dot_product":
                # Dot product similarity
                return float(np.dot(embedding1, embedding2))
            elif metric == "manhattan":
                # Manhattan distance converted to similarity
                distance = np.sum(np.abs(embedding1 - embedding2))
                return float(np.exp(-distance))
            else:
                raise ValueError(f"Unsupported similarity metric: {metric}")
                
        except Exception as e:
            logger.error(f"Failed to compute {metric} similarity: {e}")
            return 0.0

    async def clear_cache(self) -> None:
        """Clear the embedding cache."""
        async with self._lock:
            self._cache.clear()
            logger.info("Embedding cache cleared")

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Dictionary containing memory usage information
        """
        cache_size_mb = 0.0
        if self._cache:
            # Estimate cache memory usage (embedding vectors + metadata)
            embedding_bytes = len(self._cache) * self.embedding_dimension * 4  # 4 bytes per float32
            cache_size_mb = embedding_bytes / (1024 * 1024)
        
        model_memory_mb = 0.0
        if self.model is not None:
            # Estimate model memory (rough approximation)
            model_memory_mb = 100.0  # Base estimate for all-MiniLM-L6-v2
        
        return {
            "cache_size": len(self._cache),
            "cache_size_mb": round(cache_size_mb, 2),
            "model_memory_mb": round(model_memory_mb, 2),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_generations": self._total_generations,
            "hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0 else 0.0
            )
        }

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": self._cache_hits + self._cache_misses,
            "hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0 else 0.0
            ),
            "total_generations": self._total_generations
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "initialized": self._initialized
        }