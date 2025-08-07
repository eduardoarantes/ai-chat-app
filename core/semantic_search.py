"""
Semantic Search Service for processing search queries and ranking results.

This service handles:
- Query processing and expansion
- Semantic search operations using embeddings
- Result ranking and filtering algorithms
- Search result caching and optimization
- Search analytics and performance monitoring
"""

import asyncio
import hashlib
import logging
import re
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.embedding_service import EmbeddingService
from core.vector_store import VectorStore
from models.search import (
    SearchAnalytics,
    SearchResult,
    SearchType,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResult,
    SimilarityMetric,
)
from models.session import DocumentChunk

logger = logging.getLogger(__name__)


class SemanticSearchError(Exception):
    """Custom exception for semantic search operations."""

    pass


class SemanticSearchService:
    """Service for semantic search operations and query processing."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        cache_size: int = 1000,
        cache_ttl_minutes: int = 30,
        enable_query_expansion: bool = True,
        enable_deduplication: bool = True,
        similarity_threshold: float = 0.1,
    ):
        """Initialize the semantic search service.

        Args:
            embedding_service: EmbeddingService instance for query embeddings
            vector_store: VectorStore instance for similarity search
            cache_size: Maximum number of cached search results
            cache_ttl_minutes: Cache TTL in minutes
            enable_query_expansion: Whether to enable query expansion
            enable_deduplication: Whether to deduplicate similar results
            similarity_threshold: Minimum similarity threshold for deduplication
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.cache_size = cache_size
        self.cache_ttl_minutes = cache_ttl_minutes
        self.enable_query_expansion = enable_query_expansion
        self.enable_deduplication = enable_deduplication
        self.similarity_threshold = similarity_threshold

        # State variables
        self._cache: OrderedDict[str, Tuple[SemanticSearchResponse, datetime]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._initialized = False

        # Analytics tracking
        self._total_searches = 0
        self._cache_hits = 0
        self._search_times = []
        self._analytics_history: List[SearchAnalytics] = []

    async def initialize(self) -> None:
        """Initialize the semantic search service."""
        async with self._lock:
            if self._initialized:
                return

            try:
                logger.info("Initializing semantic search service")
                start_time = time.time()

                # Ensure dependencies are initialized
                if not self.embedding_service._initialized:
                    await self.embedding_service.initialize()

                if not self.vector_store._initialized:
                    await self.vector_store.initialize()

                # Validate configuration
                if self.cache_size <= 0:
                    raise SemanticSearchError("Cache size must be positive")

                if self.cache_ttl_minutes <= 0:
                    raise SemanticSearchError("Cache TTL must be positive")

                init_time = time.time() - start_time
                logger.info(f"Semantic search service initialized successfully in {init_time:.2f}s")

                self._initialized = True

            except Exception as e:
                logger.error(f"Failed to initialize semantic search service: {e}")
                raise SemanticSearchError(f"Failed to initialize semantic search service: {e}")

    def _check_initialized(self) -> None:
        """Check if the service is initialized."""
        if not self._initialized:
            raise SemanticSearchError("Semantic search service not initialized. Call initialize() first.")

    async def _process_query(self, query: str) -> str:
        """Process and normalize the search query.

        Args:
            query: Raw search query

        Returns:
            Processed query string

        Raises:
            SemanticSearchError: If query is invalid
        """
        if not query or not query.strip():
            raise SemanticSearchError("Query cannot be empty")

        # Basic text cleaning
        processed_query = query.strip()

        # Remove excessive whitespace
        processed_query = re.sub(r"\s+", " ", processed_query)

        # Basic normalization (preserve case for semantic meaning)
        processed_query = processed_query.replace("\n", " ").replace("\t", " ")

        return processed_query

    async def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms.

        Args:
            query: Original query string

        Returns:
            List of expanded query terms
        """
        if not self.enable_query_expansion:
            return [query]

        # Simple expansion - in production, this could use a thesaurus or ML model
        expanded_terms = [query]

        # Common abbreviations and synonyms
        expansions = {
            "ML": ["Machine Learning", "ML"],
            "AI": ["Artificial Intelligence", "AI"],
            "DL": ["Deep Learning", "DL"],
            "NN": ["Neural Networks", "NN"],
            "CNN": ["Convolutional Neural Networks", "CNN"],
            "RNN": ["Recurrent Neural Networks", "RNN"],
            "API": ["Application Programming Interface", "API"],
            "UI": ["User Interface", "UI"],
            "UX": ["User Experience", "UX"],
        }

        query_upper = query.upper()
        for abbrev, full_forms in expansions.items():
            if abbrev in query_upper:
                expanded_terms.extend(full_forms)

        return list(set(expanded_terms))  # Remove duplicates

    def _generate_cache_key(self, request: SemanticSearchRequest) -> str:
        """Generate a cache key for a search request.

        Args:
            request: Search request

        Returns:
            Cache key string
        """
        # Create a hash based on request parameters
        key_data = {
            "query": request.query,
            "session_id": request.session_id,
            "document_ids": sorted(request.document_ids or []),
            "limit": request.limit,
            "min_similarity": request.min_similarity,
            "search_type": request.search_type.value,
            "similarity_metric": request.similarity_metric.value,
            "boost_recent": request.boost_recent,
        }

        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()

    async def _get_cached_result(self, cache_key: str) -> Optional[SemanticSearchResponse]:
        """Get cached search result if valid.

        Args:
            cache_key: Cache key

        Returns:
            Cached response if valid, None otherwise
        """
        async with self._lock:
            if cache_key not in self._cache:
                return None

            response, cached_at = self._cache[cache_key]

            # Check TTL
            if datetime.utcnow() - cached_at > timedelta(minutes=self.cache_ttl_minutes):
                del self._cache[cache_key]
                return None

            # Move to end (LRU)
            self._cache.move_to_end(cache_key)
            self._cache_hits += 1

            return response

    async def _cache_result(self, cache_key: str, response: SemanticSearchResponse) -> None:
        """Cache a search result.

        Args:
            cache_key: Cache key
            response: Response to cache
        """
        async with self._lock:
            # Ensure cache size limit
            while len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)  # Remove oldest

            self._cache[cache_key] = (response, datetime.utcnow())

    def _validate_request(self, request: SemanticSearchRequest) -> None:
        """Validate search request parameters.

        Args:
            request: Search request to validate

        Raises:
            SemanticSearchError: If request is invalid
        """
        if request.limit <= 0:
            raise SemanticSearchError("Invalid limit: must be positive")

        if request.limit > 1000:
            raise SemanticSearchError("Invalid limit: maximum is 1000")

        if not (0.0 <= request.min_similarity <= 1.0):
            raise SemanticSearchError("Invalid similarity threshold: must be between 0.0 and 1.0")

    async def _rank_results(self, results: List[SearchResult], request: SemanticSearchRequest) -> List[SearchResult]:
        """Rank and filter search results.

        Args:
            results: Raw search results
            request: Original search request

        Returns:
            Ranked and filtered results
        """
        if not results:
            return results

        # Apply similarity threshold
        filtered_results = [result for result in results if result.similarity_score >= request.min_similarity]

        # Apply recency boost if requested
        if request.boost_recent:
            filtered_results = self._apply_recency_boost(filtered_results)

        # Apply deduplication if enabled
        if self.enable_deduplication:
            filtered_results = self._deduplicate_results(filtered_results)

        # Sort by similarity score (highest first)
        filtered_results.sort(key=lambda x: x.similarity_score, reverse=True)

        # Apply limit
        return filtered_results[: request.limit]

    def _apply_recency_boost(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply recency boost to search results.

        Args:
            results: Search results

        Returns:
            Results with recency boost applied
        """
        now = datetime.utcnow()

        for result in results:
            # Check if result has creation timestamp
            created_at_str = result.metadata.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    age_days = (now - created_at).days

                    # Apply exponential decay boost (more recent = higher boost)
                    recency_factor = np.exp(-age_days / 30.0)  # 30-day half-life
                    boosted_score = result.similarity_score * (1.0 + 0.2 * recency_factor)

                    # Ensure we don't exceed 1.0
                    result.similarity_score = min(1.0, boosted_score)

                except (ValueError, TypeError):
                    pass  # Skip invalid timestamps

        return results

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate or very similar results.

        Args:
            results: Search results

        Returns:
            Deduplicated results
        """
        if not results:
            return results

        deduplicated = []
        seen_content = set()

        for result in results:
            # Simple content-based deduplication
            content_lower = result.content.lower().strip()
            content_hash = hashlib.md5(content_lower.encode()).hexdigest()

            # Check for exact content matches
            if content_hash in seen_content:
                continue

            # Check for high similarity with existing results
            is_duplicate = False
            for existing in deduplicated:
                existing_content = existing.content.lower().strip()

                # Simple similarity check based on character overlap
                if self._content_similarity(content_lower, existing_content) > 0.9:
                    # Keep the result with higher similarity score
                    if result.similarity_score > existing.similarity_score:
                        deduplicated.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break

            if not is_duplicate:
                deduplicated.append(result)
                seen_content.add(content_hash)

        return deduplicated

    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity between two strings.

        Args:
            content1: First content string
            content2: Second content string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not content1 or not content2:
            return 0.0

        # Simple word-based similarity
        words1 = set(content1.split())
        words2 = set(content2.split())

        if not words1 and not words2:
            return 1.0

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    async def search(self, request: SemanticSearchRequest) -> SemanticSearchResponse:
        """Perform semantic search operation.

        Args:
            request: Semantic search request

        Returns:
            Search response with ranked results

        Raises:
            SemanticSearchError: If search fails
        """
        self._check_initialized()
        self._validate_request(request)

        start_time = time.time()
        embedding_start_time = start_time

        try:
            # Process query
            processed_query = await self._process_query(request.query)

            # Check cache
            cache_key = self._generate_cache_key(request)
            cached_response = await self._get_cached_result(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for query: {request.query}")
                return cached_response

            # Generate query embedding
            query_chunk = DocumentChunk(
                chunk_id="query_temp", document_id="query_temp", content=processed_query, start_page=0, end_page=0, metadata={}
            )

            query_embedding = await self.embedding_service.generate_embedding(query_chunk)
            embedding_time = (time.time() - embedding_start_time) * 1000  # Convert to ms

            # Perform vector similarity search
            search_start_time = time.time()

            # Prepare filters
            filters = {}
            if request.document_ids:
                filters["document_id"] = {"$in": request.document_ids}

            # Perform search
            search_results = await self.vector_store.similarity_search(
                query_vector=query_embedding.tolist(),
                session_id=request.session_id or "global",
                limit=request.limit * 2,  # Get more results for better ranking
                min_similarity=request.min_similarity,
                filters=filters if filters else None,
            )

            # Rank and filter results
            ranked_results = await self._rank_results(search_results, request)

            # Convert to response format
            semantic_results = []
            for result in ranked_results:
                semantic_result = SemanticSearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    document_name=result.metadata.get("document_name", "Unknown"),
                    chunk_content=result.content,
                    similarity_score=result.similarity_score,
                    chunk_metadata=result.metadata,
                    document_metadata=result.document_metadata,
                    embedding_metadata={},  # Could include embedding info if needed
                )
                semantic_results.append(semantic_result)

            search_time = (time.time() - search_start_time) * 1000  # Convert to ms
            total_time = (time.time() - start_time) * 1000

            # Create response
            response = SemanticSearchResponse(
                query=request.query,
                results=semantic_results,
                total_results=len(semantic_results),
                search_time_ms=search_time,
                embedding_time_ms=embedding_time,
                search_metadata={
                    "processed_query": processed_query,
                    "total_time_ms": total_time,
                    "cache_used": False,
                    "search_type": request.search_type.value,
                    "similarity_metric": request.similarity_metric.value,
                },
            )

            # Cache the result
            await self._cache_result(cache_key, response)

            # Record analytics
            await self._record_search_analytics(request, response)

            self._total_searches += 1
            self._search_times.append(total_time)

            logger.debug(
                f"Search completed: query='{request.query}', " f"results={len(semantic_results)}, time={total_time:.1f}ms"
            )

            return response

        except Exception as e:
            if "generate_embedding" in str(e) or "embedding" in str(e).lower():
                raise SemanticSearchError(f"Failed to generate query embedding: {e}")
            elif "similarity_search" in str(e) or "vector" in str(e).lower():
                raise SemanticSearchError(f"Vector similarity search failed: {e}")
            else:
                logger.error(f"Semantic search failed: {e}")
                raise SemanticSearchError(f"Search operation failed: {e}")

    async def find_similar_documents(
        self,
        document_id: str,
        session_id: str,
        limit: int = 5,
        min_similarity: float = 0.3,
        exclude_same_document: bool = True,
    ) -> List[SemanticSearchResult]:
        """Find documents similar to a reference document.

        Args:
            document_id: Reference document ID
            session_id: Session ID
            limit: Maximum number of similar documents
            min_similarity: Minimum similarity threshold
            exclude_same_document: Whether to exclude chunks from same document

        Returns:
            List of similar document chunks

        Raises:
            SemanticSearchError: If operation fails
        """
        self._check_initialized()

        try:
            # Get embeddings for the reference document
            reference_embeddings = await self.vector_store.get_embeddings_by_document(document_id, session_id)

            if not reference_embeddings:
                raise SemanticSearchError(f"No embeddings found for document {document_id}")

            # Use the average of document embeddings as query vector
            embedding_vectors = [emb.embedding_vector for emb in reference_embeddings]
            avg_embedding = np.mean(embedding_vectors, axis=0)

            # Perform similarity search
            search_results = await self.vector_store.similarity_search(
                query_vector=avg_embedding.tolist(),
                session_id=session_id,
                limit=limit * 2,  # Get more for filtering
                min_similarity=min_similarity,
            )

            # Filter out same document if requested
            if exclude_same_document:
                search_results = [result for result in search_results if result.document_id != document_id]

            # Convert to semantic search results
            similar_documents = []
            for result in search_results[:limit]:
                similar_doc = SemanticSearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    document_name=result.metadata.get("document_name", "Unknown"),
                    chunk_content=result.content,
                    similarity_score=result.similarity_score,
                    chunk_metadata=result.metadata,
                    document_metadata=result.document_metadata,
                )
                similar_documents.append(similar_doc)

            logger.debug(f"Found {len(similar_documents)} similar documents " f"for document {document_id}")

            return similar_documents

        except Exception as e:
            logger.error(f"Failed to find similar documents: {e}")
            raise SemanticSearchError(f"Failed to find similar documents: {e}")

    async def _record_search_analytics(self, request: SemanticSearchRequest, response: SemanticSearchResponse) -> None:
        """Record search analytics.

        Args:
            request: Original search request
            response: Search response
        """
        try:
            analytics = SearchAnalytics(
                query=request.query,
                timestamp=datetime.utcnow(),
                results_count=len(response.results),
                search_time_ms=response.search_time_ms,
                similarity_scores=[result.similarity_score for result in response.results],
                user_session=request.session_id,
                search_type=request.search_type,
            )

            # Keep limited history
            async with self._lock:
                self._analytics_history.append(analytics)
                if len(self._analytics_history) > 1000:
                    self._analytics_history = self._analytics_history[-1000:]

        except Exception as e:
            logger.warning(f"Failed to record search analytics: {e}")

    async def clear_cache(self) -> None:
        """Clear the search result cache."""
        async with self._lock:
            self._cache.clear()
            logger.info("Search cache cleared")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary containing cache statistics
        """
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self._cache_hits,
            "total_searches": self._total_searches,
            "cache_hit_rate": (self._cache_hits / self._total_searches if self._total_searches > 0 else 0.0),
        }

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information.

        Returns:
            Dictionary containing memory usage statistics
        """
        cache_size_mb = len(self._cache) * 0.01  # Rough estimate

        return {
            "cache_size": len(self._cache),
            "cache_size_mb": round(cache_size_mb, 2),
            "total_searches": self._total_searches,
            "analytics_history_size": len(self._analytics_history),
        }

    async def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics.

        Returns:
            Dictionary containing search statistics
        """
        cache_stats = await self.get_cache_stats()

        avg_search_time = sum(self._search_times) / len(self._search_times) if self._search_times else 0.0

        # Analyze recent searches
        recent_analytics = self._analytics_history[-100:] if self._analytics_history else []

        search_types_distribution = {}
        avg_results_per_search = 0.0
        avg_similarity_scores = []

        if recent_analytics:
            for analytics in recent_analytics:
                search_type = analytics.search_type.value
                search_types_distribution[search_type] = search_types_distribution.get(search_type, 0) + 1
                avg_similarity_scores.extend(analytics.similarity_scores)

            avg_results_per_search = sum(a.results_count for a in recent_analytics) / len(recent_analytics)

        return {
            **cache_stats,
            "average_search_time_ms": round(avg_search_time, 2),
            "average_results_per_search": round(avg_results_per_search, 1),
            "search_types_distribution": search_types_distribution,
            "average_similarity_score": (
                round(sum(avg_similarity_scores) / len(avg_similarity_scores), 3) if avg_similarity_scores else 0.0
            ),
            "total_analytics_recorded": len(self._analytics_history),
        }
