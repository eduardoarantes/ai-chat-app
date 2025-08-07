"""
FastAPI endpoints for vector embeddings and semantic search functionality.

This module provides API endpoints for:
- Semantic search operations
- Document similarity search
- Embedding generation and management
- Vector database statistics and health monitoring
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse

from core.embedding_service import EmbeddingError, EmbeddingService
from core.semantic_search import SemanticSearchError, SemanticSearchService
from core.session_document_manager import SessionDocumentManager
from core.vector_store import VectorStore, VectorStoreError
from models.search import (
    EmbeddingGenerationRequest,
    EmbeddingGenerationResponse,
    EmbeddingStatus,
    EmbeddingStatusResponse,
    SearchType,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SimilarDocumentsRequest,
    SimilarDocumentsResponse,
    SimilarityMetric,
    VectorDatabaseStats,
)
from models.session import DocumentChunk

logger = logging.getLogger(__name__)

# Global service instances (will be initialized in main.py)
embedding_service: Optional[EmbeddingService] = None
vector_store: Optional[VectorStore] = None
semantic_search_service: Optional[SemanticSearchService] = None
session_document_manager: Optional[SessionDocumentManager] = None


def initialize_vector_services(
    emb_service: EmbeddingService,
    vec_store: VectorStore,
    search_service: SemanticSearchService,
    doc_manager: SessionDocumentManager,
) -> None:
    """Initialize vector services for the API endpoints.

    Args:
        emb_service: EmbeddingService instance
        vec_store: VectorStore instance
        search_service: SemanticSearchService instance
        doc_manager: SessionDocumentManager instance
    """
    global embedding_service, vector_store, semantic_search_service, session_document_manager
    embedding_service = emb_service
    vector_store = vec_store
    semantic_search_service = search_service
    session_document_manager = doc_manager
    logger.info("Vector services initialized for API endpoints")


def _check_services_initialized() -> None:
    """Check that all required services are initialized."""
    if not embedding_service or not vector_store or not semantic_search_service:
        raise HTTPException(status_code=500, detail="Vector services not initialized")


async def semantic_search(request: SemanticSearchRequest) -> SemanticSearchResponse:
    """
    Perform semantic search across documents.

    This endpoint allows searching for content using natural language queries.
    The search uses vector embeddings to find semantically similar content.
    """
    _check_services_initialized()

    try:
        logger.info(f"Semantic search request: query='{request.query}', session_id={request.session_id}")

        # Perform semantic search
        response = await semantic_search_service.search(request)

        logger.info(f"Semantic search completed: {len(response.results)} results in {response.search_time_ms:.1f}ms")

        return response

    except SemanticSearchError as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in semantic search: {e}")
        raise HTTPException(status_code=500, detail="Internal search error")


async def find_similar_documents(request: SimilarDocumentsRequest) -> SimilarDocumentsResponse:
    """
    Find documents similar to a reference document.

    This endpoint finds other documents that are semantically similar to a given document,
    useful for content discovery and recommendation.
    """
    _check_services_initialized()

    try:
        start_time = time.time()

        logger.info(f"Finding similar documents for: {request.document_id}")

        # Find similar documents
        similar_docs = await semantic_search_service.find_similar_documents(
            document_id=request.document_id,
            session_id=request.session_id,
            limit=request.limit,
            min_similarity=request.min_similarity,
            exclude_same_document=request.exclude_same_document,
        )

        search_time = (time.time() - start_time) * 1000

        response = SimilarDocumentsResponse(
            reference_document_id=request.document_id, similar_documents=similar_docs, search_time_ms=search_time
        )

        logger.info(f"Found {len(similar_docs)} similar documents in {search_time:.1f}ms")
        return response

    except SemanticSearchError as e:
        logger.error(f"Similar document search failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in similar document search: {e}")
        raise HTTPException(status_code=500, detail="Internal search error")


async def generate_embeddings(
    request: EmbeddingGenerationRequest, background_tasks: BackgroundTasks
) -> EmbeddingGenerationResponse:
    """
    Generate embeddings for a document.

    This endpoint generates vector embeddings for all chunks of a document.
    Processing happens in the background with optional progress updates via SSE.
    """
    _check_services_initialized()

    try:
        logger.info(f"Embedding generation request for document: {request.document_id}")

        # Check if document exists
        if not session_document_manager:
            raise HTTPException(status_code=500, detail="Session manager not available")

        session_info = session_document_manager.get_session_info(request.session_id)
        if not session_info or request.document_id not in session_info.documents:
            raise HTTPException(status_code=404, detail="Document not found")

        document = session_info.documents[request.document_id]

        # Check if embeddings already exist and force_regenerate is False
        if not request.force_regenerate:
            # Check existing embeddings
            existing_embeddings = await vector_store.get_embeddings_by_document(request.document_id, request.session_id)
            if existing_embeddings:
                return EmbeddingGenerationResponse(
                    document_id=request.document_id,
                    status=EmbeddingStatus.COMPLETED,
                    embedding_count=len(existing_embeddings),
                    generation_time=0.0,
                    error_message=None,
                )

        # Schedule background embedding generation
        background_tasks.add_task(
            _generate_embeddings_background, request.document_id, request.session_id, document.chunks, request.batch_size or 32
        )

        return EmbeddingGenerationResponse(
            document_id=request.document_id,
            status=EmbeddingStatus.PROCESSING,
            embedding_count=0,
            generation_time=0.0,
            error_message=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation request failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to start embedding generation")


async def _generate_embeddings_background(
    document_id: str, session_id: str, chunks: List[DocumentChunk], batch_size: int
) -> None:
    """Background task for generating embeddings."""
    try:
        logger.info(f"Starting background embedding generation for document {document_id}")
        start_time = time.time()

        # Generate embeddings
        embeddings = await embedding_service.generate_embeddings(chunks=chunks, batch_size=batch_size)

        # Create ChunkEmbedding objects
        from models.search import ChunkEmbedding, EmbeddingMetadata

        chunk_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            metadata = EmbeddingMetadata(
                model_name=embedding_service.model_name, dimension=len(embedding), generation_time=time.time() - start_time
            )

            chunk_embedding = ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                document_id=document_id,
                session_id=session_id,
                content_hash=chunk.metadata.get("content_hash", ""),
                embedding_vector=embedding.tolist(),
                metadata=metadata,
            )
            chunk_embeddings.append(chunk_embedding)

        # Store embeddings
        await vector_store.store_embeddings(chunk_embeddings, session_id)

        generation_time = time.time() - start_time
        logger.info(
            f"Background embedding generation completed for document {document_id}: "
            f"{len(embeddings)} embeddings in {generation_time:.2f}s"
        )

    except Exception as e:
        logger.error(f"Background embedding generation failed for document {document_id}: {e}")


async def generate_embeddings_stream(
    document_id: str = Path(..., description="Document ID"),
    session_id: str = Query(..., description="Session ID"),
    force_regenerate: bool = Query(False, description="Force regeneration"),
    batch_size: int = Query(32, description="Batch size for processing"),
) -> EventSourceResponse:
    """
    Generate embeddings with progress streaming via Server-Sent Events.

    This endpoint provides real-time progress updates during embedding generation.
    """
    _check_services_initialized()

    async def generate_with_progress():
        try:
            # Validate document exists
            if not session_document_manager:
                yield f"event: error\ndata: Session manager not available\n\n"
                return

            session_info = session_document_manager.get_session_info(session_id)
            if not session_info or document_id not in session_info.documents:
                yield f"event: error\ndata: Document not found\n\n"
                return

            document = session_info.documents[document_id]
            chunks = document.chunks

            yield f"event: start\ndata: Starting embedding generation for {len(chunks)} chunks\n\n"

            start_time = time.time()
            completed = 0

            # Progress callback
            def progress_callback(current: int, total: int):
                nonlocal completed
                completed = current
                progress = (current / total) * 100
                yield f"event: progress\ndata: {progress:.1f}% ({current}/{total})\n\n"

            # Generate embeddings
            embeddings = await embedding_service.generate_embeddings(
                chunks=chunks, batch_size=batch_size, progress_callback=progress_callback
            )

            # Create and store ChunkEmbedding objects
            from models.search import ChunkEmbedding, EmbeddingMetadata

            chunk_embeddings = []
            for chunk, embedding in zip(chunks, embeddings):
                metadata = EmbeddingMetadata(
                    model_name=embedding_service.model_name, dimension=len(embedding), generation_time=time.time() - start_time
                )

                chunk_embedding = ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    document_id=document_id,
                    session_id=session_id,
                    content_hash=chunk.metadata.get("content_hash", ""),
                    embedding_vector=embedding.tolist(),
                    metadata=metadata,
                )
                chunk_embeddings.append(chunk_embedding)

            await vector_store.store_embeddings(chunk_embeddings, session_id)

            generation_time = time.time() - start_time

            yield f"event: complete\ndata: Generated {len(embeddings)} embeddings in {generation_time:.2f}s\n\n"

        except Exception as e:
            logger.error(f"Streaming embedding generation failed: {e}")
            yield f"event: error\ndata: {str(e)}\n\n"

    return EventSourceResponse(generate_with_progress())


async def get_embedding_status(
    document_id: str = Path(..., description="Document ID"), session_id: str = Query(..., description="Session ID")
) -> EmbeddingStatusResponse:
    """
    Get the embedding generation status for a document.

    Returns information about whether embeddings exist and their status.
    """
    _check_services_initialized()

    try:
        # Get existing embeddings
        embeddings = await vector_store.get_embeddings_by_document(document_id, session_id)

        if not embeddings:
            # Check if document exists
            if session_document_manager:
                session_info = session_document_manager.get_session_info(session_id)
                if not session_info or document_id not in session_info.documents:
                    raise HTTPException(status_code=404, detail="Document not found")

            return EmbeddingStatusResponse(
                document_id=document_id,
                status=EmbeddingStatus.PENDING,
                embedding_count=0,
                failed_chunks=0,
                model_name="",
                last_updated=datetime.utcnow(),
                progress_percentage=0.0,
            )

        # Get document info for progress calculation
        total_chunks = 0
        if session_document_manager:
            session_info = session_document_manager.get_session_info(session_id)
            if session_info and document_id in session_info.documents:
                total_chunks = len(session_info.documents[document_id].chunks)

        progress_percentage = (len(embeddings) / total_chunks) * 100.0 if total_chunks > 0 else 100.0

        return EmbeddingStatusResponse(
            document_id=document_id,
            status=EmbeddingStatus.COMPLETED,
            embedding_count=len(embeddings),
            failed_chunks=0,  # Would need to track this in real implementation
            model_name=embeddings[0].metadata.model_name if embeddings else "",
            last_updated=embeddings[0].metadata.created_at if embeddings else datetime.utcnow(),
            progress_percentage=progress_percentage,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get embedding status for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get embedding status")


async def regenerate_embeddings(
    background_tasks: BackgroundTasks,
    document_id: str = Path(..., description="Document ID"),
    session_id: str = Query(..., description="Session ID"),
) -> EmbeddingGenerationResponse:
    """
    Regenerate embeddings for a document.

    Forces regeneration of embeddings even if they already exist.
    """
    request = EmbeddingGenerationRequest(document_id=document_id, session_id=session_id, force_regenerate=True)

    return await generate_embeddings(request, background_tasks)


async def get_vector_database_stats() -> VectorDatabaseStats:
    """
    Get comprehensive vector database statistics.

    Returns information about the database size, number of embeddings,
    and other performance metrics.
    """
    _check_services_initialized()

    try:
        stats = await vector_store.get_database_stats()

        # Convert to API response model
        return VectorDatabaseStats(
            total_embeddings=stats.total_embeddings,
            total_documents=stats.total_documents,
            total_sessions=stats.total_sessions,
            database_size_mb=stats.database_size_mb,
            oldest_embedding=stats.oldest_embedding,
            newest_embedding=stats.newest_embedding,
            model_distribution=stats.model_distribution,
        )

    except Exception as e:
        logger.error(f"Failed to get vector database stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get database statistics")


async def get_vector_database_health() -> Dict[str, Any]:
    """
    Get vector database health status.

    Returns health information including status, errors, and performance metrics.
    """
    _check_services_initialized()

    try:
        health = await vector_store.health_check()
        return health

    except Exception as e:
        logger.error(f"Vector database health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}


async def get_search_analytics() -> Dict[str, Any]:
    """
    Get semantic search analytics and performance statistics.

    Returns information about search performance, cache hit rates, and usage patterns.
    """
    _check_services_initialized()

    try:
        # Get analytics from semantic search service
        search_stats = await semantic_search_service.get_search_statistics()

        # Get cache statistics
        cache_stats = await semantic_search_service.get_cache_stats()

        # Get embedding service statistics
        embedding_stats = await embedding_service.get_cache_stats()

        return {
            "search_statistics": search_stats,
            "search_cache": cache_stats,
            "embedding_cache": embedding_stats,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get search analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get search analytics")


async def clear_search_cache() -> JSONResponse:
    """
    Clear the semantic search cache.

    Clears both search result cache and embedding cache.
    """
    _check_services_initialized()

    try:
        await semantic_search_service.clear_cache()
        await embedding_service.clear_cache()

        return JSONResponse({"message": "Search caches cleared successfully", "timestamp": datetime.utcnow().isoformat()})

    except Exception as e:
        logger.error(f"Failed to clear search cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")
