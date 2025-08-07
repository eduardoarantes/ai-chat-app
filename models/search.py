"""
Data models for vector embeddings and semantic search functionality.

This module defines the data structures used for embedding generation,
vector storage, and semantic search operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import uuid

from pydantic import BaseModel, Field


class SearchType(str, Enum):
    """Types of search operations supported."""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    SIMILARITY = "similarity"


class SimilarityMetric(str, Enum):
    """Similarity metrics for vector comparisons."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class EmbeddingStatus(str, Enum):
    """Status of embedding generation process."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EmbeddingMetadata:
    """Metadata for generated embeddings."""
    embedding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    generation_time: float = 0.0  # Time taken to generate embedding in seconds
    cache_hit: bool = False  # Whether this embedding was retrieved from cache
    device: str = "cpu"  # Device used for generation (cpu/cuda)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "embedding_id": self.embedding_id,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "generation_time": self.generation_time,
            "cache_hit": self.cache_hit,
            "device": self.device
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingMetadata':
        """Create from dictionary representation."""
        return cls(
            embedding_id=data["embedding_id"],
            model_name=data["model_name"],
            dimension=data["dimension"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            generation_time=data["generation_time"],
            cache_hit=data.get("cache_hit", False),
            device=data.get("device", "cpu")
        )


@dataclass
class ChunkEmbedding:
    """Represents an embedding for a document chunk."""
    chunk_id: str
    document_id: str
    session_id: str
    content_hash: str  # Hash of the chunk content for cache key
    embedding_vector: List[float]
    metadata: EmbeddingMetadata
    search_score: Optional[float] = None  # Similarity score when used in search results
    
    def __post_init__(self):
        """Validate embedding dimensions."""
        if len(self.embedding_vector) != self.metadata.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.metadata.dimension}, "
                f"got {len(self.embedding_vector)}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "session_id": self.session_id,
            "content_hash": self.content_hash,
            "embedding_vector": self.embedding_vector,
            "metadata": self.metadata.to_dict(),
            "search_score": self.search_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkEmbedding':
        """Create from dictionary representation."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            session_id=data["session_id"],
            content_hash=data["content_hash"],
            embedding_vector=data["embedding_vector"],
            metadata=EmbeddingMetadata.from_dict(data["metadata"]),
            search_score=data.get("search_score")
        )


@dataclass
class DocumentEmbeddingStatus:
    """Status of embeddings for a document."""
    document_id: str
    status: EmbeddingStatus
    model_name: str
    embedding_count: int = 0
    failed_chunks: int = 0
    generation_time: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "status": self.status.value,
            "model_name": self.model_name,
            "embedding_count": self.embedding_count,
            "failed_chunks": self.failed_chunks,
            "generation_time": self.generation_time,
            "last_updated": self.last_updated.isoformat(),
            "error_message": self.error_message
        }


# Pydantic models for API requests/responses

class SemanticSearchRequest(BaseModel):
    """Request model for semantic search operations."""
    query: str = Field(..., description="Search query text")
    session_id: Optional[str] = Field(None, description="Limit search to specific session")
    document_ids: Optional[List[str]] = Field(None, description="Limit search to specific documents")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    min_similarity: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")
    search_type: SearchType = Field(SearchType.SEMANTIC, description="Type of search to perform")
    similarity_metric: SimilarityMetric = Field(SimilarityMetric.COSINE, description="Similarity metric to use")
    include_metadata: bool = Field(True, description="Include chunk metadata in results")
    boost_recent: bool = Field(False, description="Boost scores for more recent documents")
    
    class Config:
        json_encoders = {
            SearchType: lambda v: v.value,
            SimilarityMetric: lambda v: v.value
        }


class SemanticSearchResult(BaseModel):
    """Individual search result from semantic search."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="Parent document identifier")
    document_name: str = Field(..., description="Name of the parent document")
    chunk_content: str = Field(..., description="Text content of the chunk")
    similarity_score: float = Field(..., description="Similarity score (0.0 to 1.0)")
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk-level metadata")
    document_metadata: Dict[str, Any] = Field(default_factory=dict, description="Document-level metadata")
    embedding_metadata: Optional[Dict[str, Any]] = Field(None, description="Embedding generation metadata")


class SemanticSearchResponse(BaseModel):
    """Response model for semantic search operations."""
    query: str = Field(..., description="Original search query")
    results: List[SemanticSearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of matching chunks")
    search_time_ms: float = Field(..., description="Time taken for search in milliseconds")
    embedding_time_ms: float = Field(..., description="Time taken to embed query in milliseconds")
    search_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional search metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SimilarDocumentsRequest(BaseModel):
    """Request model for finding similar documents."""
    document_id: str = Field(..., description="Reference document ID")
    session_id: Optional[str] = Field(None, description="Limit search to specific session")
    limit: int = Field(5, ge=1, le=20, description="Maximum number of similar documents")
    min_similarity: float = Field(0.3, ge=0.0, le=1.0, description="Minimum similarity threshold")
    similarity_metric: SimilarityMetric = Field(SimilarityMetric.COSINE, description="Similarity metric to use")
    exclude_same_document: bool = Field(True, description="Exclude chunks from the same document")


class SimilarDocumentsResponse(BaseModel):
    """Response model for similar documents search."""
    reference_document_id: str = Field(..., description="Reference document ID")
    similar_documents: List[SemanticSearchResult] = Field(..., description="Similar document chunks")
    search_time_ms: float = Field(..., description="Time taken for search in milliseconds")


class EmbeddingGenerationRequest(BaseModel):
    """Request model for embedding generation."""
    document_id: str = Field(..., description="Document ID to generate embeddings for")
    session_id: str = Field(..., description="Session ID")
    force_regenerate: bool = Field(False, description="Force regeneration even if embeddings exist")
    batch_size: Optional[int] = Field(None, description="Batch size for processing")
    notify_progress: bool = Field(True, description="Send progress updates via SSE")


class EmbeddingGenerationResponse(BaseModel):
    """Response model for embedding generation."""
    document_id: str = Field(..., description="Document ID")
    status: EmbeddingStatus = Field(..., description="Current status")
    embedding_count: int = Field(..., description="Number of embeddings generated")
    generation_time: float = Field(..., description="Time taken in seconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class EmbeddingStatusResponse(BaseModel):
    """Response model for embedding status queries."""
    document_id: str = Field(..., description="Document ID")
    status: EmbeddingStatus = Field(..., description="Current status")
    embedding_count: int = Field(..., description="Number of embeddings generated")
    failed_chunks: int = Field(..., description="Number of chunks that failed")
    model_name: str = Field(..., description="Model used for embeddings")
    last_updated: datetime = Field(..., description="Last update timestamp")
    progress_percentage: float = Field(..., description="Completion percentage (0.0 to 100.0)")


class VectorDatabaseStats(BaseModel):
    """Statistics about the vector database."""
    total_embeddings: int = Field(..., description="Total number of embeddings stored")
    total_documents: int = Field(..., description="Total number of documents with embeddings")
    total_sessions: int = Field(..., description="Total number of sessions with embeddings")
    database_size_mb: float = Field(..., description="Database size in megabytes")
    oldest_embedding: Optional[datetime] = Field(None, description="Timestamp of oldest embedding")
    newest_embedding: Optional[datetime] = Field(None, description="Timestamp of newest embedding")
    model_distribution: Dict[str, int] = Field(default_factory=dict, description="Count of embeddings by model")


@dataclass
class SearchResult:
    """Individual search result from vector similarity search."""
    chunk_id: str
    document_id: str
    session_id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "session_id": self.session_id,
            "content": self.content,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
            "document_metadata": self.document_metadata
        }


@dataclass
class DatabaseStats:
    """Statistics about the vector database."""
    total_embeddings: int
    total_documents: int
    total_sessions: int
    database_size_mb: float
    oldest_embedding: Optional[datetime] = None
    newest_embedding: Optional[datetime] = None
    model_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_embeddings": self.total_embeddings,
            "total_documents": self.total_documents,
            "total_sessions": self.total_sessions,
            "database_size_mb": self.database_size_mb,
            "oldest_embedding": self.oldest_embedding.isoformat() if self.oldest_embedding else None,
            "newest_embedding": self.newest_embedding.isoformat() if self.newest_embedding else None,
            "model_distribution": self.model_distribution
        }


class SearchAnalytics(BaseModel):
    """Analytics data for search operations."""
    query: str = Field(..., description="Search query")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Search timestamp")
    results_count: int = Field(..., description="Number of results returned")
    search_time_ms: float = Field(..., description="Time taken for search")
    similarity_scores: List[float] = Field(..., description="Similarity scores of results")
    user_session: Optional[str] = Field(None, description="User session ID")
    search_type: SearchType = Field(..., description="Type of search performed")
    
    def get_average_similarity(self) -> float:
        """Calculate average similarity score."""
        return sum(self.similarity_scores) / len(self.similarity_scores) if self.similarity_scores else 0.0
    
    def get_score_distribution(self) -> Dict[str, int]:
        """Get distribution of similarity scores by ranges."""
        ranges = {
            "0.9-1.0": 0, "0.8-0.9": 0, "0.7-0.8": 0, 
            "0.6-0.7": 0, "0.5-0.6": 0, "0.0-0.5": 0
        }
        
        for score in self.similarity_scores:
            if 0.9 <= score <= 1.0:
                ranges["0.9-1.0"] += 1
            elif 0.8 <= score < 0.9:
                ranges["0.8-0.9"] += 1
            elif 0.7 <= score < 0.8:
                ranges["0.7-0.8"] += 1
            elif 0.6 <= score < 0.7:
                ranges["0.6-0.7"] += 1
            elif 0.5 <= score < 0.6:
                ranges["0.5-0.6"] += 1
            else:
                ranges["0.0-0.5"] += 1
        
        return ranges