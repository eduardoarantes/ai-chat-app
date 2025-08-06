"""Chunking models and data structures for intelligent document processing."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union


class ChunkingStrategy(Enum):
    """Enumeration for chunking strategies."""
    RECURSIVE_CHARACTER = "recursive_character"
    PDF_AWARE = "pdf_aware"
    TEXT_PARAGRAPH = "text_paragraph"
    CODE_BOUNDARY = "code_boundary"
    STRUCTURED_DOCUMENT = "structured_document"


class BoundaryQuality(Enum):
    """Enumeration for chunk boundary quality assessment."""
    EXCELLENT = "excellent"  # Perfect semantic boundaries
    GOOD = "good"           # Good boundaries with minimal context loss
    FAIR = "fair"           # Acceptable boundaries with some context loss
    POOR = "poor"           # Suboptimal boundaries with significant context loss


@dataclass
class ChunkMetadata:
    """Comprehensive metadata for document chunks."""
    chunk_id: str
    position: Dict[str, int]  # start_index, end_index, start_line, end_line
    relationships: Dict[str, Any]  # previous_chunk_id, next_chunk_id, parent_document_id
    content_analysis: Dict[str, Any]  # word_count, char_count, complexity_score, topic_keywords
    processing_metadata: Dict[str, Any]  # strategy_used, optimization_applied, boundary_quality
    quality_metrics: Dict[str, float]  # coherence_score, boundary_quality_score, semantic_completeness
    
    def __post_init__(self):
        """Ensure chunk_id is set if not provided."""
        if not self.chunk_id:
            self.chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"


@dataclass
class Chunk:
    """Individual document chunk with content and metadata."""
    content: str
    metadata: ChunkMetadata
    
    @property
    def chunk_id(self) -> str:
        """Get chunk ID from metadata."""
        return self.metadata.chunk_id
    
    @property
    def size(self) -> int:
        """Get chunk size in characters."""
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        """Get word count for the chunk."""
        return len(self.content.split())


@dataclass
class ChunkingConfig:
    """Configuration parameters for document chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 4000
    enable_semantic_boundaries: bool = True
    enable_overlap_optimization: bool = True
    enable_metadata_extraction: bool = True
    separators: Optional[List[str]] = None
    keep_separator: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size < 0:
            raise ValueError("min_chunk_size cannot be negative")
        if self.max_chunk_size < self.chunk_size:
            raise ValueError("max_chunk_size must be >= chunk_size")


@dataclass
class ChunkIndex:
    """Index for efficient chunk lookup and cross-referencing."""
    chunks_by_id: Dict[str, Chunk] = field(default_factory=dict)
    chunks_by_position: Dict[int, List[str]] = field(default_factory=dict)  # start_position -> chunk_ids
    chunks_by_document: Dict[str, List[str]] = field(default_factory=dict)  # document_id -> chunk_ids
    parent_child_relationships: Dict[str, List[str]] = field(default_factory=dict)
    chunk_metadata_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add_chunk(self, chunk: Chunk, document_id: Optional[str] = None) -> None:
        """Add chunk to the index."""
        chunk_id = chunk.chunk_id
        self.chunks_by_id[chunk_id] = chunk
        
        # Index by position
        start_pos = chunk.metadata.position.get("start_index", 0)
        if start_pos not in self.chunks_by_position:
            self.chunks_by_position[start_pos] = []
        self.chunks_by_position[start_pos].append(chunk_id)
        
        # Index by document
        if document_id:
            if document_id not in self.chunks_by_document:
                self.chunks_by_document[document_id] = []
            self.chunks_by_document[document_id].append(chunk_id)
        
        # Index metadata for fast searching
        self.chunk_metadata_index[chunk_id] = chunk.metadata.content_analysis
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID."""
        return self.chunks_by_id.get(chunk_id)
    
    def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        """Get all chunks for a document."""
        chunk_ids = self.chunks_by_document.get(document_id, [])
        return [self.chunks_by_id[cid] for cid in chunk_ids if cid in self.chunks_by_id]
    
    def get_adjacent_chunks(self, chunk_id: str) -> Dict[str, Optional[Chunk]]:
        """Get previous and next chunks."""
        if chunk_id not in self.chunks_by_id:
            return {"previous": None, "next": None}
        
        chunk = self.chunks_by_id[chunk_id]
        prev_id = chunk.metadata.relationships.get("previous_chunk_id")
        next_id = chunk.metadata.relationships.get("next_chunk_id")
        
        return {
            "previous": self.chunks_by_id.get(prev_id) if prev_id else None,
            "next": self.chunks_by_id.get(next_id) if next_id else None
        }


@dataclass
class ChunkedDocument:
    """Document with intelligent chunking applied."""
    original_document: Any  # Reference to ProcessedDocument
    chunks: List[Chunk]
    chunk_index: ChunkIndex
    chunking_metadata: Dict[str, Any]
    
    @property
    def chunk_count(self) -> int:
        """Get total number of chunks."""
        return len(self.chunks)
    
    @property
    def total_content_length(self) -> int:
        """Get total content length across all chunks."""
        return sum(chunk.size for chunk in self.chunks)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID."""
        return self.chunk_index.get_chunk(chunk_id)
    
    def get_chunks_in_range(self, start_pos: int, end_pos: int) -> List[Chunk]:
        """Get chunks that overlap with the specified position range."""
        overlapping_chunks = []
        for chunk in self.chunks:
            chunk_start = chunk.metadata.position.get("start_index", 0)
            chunk_end = chunk.metadata.position.get("end_index", chunk_start + chunk.size)
            
            # Check for overlap
            if chunk_start < end_pos and chunk_end > start_pos:
                overlapping_chunks.append(chunk)
        
        return overlapping_chunks


class BaseChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize strategy with configuration."""
        self.config = config
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk text using the specific strategy.
        
        Args:
            text: Text to chunk
            metadata: Document metadata for context
            
        Returns:
            List of chunks with metadata
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this chunking strategy."""
        pass
    
    def _create_chunk_metadata(
        self, 
        content: str, 
        start_index: int, 
        end_index: int,
        chunk_number: int,
        total_chunks: int,
        document_metadata: Dict[str, Any]
    ) -> ChunkMetadata:
        """Create comprehensive metadata for a chunk."""
        return ChunkMetadata(
            chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
            position={
                "start_index": start_index,
                "end_index": end_index,
                "chunk_number": chunk_number,
                "total_chunks": total_chunks,
                "start_line": content[:start_index].count('\n') + 1 if start_index > 0 else 1,
                "end_line": content[:end_index].count('\n') + 1
            },
            relationships={
                "parent_document_id": document_metadata.get("document_id", "unknown"),
                "previous_chunk_id": None,  # Set by post-processing
                "next_chunk_id": None      # Set by post-processing
            },
            content_analysis={
                "word_count": len(content[start_index:end_index].split()),
                "char_count": end_index - start_index,
                "line_count": content[start_index:end_index].count('\n') + 1,
                "paragraph_count": content[start_index:end_index].count('\n\n') + 1,
                "complexity_score": self._calculate_complexity_score(content[start_index:end_index]),
                "topic_keywords": self._extract_keywords(content[start_index:end_index])
            },
            processing_metadata={
                "strategy_used": self.get_strategy_name(),
                "chunk_size_target": self.config.chunk_size,
                "overlap_size": self.config.chunk_overlap,
                "boundary_quality": BoundaryQuality.GOOD,  # Default, can be refined
                "processing_timestamp": None  # Set by processor
            },
            quality_metrics={
                "coherence_score": 0.8,  # Default, can be calculated
                "boundary_quality_score": 0.75,  # Default, can be calculated
                "semantic_completeness": 0.85  # Default, can be calculated
            }
        )
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity score (0-1)."""
        # Simple heuristic based on sentence length, word length, and punctuation
        words = text.split()
        if not words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentences = text.split('.')
        avg_sentence_length = len(words) / len(sentences) if sentences else len(words)
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (avg_word_length / 10.0 + avg_sentence_length / 20.0) / 2.0)
        return complexity
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract simple keywords from text."""
        # Simple keyword extraction based on word frequency
        words = text.lower().split()
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        filtered_words = [word.strip('.,!?;:"()[]') for word in words if word not in stop_words and len(word) > 2]
        
        # Count frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word[0] for word in sorted_words[:max_keywords]]
    
    def _link_adjacent_chunks(self, chunks: List[Chunk]) -> None:
        """Link chunks with previous/next relationships."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.metadata.relationships["previous_chunk_id"] = chunks[i-1].chunk_id
            if i < len(chunks) - 1:
                chunk.metadata.relationships["next_chunk_id"] = chunks[i+1].chunk_id