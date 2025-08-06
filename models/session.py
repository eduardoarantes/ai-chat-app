"""Session-based document persistence models and data structures."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set

from models.chunking import ChunkedDocument


class DocumentProcessingStatus(Enum):
    """Enumeration for document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentAccessLevel(Enum):
    """Enumeration for document access levels."""
    PRIVATE = "private"      # Only accessible within owning session
    SHARED = "shared"        # Accessible to specific sessions
    PUBLIC = "public"        # Accessible to all sessions (future feature)


@dataclass
class DocumentMetadata:
    """Comprehensive metadata for session documents."""
    id: str
    hash: str  # SHA256 hash for deduplication
    filename: str
    mime_type: str
    size: int
    uploaded_at: datetime
    last_accessed: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)  # Document-specific metadata (title, author, etc.)
    processing_status: DocumentProcessingStatus = DocumentProcessingStatus.PENDING
    processing_error: Optional[str] = None
    
    def __post_init__(self):
        """Ensure ID is set if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class SessionDocument:
    """Document reference within a session with access controls."""
    document_metadata: DocumentMetadata
    chunks: Optional[ChunkedDocument] = None  # Integration with TASK-005
    access_level: DocumentAccessLevel = DocumentAccessLevel.PRIVATE
    owned: bool = True  # True if this session uploaded the document
    shared_from: Optional[str] = None  # Session ID if document was shared from another session
    shared_to: Set[str] = field(default_factory=set)  # Set of session IDs this document is shared to
    
    @property
    def document_id(self) -> str:
        """Get document ID."""
        return self.document_metadata.id
    
    @property
    def document_hash(self) -> str:
        """Get document hash."""
        return self.document_metadata.hash
    
    def can_share(self) -> bool:
        """Check if document can be shared to other sessions."""
        return self.owned and self.access_level != DocumentAccessLevel.PRIVATE
    
    def add_shared_session(self, session_id: str) -> bool:
        """Add a session to the shared list."""
        if self.can_share():
            self.shared_to.add(session_id)
            return True
        return False
    
    def remove_shared_session(self, session_id: str) -> bool:
        """Remove a session from the shared list."""
        if session_id in self.shared_to:
            self.shared_to.remove(session_id)
            return True
        return False
    
    def is_accessible_by_session(self, session_id: str) -> bool:
        """Check if document is accessible by a specific session."""
        # Document is accessible if:
        # 1. Session owns the document
        # 2. Document was shared with this session
        # 3. Document is shared from this session
        return (self.owned or 
                session_id in self.shared_to or 
                self.shared_from == session_id)


@dataclass
class SessionAccessControls:
    """Access control settings for session-level document sharing."""
    sharing_enabled: bool = False
    allowed_sessions: Set[str] = field(default_factory=set)
    max_shared_documents: int = 10  # Limit number of documents that can be shared
    
    def can_share_with_session(self, session_id: str) -> bool:
        """Check if sharing is allowed with a specific session."""
        return (self.sharing_enabled and 
                (not self.allowed_sessions or session_id in self.allowed_sessions))
    
    def add_allowed_session(self, session_id: str) -> None:
        """Add a session to the allowed list."""
        self.allowed_sessions.add(session_id)
    
    def remove_allowed_session(self, session_id: str) -> None:
        """Remove a session from the allowed list."""
        self.allowed_sessions.discard(session_id)


@dataclass
class EnhancedSession:
    """Enhanced session structure with document persistence support."""
    id: str
    title: str = "New Chat"
    messages: List[Any] = field(default_factory=list)  # LangChain messages
    documents: Dict[str, SessionDocument] = field(default_factory=dict)  # document_hash -> SessionDocument
    document_access_controls: SessionAccessControls = field(default_factory=SessionAccessControls)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Ensure session ID is set if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def add_document(self, document: SessionDocument) -> bool:
        """Add a document to the session."""
        doc_hash = document.document_hash
        if doc_hash not in self.documents:
            self.documents[doc_hash] = document
            self.update_activity()
            return True
        return False
    
    def get_document(self, document_hash: str) -> Optional[SessionDocument]:
        """Get a document by hash."""
        return self.documents.get(document_hash)
    
    def remove_document(self, document_hash: str) -> bool:
        """Remove a document from the session."""
        if document_hash in self.documents:
            del self.documents[document_hash]
            self.update_activity()
            return True
        return False
    
    def get_owned_documents(self) -> List[SessionDocument]:
        """Get all documents owned by this session."""
        return [doc for doc in self.documents.values() if doc.owned]
    
    def get_shared_documents(self) -> List[SessionDocument]:
        """Get all documents shared with this session."""
        return [doc for doc in self.documents.values() if not doc.owned]
    
    def get_shareable_documents(self) -> List[SessionDocument]:
        """Get all documents that can be shared from this session."""
        return [doc for doc in self.documents.values() if doc.can_share()]
    
    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def get_document_count(self) -> int:
        """Get total number of documents in session."""
        return len(self.documents)
    
    def get_total_document_size(self) -> int:
        """Get total size of all documents in session."""
        return sum(doc.document_metadata.size for doc in self.documents.values())
    
    def cleanup_failed_documents(self) -> int:
        """Remove documents with failed processing status."""
        failed_docs = [
            doc_hash for doc_hash, doc in self.documents.items()
            if doc.document_metadata.processing_status == DocumentProcessingStatus.FAILED
        ]
        
        for doc_hash in failed_docs:
            del self.documents[doc_hash]
        
        if failed_docs:
            self.update_activity()
        
        return len(failed_docs)
    
    def to_dict(self, include_messages: bool = True, include_documents: bool = True) -> Dict[str, Any]:
        """Convert session to dictionary format for API responses."""
        result = {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "document_count": self.get_document_count(),
            "total_document_size": self.get_total_document_size()
        }
        
        if include_messages:
            # Note: messages will need serialization handling in endpoints
            result["messages"] = self.messages
        
        if include_documents:
            result["documents"] = {
                doc_hash: {
                    "id": doc.document_metadata.id,
                    "filename": doc.document_metadata.filename,
                    "mime_type": doc.document_metadata.mime_type,
                    "size": doc.document_metadata.size,
                    "uploaded_at": doc.document_metadata.uploaded_at.isoformat(),
                    "last_accessed": doc.document_metadata.last_accessed.isoformat(),
                    "processing_status": doc.document_metadata.processing_status.value,
                    "access_level": doc.access_level.value,
                    "owned": doc.owned,
                    "shared_to_count": len(doc.shared_to),
                    "chunks_available": doc.chunks is not None
                }
                for doc_hash, doc in self.documents.items()
            }
        
        return result


# Type aliases for backwards compatibility and clarity
DocumentRegistry = Dict[str, DocumentMetadata]  # document_hash -> DocumentMetadata
SessionRegistry = Dict[str, EnhancedSession]   # session_id -> EnhancedSession