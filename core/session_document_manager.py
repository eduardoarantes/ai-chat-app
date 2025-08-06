"""Session document manager for handling document-session relationships."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from models.session import (
    EnhancedSession, SessionDocument, DocumentMetadata, 
    DocumentProcessingStatus, DocumentAccessLevel, SessionRegistry
)
from models.chunking import ChunkedDocument
from core.document_store import DocumentStore


class SessionDocumentManager:
    """
    Manages document-session relationships and provides high-level document operations.
    
    This class provides:
    - Session document lifecycle management
    - Document sharing between sessions
    - Document access validation
    - Integration with document store for deduplication
    - Document processing result storage
    """
    
    def __init__(self, document_store: DocumentStore, config: dict):
        """
        Initialize the session document manager.
        
        Args:
            document_store: Document store instance
            config: Document persistence configuration
        """
        self._document_store = document_store
        self._sessions: SessionRegistry = {}
        self._config = config
        self._logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._max_documents_per_session = config.get("max_documents_per_session", 50)
        self._max_total_size_per_session = config.get("max_total_document_size_per_session", 500 * 1024 * 1024)
        self._enable_document_persistence = config.get("enable_document_persistence", True)
        self._enable_cross_session_sharing = config.get("enable_cross_session_sharing", False)
    
    def create_session(self, session_id: str, title: str = "New Chat") -> EnhancedSession:
        """
        Create a new enhanced session.
        
        Args:
            session_id: Unique session identifier
            title: Session title
            
        Returns:
            Created EnhancedSession instance
        """
        if session_id in self._sessions:
            self._logger.warning(f"Session {session_id} already exists")
            return self._sessions[session_id]
        
        session = EnhancedSession(id=session_id, title=title)
        self._sessions[session_id] = session
        
        self._logger.info(f"Created new session {session_id} with title '{title}'")
        return session
    
    def get_session(self, session_id: str) -> Optional[EnhancedSession]:
        """Get session by ID."""
        return self._sessions.get(session_id)
    
    def get_or_create_session(self, session_id: str, title: str = "New Chat") -> EnhancedSession:
        """Get existing session or create new one."""
        session = self.get_session(session_id)
        if session is None:
            session = self.create_session(session_id, title)
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and remove all its document references.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if session was deleted, False if not found
        """
        if session_id not in self._sessions:
            return False
        
        # Remove all document references from the document store
        orphaned_count = self._document_store.remove_session_references(session_id)
        
        # Remove session
        del self._sessions[session_id]
        
        self._logger.info(f"Deleted session {session_id}, {orphaned_count} documents became orphaned")
        return True
    
    def add_document_to_session(
        self, 
        session_id: str, 
        document_metadata: DocumentMetadata,
        chunked_document: Optional[ChunkedDocument] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Add a document to a session with validation.
        
        Args:
            session_id: Session ID
            document_metadata: Document metadata
            chunked_document: Optional chunked document from processing
            
        Returns:
            Tuple of (success, error_message)
        """
        if not self._enable_document_persistence:
            return False, "Document persistence is disabled"
        
        session = self.get_session(session_id)
        if not session:
            return False, f"Session {session_id} not found"
        
        # Validate session limits
        validation_error = self._validate_session_limits(session, document_metadata.size)
        if validation_error:
            return False, validation_error
        
        # Add document to document store
        is_new_document = self._document_store.add_document(document_metadata, session_id)
        
        # Create session document
        session_document = SessionDocument(
            document_metadata=document_metadata,
            chunks=chunked_document,
            access_level=DocumentAccessLevel.PRIVATE,
            owned=True
        )
        
        # Add to session
        success = session.add_document(session_document)
        if not success:
            return False, "Document already exists in session"
        
        action = "Added new" if is_new_document else "Referenced existing"
        self._logger.info(f"{action} document {document_metadata.filename} to session {session_id}")
        
        return True, None
    
    def update_document_processing_status(
        self, 
        document_hash: str, 
        status: DocumentProcessingStatus,
        error_message: Optional[str] = None,
        chunked_document: Optional[ChunkedDocument] = None
    ) -> bool:
        """
        Update document processing status.
        
        Args:
            document_hash: Document hash
            status: New processing status
            error_message: Error message if status is FAILED
            chunked_document: Chunked document result if status is COMPLETED
            
        Returns:
            True if updated successfully, False otherwise
        """
        # Update in document store
        document_metadata = self._document_store.get_document_by_hash(document_hash)
        if not document_metadata:
            return False
        
        document_metadata.processing_status = status
        document_metadata.processing_error = error_message
        document_metadata.last_accessed = datetime.utcnow()
        
        self._document_store.update_document(document_metadata)
        
        # Update in all sessions that reference this document
        sessions_with_document = self._document_store.get_sessions_for_document(document_hash)
        
        for session_id in sessions_with_document:
            session = self.get_session(session_id)
            if session:
                session_document = session.get_document(document_hash)
                if session_document:
                    session_document.document_metadata = document_metadata
                    if chunked_document and status == DocumentProcessingStatus.COMPLETED:
                        session_document.chunks = chunked_document
                    session.update_activity()
        
        self._logger.info(f"Updated processing status for document {document_hash[:8]}... to {status.value}")
        return True
    
    def get_document_from_session(self, session_id: str, document_hash: str) -> Optional[SessionDocument]:
        """Get a document from a specific session."""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return session.get_document(document_hash)
    
    def remove_document_from_session(self, session_id: str, document_hash: str) -> bool:
        """
        Remove a document from a session.
        
        Args:
            session_id: Session ID
            document_hash: Document hash to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Remove from session
        if not session.remove_document(document_hash):
            return False
        
        # Remove reference from document store
        self._document_store.remove_document_reference(document_hash, session_id)
        
        self._logger.info(f"Removed document {document_hash[:8]}... from session {session_id}")
        return True
    
    def share_document_between_sessions(
        self, 
        source_session_id: str, 
        target_session_id: str, 
        document_hash: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Share a document between sessions.
        
        Args:
            source_session_id: Source session ID
            target_session_id: Target session ID  
            document_hash: Document hash to share
            
        Returns:
            Tuple of (success, error_message)
        """
        if not self._enable_cross_session_sharing:
            return False, "Cross-session sharing is disabled"
        
        # Get sessions
        source_session = self.get_session(source_session_id)
        target_session = self.get_session(target_session_id)
        
        if not source_session:
            return False, f"Source session {source_session_id} not found"
        if not target_session:
            return False, f"Target session {target_session_id} not found"
        
        # Get document from source session
        source_document = source_session.get_document(document_hash)
        if not source_document:
            return False, "Document not found in source session"
        
        # Validate sharing permissions
        if not source_document.can_share():
            return False, "Document cannot be shared (private or not owned)"
        
        if not source_session.document_access_controls.can_share_with_session(target_session_id):
            return False, "Source session does not allow sharing with target session"
        
        # Check target session limits
        validation_error = self._validate_session_limits(target_session, source_document.document_metadata.size)
        if validation_error:
            return False, f"Target session: {validation_error}"
        
        # Create shared document reference
        shared_document = SessionDocument(
            document_metadata=source_document.document_metadata,
            chunks=source_document.chunks,
            access_level=source_document.access_level,
            owned=False,
            shared_from=source_session_id
        )
        
        # Add to target session
        if not target_session.add_document(shared_document):
            return False, "Document already exists in target session"
        
        # Update sharing references
        source_document.add_shared_session(target_session_id)
        
        # Add document store reference
        self._document_store._add_session_reference(document_hash, target_session_id)
        
        self._logger.info(f"Shared document {document_hash[:8]}... from session {source_session_id} to {target_session_id}")
        return True, None
    
    def get_session_documents(self, session_id: str, include_chunks: bool = False) -> List[Dict[str, Any]]:
        """
        Get all documents for a session.
        
        Args:
            session_id: Session ID
            include_chunks: Whether to include chunked document information
            
        Returns:
            List of document information dictionaries
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        documents = []
        for doc_hash, session_document in session.documents.items():
            doc_info = {
                "id": session_document.document_metadata.id,
                "hash": doc_hash,
                "filename": session_document.document_metadata.filename,
                "mime_type": session_document.document_metadata.mime_type,
                "size": session_document.document_metadata.size,
                "uploaded_at": session_document.document_metadata.uploaded_at.isoformat(),
                "last_accessed": session_document.document_metadata.last_accessed.isoformat(),
                "processing_status": session_document.document_metadata.processing_status.value,
                "processing_error": session_document.document_metadata.processing_error,
                "access_level": session_document.access_level.value,
                "owned": session_document.owned,
                "shared_from": session_document.shared_from,
                "shared_to_count": len(session_document.shared_to),
                "metadata": session_document.document_metadata.metadata
            }
            
            if include_chunks and session_document.chunks:
                doc_info["chunks"] = {
                    "chunk_count": session_document.chunks.chunk_count,
                    "total_content_length": session_document.chunks.total_content_length,
                    "chunking_metadata": session_document.chunks.chunking_metadata
                }
            
            documents.append(doc_info)
        
        return documents
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get information about all sessions."""
        sessions = []
        for session_id, session in self._sessions.items():
            session_info = session.to_dict(include_messages=False, include_documents=False)
            sessions.append(session_info)
        return sessions
    
    def cleanup_failed_documents(self) -> int:
        """
        Clean up documents with failed processing status across all sessions.
        
        Returns:
            Total number of documents cleaned up
        """
        total_cleaned = 0
        
        for session in self._sessions.values():
            cleaned = session.cleanup_failed_documents()
            total_cleaned += cleaned
        
        # Also cleanup orphaned documents from the store
        orphaned_cleaned = self._document_store.cleanup_orphaned_documents()
        
        if total_cleaned > 0 or orphaned_cleaned > 0:
            self._logger.info(f"Cleaned up {total_cleaned} failed documents from sessions, "
                            f"{orphaned_cleaned} orphaned documents from store")
        
        return total_cleaned + orphaned_cleaned
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the session document manager."""
        store_stats = self._document_store.get_store_statistics()
        
        session_stats = {
            "total_sessions": len(self._sessions),
            "sessions_with_documents": len([s for s in self._sessions.values() if s.get_document_count() > 0]),
            "total_session_documents": sum(s.get_document_count() for s in self._sessions.values()),
            "total_session_document_size": sum(s.get_total_document_size() for s in self._sessions.values()),
        }
        
        # Configuration stats
        config_stats = {
            "document_persistence_enabled": self._enable_document_persistence,
            "cross_session_sharing_enabled": self._enable_cross_session_sharing,
            "max_documents_per_session": self._max_documents_per_session,
            "max_total_size_per_session": self._max_total_size_per_session
        }
        
        return {
            "document_store": store_stats,
            "sessions": session_stats,
            "configuration": config_stats
        }
    
    def _validate_session_limits(self, session: EnhancedSession, new_document_size: int) -> Optional[str]:
        """
        Validate that adding a document won't exceed session limits.
        
        Args:
            session: Session to validate
            new_document_size: Size of document to add
            
        Returns:
            Error message if validation fails, None if valid
        """
        current_count = session.get_document_count()
        current_size = session.get_total_document_size()
        
        if current_count >= self._max_documents_per_session:
            return f"Session has reached maximum document limit ({self._max_documents_per_session})"
        
        if current_size + new_document_size > self._max_total_size_per_session:
            return (f"Adding document would exceed size limit "
                   f"({current_size + new_document_size} > {self._max_total_size_per_session} bytes)")
        
        return None