"""Document store for managing document metadata and deduplication."""

import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict

from models.session import DocumentMetadata, DocumentRegistry


class DocumentStore:
    """
    Document store for managing document metadata with deduplication and indexing.
    
    This class provides:
    - Document deduplication via hash-based storage
    - Fast lookup by document ID and hash
    - Session-to-document relationship tracking
    - Storage optimization through reference counting
    """
    
    def __init__(self):
        """Initialize the document store."""
        # Primary storage: hash -> DocumentMetadata
        self._documents: DocumentRegistry = {}
        
        # Indexes for fast lookup
        self._id_to_hash_index: Dict[str, str] = {}  # document_id -> hash
        self._hash_to_id_index: Dict[str, str] = {}  # hash -> document_id
        
        # Session references: hash -> set of session_ids that reference this document
        self._document_sessions: Dict[str, Set[str]] = defaultdict(set)
        
        # Session documents: session_id -> set of document hashes
        self._session_documents: Dict[str, Set[str]] = defaultdict(set)
        
        # Reference counting for cleanup optimization
        self._reference_counts: Dict[str, int] = defaultdict(int)
        
        self._logger = logging.getLogger(__name__)
    
    def add_document(self, document_metadata: DocumentMetadata, session_id: str) -> bool:
        """
        Add a document to the store.
        
        Args:
            document_metadata: Document metadata to store
            session_id: Session ID that owns/references this document
            
        Returns:
            True if document was added, False if it already exists
        """
        doc_hash = document_metadata.hash
        doc_id = document_metadata.id
        
        # Check if document already exists
        if doc_hash in self._documents:
            # Document exists, just add session reference
            self._add_session_reference(doc_hash, session_id)
            self._logger.info(f"Document {doc_id} (hash: {doc_hash[:8]}...) already exists, added session reference")
            return False
        
        # Add new document
        self._documents[doc_hash] = document_metadata
        self._id_to_hash_index[doc_id] = doc_hash
        self._hash_to_id_index[doc_hash] = doc_id
        
        # Add session reference
        self._add_session_reference(doc_hash, session_id)
        
        self._logger.info(f"Added new document {doc_id} (hash: {doc_hash[:8]}...) from session {session_id}")
        return True
    
    def get_document_by_id(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get document by ID."""
        doc_hash = self._id_to_hash_index.get(document_id)
        if doc_hash:
            return self._documents.get(doc_hash)
        return None
    
    def get_document_by_hash(self, document_hash: str) -> Optional[DocumentMetadata]:
        """Get document by hash."""
        return self._documents.get(document_hash)
    
    def has_document(self, document_hash: str) -> bool:
        """Check if document exists in store."""
        return document_hash in self._documents
    
    def update_document(self, document_metadata: DocumentMetadata) -> bool:
        """
        Update existing document metadata.
        
        Args:
            document_metadata: Updated document metadata
            
        Returns:
            True if document was updated, False if not found
        """
        doc_hash = document_metadata.hash
        
        if doc_hash not in self._documents:
            self._logger.warning(f"Attempted to update non-existent document with hash {doc_hash[:8]}...")
            return False
        
        # Update the document while preserving hash consistency
        old_metadata = self._documents[doc_hash]
        if old_metadata.id != document_metadata.id:
            # Update ID indexes if ID changed
            self._id_to_hash_index.pop(old_metadata.id, None)
            self._id_to_hash_index[document_metadata.id] = doc_hash
            self._hash_to_id_index[doc_hash] = document_metadata.id
        
        self._documents[doc_hash] = document_metadata
        self._logger.info(f"Updated document {document_metadata.id} (hash: {doc_hash[:8]}...)")
        return True
    
    def remove_document_reference(self, document_hash: str, session_id: str) -> bool:
        """
        Remove a session's reference to a document.
        
        Args:
            document_hash: Document hash
            session_id: Session ID to remove reference for
            
        Returns:
            True if reference was removed, False if not found
        """
        if document_hash not in self._documents:
            return False
        
        # Remove session reference
        self._document_sessions[document_hash].discard(session_id)
        self._session_documents[session_id].discard(document_hash)
        self._reference_counts[document_hash] = max(0, self._reference_counts[document_hash] - 1)
        
        # If no more references, remove document entirely
        if not self._document_sessions[document_hash]:
            return self._remove_document_completely(document_hash)
        
        self._logger.info(f"Removed session {session_id} reference to document {document_hash[:8]}...")
        return True
    
    def get_documents_for_session(self, session_id: str) -> List[DocumentMetadata]:
        """Get all documents referenced by a session."""
        document_hashes = self._session_documents.get(session_id, set())
        return [self._documents[doc_hash] for doc_hash in document_hashes if doc_hash in self._documents]
    
    def get_sessions_for_document(self, document_hash: str) -> Set[str]:
        """Get all sessions that reference a document."""
        return self._document_sessions.get(document_hash, set()).copy()
    
    def get_orphaned_documents(self) -> List[str]:
        """Get list of document hashes that have no session references."""
        orphaned = []
        for doc_hash in self._documents:
            if not self._document_sessions.get(doc_hash):
                orphaned.append(doc_hash)
        return orphaned
    
    def cleanup_orphaned_documents(self) -> int:
        """
        Remove documents that have no session references.
        
        Returns:
            Number of documents removed
        """
        orphaned = self.get_orphaned_documents()
        removed_count = 0
        
        for doc_hash in orphaned:
            if self._remove_document_completely(doc_hash):
                removed_count += 1
        
        if removed_count > 0:
            self._logger.info(f"Cleaned up {removed_count} orphaned documents")
        
        return removed_count
    
    def remove_session_references(self, session_id: str) -> int:
        """
        Remove all document references for a session.
        
        Args:
            session_id: Session ID to remove references for
            
        Returns:
            Number of documents that became orphaned and were removed
        """
        document_hashes = self._session_documents.get(session_id, set()).copy()
        removed_count = 0
        
        for doc_hash in document_hashes:
            # Remove the reference
            self._document_sessions[doc_hash].discard(session_id)
            self._reference_counts[doc_hash] = max(0, self._reference_counts[doc_hash] - 1)
            
            # If document became orphaned, remove it
            if not self._document_sessions[doc_hash]:
                if self._remove_document_completely(doc_hash):
                    removed_count += 1
        
        # Clear session's document references
        self._session_documents.pop(session_id, None)
        
        if document_hashes:
            self._logger.info(f"Removed {len(document_hashes)} document references for session {session_id}, "
                            f"{removed_count} documents became orphaned and were removed")
        
        return removed_count
    
    def get_store_statistics(self) -> Dict[str, Any]:
        """Get statistics about the document store."""
        total_size = sum(doc.size for doc in self._documents.values())
        
        return {
            "total_documents": len(self._documents),
            "total_size_bytes": total_size,
            "total_sessions_with_documents": len([s for s in self._session_documents.values() if s]),
            "average_references_per_document": (
                sum(self._reference_counts.values()) / len(self._documents) 
                if self._documents else 0
            ),
            "documents_by_mime_type": self._get_documents_by_mime_type(),
            "largest_document_size": max((doc.size for doc in self._documents.values()), default=0),
            "oldest_document": self._get_oldest_document_date(),
            "newest_document": self._get_newest_document_date(),
        }
    
    def calculate_content_hash(self, content: bytes) -> str:
        """
        Calculate SHA256 hash of content.
        
        Args:
            content: Raw content bytes
            
        Returns:
            SHA256 hash string
        """
        return hashlib.sha256(content).hexdigest()
    
    def find_duplicate_documents(self) -> Dict[str, List[str]]:
        """
        Find documents with identical hashes (should not happen in normal operation).
        
        Returns:
            Dict mapping hashes to lists of document IDs
        """
        duplicates = {}
        hash_to_ids = defaultdict(list)
        
        for doc_id, doc_hash in self._id_to_hash_index.items():
            hash_to_ids[doc_hash].append(doc_id)
        
        for doc_hash, doc_ids in hash_to_ids.items():
            if len(doc_ids) > 1:
                duplicates[doc_hash] = doc_ids
        
        return duplicates
    
    def validate_consistency(self) -> Dict[str, List[str]]:
        """
        Validate internal consistency of the store.
        
        Returns:
            Dict with any consistency issues found
        """
        issues = {
            "missing_documents": [],
            "missing_indexes": [],
            "orphaned_references": [],
            "inconsistent_references": []
        }
        
        # Check that all indexed documents exist
        for doc_id, doc_hash in self._id_to_hash_index.items():
            if doc_hash not in self._documents:
                issues["missing_documents"].append(f"Document ID {doc_id} points to missing hash {doc_hash}")
        
        # Check that all documents have proper indexes
        for doc_hash, doc_metadata in self._documents.items():
            if doc_metadata.id not in self._id_to_hash_index:
                issues["missing_indexes"].append(f"Document {doc_metadata.id} missing from ID index")
            if doc_hash not in self._hash_to_id_index:
                issues["missing_indexes"].append(f"Hash {doc_hash} missing from hash index")
        
        # Check reference consistency
        for doc_hash, sessions in self._document_sessions.items():
            if doc_hash not in self._documents:
                issues["orphaned_references"].append(f"Hash {doc_hash} has session references but no document")
            
            for session_id in sessions:
                if doc_hash not in self._session_documents.get(session_id, set()):
                    issues["inconsistent_references"].append(
                        f"Document {doc_hash} references session {session_id} but not vice versa"
                    )
        
        return {k: v for k, v in issues.items() if v}
    
    def _add_session_reference(self, document_hash: str, session_id: str) -> None:
        """Add a session reference to a document."""
        self._document_sessions[document_hash].add(session_id)
        self._session_documents[session_id].add(document_hash)
        self._reference_counts[document_hash] += 1
    
    def _remove_document_completely(self, document_hash: str) -> bool:
        """Remove a document completely from the store."""
        document = self._documents.get(document_hash)
        if not document:
            return False
        
        # Remove from all indexes
        self._documents.pop(document_hash, None)
        self._id_to_hash_index.pop(document.id, None)
        self._hash_to_id_index.pop(document_hash, None)
        self._document_sessions.pop(document_hash, None)
        self._reference_counts.pop(document_hash, None)
        
        self._logger.info(f"Completely removed document {document.id} (hash: {document_hash[:8]}...)")
        return True
    
    def _get_documents_by_mime_type(self) -> Dict[str, int]:
        """Get count of documents by MIME type."""
        mime_counts = defaultdict(int)
        for doc in self._documents.values():
            mime_counts[doc.mime_type] += 1
        return dict(mime_counts)
    
    def _get_oldest_document_date(self) -> Optional[str]:
        """Get the oldest document upload date."""
        if not self._documents:
            return None
        oldest = min(doc.uploaded_at for doc in self._documents.values())
        return oldest.isoformat()
    
    def _get_newest_document_date(self) -> Optional[str]:
        """Get the newest document upload date."""
        if not self._documents:
            return None
        newest = max(doc.uploaded_at for doc in self._documents.values())
        return newest.isoformat()