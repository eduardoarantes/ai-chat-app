"""Unit tests for session document models."""

import uuid
from datetime import datetime, timedelta

import pytest

from models.chunking import ChunkedDocument
from models.session import (
    DocumentAccessLevel,
    DocumentMetadata,
    DocumentProcessingStatus,
    EnhancedSession,
    SessionAccessControls,
    SessionDocument,
)


class TestDocumentMetadata:
    """Test cases for DocumentMetadata dataclass."""

    def test_document_metadata_creation(self):
        """Test basic DocumentMetadata creation."""
        now = datetime.utcnow()
        metadata = DocumentMetadata(
            id="test-id",
            hash="test-hash",
            filename="test.pdf",
            mime_type="application/pdf",
            size=1024,
            uploaded_at=now,
            last_accessed=now,
        )

        assert metadata.id == "test-id"
        assert metadata.hash == "test-hash"
        assert metadata.filename == "test.pdf"
        assert metadata.mime_type == "application/pdf"
        assert metadata.size == 1024
        assert metadata.uploaded_at == now
        assert metadata.last_accessed == now
        assert metadata.processing_status == DocumentProcessingStatus.PENDING
        assert metadata.processing_error is None
        assert metadata.metadata == {}

    def test_document_metadata_auto_id_generation(self):
        """Test automatic ID generation when not provided."""
        now = datetime.utcnow()
        metadata = DocumentMetadata(
            id="",  # Empty ID should trigger auto-generation
            hash="test-hash",
            filename="test.pdf",
            mime_type="application/pdf",
            size=1024,
            uploaded_at=now,
            last_accessed=now,
        )

        assert metadata.id is not None
        assert metadata.id != ""
        assert len(metadata.id) > 0

    def test_document_metadata_with_custom_metadata(self):
        """Test DocumentMetadata with custom metadata fields."""
        now = datetime.utcnow()
        custom_metadata = {"title": "Test Document", "author": "Test Author", "page_count": 10}

        metadata = DocumentMetadata(
            id="test-id",
            hash="test-hash",
            filename="test.pdf",
            mime_type="application/pdf",
            size=1024,
            uploaded_at=now,
            last_accessed=now,
            metadata=custom_metadata,
            processing_status=DocumentProcessingStatus.COMPLETED,
        )

        assert metadata.metadata == custom_metadata
        assert metadata.processing_status == DocumentProcessingStatus.COMPLETED

    def test_document_metadata_processing_statuses(self):
        """Test all processing status values."""
        now = datetime.utcnow()

        for status in DocumentProcessingStatus:
            metadata = DocumentMetadata(
                id="test-id",
                hash="test-hash",
                filename="test.pdf",
                mime_type="application/pdf",
                size=1024,
                uploaded_at=now,
                last_accessed=now,
                processing_status=status,
            )
            assert metadata.processing_status == status


class TestSessionDocument:
    """Test cases for SessionDocument dataclass."""

    def create_test_document_metadata(self) -> DocumentMetadata:
        """Helper to create test document metadata."""
        now = datetime.utcnow()
        return DocumentMetadata(
            id="test-doc-id",
            hash="test-hash",
            filename="test.pdf",
            mime_type="application/pdf",
            size=1024,
            uploaded_at=now,
            last_accessed=now,
        )

    def test_session_document_creation(self):
        """Test basic SessionDocument creation."""
        doc_metadata = self.create_test_document_metadata()
        session_doc = SessionDocument(document_metadata=doc_metadata)

        assert session_doc.document_metadata == doc_metadata
        assert session_doc.chunks is None
        assert session_doc.access_level == DocumentAccessLevel.PRIVATE
        assert session_doc.owned is True
        assert session_doc.shared_from is None
        assert len(session_doc.shared_to) == 0

    def test_session_document_properties(self):
        """Test SessionDocument properties."""
        doc_metadata = self.create_test_document_metadata()
        session_doc = SessionDocument(document_metadata=doc_metadata)

        assert session_doc.document_id == "test-doc-id"
        assert session_doc.document_hash == "test-hash"

    def test_session_document_can_share(self):
        """Test can_share logic."""
        doc_metadata = self.create_test_document_metadata()

        # Owned private document - cannot share
        private_doc = SessionDocument(document_metadata=doc_metadata, access_level=DocumentAccessLevel.PRIVATE, owned=True)
        assert not private_doc.can_share()

        # Owned shared document - can share
        shared_doc = SessionDocument(document_metadata=doc_metadata, access_level=DocumentAccessLevel.SHARED, owned=True)
        assert shared_doc.can_share()

        # Not owned document - cannot share
        not_owned_doc = SessionDocument(document_metadata=doc_metadata, access_level=DocumentAccessLevel.SHARED, owned=False)
        assert not not_owned_doc.can_share()

    def test_session_document_sharing_operations(self):
        """Test adding and removing shared sessions."""
        doc_metadata = self.create_test_document_metadata()
        session_doc = SessionDocument(document_metadata=doc_metadata, access_level=DocumentAccessLevel.SHARED, owned=True)

        # Add shared session
        assert session_doc.add_shared_session("session-1")
        assert "session-1" in session_doc.shared_to

        # Add another shared session
        assert session_doc.add_shared_session("session-2")
        assert "session-2" in session_doc.shared_to
        assert len(session_doc.shared_to) == 2

        # Remove shared session
        assert session_doc.remove_shared_session("session-1")
        assert "session-1" not in session_doc.shared_to
        assert "session-2" in session_doc.shared_to
        assert len(session_doc.shared_to) == 1

        # Try to remove non-existent session
        assert not session_doc.remove_shared_session("session-3")

    def test_session_document_sharing_private_document(self):
        """Test that private documents cannot be shared."""
        doc_metadata = self.create_test_document_metadata()
        private_doc = SessionDocument(document_metadata=doc_metadata, access_level=DocumentAccessLevel.PRIVATE, owned=True)

        # Cannot add shared session to private document
        assert not private_doc.add_shared_session("session-1")
        assert len(private_doc.shared_to) == 0

    def test_session_document_accessibility(self):
        """Test is_accessible_by_session logic."""
        doc_metadata = self.create_test_document_metadata()

        # Owned document - accessible
        owned_doc = SessionDocument(document_metadata=doc_metadata, owned=True)
        assert owned_doc.is_accessible_by_session("any-session")

        # Document shared with specific session
        shared_doc = SessionDocument(document_metadata=doc_metadata, access_level=DocumentAccessLevel.SHARED, owned=False)
        shared_doc.shared_to.add("session-1")
        assert shared_doc.is_accessible_by_session("session-1")
        assert not shared_doc.is_accessible_by_session("session-2")

        # Document shared from specific session
        shared_from_doc = SessionDocument(document_metadata=doc_metadata, owned=False, shared_from="source-session")
        assert shared_from_doc.is_accessible_by_session("source-session")
        assert not shared_from_doc.is_accessible_by_session("other-session")


class TestSessionAccessControls:
    """Test cases for SessionAccessControls dataclass."""

    def test_session_access_controls_defaults(self):
        """Test default SessionAccessControls values."""
        controls = SessionAccessControls()

        assert not controls.sharing_enabled
        assert len(controls.allowed_sessions) == 0
        assert controls.max_shared_documents == 10

    def test_session_access_controls_custom(self):
        """Test custom SessionAccessControls values."""
        allowed_sessions = {"session-1", "session-2"}
        controls = SessionAccessControls(sharing_enabled=True, allowed_sessions=allowed_sessions, max_shared_documents=5)

        assert controls.sharing_enabled
        assert controls.allowed_sessions == allowed_sessions
        assert controls.max_shared_documents == 5

    def test_can_share_with_session(self):
        """Test can_share_with_session logic."""
        # Sharing disabled
        controls = SessionAccessControls(sharing_enabled=False)
        assert not controls.can_share_with_session("any-session")

        # Sharing enabled, no restrictions
        controls = SessionAccessControls(sharing_enabled=True)
        assert controls.can_share_with_session("any-session")

        # Sharing enabled with allowed sessions list
        controls = SessionAccessControls(sharing_enabled=True, allowed_sessions={"session-1", "session-2"})
        assert controls.can_share_with_session("session-1")
        assert controls.can_share_with_session("session-2")
        assert not controls.can_share_with_session("session-3")

    def test_allowed_sessions_management(self):
        """Test adding and removing allowed sessions."""
        controls = SessionAccessControls()

        # Add allowed sessions
        controls.add_allowed_session("session-1")
        assert "session-1" in controls.allowed_sessions

        controls.add_allowed_session("session-2")
        assert "session-2" in controls.allowed_sessions
        assert len(controls.allowed_sessions) == 2

        # Remove allowed session
        controls.remove_allowed_session("session-1")
        assert "session-1" not in controls.allowed_sessions
        assert "session-2" in controls.allowed_sessions

        # Remove non-existent session (should not raise error)
        controls.remove_allowed_session("session-3")
        assert len(controls.allowed_sessions) == 1


class TestEnhancedSession:
    """Test cases for EnhancedSession dataclass."""

    def create_test_session_document(self, doc_id: str = "test-doc") -> SessionDocument:
        """Helper to create test session document."""
        now = datetime.utcnow()
        doc_metadata = DocumentMetadata(
            id=doc_id,
            hash=f"hash-{doc_id}",
            filename=f"{doc_id}.pdf",
            mime_type="application/pdf",
            size=1024,
            uploaded_at=now,
            last_accessed=now,
        )
        return SessionDocument(document_metadata=doc_metadata)

    def test_enhanced_session_creation(self):
        """Test basic EnhancedSession creation."""
        session = EnhancedSession(id="test-session")

        assert session.id == "test-session"
        assert session.title == "New Chat"
        assert len(session.messages) == 0
        assert len(session.documents) == 0
        assert isinstance(session.document_access_controls, SessionAccessControls)
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)

    def test_enhanced_session_auto_id_generation(self):
        """Test automatic ID generation when not provided."""
        session = EnhancedSession(id="")

        assert session.id is not None
        assert session.id != ""
        assert len(session.id) > 0

    def test_enhanced_session_document_operations(self):
        """Test document management operations."""
        session = EnhancedSession(id="test-session")
        doc = self.create_test_session_document("doc-1")

        # Add document
        assert session.add_document(doc)
        assert len(session.documents) == 1
        assert "hash-doc-1" in session.documents

        # Try to add same document again (should fail)
        assert not session.add_document(doc)
        assert len(session.documents) == 1

        # Get document
        retrieved_doc = session.get_document("hash-doc-1")
        assert retrieved_doc is not None
        assert retrieved_doc.document_id == "doc-1"

        # Get non-existent document
        assert session.get_document("non-existent") is None

        # Remove document
        assert session.remove_document("hash-doc-1")
        assert len(session.documents) == 0

        # Try to remove non-existent document
        assert not session.remove_document("non-existent")

    def test_enhanced_session_document_filtering(self):
        """Test document filtering methods."""
        session = EnhancedSession(id="test-session")

        # Add owned document
        owned_doc = self.create_test_session_document("owned-doc")
        session.add_document(owned_doc)

        # Add shared document
        shared_doc = self.create_test_session_document("shared-doc")
        shared_doc.owned = False
        shared_doc.shared_from = "other-session"
        session.add_document(shared_doc)

        # Add shareable document
        shareable_doc = self.create_test_session_document("shareable-doc")
        shareable_doc.access_level = DocumentAccessLevel.SHARED
        session.add_document(shareable_doc)

        # Test filtering
        owned_docs = session.get_owned_documents()
        assert len(owned_docs) == 2  # owned-doc and shareable-doc
        assert all(doc.owned for doc in owned_docs)

        shared_docs = session.get_shared_documents()
        assert len(shared_docs) == 1
        assert shared_docs[0].document_id == "shared-doc"

        shareable_docs = session.get_shareable_documents()
        assert len(shareable_docs) == 1
        assert shareable_docs[0].document_id == "shareable-doc"

    def test_enhanced_session_statistics(self):
        """Test session statistics methods."""
        session = EnhancedSession(id="test-session")

        # Initially empty
        assert session.get_document_count() == 0
        assert session.get_total_document_size() == 0

        # Add documents
        doc1 = self.create_test_session_document("doc-1")
        doc1.document_metadata.size = 1000
        doc2 = self.create_test_session_document("doc-2")
        doc2.document_metadata.size = 2000

        session.add_document(doc1)
        session.add_document(doc2)

        assert session.get_document_count() == 2
        assert session.get_total_document_size() == 3000

    def test_enhanced_session_failed_document_cleanup(self):
        """Test cleanup of failed documents."""
        session = EnhancedSession(id="test-session")

        # Add successful document
        success_doc = self.create_test_session_document("success-doc")
        success_doc.document_metadata.processing_status = DocumentProcessingStatus.COMPLETED
        session.add_document(success_doc)

        # Add failed documents
        failed_doc1 = self.create_test_session_document("failed-doc-1")
        failed_doc1.document_metadata.processing_status = DocumentProcessingStatus.FAILED
        session.add_document(failed_doc1)

        failed_doc2 = self.create_test_session_document("failed-doc-2")
        failed_doc2.document_metadata.processing_status = DocumentProcessingStatus.FAILED
        session.add_document(failed_doc2)

        assert session.get_document_count() == 3

        # Cleanup failed documents
        cleaned_count = session.cleanup_failed_documents()
        assert cleaned_count == 2
        assert session.get_document_count() == 1

        # Verify successful document remains
        remaining_docs = list(session.documents.values())
        assert len(remaining_docs) == 1
        assert remaining_docs[0].document_id == "success-doc"

    def test_enhanced_session_activity_tracking(self):
        """Test activity timestamp tracking."""
        session = EnhancedSession(id="test-session")
        initial_activity = session.last_activity

        # Simulate some delay
        import time

        time.sleep(0.01)

        # Activity should update when documents are modified
        doc = self.create_test_session_document("doc-1")
        session.add_document(doc)

        assert session.last_activity > initial_activity

        # Update activity manually
        time.sleep(0.01)
        manual_update_time = datetime.utcnow()
        session.update_activity()

        assert session.last_activity >= manual_update_time

    def test_enhanced_session_to_dict(self):
        """Test session serialization to dictionary."""
        session = EnhancedSession(id="test-session", title="Test Session")
        doc = self.create_test_session_document("doc-1")
        session.add_document(doc)

        # Test full serialization
        session_dict = session.to_dict()

        assert session_dict["id"] == "test-session"
        assert session_dict["title"] == "Test Session"
        assert "created_at" in session_dict
        assert "last_activity" in session_dict
        assert session_dict["document_count"] == 1
        assert session_dict["total_document_size"] == 1024
        assert "messages" in session_dict
        assert "documents" in session_dict

        # Test partial serialization
        session_dict_no_messages = session.to_dict(include_messages=False)
        assert "messages" not in session_dict_no_messages

        session_dict_no_docs = session.to_dict(include_documents=False)
        assert "documents" not in session_dict_no_docs

        # Verify document serialization structure
        doc_dict = session_dict["documents"]["hash-doc-1"]
        assert doc_dict["id"] == "doc-1"
        assert doc_dict["filename"] == "doc-1.pdf"
        assert doc_dict["mime_type"] == "application/pdf"
        assert doc_dict["size"] == 1024
        assert doc_dict["processing_status"] == "pending"
        assert doc_dict["access_level"] == "private"
        assert doc_dict["owned"] is True
        assert doc_dict["shared_to_count"] == 0
        assert doc_dict["chunks_available"] is False
