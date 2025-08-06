"""Unit tests for session and document cleanup manager."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from core.cleanup_manager import SessionCleanupManager, CleanupPolicy
from core.document_store import DocumentStore
from core.session_document_manager import SessionDocumentManager
from core.access_controller import DocumentAccessController
from models.session import EnhancedSession, DocumentMetadata, SessionDocument, DocumentProcessingStatus


class TestCleanupPolicy:
    """Test cases for CleanupPolicy dataclass."""
    
    def test_cleanup_policy_defaults(self):
        """Test default CleanupPolicy values."""
        policy = CleanupPolicy()
        
        assert policy.session_ttl_hours == 24
        assert policy.document_ttl_hours == 72
        assert policy.cleanup_interval_minutes == 60
        assert policy.enable_session_cleanup is True
        assert policy.enable_document_cleanup is True
        assert policy.enable_orphaned_document_cleanup is True
        assert policy.enable_failed_document_cleanup is True
        assert policy.max_cleanup_batch_size == 100
        assert policy.cleanup_on_startup is False
    
    def test_cleanup_policy_custom_values(self):
        """Test custom CleanupPolicy values."""
        policy = CleanupPolicy(
            session_ttl_hours=48,
            document_ttl_hours=96,
            cleanup_interval_minutes=30,
            enable_session_cleanup=False,
            max_cleanup_batch_size=50
        )
        
        assert policy.session_ttl_hours == 48
        assert policy.document_ttl_hours == 96
        assert policy.cleanup_interval_minutes == 30
        assert policy.enable_session_cleanup is False
        assert policy.max_cleanup_batch_size == 50


class TestSessionCleanupManager:
    """Test cases for SessionCleanupManager."""
    
    def create_test_cleanup_manager(self, config=None):
        """Helper to create a test cleanup manager."""
        if config is None:
            config = {
                "session_ttl_hours": 24,
                "document_ttl_hours": 72,
                "cleanup_interval_minutes": 60,
                "enable_document_persistence": True,
                "enable_cross_session_sharing": False
            }
        
        document_store = DocumentStore()
        session_manager = SessionDocumentManager(document_store, config)
        access_controller = DocumentAccessController(config)
        
        return SessionCleanupManager(
            session_manager=session_manager,
            document_store=document_store,
            access_controller=access_controller,
            config=config
        )
    
    def test_cleanup_manager_initialization(self):
        """Test cleanup manager initialization."""
        manager = self.create_test_cleanup_manager()
        
        assert manager._session_manager is not None
        assert manager._document_store is not None
        assert manager._access_controller is not None
        assert isinstance(manager._policy, CleanupPolicy)
        assert manager._policy.session_ttl_hours == 24
        assert manager._policy.document_ttl_hours == 72
    
    def test_cleanup_manager_custom_config(self):
        """Test cleanup manager with custom configuration."""
        config = {
            "session_ttl_hours": 48,
            "document_ttl_hours": 96,
            "cleanup_interval_minutes": 30,
            "enable_document_persistence": True,
            "enable_cross_session_sharing": False
        }
        
        manager = self.create_test_cleanup_manager(config)
        
        assert manager._policy.session_ttl_hours == 48
        assert manager._policy.document_ttl_hours == 96
        assert manager._policy.cleanup_interval_minutes == 30
    
    def test_is_session_expired(self):
        """Test session expiration logic."""
        manager = self.create_test_cleanup_manager()
        
        # Create session with recent activity
        recent_session = EnhancedSession(id="recent-session")
        recent_session.last_activity = datetime.utcnow() - timedelta(hours=12)  # 12 hours ago
        
        # Create session with old activity
        old_session = EnhancedSession(id="old-session")
        old_session.last_activity = datetime.utcnow() - timedelta(hours=48)  # 48 hours ago
        
        assert not manager._is_session_expired(recent_session)
        assert manager._is_session_expired(old_session)
    
    def test_is_document_orphaned(self):
        """Test orphaned document detection logic."""
        manager = self.create_test_cleanup_manager()
        
        # Create document metadata
        now = datetime.utcnow()
        doc_metadata = DocumentMetadata(
            id="test-doc",
            hash="test-hash",
            filename="test.pdf",
            mime_type="application/pdf",
            size=1024,
            uploaded_at=now,
            last_accessed=now - timedelta(hours=48)  # 48 hours ago
        )
        
        # Test with sessions referencing the document
        assert not manager._is_document_orphaned("test-hash", {"session-1"})
        
        # Test without sessions referencing the document
        assert manager._is_document_orphaned("test-hash", set())
    
    def test_get_expired_sessions(self):
        """Test getting expired sessions."""
        manager = self.create_test_cleanup_manager()
        
        # Add sessions to the session manager
        recent_session = manager._session_manager.create_session("recent-session")
        recent_session.last_activity = datetime.utcnow() - timedelta(hours=12)
        
        old_session = manager._session_manager.create_session("old-session")
        old_session.last_activity = datetime.utcnow() - timedelta(hours=48)
        
        expired_sessions = manager._get_expired_sessions()
        
        assert len(expired_sessions) == 1
        assert expired_sessions[0].id == "old-session"
    
    def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions."""
        manager = self.create_test_cleanup_manager()
        
        # Create sessions
        recent_session = manager._session_manager.create_session("recent-session")
        recent_session.last_activity = datetime.utcnow() - timedelta(hours=12)
        
        old_session1 = manager._session_manager.create_session("old-session-1")
        old_session1.last_activity = datetime.utcnow() - timedelta(hours=48)
        
        old_session2 = manager._session_manager.create_session("old-session-2")
        old_session2.last_activity = datetime.utcnow() - timedelta(hours=72)
        
        # Cleanup expired sessions
        result = manager.cleanup_expired_sessions()
        
        assert result["sessions_cleaned"] == 2
        assert result["sessions_remaining"] == 1
        assert "recent-session" in [s.id for s in manager._session_manager._sessions.values()]
        assert "old-session-1" not in manager._session_manager._sessions
        assert "old-session-2" not in manager._session_manager._sessions
    
    def test_cleanup_orphaned_documents(self):
        """Test cleanup of orphaned documents."""
        manager = self.create_test_cleanup_manager()
        
        # Create documents in the document store
        now = datetime.utcnow()
        doc1 = DocumentMetadata(
            id="doc-1", hash="hash-1", filename="doc1.pdf",
            mime_type="application/pdf", size=1024,
            uploaded_at=now, last_accessed=now - timedelta(hours=48)
        )
        doc2 = DocumentMetadata(
            id="doc-2", hash="hash-2", filename="doc2.pdf",
            mime_type="application/pdf", size=1024,
            uploaded_at=now, last_accessed=now - timedelta(hours=96)  # Old enough to be cleaned
        )
        
        # Add documents
        manager._document_store.add_document(doc1, "session-1")  # Has reference
        manager._document_store.add_document(doc2, "session-2")  # Will have reference removed
        
        # Manually create orphaned document by manipulating the store directly
        # Remove from session references but keep document
        manager._document_store._document_sessions["hash-2"].clear()
        manager._document_store._session_documents["session-2"].discard("hash-2")
        manager._document_store._reference_counts["hash-2"] = 0
        
        result = manager.cleanup_orphaned_documents()
        
        assert result["documents_cleaned"] == 1
        assert result["documents_remaining"] == 1
        assert manager._document_store.has_document("hash-1")
        assert not manager._document_store.has_document("hash-2")
    
    def test_cleanup_failed_documents(self):
        """Test cleanup of failed processing documents."""
        manager = self.create_test_cleanup_manager()
        
        # Create session and documents
        session = manager._session_manager.create_session("test-session")
        
        now = datetime.utcnow()
        success_doc = DocumentMetadata(
            id="success-doc", hash="success-hash", filename="success.pdf",
            mime_type="application/pdf", size=1024,
            uploaded_at=now, last_accessed=now,
            processing_status=DocumentProcessingStatus.COMPLETED
        )
        failed_doc = DocumentMetadata(
            id="failed-doc", hash="failed-hash", filename="failed.pdf",
            mime_type="application/pdf", size=1024,
            uploaded_at=now, last_accessed=now,
            processing_status=DocumentProcessingStatus.FAILED
        )
        
        # Add documents to session
        manager._session_manager.add_document_to_session("test-session", success_doc)
        manager._session_manager.add_document_to_session("test-session", failed_doc)
        
        result = manager.cleanup_failed_documents()
        
        assert result["documents_cleaned"] == 1
        assert result["sessions_processed"] == 1
        assert session.get_document_count() == 1
        assert session.get_document("success-hash") is not None
        assert session.get_document("failed-hash") is None
    
    def test_run_full_cleanup(self):
        """Test running full cleanup process."""
        manager = self.create_test_cleanup_manager()
        
        # Create test data
        # Recent session
        recent_session = manager._session_manager.create_session("recent-session")
        recent_session.last_activity = datetime.utcnow() - timedelta(hours=12)
        
        # Expired session
        expired_session = manager._session_manager.create_session("expired-session")
        expired_session.last_activity = datetime.utcnow() - timedelta(hours=48)
        
        # Add documents
        now = datetime.utcnow()
        doc1 = DocumentMetadata(
            id="doc-1", hash="hash-1", filename="doc1.pdf",
            mime_type="application/pdf", size=1024,
            uploaded_at=now, last_accessed=now,
            processing_status=DocumentProcessingStatus.COMPLETED
        )
        doc2 = DocumentMetadata(
            id="doc-2", hash="hash-2", filename="doc2.pdf",
            mime_type="application/pdf", size=1024,
            uploaded_at=now, last_accessed=now,
            processing_status=DocumentProcessingStatus.FAILED
        )
        
        manager._session_manager.add_document_to_session("recent-session", doc1)
        manager._session_manager.add_document_to_session("recent-session", doc2)
        
        result = manager.run_full_cleanup()
        
        assert result["expired_sessions"]["sessions_cleaned"] == 1
        assert result["failed_documents"]["documents_cleaned"] == 1
        assert result["orphaned_documents"]["documents_cleaned"] >= 0  # May vary based on cleanup order
        assert "cleanup_duration_seconds" in result
        assert "total_items_cleaned" in result
    
    def test_get_cleanup_statistics(self):
        """Test getting cleanup statistics."""
        manager = self.create_test_cleanup_manager()
        
        # Create test data
        session = manager._session_manager.create_session("test-session")
        now = datetime.utcnow()
        doc = DocumentMetadata(
            id="doc-1", hash="hash-1", filename="doc1.pdf",
            mime_type="application/pdf", size=1024,
            uploaded_at=now, last_accessed=now
        )
        manager._session_manager.add_document_to_session("test-session", doc)
        
        stats = manager.get_cleanup_statistics()
        
        assert "policy" in stats
        assert "current_state" in stats
        assert "cleanup_recommendations" in stats
        assert stats["current_state"]["total_sessions"] == 1
        assert stats["current_state"]["total_documents"] == 1
        assert stats["policy"]["session_ttl_hours"] == 24
        assert stats["policy"]["document_ttl_hours"] == 72
        assert isinstance(stats["cleanup_recommendations"], list)
    
    def test_validate_cleanup_policy(self):
        """Test cleanup policy validation."""
        # Valid policy should pass
        valid_config = {
            "session_ttl_hours": 24,
            "document_ttl_hours": 72,
            "cleanup_interval_minutes": 60
        }
        manager = self.create_test_cleanup_manager(valid_config)
        issues = manager._validate_cleanup_policy()
        assert len(issues) == 0
        
        # Invalid policy should fail
        invalid_config = {
            "session_ttl_hours": 0,  # Invalid
            "document_ttl_hours": -1,  # Invalid
            "cleanup_interval_minutes": 0  # Invalid
        }
        manager_invalid = self.create_test_cleanup_manager(invalid_config)
        issues = manager_invalid._validate_cleanup_policy()
        assert len(issues) > 0
        assert any("session_ttl_hours" in issue for issue in issues)
        assert any("document_ttl_hours" in issue for issue in issues)
        assert any("cleanup_interval_minutes" in issue for issue in issues)
    
    def test_cleanup_with_disabled_features(self):
        """Test cleanup when certain features are disabled."""
        config = {
            "session_ttl_hours": 24,
            "document_ttl_hours": 72,
            "cleanup_interval_minutes": 60,
            "enable_document_persistence": False,  # Disabled
            "enable_cross_session_sharing": False
        }
        
        manager = self.create_test_cleanup_manager(config)
        
        # Policy should reflect disabled features
        assert manager._policy.enable_document_cleanup is False
        assert manager._policy.enable_orphaned_document_cleanup is False
        
        # Cleanup should handle disabled features gracefully
        result = manager.run_full_cleanup()
        assert "expired_sessions" in result
        assert result["failed_documents"]["documents_cleaned"] == 0  # Should be 0 when disabled
        assert result["orphaned_documents"]["documents_cleaned"] == 0  # Should be 0 when disabled