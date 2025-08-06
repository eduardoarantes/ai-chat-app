"""Session and document cleanup manager for maintaining system health."""

import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

from models.session import EnhancedSession, DocumentProcessingStatus


@dataclass
class CleanupPolicy:
    """Configuration policy for cleanup operations."""
    session_ttl_hours: int = 24
    document_ttl_hours: int = 72
    cleanup_interval_minutes: int = 60
    enable_session_cleanup: bool = True
    enable_document_cleanup: bool = True
    enable_orphaned_document_cleanup: bool = True
    enable_failed_document_cleanup: bool = True
    max_cleanup_batch_size: int = 100
    cleanup_on_startup: bool = False


class SessionCleanupManager:
    """
    Manages cleanup of expired sessions and orphaned documents.
    
    This class provides:
    - TTL-based session expiration and cleanup
    - Orphaned document detection and removal
    - Failed document processing cleanup
    - Resource usage monitoring and optimization
    - Configurable cleanup policies
    - Audit logging for maintenance operations
    """
    
    def __init__(self, session_manager, document_store, access_controller, config: dict):
        """
        Initialize the cleanup manager.
        
        Args:
            session_manager: SessionDocumentManager instance
            document_store: DocumentStore instance
            access_controller: DocumentAccessController instance
            config: Cleanup configuration dictionary
        """
        self._session_manager = session_manager
        self._document_store = document_store
        self._access_controller = access_controller
        self._config = config
        self._logger = logging.getLogger(__name__)
        
        # Create cleanup policy from config
        self._policy = self._create_cleanup_policy(config)
        
        # Runtime statistics
        self._cleanup_stats = {
            "total_cleanups_run": 0,
            "last_cleanup_time": None,
            "total_sessions_cleaned": 0,
            "total_documents_cleaned": 0,
            "total_cleanup_time_seconds": 0.0,
            "last_cleanup_duration": 0.0
        }
        
        # Validate policy
        policy_issues = self._validate_cleanup_policy()
        if policy_issues:
            self._logger.warning(f"Cleanup policy validation issues: {policy_issues}")
        
        self._logger.info(f"Initialized cleanup manager with policy: {self._policy}")
    
    def cleanup_expired_sessions(self) -> Dict[str, Any]:
        """
        Clean up expired sessions based on TTL policy.
        
        Returns:
            Dictionary with cleanup results
        """
        if not self._policy.enable_session_cleanup:
            return {"sessions_cleaned": 0, "sessions_remaining": len(self._session_manager._sessions), "disabled": True}
        
        start_time = time.time()
        expired_sessions = self._get_expired_sessions()
        
        if not expired_sessions:
            return {
                "sessions_cleaned": 0,
                "sessions_remaining": len(self._session_manager._sessions),
                "duration_seconds": time.time() - start_time
            }
        
        # Limit batch size to prevent performance issues
        sessions_to_clean = expired_sessions[:self._policy.max_cleanup_batch_size]
        
        cleaned_count = 0
        for session in sessions_to_clean:
            try:
                success = self._session_manager.delete_session(session.id)
                if success:
                    cleaned_count += 1
                    self._logger.info(f"Cleaned up expired session {session.id} (inactive for "
                                    f"{(datetime.utcnow() - session.last_activity).days} days)")
            except Exception as e:
                self._logger.error(f"Failed to cleanup session {session.id}: {e}")
        
        duration = time.time() - start_time
        self._update_cleanup_stats("sessions", cleaned_count, duration)
        
        return {
            "sessions_cleaned": cleaned_count,
            "sessions_remaining": len(self._session_manager._sessions),
            "duration_seconds": duration,
            "batch_limited": len(expired_sessions) > self._policy.max_cleanup_batch_size
        }
    
    def cleanup_orphaned_documents(self) -> Dict[str, Any]:
        """
        Clean up documents that have no session references.
        
        Returns:
            Dictionary with cleanup results
        """
        if not self._policy.enable_orphaned_document_cleanup:
            return {"documents_cleaned": 0, "documents_remaining": len(self._document_store._documents), "disabled": True}
        
        start_time = time.time()
        orphaned_hashes = self._document_store.get_orphaned_documents()
        
        if not orphaned_hashes:
            return {
                "documents_cleaned": 0,
                "documents_remaining": len(self._document_store._documents),
                "duration_seconds": time.time() - start_time
            }
        
        # Filter orphaned documents by TTL
        ttl_cutoff = datetime.utcnow() - timedelta(hours=self._policy.document_ttl_hours)
        eligible_hashes = []
        
        for doc_hash in orphaned_hashes:
            document = self._document_store.get_document_by_hash(doc_hash)
            if document and document.last_accessed < ttl_cutoff:
                eligible_hashes.append(doc_hash)
        
        # Limit batch size
        hashes_to_clean = eligible_hashes[:self._policy.max_cleanup_batch_size]
        
        cleaned_count = 0
        for doc_hash in hashes_to_clean:
            try:
                if self._document_store._remove_document_completely(doc_hash):
                    cleaned_count += 1
                    self._logger.info(f"Cleaned up orphaned document {doc_hash[:8]}...")
            except Exception as e:
                self._logger.error(f"Failed to cleanup orphaned document {doc_hash[:8]}...: {e}")
        
        duration = time.time() - start_time
        self._update_cleanup_stats("documents", cleaned_count, duration)
        
        return {
            "documents_cleaned": cleaned_count,
            "documents_remaining": len(self._document_store._documents),
            "duration_seconds": duration,
            "batch_limited": len(eligible_hashes) > self._policy.max_cleanup_batch_size
        }
    
    def cleanup_failed_documents(self) -> Dict[str, Any]:
        """
        Clean up documents with failed processing status across all sessions.
        
        Returns:
            Dictionary with cleanup results
        """
        if not self._policy.enable_failed_document_cleanup:
            return {"documents_cleaned": 0, "sessions_processed": 0, "disabled": True}
        
        start_time = time.time()
        total_cleaned = 0
        sessions_processed = 0
        
        for session in self._session_manager._sessions.values():
            try:
                cleaned = session.cleanup_failed_documents()
                total_cleaned += cleaned
                if cleaned > 0:
                    sessions_processed += 1
                    self._logger.info(f"Cleaned up {cleaned} failed documents from session {session.id}")
            except Exception as e:
                self._logger.error(f"Failed to cleanup failed documents in session {session.id}: {e}")
        
        # Also cleanup orphaned documents from the store
        try:
            orphaned_cleaned = self._document_store.cleanup_orphaned_documents()
            total_cleaned += orphaned_cleaned
        except Exception as e:
            self._logger.error(f"Failed to cleanup orphaned documents from store: {e}")
        
        duration = time.time() - start_time
        self._update_cleanup_stats("failed_documents", total_cleaned, duration)
        
        return {
            "documents_cleaned": total_cleaned,
            "sessions_processed": sessions_processed,
            "duration_seconds": duration
        }
    
    def cleanup_expired_sharing_requests(self) -> Dict[str, Any]:
        """
        Clean up expired sharing requests from access controller.
        
        Returns:
            Dictionary with cleanup results
        """
        start_time = time.time()
        
        try:
            expired_count = self._access_controller.cleanup_expired_requests()
            duration = time.time() - start_time
            
            if expired_count > 0:
                self._logger.info(f"Cleaned up {expired_count} expired sharing requests")
            
            return {
                "requests_cleaned": expired_count,
                "duration_seconds": duration
            }
        except Exception as e:
            self._logger.error(f"Failed to cleanup expired sharing requests: {e}")
            return {
                "requests_cleaned": 0,
                "duration_seconds": time.time() - start_time,
                "error": str(e)
            }
    
    def run_full_cleanup(self) -> Dict[str, Any]:
        """
        Run complete cleanup process including all cleanup operations.
        
        Returns:
            Dictionary with comprehensive cleanup results
        """
        overall_start_time = time.time()
        self._logger.info("Starting full cleanup process")
        
        results = {
            "cleanup_started_at": datetime.utcnow().isoformat(),
            "expired_sessions": {},
            "orphaned_documents": {},
            "failed_documents": {},
            "expired_sharing_requests": {}
        }
        
        # Run each cleanup operation
        try:
            results["expired_sessions"] = self.cleanup_expired_sessions()
        except Exception as e:
            self._logger.error(f"Failed to cleanup expired sessions: {e}")
            results["expired_sessions"] = {"error": str(e), "sessions_cleaned": 0}
        
        try:
            results["failed_documents"] = self.cleanup_failed_documents()
        except Exception as e:
            self._logger.error(f"Failed to cleanup failed documents: {e}")
            results["failed_documents"] = {"error": str(e), "documents_cleaned": 0}
        
        try:
            results["orphaned_documents"] = self.cleanup_orphaned_documents()
        except Exception as e:
            self._logger.error(f"Failed to cleanup orphaned documents: {e}")
            results["orphaned_documents"] = {"error": str(e), "documents_cleaned": 0}
        
        try:
            results["expired_sharing_requests"] = self.cleanup_expired_sharing_requests()
        except Exception as e:
            self._logger.error(f"Failed to cleanup expired sharing requests: {e}")
            results["expired_sharing_requests"] = {"error": str(e), "requests_cleaned": 0}
        
        # Calculate totals
        overall_duration = time.time() - overall_start_time
        total_sessions_cleaned = results["expired_sessions"].get("sessions_cleaned", 0)
        total_documents_cleaned = (
            results["orphaned_documents"].get("documents_cleaned", 0) +
            results["failed_documents"].get("documents_cleaned", 0)
        )
        total_requests_cleaned = results["expired_sharing_requests"].get("requests_cleaned", 0)
        
        results.update({
            "cleanup_completed_at": datetime.utcnow().isoformat(),
            "cleanup_duration_seconds": overall_duration,
            "total_items_cleaned": total_sessions_cleaned + total_documents_cleaned + total_requests_cleaned,
            "total_sessions_cleaned": total_sessions_cleaned,
            "total_documents_cleaned": total_documents_cleaned,
            "total_requests_cleaned": total_requests_cleaned
        })
        
        # Update global stats
        self._cleanup_stats["total_cleanups_run"] += 1
        self._cleanup_stats["last_cleanup_time"] = datetime.utcnow()
        self._cleanup_stats["total_sessions_cleaned"] += total_sessions_cleaned
        self._cleanup_stats["total_documents_cleaned"] += total_documents_cleaned
        self._cleanup_stats["total_cleanup_time_seconds"] += overall_duration
        self._cleanup_stats["last_cleanup_duration"] = overall_duration
        
        self._logger.info(f"Full cleanup completed in {overall_duration:.2f}s: "
                         f"{total_sessions_cleaned} sessions, {total_documents_cleaned} documents, "
                         f"{total_requests_cleaned} requests cleaned")
        
        return results
    
    def get_cleanup_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cleanup statistics and recommendations.
        
        Returns:
            Dictionary with statistics and recommendations
        """
        current_sessions = len(self._session_manager._sessions)
        current_documents = len(self._document_store._documents)
        
        # Get expired counts without cleaning
        expired_sessions_count = len(self._get_expired_sessions())
        orphaned_documents_count = len(self._document_store.get_orphaned_documents())
        
        # Calculate resource usage
        total_document_size = sum(
            doc.size for doc in self._document_store._documents.values()
        )
        
        # Generate recommendations
        recommendations = self._generate_cleanup_recommendations(
            expired_sessions_count, orphaned_documents_count, total_document_size
        )
        
        return {
            "policy": {
                "session_ttl_hours": self._policy.session_ttl_hours,
                "document_ttl_hours": self._policy.document_ttl_hours,
                "cleanup_interval_minutes": self._policy.cleanup_interval_minutes,
                "max_cleanup_batch_size": self._policy.max_cleanup_batch_size,
                "features_enabled": {
                    "session_cleanup": self._policy.enable_session_cleanup,
                    "document_cleanup": self._policy.enable_document_cleanup,
                    "orphaned_document_cleanup": self._policy.enable_orphaned_document_cleanup,
                    "failed_document_cleanup": self._policy.enable_failed_document_cleanup
                }
            },
            "current_state": {
                "total_sessions": current_sessions,
                "total_documents": current_documents,
                "expired_sessions": expired_sessions_count,
                "orphaned_documents": orphaned_documents_count,
                "total_document_size_mb": total_document_size / (1024 * 1024),
                "pending_sharing_requests": len(self._access_controller._pending_requests)
            },
            "runtime_stats": self._cleanup_stats.copy(),
            "cleanup_recommendations": recommendations,
            "store_statistics": self._document_store.get_store_statistics(),
            "manager_statistics": self._session_manager.get_manager_statistics()
        }
    
    def _get_expired_sessions(self) -> List[EnhancedSession]:
        """Get list of expired sessions based on TTL."""
        if not self._policy.enable_session_cleanup:
            return []
        
        expired_sessions = []
        ttl_cutoff = datetime.utcnow() - timedelta(hours=self._policy.session_ttl_hours)
        
        for session in self._session_manager._sessions.values():
            if self._is_session_expired(session):
                expired_sessions.append(session)
        
        return expired_sessions
    
    def _is_session_expired(self, session: EnhancedSession) -> bool:
        """Check if a session is expired based on TTL policy."""
        ttl_cutoff = datetime.utcnow() - timedelta(hours=self._policy.session_ttl_hours)
        return session.last_activity < ttl_cutoff
    
    def _is_document_orphaned(self, document_hash: str, referencing_sessions: Set[str]) -> bool:
        """Check if a document is orphaned (no session references)."""
        return len(referencing_sessions) == 0
    
    def _create_cleanup_policy(self, config: dict) -> CleanupPolicy:
        """Create cleanup policy from configuration."""
        return CleanupPolicy(
            session_ttl_hours=config.get("session_ttl_hours", 24),
            document_ttl_hours=config.get("document_ttl_hours", 72),
            cleanup_interval_minutes=config.get("cleanup_interval_minutes", 60),
            enable_session_cleanup=config.get("enable_document_persistence", True),
            enable_document_cleanup=config.get("enable_document_persistence", True),
            enable_orphaned_document_cleanup=config.get("enable_document_persistence", True),
            enable_failed_document_cleanup=True,  # Always enabled for system health
            max_cleanup_batch_size=config.get("max_cleanup_batch_size", 100),
            cleanup_on_startup=config.get("cleanup_on_startup", False)
        )
    
    def _validate_cleanup_policy(self) -> List[str]:
        """
        Validate cleanup policy settings.
        
        Returns:
            List of validation issues found
        """
        issues = []
        
        if self._policy.session_ttl_hours <= 0:
            issues.append("session_ttl_hours must be positive")
        
        if self._policy.document_ttl_hours <= 0:
            issues.append("document_ttl_hours must be positive")
        
        if self._policy.cleanup_interval_minutes <= 0:
            issues.append("cleanup_interval_minutes must be positive")
        
        if self._policy.max_cleanup_batch_size <= 0:
            issues.append("max_cleanup_batch_size must be positive")
        
        # Logical validations
        if self._policy.document_ttl_hours < self._policy.session_ttl_hours:
            issues.append("document_ttl_hours should be >= session_ttl_hours to prevent premature document cleanup")
        
        if self._policy.cleanup_interval_minutes < 5:
            issues.append("cleanup_interval_minutes should be >= 5 to prevent excessive cleanup overhead")
        
        return issues
    
    def _update_cleanup_stats(self, operation: str, items_cleaned: int, duration: float) -> None:
        """Update runtime cleanup statistics."""
        if operation == "sessions":
            self._cleanup_stats["total_sessions_cleaned"] += items_cleaned
        elif operation in ["documents", "failed_documents"]:
            self._cleanup_stats["total_documents_cleaned"] += items_cleaned
        
        self._cleanup_stats["total_cleanup_time_seconds"] += duration
    
    def _generate_cleanup_recommendations(
        self, 
        expired_sessions: int, 
        orphaned_documents: int, 
        total_document_size: int
    ) -> List[str]:
        """Generate cleanup recommendations based on current state."""
        recommendations = []
        
        if expired_sessions > 10:
            recommendations.append(f"Consider running session cleanup - {expired_sessions} expired sessions found")
        
        if orphaned_documents > 20:
            recommendations.append(f"Consider running document cleanup - {orphaned_documents} orphaned documents found")
        
        if total_document_size > 1024 * 1024 * 1024:  # 1GB
            recommendations.append(f"High document storage usage: {total_document_size / (1024**3):.1f}GB - consider cleanup")
        
        if self._policy.session_ttl_hours > 168:  # 1 week
            recommendations.append("Session TTL is quite long - consider shorter TTL to free up memory")
        
        if not self._policy.enable_session_cleanup:
            recommendations.append("Session cleanup is disabled - enable for better resource management")
        
        if not self._policy.enable_orphaned_document_cleanup:
            recommendations.append("Orphaned document cleanup is disabled - enable to prevent storage leaks")
        
        return recommendations