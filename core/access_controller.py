"""Access controller for managing cross-session document sharing permissions."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from models.session import DocumentAccessLevel, EnhancedSession, SessionDocument


@dataclass
class SharingRequest:
    """Represents a document sharing request."""

    id: str
    source_session_id: str
    target_session_id: str
    document_hash: str
    requested_at: datetime
    expires_at: Optional[datetime] = None
    approved: Optional[bool] = None
    approved_at: Optional[datetime] = None
    rejected_reason: Optional[str] = None


@dataclass
class AccessPolicy:
    """Access control policy for a session."""

    session_id: str
    allow_sharing_requests: bool = True
    allow_receiving_shares: bool = True
    max_shared_documents: int = 10
    max_received_documents: int = 20
    trusted_sessions: Set[str] = field(default_factory=set)
    blocked_sessions: Set[str] = field(default_factory=set)
    auto_approve_from_trusted: bool = False
    sharing_request_ttl_hours: int = 24


class DocumentAccessController:
    """
    Controls access and sharing permissions for documents across sessions.

    This class provides:
    - Permission validation for document access
    - Cross-session sharing request management
    - Access policy enforcement
    - Audit logging for security compliance
    - Sharing quotas and limits
    """

    def __init__(self, config: dict):
        """
        Initialize the access controller.

        Args:
            config: Access control configuration
        """
        self._config = config
        self._logger = logging.getLogger(__name__)

        # Configuration
        self._enable_cross_session_sharing = config.get("enable_cross_session_sharing", False)
        self._max_shared_documents_per_session = config.get("max_shared_documents_per_session", 10)
        self._default_sharing_request_ttl_hours = config.get("sharing_request_ttl_hours", 24)

        # Storage
        self._access_policies: Dict[str, AccessPolicy] = {}
        self._pending_requests: Dict[str, SharingRequest] = {}  # request_id -> SharingRequest
        self._sharing_history: List[Dict[str, Any]] = []  # Audit log

        # Indexes for fast lookup
        self._requests_by_target_session: Dict[str, Set[str]] = {}  # session_id -> request_ids
        self._requests_by_source_session: Dict[str, Set[str]] = {}  # session_id -> request_ids
        self._shared_documents: Dict[str, Set[str]] = {}  # session_id -> document_hashes

    def get_access_policy(self, session_id: str) -> AccessPolicy:
        """
        Get access policy for a session, creating default if not exists.

        Args:
            session_id: Session ID

        Returns:
            AccessPolicy for the session
        """
        if session_id not in self._access_policies:
            self._access_policies[session_id] = AccessPolicy(
                session_id=session_id,
                max_shared_documents=self._max_shared_documents_per_session,
                sharing_request_ttl_hours=self._default_sharing_request_ttl_hours,
            )

        return self._access_policies[session_id]

    def update_access_policy(self, session_id: str, policy_updates: Dict[str, Any]) -> bool:
        """
        Update access policy for a session.

        Args:
            session_id: Session ID
            policy_updates: Dictionary of policy fields to update

        Returns:
            True if updated successfully
        """
        policy = self.get_access_policy(session_id)

        # Update allowed fields
        allowed_fields = {
            "allow_sharing_requests",
            "allow_receiving_shares",
            "max_shared_documents",
            "max_received_documents",
            "trusted_sessions",
            "blocked_sessions",
            "auto_approve_from_trusted",
            "sharing_request_ttl_hours",
        }

        updated_fields = []
        for field, value in policy_updates.items():
            if field in allowed_fields and hasattr(policy, field):
                setattr(policy, field, value)
                updated_fields.append(field)

        if updated_fields:
            self._logger.info(f"Updated access policy for session {session_id}: {updated_fields}")

        return len(updated_fields) > 0

    def can_access_document(self, session_id: str, document: SessionDocument) -> Tuple[bool, Optional[str]]:
        """
        Check if a session can access a specific document.

        Args:
            session_id: Session requesting access
            document: Document to check access for

        Returns:
            Tuple of (can_access, reason_if_denied)
        """
        # Owner always has access
        if document.owned:
            return True, None

        # Check if document was shared with this session
        if document.is_accessible_by_session(session_id):
            return True, None

        # Check access policy
        policy = self.get_access_policy(session_id)

        if not policy.allow_receiving_shares:
            return False, "Session does not allow receiving shared documents"

        # Check if source session is blocked
        if document.shared_from and document.shared_from in policy.blocked_sessions:
            return False, "Source session is blocked"

        return False, "Document is not shared with this session"

    def can_share_document(
        self, source_session_id: str, target_session_id: str, document: SessionDocument
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a document can be shared between sessions.

        Args:
            source_session_id: Session sharing the document
            target_session_id: Session receiving the document
            document: Document to share

        Returns:
            Tuple of (can_share, reason_if_denied)
        """
        if not self._enable_cross_session_sharing:
            return False, "Cross-session sharing is disabled globally"

        # Get policies
        source_policy = self.get_access_policy(source_session_id)
        target_policy = self.get_access_policy(target_session_id)

        # Check source session permissions
        if not document.can_share():
            return False, "Document cannot be shared (private access level or not owned)"

        # Check if target session is blocked
        if target_session_id in source_policy.blocked_sessions:
            return False, "Target session is blocked by source session"

        if source_session_id in target_policy.blocked_sessions:
            return False, "Source session is blocked by target session"

        # Check target session permissions
        if not target_policy.allow_receiving_shares:
            return False, "Target session does not allow receiving shared documents"

        # Check sharing limits
        current_shared = len(self._shared_documents.get(source_session_id, set()))
        if current_shared >= source_policy.max_shared_documents:
            return False, f"Source session has reached sharing limit ({source_policy.max_shared_documents})"

        return True, None

    def create_sharing_request(
        self, source_session_id: str, target_session_id: str, document_hash: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Create a document sharing request.

        Args:
            source_session_id: Session requesting to share
            target_session_id: Session to share with
            document_hash: Hash of document to share

        Returns:
            Tuple of (success, error_message, request_id)
        """
        if not self._enable_cross_session_sharing:
            return False, "Cross-session sharing is disabled", None

        # Generate request ID
        request_id = f"req_{source_session_id[:8]}_{target_session_id[:8]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Get policies
        source_policy = self.get_access_policy(source_session_id)
        target_policy = self.get_access_policy(target_session_id)

        # Check if target allows sharing requests
        if not target_policy.allow_sharing_requests:
            return False, "Target session does not accept sharing requests", None

        # Create request
        ttl_hours = source_policy.sharing_request_ttl_hours
        expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

        sharing_request = SharingRequest(
            id=request_id,
            source_session_id=source_session_id,
            target_session_id=target_session_id,
            document_hash=document_hash,
            requested_at=datetime.utcnow(),
            expires_at=expires_at,
        )

        # Store request
        self._pending_requests[request_id] = sharing_request

        # Update indexes
        if target_session_id not in self._requests_by_target_session:
            self._requests_by_target_session[target_session_id] = set()
        self._requests_by_target_session[target_session_id].add(request_id)

        if source_session_id not in self._requests_by_source_session:
            self._requests_by_source_session[source_session_id] = set()
        self._requests_by_source_session[source_session_id].add(request_id)

        # Auto-approve if from trusted session
        if target_policy.auto_approve_from_trusted and source_session_id in target_policy.trusted_sessions:
            self.approve_sharing_request(request_id, target_session_id)
            self._logger.info(f"Auto-approved sharing request {request_id} from trusted session")

        self._logger.info(f"Created sharing request {request_id}: {source_session_id} -> {target_session_id}")
        return True, None, request_id

    def approve_sharing_request(self, request_id: str, approving_session_id: str) -> Tuple[bool, Optional[str]]:
        """
        Approve a sharing request.

        Args:
            request_id: Request ID to approve
            approving_session_id: Session ID of the approver

        Returns:
            Tuple of (success, error_message)
        """
        request = self._pending_requests.get(request_id)
        if not request:
            return False, "Sharing request not found"

        # Verify approver is the target session
        if approving_session_id != request.target_session_id:
            return False, "Only target session can approve sharing requests"

        # Check if request has expired
        if request.expires_at and datetime.utcnow() > request.expires_at:
            return False, "Sharing request has expired"

        # Approve request
        request.approved = True
        request.approved_at = datetime.utcnow()

        # Update shared documents index
        if request.source_session_id not in self._shared_documents:
            self._shared_documents[request.source_session_id] = set()
        self._shared_documents[request.source_session_id].add(request.document_hash)

        # Log to audit history
        self._add_to_audit_log(
            "request_approved",
            {
                "request_id": request_id,
                "source_session": request.source_session_id,
                "target_session": request.target_session_id,
                "document_hash": request.document_hash,
                "approved_by": approving_session_id,
                "approved_at": request.approved_at.isoformat(),
            },
        )

        self._logger.info(f"Approved sharing request {request_id}")
        return True, None

    def reject_sharing_request(
        self, request_id: str, rejecting_session_id: str, reason: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Reject a sharing request.

        Args:
            request_id: Request ID to reject
            rejecting_session_id: Session ID of the rejector
            reason: Optional rejection reason

        Returns:
            Tuple of (success, error_message)
        """
        request = self._pending_requests.get(request_id)
        if not request:
            return False, "Sharing request not found"

        # Verify rejector is the target session
        if rejecting_session_id != request.target_session_id:
            return False, "Only target session can reject sharing requests"

        # Reject request
        request.approved = False
        request.approved_at = datetime.utcnow()
        request.rejected_reason = reason or "Request rejected by user"

        # Log to audit history
        self._add_to_audit_log(
            "request_rejected",
            {
                "request_id": request_id,
                "source_session": request.source_session_id,
                "target_session": request.target_session_id,
                "document_hash": request.document_hash,
                "rejected_by": rejecting_session_id,
                "rejected_at": request.approved_at.isoformat(),
                "reason": request.rejected_reason,
            },
        )

        self._logger.info(f"Rejected sharing request {request_id}: {reason}")
        return True, None

    def get_pending_requests_for_session(self, session_id: str, as_target: bool = True) -> List[Dict[str, Any]]:
        """
        Get pending sharing requests for a session.

        Args:
            session_id: Session ID
            as_target: If True, get requests where session is target; if False, get requests where session is source

        Returns:
            List of request information dictionaries
        """
        if as_target:
            request_ids = self._requests_by_target_session.get(session_id, set())
        else:
            request_ids = self._requests_by_source_session.get(session_id, set())

        requests = []
        for request_id in request_ids:
            request = self._pending_requests.get(request_id)
            if request and request.approved is None:  # Still pending
                # Check if expired
                if request.expires_at and datetime.utcnow() > request.expires_at:
                    continue

                requests.append(
                    {
                        "id": request.id,
                        "source_session_id": request.source_session_id,
                        "target_session_id": request.target_session_id,
                        "document_hash": request.document_hash,
                        "requested_at": request.requested_at.isoformat(),
                        "expires_at": request.expires_at.isoformat() if request.expires_at else None,
                    }
                )

        return requests

    def cleanup_expired_requests(self) -> int:
        """
        Clean up expired sharing requests.

        Returns:
            Number of expired requests removed
        """
        now = datetime.utcnow()
        expired_requests = []

        for request_id, request in self._pending_requests.items():
            if request.expires_at and now > request.expires_at and request.approved is None:
                expired_requests.append(request_id)

        # Remove expired requests
        for request_id in expired_requests:
            request = self._pending_requests.pop(request_id, None)
            if request:
                # Clean up indexes
                self._requests_by_target_session.get(request.target_session_id, set()).discard(request_id)
                self._requests_by_source_session.get(request.source_session_id, set()).discard(request_id)

                # Log cleanup
                self._add_to_audit_log(
                    "request_expired",
                    {
                        "request_id": request_id,
                        "source_session": request.source_session_id,
                        "target_session": request.target_session_id,
                        "document_hash": request.document_hash,
                        "expired_at": now.isoformat(),
                    },
                )

        if expired_requests:
            self._logger.info(f"Cleaned up {len(expired_requests)} expired sharing requests")

        return len(expired_requests)

    def revoke_document_sharing(self, source_session_id: str, document_hash: str) -> int:
        """
        Revoke sharing of a document from a source session.

        Args:
            source_session_id: Session that originally shared the document
            document_hash: Hash of document to revoke sharing for

        Returns:
            Number of sharing relationships revoked
        """
        if source_session_id not in self._shared_documents:
            return 0

        if document_hash not in self._shared_documents[source_session_id]:
            return 0

        # Remove from shared documents
        self._shared_documents[source_session_id].discard(document_hash)

        # Log revocation
        self._add_to_audit_log(
            "sharing_revoked",
            {"source_session": source_session_id, "document_hash": document_hash, "revoked_at": datetime.utcnow().isoformat()},
        )

        self._logger.info(f"Revoked sharing for document {document_hash[:8]}... from session {source_session_id}")
        return 1

    def get_access_statistics(self) -> Dict[str, Any]:
        """Get access control statistics."""
        now = datetime.utcnow()

        # Count pending, approved, and rejected requests
        pending_count = sum(1 for req in self._pending_requests.values() if req.approved is None)
        approved_count = sum(1 for req in self._pending_requests.values() if req.approved is True)
        rejected_count = sum(1 for req in self._pending_requests.values() if req.approved is False)
        expired_count = sum(
            1 for req in self._pending_requests.values() if req.expires_at and now > req.expires_at and req.approved is None
        )

        return {
            "cross_session_sharing_enabled": self._enable_cross_session_sharing,
            "total_access_policies": len(self._access_policies),
            "sharing_requests": {
                "total": len(self._pending_requests),
                "pending": pending_count,
                "approved": approved_count,
                "rejected": rejected_count,
                "expired": expired_count,
            },
            "shared_documents": {
                "total_sharing_sessions": len(self._shared_documents),
                "total_shared_documents": sum(len(docs) for docs in self._shared_documents.values()),
            },
            "audit_log_entries": len(self._sharing_history),
        }

    def _add_to_audit_log(self, action: str, details: Dict[str, Any]) -> None:
        """Add entry to audit log."""
        log_entry = {"timestamp": datetime.utcnow().isoformat(), "action": action, "details": details}

        self._sharing_history.append(log_entry)

        # Keep audit log size manageable (keep last 1000 entries)
        if len(self._sharing_history) > 1000:
            self._sharing_history = self._sharing_history[-1000:]
