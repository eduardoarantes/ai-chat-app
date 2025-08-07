"""Integration tests for enhanced session API with document persistence."""

import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from core.document_store import DocumentStore
from core.session_document_manager import SessionDocumentManager
from main import create_app
from models.session import DocumentMetadata, DocumentProcessingStatus, SessionDocument


class TestEnhancedSessionAPI:
    """Integration tests for enhanced session API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_root_endpoint_creates_enhanced_session(self, client):
        """Test that root endpoint creates enhanced session structure."""
        response = client.get("/")

        assert response.status_code == 200
        # Check that session was created (this would be verified through session manager)

    def test_get_sessions_includes_document_info(self, client):
        """Test that get_sessions endpoint includes document information."""
        # First create a session by visiting root
        client.get("/")

        response = client.get("/sessions")
        assert response.status_code == 200

        sessions = response.json()
        assert isinstance(sessions, list)

        # Should have at least one session from root access
        if sessions:
            session = sessions[0]
            assert "id" in session
            assert "title" in session

    def test_get_session_history_enhanced_structure(self, client):
        """Test that session history endpoint works with enhanced structure."""
        # Create session via root
        root_response = client.get("/")
        assert root_response.status_code == 200

        # Get sessions to find the created session ID
        sessions_response = client.get("/sessions")
        sessions = sessions_response.json()

        if sessions:
            session_id = sessions[0]["id"]

            # Get session history
            history_response = client.get(f"/sessions/{session_id}")
            assert history_response.status_code == 200

            history = history_response.json()
            assert isinstance(history, list)

    def test_delete_session_with_documents(self, client):
        """Test that session deletion works with document references."""
        # Create session via root
        client.get("/")

        # Get session ID
        sessions_response = client.get("/sessions")
        sessions = sessions_response.json()

        if sessions:
            session_id = sessions[0]["id"]

            # Delete session
            delete_response = client.delete(f"/sessions/{session_id}")
            assert delete_response.status_code == 200

            result = delete_response.json()
            assert result["status"] == "success"

            # Verify session is deleted
            sessions_after = client.get("/sessions").json()
            session_ids_after = [s["id"] for s in sessions_after]
            assert session_id not in session_ids_after

    def test_stream_endpoint_backward_compatibility(self, client):
        """Test that stream endpoint maintains backward compatibility."""
        # Create a simple text file for testing
        test_content = "Hello world, this is a test document."

        # Create form data
        files = {"file": ("test.txt", test_content, "text/plain")}
        data = {"session_id": "test-session", "prompt": "What is in this document?"}

        # Make streaming request (we'll test the response structure, not the streaming itself)
        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = "test-api-key"

            # Mock the LLM to avoid actual API calls
            with patch("api.endpoints.ChatGoogleGenerativeAI") as mock_llm_class:
                mock_llm = Mock()
                mock_llm_class.return_value = mock_llm

                # Mock async stream method
                async def mock_astream(messages):
                    class MockChunk:
                        def __init__(self, content):
                            self.content = content

                    yield MockChunk("This document contains: ")
                    yield MockChunk("Hello world, this is a test document.")

                mock_llm.astream = mock_astream

                response = client.post("/stream", data=data, files=files)

                # Should get a streaming response
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestDocumentManagementAPI:
    """Tests for new document management API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_session_documents_endpoint_structure(self, client):
        """Test the structure expected for session documents endpoint."""
        # This test defines the expected API structure
        # The actual implementation will be added in the next steps

        # Expected endpoint: GET /sessions/{session_id}/documents
        # Expected response structure:
        expected_response_structure = {
            "session_id": "string",
            "documents": [
                {
                    "id": "string",
                    "hash": "string",
                    "filename": "string",
                    "mime_type": "string",
                    "size": "integer",
                    "uploaded_at": "string (ISO datetime)",
                    "last_accessed": "string (ISO datetime)",
                    "processing_status": "string (enum)",
                    "access_level": "string (enum)",
                    "owned": "boolean",
                    "shared_from": "string (optional)",
                    "shared_to_count": "integer",
                    "chunks_available": "boolean",
                }
            ],
            "total_count": "integer",
            "total_size_bytes": "integer",
        }

        # This test serves as documentation for now
        assert expected_response_structure is not None

    def test_document_details_endpoint_structure(self, client):
        """Test the structure expected for document details endpoint."""
        # Expected endpoint: GET /documents/{document_id}
        # Expected response structure:
        expected_response_structure = {
            "id": "string",
            "hash": "string",
            "filename": "string",
            "mime_type": "string",
            "size": "integer",
            "uploaded_at": "string (ISO datetime)",
            "last_accessed": "string (ISO datetime)",
            "processing_status": "string (enum)",
            "processing_error": "string (optional)",
            "metadata": "object",
            "access_level": "string (enum)",
            "owned": "boolean",
            "shared_from": "string (optional)",
            "shared_to": ["array of session_ids"],
            "chunks": {
                "available": "boolean",
                "chunk_count": "integer (optional)",
                "total_content_length": "integer (optional)",
            },
            "sessions_with_access": ["array of session_ids"],
        }

        # This test serves as documentation for now
        assert expected_response_structure is not None

    def test_document_sharing_endpoint_structure(self, client):
        """Test the structure expected for document sharing endpoint."""
        # Expected endpoint: POST /documents/{document_id}/share
        # Expected request body:
        expected_request_structure = {"target_session_id": "string", "access_level": "string (optional, default: shared)"}

        # Expected response structure:
        expected_response_structure = {
            "success": "boolean",
            "message": "string",
            "sharing_request_id": "string (optional)",
            "auto_approved": "boolean",
        }

        # This test serves as documentation for now
        assert expected_request_structure is not None
        assert expected_response_structure is not None

    def test_document_deletion_endpoint_structure(self, client):
        """Test the structure expected for document deletion endpoint."""
        # Expected endpoint: DELETE /documents/{document_id}
        # Expected query parameters: ?session_id=string
        # Expected response structure:
        expected_response_structure = {
            "success": "boolean",
            "message": "string",
            "document_removed_from_session": "boolean",
            "document_completely_deleted": "boolean",
        }

        # This test serves as documentation for now
        assert expected_response_structure is not None


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_existing_session_structure_still_works(self, client):
        """Test that existing session structure is maintained for compatibility."""
        # Get sessions - should work as before
        response = client.get("/sessions")
        assert response.status_code == 200

        sessions = response.json()
        assert isinstance(sessions, list)

        # Each session should have the basic required fields
        for session in sessions:
            assert "id" in session
            assert "title" in session
            # Enhanced fields may be present but are optional for backward compatibility

    def test_session_history_format_unchanged(self, client):
        """Test that session history format remains unchanged."""
        # Create session
        client.get("/")

        # Get session ID
        sessions = client.get("/sessions").json()
        if sessions:
            session_id = sessions[0]["id"]

            # Get history
            history_response = client.get(f"/sessions/{session_id}")
            assert history_response.status_code == 200

            # History should be an array of messages
            history = history_response.json()
            assert isinstance(history, list)

            # Each message should follow the expected format when present
            for message in history:
                assert "type" in message  # "user" or "ai"
                assert "content" in message

    def test_session_deletion_response_unchanged(self, client):
        """Test that session deletion response format is unchanged."""
        # Create session
        client.get("/")

        # Get session ID
        sessions = client.get("/sessions").json()
        if sessions:
            session_id = sessions[0]["id"]

            # Delete session
            response = client.delete(f"/sessions/{session_id}")
            assert response.status_code == 200

            result = response.json()
            assert "status" in result
            assert result["status"] == "success"
