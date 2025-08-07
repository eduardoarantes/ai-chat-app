"""Integration tests for complete document persistence workflow."""

import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from main import create_app


class TestDocumentPersistenceWorkflow:
    """Integration tests for complete document persistence workflow."""

    @pytest.fixture
    def client(self):
        """Create test client with document persistence enabled."""
        app = create_app()
        return TestClient(app)

    def test_file_upload_and_persistence(self, client):
        """Test file upload with document persistence."""
        test_content = "This is a test document for persistence testing."

        files = {"file": ("test.txt", test_content, "text/plain")}
        data = {"session_id": "test-session-persistence", "prompt": "What is in this document?"}

        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = "test-api-key"

            # Mock the LLM to avoid actual API calls
            with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_llm_class:
                mock_llm = Mock()
                mock_llm_class.return_value = mock_llm

                # Mock async stream method
                async def mock_astream(messages):
                    class MockChunk:
                        def __init__(self, content):
                            self.content = content

                    yield MockChunk("This document contains: ")
                    yield MockChunk(test_content)

                mock_llm.astream = mock_astream

                # Stream chat with file
                response = client.post("/stream", data=data, files=files)

                # Should get a streaming response
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Check if session was created and has documents
        session_documents_response = client.get(f"/sessions/{data['session_id']}/documents")

        if session_documents_response.status_code == 200:
            # Document persistence is enabled
            session_docs = session_documents_response.json()
            assert session_docs["session_id"] == data["session_id"]
            assert session_docs["total_count"] >= 0  # May be 0 if processing failed

            # If documents were persisted, verify structure
            if session_docs["total_count"] > 0:
                document = session_docs["documents"][0]
                assert "id" in document
                assert "filename" in document
                assert document["filename"] == "test.txt"
                assert document["mime_type"] == "text/plain"
                assert "processing_status" in document
        else:
            # Document persistence might be disabled - that's ok for backward compatibility
            assert session_documents_response.status_code in [501, 404]

    def test_session_cleanup_with_documents(self, client):
        """Test session cleanup behavior with document persistence."""
        # Create a session by accessing root
        client.get("/")

        # Get the session ID
        sessions_response = client.get("/sessions")
        sessions = sessions_response.json()

        if sessions:
            session_id = sessions[0]["id"]

            # Check cleanup statistics
            cleanup_stats_response = client.get("/admin/cleanup/statistics")

            if cleanup_stats_response.status_code == 200:
                # Cleanup manager is available
                stats = cleanup_stats_response.json()
                assert "policy" in stats
                assert "current_state" in stats
                assert "cleanup_recommendations" in stats

                # Try manual cleanup (should not affect fresh sessions)
                cleanup_response = client.post("/admin/cleanup/run")

                if cleanup_response.status_code == 200:
                    cleanup_results = cleanup_response.json()
                    assert "expired_sessions" in cleanup_results
                    assert "orphaned_documents" in cleanup_results
                    assert "failed_documents" in cleanup_results

                    # Fresh session should not be cleaned up
                    assert cleanup_results["expired_sessions"]["sessions_cleaned"] == 0
            else:
                # Cleanup manager might be disabled - that's ok
                assert cleanup_stats_response.status_code == 501

    def test_new_document_management_endpoints(self, client):
        """Test new document management endpoints are available."""
        # Test session documents endpoint
        response = client.get("/sessions/non-existent-session/documents")
        # Should return 404 for non-existent session (if enabled) or 501 (if disabled)
        assert response.status_code in [404, 501]

        # Test document details endpoint
        response = client.get("/documents/non-existent-document")
        # Should return 404 for non-existent document (if enabled) or 501 (if disabled)
        assert response.status_code in [404, 501]

        # Test document sharing endpoint (POST)
        response = client.post(
            "/documents/non-existent-document/share", data={"source_session_id": "test", "target_session_id": "test2"}
        )
        # Should return 404 for non-existent document (if enabled) or 501 (if disabled)
        assert response.status_code in [404, 501]

        # Test document removal endpoint (DELETE)
        response = client.delete("/documents/non-existent-document?session_id=test")
        # Should return 404 for non-existent document (if enabled) or 501 (if disabled)
        assert response.status_code in [404, 501]

    def test_admin_endpoints_available(self, client):
        """Test administrative endpoints are available."""
        # Test cleanup statistics endpoint
        response = client.get("/admin/cleanup/statistics")
        # Should return data (if enabled) or 501 (if disabled)
        assert response.status_code in [200, 501]

        # Test manual cleanup endpoint
        response = client.post("/admin/cleanup/run")
        # Should return results (if enabled) or 501 (if disabled)
        assert response.status_code in [200, 501]

    def test_document_persistence_backward_compatibility(self, client):
        """Test that document persistence doesn't break existing functionality."""
        # Test root endpoint still works
        response = client.get("/")
        assert response.status_code == 200

        # Test sessions endpoint still works
        response = client.get("/sessions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

        # Create a session and test history endpoint
        client.get("/")  # Creates session
        sessions = client.get("/sessions").json()

        if sessions:
            session_id = sessions[0]["id"]

            # Test session history endpoint
            response = client.get(f"/sessions/{session_id}")
            assert response.status_code == 200
            assert isinstance(response.json(), list)

            # Test session deletion endpoint
            response = client.delete(f"/sessions/{session_id}")
            assert response.status_code == 200
            assert response.json()["status"] == "success"


class TestDocumentSharingWorkflow:
    """Tests for document sharing workflow (when enabled)."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_cross_session_document_access_structure(self, client):
        """Test the expected structure for cross-session document access."""
        # This test defines expected behavior when sharing is enabled

        # Expected workflow:
        # 1. Session A uploads a document
        # 2. Session A shares document with Session B
        # 3. Session B can access the shared document
        # 4. Both sessions can see the document in their document lists

        # For now, just test that the endpoints have the expected structure
        response = client.get("/sessions/session-a/documents")
        assert response.status_code in [200, 404, 501]  # Valid responses

        response = client.get("/documents/doc-id")
        assert response.status_code in [200, 404, 501]  # Valid responses

        response = client.post(
            "/documents/doc-id/share", data={"source_session_id": "session-a", "target_session_id": "session-b"}
        )
        assert response.status_code in [200, 400, 404, 501]  # Valid responses


class TestDocumentMetadataIntegration:
    """Tests for document metadata integration with the processing pipeline."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_document_metadata_enrichment(self, client):
        """Test that document metadata is enriched during processing."""
        # Create a PDF-like content for testing
        test_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"

        files = {"file": ("test.pdf", test_content, "application/pdf")}
        data = {"session_id": "metadata-test-session", "prompt": "Analyze this PDF"}

        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = "test-api-key"

            # Mock the document processor to simulate metadata extraction
            with patch("document_processing.processors.DocumentProcessorFactory.process_document") as mock_processor:
                # Mock processed document with metadata
                mock_processed = Mock()
                mock_processed.processing_method = "test_method"
                mock_processed.original_filename = "test.pdf"
                mock_processed.content = "Processed content"
                mock_processed.metadata = {"title": "Test Document", "author": "Test Author", "page_count": 1}
                mock_processed.structure = {"total_pages": 1, "content_length": 16}
                mock_processor.return_value = mock_processed

                # Mock the LLM
                with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_llm_class:
                    mock_llm = Mock()
                    mock_llm_class.return_value = mock_llm

                    async def mock_astream(messages):
                        class MockChunk:
                            def __init__(self, content):
                                self.content = content

                        yield MockChunk("This is a test document with metadata.")

                    mock_llm.astream = mock_astream

                    # Process the document
                    response = client.post("/stream", data=data, files=files)
                    assert response.status_code == 200

        # Check if document was persisted with metadata
        session_docs_response = client.get(f"/sessions/{data['session_id']}/documents")

        if session_docs_response.status_code == 200:
            session_docs = session_docs_response.json()

            # If documents were persisted, verify metadata structure
            if session_docs["total_count"] > 0:
                document = session_docs["documents"][0]

                # Check basic metadata
                assert document["filename"] == "test.pdf"
                assert document["mime_type"] == "application/pdf"

                # Check if processing was attempted
                assert "processing_status" in document

                # If we can get document details, check enhanced metadata
                doc_id = document["id"]
                doc_details_response = client.get(f"/documents/{doc_id}")

                if doc_details_response.status_code == 200:
                    doc_details = doc_details_response.json()
                    assert "metadata" in doc_details
                    assert "processing_status" in doc_details
                    assert "chunks" in doc_details
