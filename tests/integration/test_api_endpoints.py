"""
Integration tests for API endpoints.

Tests the API endpoints to ensure they respond correctly and handle
various scenarios including success cases, error cases, and edge cases.
"""

import pytest
import json
from fastapi.testclient import TestClient
from main import app, chat_sessions


class TestRootEndpoint:
    """Test class for the root endpoint."""
    
    def test_root_returns_html(self, test_client: TestClient):
        """Test that the root endpoint returns HTML content."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        # Basic check that it's likely HTML content
        assert "<html" in response.text.lower() or "<!doctype" in response.text.lower()


class TestSessionsEndpoint:
    """Test class for sessions management endpoints."""
    
    def test_get_empty_sessions(self, test_client: TestClient):
        """Test getting sessions when none exist."""
        response = test_client.get("/sessions")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0
    
    def test_get_sessions_with_data(self, test_client: TestClient, populated_session: dict):
        """Test getting sessions when data exists."""
        response = test_client.get("/sessions")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        
        session = data[0]
        assert session["id"] == populated_session["id"]
        assert session["title"] == populated_session["title"]
        # Note: /sessions endpoint only returns id and title, not messages
    
    def test_get_specific_session(self, test_client: TestClient, populated_session: dict):
        """Test getting a specific session by ID."""
        session_id = populated_session["id"]
        response = test_client.get(f"/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        # /sessions/{id} endpoint returns serialized messages only, not the full session
        assert isinstance(data, list)
        assert len(data) == len(populated_session["messages"])
    
    def test_get_nonexistent_session(self, test_client: TestClient):
        """Test getting a session that doesn't exist."""
        response = test_client.get("/sessions/nonexistent-id")
        
        assert response.status_code == 404
        data = response.json()
        # API returns empty list for non-existent sessions
        assert isinstance(data, list)
        assert len(data) == 0
    
    def test_delete_existing_session(self, test_client: TestClient, populated_session: dict):
        """Test deleting an existing session."""
        session_id = populated_session["id"]
        
        # Verify session exists
        response = test_client.get(f"/sessions/{session_id}")
        assert response.status_code == 200
        
        # Delete session
        response = test_client.delete(f"/sessions/{session_id}")
        assert response.status_code == 200
        
        # Verify session is deleted
        response = test_client.get(f"/sessions/{session_id}")
        assert response.status_code == 404
    
    def test_delete_nonexistent_session(self, test_client: TestClient):
        """Test deleting a session that doesn't exist."""
        response = test_client.delete("/sessions/nonexistent-id")
        
        assert response.status_code == 404
        data = response.json()
        assert data["status"] == "error"
        assert "not found" in data["message"].lower()


class TestStreamEndpoint:
    """Test class for the streaming endpoint."""
    
    def test_stream_endpoint_exists(self, test_client: TestClient):
        """Test that the stream endpoint exists and responds to POST."""
        # This test will fail initially since we're not providing required data
        # but it validates the endpoint is accessible
        response = test_client.post("/stream")
        
        # Should return an error but not 404 (which would indicate endpoint doesn't exist)
        assert response.status_code != 404
    
    def test_stream_requires_session_id(self, test_client: TestClient):
        """Test that stream endpoint requires session_id."""
        data = {"prompt": "Hello"}
        response = test_client.post("/stream", data=data)
        
        # Should return an error indicating missing session_id
        assert response.status_code == 422  # Validation error
    
    def test_stream_requires_prompt(self, test_client: TestClient):
        """Test that stream endpoint requires prompt."""
        data = {"session_id": "test-123"}
        response = test_client.post("/stream", data=data)
        
        # Should return an error indicating missing prompt
        assert response.status_code == 422  # Validation error


class TestApplicationHealth:
    """Test class for application health and basic functionality."""
    
    def test_app_starts_successfully(self, test_client: TestClient):
        """Test that the application starts and responds to requests."""
        response = test_client.get("/")
        assert response.status_code == 200
    
    def test_sessions_storage_is_clean(self):
        """Test that sessions storage starts clean for each test."""
        assert len(chat_sessions) == 0
        assert chat_sessions == {}
    
    def test_can_create_and_retrieve_session_data(self, sample_session_id: str):
        """Test basic session CRUD operations."""
        # Create session
        test_session = {
            "id": sample_session_id,
            "title": "Test Session",
            "messages": []
        }
        chat_sessions[sample_session_id] = test_session
        
        # Retrieve session
        retrieved = chat_sessions.get(sample_session_id)
        assert retrieved is not None
        assert retrieved["id"] == sample_session_id
        assert retrieved["title"] == "Test Session"
        
        # Clean up
        del chat_sessions[sample_session_id]
        assert sample_session_id not in chat_sessions