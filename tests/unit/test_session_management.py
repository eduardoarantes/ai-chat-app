"""
Unit tests for session management functionality.

Tests the core session management operations including CRUD operations
on the in-memory chat_sessions dictionary.
"""

import pytest
from api.endpoints import chat_sessions, serialize_message
from langchain.schema import HumanMessage, AIMessage


class TestSessionManagement:
    """Test class for session management operations."""
    
    def test_chat_sessions_initially_empty(self):
        """Test that chat_sessions starts empty in each test."""
        assert len(chat_sessions) == 0
        assert chat_sessions == {}
    
    def test_can_create_session(self, sample_session_id: str, sample_chat_session: dict):
        """Test that we can create a new session."""
        # This test should initially fail since we haven't implemented session creation
        chat_sessions[sample_session_id] = sample_chat_session
        
        assert sample_session_id in chat_sessions
        assert chat_sessions[sample_session_id] == sample_chat_session
        assert len(chat_sessions) == 1
    
    def test_can_retrieve_session(self, populated_session: dict):
        """Test that we can retrieve an existing session."""
        session_id = populated_session["id"]
        
        retrieved_session = chat_sessions.get(session_id)
        
        assert retrieved_session is not None
        assert retrieved_session == populated_session
        assert retrieved_session["title"] == "Test Chat Session"
    
    def test_can_delete_session(self, populated_session: dict):
        """Test that we can delete an existing session."""
        session_id = populated_session["id"]
        
        # Verify session exists
        assert session_id in chat_sessions
        
        # Delete session
        del chat_sessions[session_id]
        
        # Verify session is deleted
        assert session_id not in chat_sessions
        assert len(chat_sessions) == 0
    
    def test_nonexistent_session_returns_none(self):
        """Test that retrieving a nonexistent session returns None."""
        nonexistent_id = "nonexistent-session-id"
        
        result = chat_sessions.get(nonexistent_id)
        
        assert result is None


class TestMessageSerialization:
    """Test class for message serialization functionality."""
    
    def test_serialize_human_message_with_text(self):
        """Test serialization of a simple text HumanMessage."""
        message = HumanMessage(content="Hello, how are you?")
        
        result = serialize_message(message)
        
        expected = {
            "type": "user",
            "content": "Hello, how are you?"
        }
        assert result == expected
    
    def test_serialize_ai_message(self):
        """Test serialization of an AIMessage."""
        message = AIMessage(content="I'm doing well, thank you!")
        
        result = serialize_message(message)
        
        expected = {
            "type": "ai",
            "content": "I'm doing well, thank you!"
        }
        assert result == expected
    
    def test_serialize_human_message_with_list_content(self):
        """Test serialization of HumanMessage with list content (multimodal)."""
        content = [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
        ]
        message = HumanMessage(content=content)
        
        result = serialize_message(message)
        
        expected = {
            "type": "user",
            "content": content
        }
        assert result == expected
    
    def test_serialize_unknown_message_type(self):
        """Test that unknown message types are returned as-is."""
        unknown_message = {"type": "unknown", "content": "test"}
        
        result = serialize_message(unknown_message)
        
        assert result == unknown_message