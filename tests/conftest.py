"""
Test configuration and fixtures for AI Foundation chat application.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import os
import sys
import base64
from typing import AsyncGenerator, Generator, List, Dict, Any
from fastapi.testclient import TestClient
import httpx

# Add the parent directory to the path so we can import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from api.endpoints import chat_sessions
from tests.utils.fixtures import MockFileUpload


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """
    FastAPI test client fixture for synchronous testing.
    
    Yields:
        TestClient: Configured FastAPI test client
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    Async HTTP client fixture for testing async endpoints.
    
    Yields:
        httpx.AsyncClient: Configured async HTTP client
    """
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(autouse=True)
def clean_sessions():
    """
    Auto-used fixture to clean chat sessions before each test.
    
    This ensures test isolation by clearing the in-memory session storage.
    """
    # Clear sessions before test
    chat_sessions.clear()
    yield
    # Clear sessions after test
    chat_sessions.clear()


@pytest.fixture
def sample_session_id() -> str:
    """
    Fixture providing a sample session ID for testing.
    
    Returns:
        str: Sample session ID
    """
    return "test-session-123"


@pytest.fixture
def sample_chat_session(sample_session_id: str) -> dict:
    """
    Fixture providing a sample chat session for testing.
    
    Args:
        sample_session_id: Session ID from fixture
        
    Returns:
        dict: Sample session data
    """
    return {
        "id": sample_session_id,
        "title": "Test Chat Session",
        "messages": [
            {"type": "user", "content": "Hello, how are you?"},
            {"type": "ai", "content": "I'm doing well, thank you! How can I help you today?"}
        ]
    }


@pytest.fixture
def populated_session(sample_chat_session: dict) -> dict:
    """
    Fixture that creates a populated session in the chat_sessions store.
    
    Args:
        sample_chat_session: Sample session data from fixture
        
    Returns:
        dict: The created session data
    """
    session_id = sample_chat_session["id"]
    chat_sessions[session_id] = sample_chat_session
    return sample_chat_session


@pytest.fixture
def mock_text_file() -> MockFileUpload:
    """
    Fixture providing a mock text file for testing.
    
    Returns:
        MockFileUpload: Mock text file data
    """
    content = b"This is a test text file content."
    return MockFileUpload(
        filename="test.txt",
        content=content,
        content_type="text/plain"
    )


@pytest.fixture
def mock_image_file() -> MockFileUpload:
    """
    Fixture providing a mock image file for testing.
    
    Returns:
        MockFileUpload: Mock image file data
    """
    # Simple 1x1 pixel PNG image in bytes
    content = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )
    return MockFileUpload(
        filename="test.png",
        content=content,
        content_type="image/png"
    )


@pytest.fixture
def multimodal_message_content() -> List[Dict[str, Any]]:
    """
    Fixture providing multimodal message content for testing.
    
    Returns:
        List[Dict[str, Any]]: Multimodal message content structure
    """
    return [
        {
            "type": "text",
            "text": "What is in this image?"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            }
        }
    ]


@pytest.fixture
def multiple_sessions() -> List[Dict[str, Any]]:
    """
    Fixture providing multiple test sessions for testing.
    
    Returns:
        List[Dict[str, Any]]: List of test sessions
    """
    import uuid
    return [
        {
            "id": str(uuid.uuid4()),
            "title": "First Test Session",
            "messages": [
                {"type": "user", "content": "Hello"},
                {"type": "ai", "content": "Hi there!"}
            ]
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Second Test Session", 
            "messages": [
                {"type": "user", "content": "How are you?"},
                {"type": "ai", "content": "I'm doing well, thank you!"}
            ]
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Empty Session",
            "messages": []
        }
    ]