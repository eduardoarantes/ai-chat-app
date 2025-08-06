"""
Common test fixtures for AI Foundation chat application.

This module provides reusable fixtures for testing various components
of the chat application including sessions, messages, files, and API responses.
"""

import pytest
import base64
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from io import BytesIO


@dataclass
class SessionData:
    """Test session data model."""
    id: str
    title: str
    messages: List[Dict[str, Any]]


class MockFileUpload:
    """Mock file upload data model that mimics FastAPI UploadFile."""
    
    def __init__(self, filename: str, content: bytes, content_type: str):
        self.filename = filename
        self.content = content
        self.content_type = content_type
        self.base64_content = base64.b64encode(content).decode()
        self._position = 0
    
    async def read(self) -> bytes:
        """Read file content."""
        return self.content
    
    async def seek(self, position: int) -> None:
        """Seek to position in file."""
        self._position = position


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
def mock_pdf_file() -> MockFileUpload:
    """
    Fixture providing a mock PDF file for testing.
    
    Returns:
        MockFileUpload: Mock PDF file data
    """
    # Minimal PDF content
    content = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj << /Type /Page /Parent 2 0 R >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer << /Size 4 /Root 1 0 R >>\nstartxref\n149\n%%EOF"
    return MockFileUpload(
        filename="test.pdf",
        content=content,
        content_type="application/pdf"
    )


@pytest.fixture
def large_mock_file() -> MockFileUpload:
    """
    Fixture providing a large mock file for testing file size limits.
    
    Returns:
        MockFileUpload: Large mock file data (1MB)
    """
    content = b"A" * (1024 * 1024)  # 1MB of 'A' characters
    return MockFileUpload(
        filename="large_file.txt",
        content=content,
        content_type="text/plain"
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
def multiple_sessions() -> List[SessionData]:
    """
    Fixture providing multiple test sessions for testing.
    
    Returns:
        List[SessionData]: List of test sessions
    """
    return [
        SessionData(
            id=str(uuid.uuid4()),
            title="First Test Session",
            messages=[
                {"type": "user", "content": "Hello"},
                {"type": "ai", "content": "Hi there!"}
            ]
        ),
        SessionData(
            id=str(uuid.uuid4()),
            title="Second Test Session",
            messages=[
                {"type": "user", "content": "How are you?"},
                {"type": "ai", "content": "I'm doing well, thank you!"}
            ]
        ),
        SessionData(
            id=str(uuid.uuid4()),
            title="Empty Session",
            messages=[]
        )
    ]


@pytest.fixture
def conversation_history() -> List[Dict[str, Any]]:
    """
    Fixture providing a realistic conversation history for testing.
    
    Returns:
        List[Dict[str, Any]]: Conversation history
    """
    return [
        {"type": "user", "content": "Hello, can you help me with Python programming?"},
        {"type": "ai", "content": "Of course! I'd be happy to help you with Python programming. What specific topic or problem would you like assistance with?"},
        {"type": "user", "content": "I want to learn about list comprehensions."},
        {"type": "ai", "content": "List comprehensions are a concise way to create lists in Python. Here's the basic syntax: [expression for item in iterable if condition]. For example: squares = [x**2 for x in range(10)] creates a list of squares from 0 to 81."},
        {"type": "user", "content": "Can you show me a more complex example?"},
        {"type": "ai", "content": "Certainly! Here's a more complex example that filters and transforms data: words = ['apple', 'banana', 'cherry', 'date']; long_words_upper = [word.upper() for word in words if len(word) > 5]. This creates ['BANANA', 'CHERRY'] - only words longer than 5 characters, converted to uppercase."}
    ]


@pytest.fixture
def gemini_api_response() -> Dict[str, Any]:
    """
    Fixture providing a mock Gemini API response for testing.
    
    Returns:
        Dict[str, Any]: Mock API response structure
    """
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "This is a mock response from the Gemini API for testing purposes."
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH", 
                        "probability": "NEGLIGIBLE"
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE"
                    }
                ]
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 25,
            "totalTokenCount": 35
        }
    }