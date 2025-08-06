"""
Test helper functions for AI Foundation chat application.

This module provides utility functions for testing async endpoints,
SSE streams, file uploads, and API interactions.
"""

import asyncio
import json
from typing import AsyncGenerator, Dict, List, Any, Optional, Union
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
import httpx


class SSEStreamParser:
    """Helper class for parsing Server-Sent Events streams."""
    
    @staticmethod
    def parse_sse_line(line: str) -> Optional[Dict[str, str]]:
        """
        Parse a single SSE line into event data.
        
        Args:
            line: SSE line to parse
            
        Returns:
            Optional[Dict[str, str]]: Parsed event data or None
        """
        line = line.strip()
        if not line:
            return None
            
        if line.startswith("data: "):
            return {"type": "data", "content": line[6:]}
        elif line.startswith("event: "):
            return {"type": "event", "content": line[7:]}
        elif line.startswith("id: "):
            return {"type": "id", "content": line[4:]}
        elif line.startswith("retry: "):
            return {"type": "retry", "content": line[7:]}
        
        return None
    
    @staticmethod
    def parse_sse_stream(stream_text: str) -> List[Dict[str, str]]:
        """
        Parse a complete SSE stream into individual events.
        
        Args:
            stream_text: Complete SSE stream text
            
        Returns:
            List[Dict[str, str]]: List of parsed events
        """
        lines = stream_text.split('\n')
        events = []
        
        for line in lines:
            parsed = SSEStreamParser.parse_sse_line(line)
            if parsed:
                events.append(parsed)
                
        return events
    
    @staticmethod
    def extract_data_content(events: List[Dict[str, str]]) -> List[str]:
        """
        Extract only the data content from SSE events.
        
        Args:
            events: List of parsed SSE events
            
        Returns:
            List[str]: List of data content strings
        """
        return [event["content"] for event in events if event["type"] == "data"]


class MockGeminiAPI:
    """Helper class for mocking Gemini API interactions."""
    
    @staticmethod
    def create_mock_chat_model(response_text: str = "Mock response") -> MagicMock:
        """
        Create a mock ChatGoogleGenerativeAI instance.
        
        Args:
            response_text: Text to return in mock response
            
        Returns:
            MagicMock: Mock chat model
        """
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = response_text
        mock_model.invoke.return_value = mock_response
        
        return mock_model
    
    @staticmethod
    def create_async_mock_chat_model(response_text: str = "Mock async response") -> MagicMock:
        """
        Create a mock async ChatGoogleGenerativeAI instance.
        
        Args:
            response_text: Text to return in mock response
            
        Returns:
            MagicMock: Mock async chat model
        """
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = response_text
        
        async def mock_ainvoke(*args, **kwargs):
            return mock_response
            
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        
        return mock_model
    
    @staticmethod
    def create_streaming_mock_chat_model(response_chunks: List[str]) -> MagicMock:
        """
        Create a mock streaming ChatGoogleGenerativeAI instance.
        
        Args:
            response_chunks: List of text chunks to stream
            
        Returns:
            MagicMock: Mock streaming chat model
        """
        mock_model = MagicMock()
        
        async def mock_astream(*args, **kwargs):
            for chunk in response_chunks:
                mock_chunk = MagicMock()
                mock_chunk.content = chunk
                yield mock_chunk
        
        mock_model.astream = mock_astream
        
        return mock_model


class FileUploadHelper:
    """Helper class for testing file uploads."""
    
    @staticmethod
    def create_multipart_data(
        session_id: str,
        prompt: str,
        file_content: bytes,
        filename: str,
        content_type: str
    ) -> Dict[str, Any]:
        """
        Create multipart form data for file upload testing.
        
        Args:
            session_id: Session ID
            prompt: User prompt
            file_content: File content bytes
            filename: Name of the file
            content_type: MIME type of the file
            
        Returns:
            Dict[str, Any]: Multipart form data
        """
        return {
            "session_id": session_id,
            "prompt": prompt,
            "file": (filename, file_content, content_type)
        }
    
    @staticmethod
    def create_httpx_files_data(
        file_content: bytes,
        filename: str,
        content_type: str
    ) -> Dict[str, Any]:
        """
        Create httpx-compatible files data for async testing.
        
        Args:
            file_content: File content bytes
            filename: Name of the file
            content_type: MIME type of the file
            
        Returns:
            Dict[str, Any]: httpx files data structure
        """
        return {
            "file": (filename, file_content, content_type)
        }


class AsyncTestHelper:
    """Helper class for async testing operations."""
    
    @staticmethod
    async def consume_async_generator(
        async_gen: AsyncGenerator[Any, None]
    ) -> List[Any]:
        """
        Consume an async generator and return all items as a list.
        
        Args:
            async_gen: Async generator to consume
            
        Returns:
            List[Any]: List of all items from the generator
        """
        items = []
        async for item in async_gen:
            items.append(item)
        return items
    
    @staticmethod
    async def wait_for_condition(
        condition_func,
        timeout: float = 5.0,
        poll_interval: float = 0.1
    ) -> bool:
        """
        Wait for a condition to become true within a timeout.
        
        Args:
            condition_func: Function that returns True when condition is met
            timeout: Maximum time to wait in seconds
            poll_interval: Time between condition checks in seconds
            
        Returns:
            bool: True if condition was met, False if timeout occurred
        """
        elapsed = 0.0
        while elapsed < timeout:
            if condition_func():
                return True
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        return False


class AssertionHelper:
    """Helper class for common test assertions."""
    
    @staticmethod
    def assert_session_structure(session: Dict[str, Any]) -> None:
        """
        Assert that a session has the expected structure.
        
        Args:
            session: Session dictionary to validate
            
        Raises:
            AssertionError: If session structure is invalid
        """
        assert "id" in session, "Session must have an 'id' field"
        assert "title" in session, "Session must have a 'title' field"
        assert "messages" in session, "Session must have a 'messages' field"
        assert isinstance(session["messages"], list), "Messages must be a list"
    
    @staticmethod
    def assert_message_structure(message: Dict[str, Any]) -> None:
        """
        Assert that a message has the expected structure.
        
        Args:
            message: Message dictionary to validate
            
        Raises:
            AssertionError: If message structure is invalid
        """
        assert "type" in message, "Message must have a 'type' field"
        assert "content" in message, "Message must have a 'content' field"
        assert message["type"] in ["user", "ai"], "Message type must be 'user' or 'ai'"
    
    @staticmethod
    def assert_sse_event_structure(event: Dict[str, str]) -> None:
        """
        Assert that an SSE event has the expected structure.
        
        Args:
            event: SSE event dictionary to validate
            
        Raises:
            AssertionError: If SSE event structure is invalid
        """
        assert "type" in event, "SSE event must have a 'type' field"
        assert "content" in event, "SSE event must have a 'content' field"
        assert event["type"] in ["data", "event", "id", "retry"], "Invalid SSE event type"
    
    @staticmethod
    def assert_api_response_success(response: Union[httpx.Response, Any]) -> None:
        """
        Assert that an API response indicates success.
        
        Args:
            response: HTTP response to validate
            
        Raises:
            AssertionError: If response indicates failure
        """
        if hasattr(response, 'status_code'):
            assert 200 <= response.status_code < 300, f"Expected success status code, got {response.status_code}"
        
        if hasattr(response, 'json'):
            try:
                json_data = response.json()
                if "error" in json_data:
                    assert False, f"Response contains error: {json_data['error']}"
            except (json.JSONDecodeError, ValueError):
                # Not a JSON response, skip JSON validation
                pass