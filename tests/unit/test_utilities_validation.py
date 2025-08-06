"""
Unit tests to validate our test utilities and fixtures are working correctly.

These tests ensure our testing framework components function as expected.
"""

import pytest
import json
from tests.utils.fixtures import MockFileUpload, SessionData
from tests.utils.helpers import SSEStreamParser, MockGeminiAPI, AssertionHelper
from tests.utils.mock_data import MockDataGenerator


class TestFixtures:
    """Test class for validating test fixtures."""
    
    def test_mock_file_upload_fixture(self, mock_text_file: MockFileUpload):
        """Test that mock text file fixture works correctly."""
        assert mock_text_file.filename == "test.txt"
        assert mock_text_file.content_type == "text/plain"
        assert isinstance(mock_text_file.content, bytes)
        assert mock_text_file.base64_content is not None
        assert len(mock_text_file.base64_content) > 0
    
    def test_mock_image_file_fixture(self, mock_image_file: MockFileUpload):
        """Test that mock image file fixture works correctly."""
        assert mock_image_file.filename == "test.png"
        assert mock_image_file.content_type == "image/png"
        assert isinstance(mock_image_file.content, bytes)
        assert mock_image_file.base64_content is not None
    
    def test_multimodal_message_content_fixture(self, multimodal_message_content: list):
        """Test that multimodal message content fixture is structured correctly."""
        assert isinstance(multimodal_message_content, list)
        assert len(multimodal_message_content) == 2
        
        text_part = multimodal_message_content[0]
        assert text_part["type"] == "text"
        assert "text" in text_part
        
        image_part = multimodal_message_content[1]
        assert image_part["type"] == "image_url"
        assert "image_url" in image_part
        assert "url" in image_part["image_url"]
    
    def test_multiple_sessions_fixture(self, multiple_sessions: list):
        """Test that multiple sessions fixture generates valid sessions."""
        assert isinstance(multiple_sessions, list)
        assert len(multiple_sessions) >= 3
        
        for session in multiple_sessions:
            assert "id" in session
            assert "title" in session
            assert "messages" in session
            assert isinstance(session["messages"], list)


class TestSSEStreamParser:
    """Test class for SSE stream parsing utilities."""
    
    def test_parse_sse_data_line(self):
        """Test parsing SSE data lines."""
        line = "data: Hello, world!"
        result = SSEStreamParser.parse_sse_line(line)
        
        assert result is not None
        assert result["type"] == "data"
        assert result["content"] == "Hello, world!"
    
    def test_parse_sse_event_line(self):
        """Test parsing SSE event lines."""
        line = "event: title"
        result = SSEStreamParser.parse_sse_line(line)
        
        assert result is not None
        assert result["type"] == "event"
        assert result["content"] == "title"
    
    def test_parse_empty_line(self):
        """Test parsing empty lines returns None."""
        result = SSEStreamParser.parse_sse_line("")
        assert result is None
        
        result = SSEStreamParser.parse_sse_line("   ")
        assert result is None
    
    def test_parse_sse_stream(self):
        """Test parsing complete SSE stream."""
        stream_text = """data: First chunk
data: Second chunk
event: title
data: New Title
data: [DONE]"""
        
        events = SSEStreamParser.parse_sse_stream(stream_text)
        assert len(events) == 5
        
        data_events = SSEStreamParser.extract_data_content(events)
        assert len(data_events) == 4
        assert data_events[0] == "First chunk"
        assert data_events[1] == "Second chunk"
        assert data_events[2] == "New Title"
        assert data_events[3] == "[DONE]"


class TestMockGeminiAPI:
    """Test class for Gemini API mocking utilities."""
    
    def test_create_mock_chat_model(self):
        """Test creating mock chat model."""
        mock_model = MockGeminiAPI.create_mock_chat_model("Test response")
        
        response = mock_model.invoke(["test prompt"])
        assert response.content == "Test response"
    
    @pytest.mark.asyncio
    async def test_create_async_mock_chat_model(self):
        """Test creating async mock chat model."""
        mock_model = MockGeminiAPI.create_async_mock_chat_model("Async test response")
        
        response = await mock_model.ainvoke(["test prompt"])
        assert response.content == "Async test response"
    
    @pytest.mark.asyncio
    async def test_create_streaming_mock_chat_model(self):
        """Test creating streaming mock chat model."""
        chunks = ["Hello", " world", "!"]
        mock_model = MockGeminiAPI.create_streaming_mock_chat_model(chunks)
        
        collected_chunks = []
        async for chunk in mock_model.astream(["test prompt"]):
            collected_chunks.append(chunk.content)
        
        assert collected_chunks == chunks


class TestAssertionHelper:
    """Test class for assertion helper utilities."""
    
    def test_assert_session_structure_valid(self, sample_chat_session: dict):
        """Test assertion helper with valid session structure."""
        # Should not raise any assertion errors
        AssertionHelper.assert_session_structure(sample_chat_session)
    
    def test_assert_session_structure_invalid(self):
        """Test assertion helper with invalid session structure."""
        invalid_session = {"id": "test", "title": "test"}  # Missing messages
        
        with pytest.raises(AssertionError, match="Session must have a 'messages' field"):
            AssertionHelper.assert_session_structure(invalid_session)
    
    def test_assert_message_structure_valid(self):
        """Test assertion helper with valid message structure."""
        valid_message = {"type": "user", "content": "Hello"}
        
        # Should not raise any assertion errors
        AssertionHelper.assert_message_structure(valid_message)
    
    def test_assert_message_structure_invalid(self):
        """Test assertion helper with invalid message structure."""
        invalid_message = {"type": "invalid", "content": "Hello"}
        
        with pytest.raises(AssertionError, match="Message type must be"):
            AssertionHelper.assert_message_structure(invalid_message)


class TestMockDataGenerator:
    """Test class for mock data generation utilities."""
    
    def test_generate_session_id(self):
        """Test session ID generation."""
        session_id = MockDataGenerator.generate_session_id()
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        # Should be UUID format
        assert len(session_id.split('-')) == 5
    
    def test_generate_session_title(self):
        """Test session title generation."""
        title = MockDataGenerator.generate_session_title()
        assert isinstance(title, str)
        assert len(title) > 0
        
        # Test specific length
        short_title = MockDataGenerator.generate_session_title(length=10)
        assert len(short_title) <= 10
    
    def test_generate_user_message(self):
        """Test user message generation."""
        message = MockDataGenerator.generate_user_message()
        
        AssertionHelper.assert_message_structure(message)
        assert message["type"] == "user"
    
    def test_generate_ai_message(self):
        """Test AI message generation."""
        message = MockDataGenerator.generate_ai_message()
        
        AssertionHelper.assert_message_structure(message)
        assert message["type"] == "ai"
    
    def test_generate_conversation(self):
        """Test conversation generation."""
        conversation = MockDataGenerator.generate_conversation(num_exchanges=3)
        
        assert len(conversation) == 6  # 3 exchanges = 6 messages
        
        # Check alternating pattern
        for i, message in enumerate(conversation):
            if i % 2 == 0:
                assert message["type"] == "user"
            else:
                assert message["type"] == "ai"
    
    def test_generate_session(self):
        """Test complete session generation."""
        session = MockDataGenerator.generate_session(num_messages=4)
        
        AssertionHelper.assert_session_structure(session)
        assert len(session["messages"]) == 4
    
    def test_generate_streaming_response_chunks(self):
        """Test streaming response chunks generation."""
        chunks = MockDataGenerator.generate_streaming_response_chunks()
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_generate_file_upload_scenarios(self):
        """Test file upload scenarios generation."""
        scenarios = MockDataGenerator.generate_file_upload_scenarios()
        
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        
        for scenario in scenarios:
            assert "filename" in scenario
            assert "content" in scenario
            assert "content_type" in scenario
            assert "prompt" in scenario
    
    def test_generate_error_scenarios(self):
        """Test error scenarios generation."""
        scenarios = MockDataGenerator.generate_error_scenarios()
        
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        
        for scenario in scenarios:
            assert "scenario" in scenario
            assert "data" in scenario
            assert "expected_error" in scenario