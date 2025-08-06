"""
Unit tests for comprehensive error handling infrastructure.

Tests cover:
- Enhanced exception hierarchy with system-wide error types
- Error context capture and structured error data
- Error severity and categorization
- User-friendly message translation
- Error recovery decision logic
"""

import pytest
import datetime
import uuid
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

# Import the error handling components we'll implement
from models.errors import (
    # Existing exceptions from TASK-001
    FileValidationError, FileSizeError, MimeTypeError, SecurityError, ContentValidationError,
    # New exceptions
    ApplicationError, SessionError, LLMError, NetworkError, ConfigurationError,
    # Error handling infrastructure 
    ErrorSeverity, ErrorCategory, ErrorContext, ErrorResult
)
from error_handling.handlers import (
    ErrorHandler, ErrorMessageTranslator, ErrorContextCapture
)


class TestErrorSeverityAndCategory:
    """Test error severity and category enums."""
    
    def test_error_severity_values(self):
        """Test ErrorSeverity enum has correct values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"
    
    def test_error_category_values(self):
        """Test ErrorCategory enum has correct values."""
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.SESSION.value == "session"
        assert ErrorCategory.LLM.value == "llm"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.SYSTEM.value == "system"


class TestErrorContext:
    """Test ErrorContext data structure."""
    
    def test_error_context_creation(self):
        """Test ErrorContext can be created with all required fields."""
        error_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now()
        
        context = ErrorContext(
            error_id=error_id,
            timestamp=timestamp,
            session_id="test-session",
            request_id="test-request",
            user_agent="Test/1.0",
            endpoint="/test",
            stack_trace="test traceback",
            request_data={"key": "value"}
        )
        
        assert context.error_id == error_id
        assert context.timestamp == timestamp
        assert context.session_id == "test-session"
        assert context.request_id == "test-request"
        assert context.user_agent == "Test/1.0"
        assert context.endpoint == "/test"
        assert context.stack_trace == "test traceback"
        assert context.request_data == {"key": "value"}
    
    def test_error_context_optional_fields(self):
        """Test ErrorContext with optional fields as None."""
        error_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now()
        
        context = ErrorContext(
            error_id=error_id,
            timestamp=timestamp,
            session_id=None,
            request_id=None,
            user_agent=None,
            endpoint=None,
            stack_trace=None,
            request_data={}
        )
        
        assert context.error_id == error_id
        assert context.timestamp == timestamp
        assert context.session_id is None
        assert context.request_id is None
        assert context.user_agent is None
        assert context.endpoint is None
        assert context.stack_trace is None
        assert context.request_data == {}


class TestErrorResult:
    """Test ErrorResult data structure."""
    
    def test_error_result_creation(self):
        """Test ErrorResult can be created with all required fields."""
        context = ErrorContext(
            error_id="test-error",
            timestamp=datetime.datetime.now(),
            session_id=None,
            request_id=None,
            user_agent=None,
            endpoint=None,
            stack_trace=None,
            request_data={}
        )
        
        result = ErrorResult(
            error_code="TEST_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            technical_message="Technical error message",
            user_message="User-friendly error message",
            suggested_actions=["Action 1", "Action 2"],
            context=context,
            recoverable=True,
            retry_after=30
        )
        
        assert result.error_code == "TEST_ERROR"
        assert result.severity == ErrorSeverity.HIGH
        assert result.category == ErrorCategory.VALIDATION
        assert result.technical_message == "Technical error message"
        assert result.user_message == "User-friendly error message"
        assert result.suggested_actions == ["Action 1", "Action 2"]
        assert result.context == context
        assert result.recoverable is True
        assert result.retry_after == 30


class TestEnhancedExceptionHierarchy:
    """Test enhanced exception hierarchy building on TASK-001."""
    
    def test_application_error_base(self):
        """Test ApplicationError base exception with all attributes."""
        error = ApplicationError(
            message="Test error",
            error_code="TEST_001",
            severity=ErrorSeverity.HIGH,
            user_message="Something went wrong",
            suggested_actions=["Try again", "Contact support"]
        )
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST_001"
        assert error.severity == ErrorSeverity.HIGH
        assert error.user_message == "Something went wrong"
        assert error.suggested_actions == ["Try again", "Contact support"]
    
    def test_application_error_defaults(self):
        """Test ApplicationError with default values."""
        error = ApplicationError(
            message="Test error",
            error_code="TEST_001",
            severity=ErrorSeverity.MEDIUM
        )
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST_001"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.user_message == "Test error"  # defaults to message
        assert error.suggested_actions == []  # defaults to empty list
    
    def test_session_error_inheritance(self):
        """Test SessionError inherits from ApplicationError."""
        error = SessionError(
            message="Session expired",
            error_code="SESSION_001",
            severity=ErrorSeverity.MEDIUM,
            user_message="Your session has expired. Please refresh the page.",
            suggested_actions=["Refresh the page", "Start a new session"]
        )
        
        assert isinstance(error, ApplicationError)
        assert str(error) == "Session expired"
        assert error.error_code == "SESSION_001"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.user_message == "Your session has expired. Please refresh the page."
        assert "Refresh the page" in error.suggested_actions
    
    def test_llm_error_inheritance(self):
        """Test LLMError inherits from ApplicationError."""
        error = LLMError(
            message="API rate limit exceeded",
            error_code="LLM_001",
            severity=ErrorSeverity.HIGH,
            user_message="The AI service is currently busy. Please wait a moment.",
            suggested_actions=["Wait and try again", "Try a shorter prompt"]
        )
        
        assert isinstance(error, ApplicationError)
        assert str(error) == "API rate limit exceeded"
        assert error.error_code == "LLM_001"
        assert error.severity == ErrorSeverity.HIGH
        assert error.user_message == "The AI service is currently busy. Please wait a moment."
        assert "Wait and try again" in error.suggested_actions
    
    def test_network_error_inheritance(self):
        """Test NetworkError inherits from ApplicationError."""
        error = NetworkError(
            message="Connection timeout",
            error_code="NETWORK_001",
            severity=ErrorSeverity.HIGH,
            user_message="Network connection failed. Check your internet connection.",
            suggested_actions=["Check your connection", "Try again later"]
        )
        
        assert isinstance(error, ApplicationError)
        assert str(error) == "Connection timeout"
        assert error.error_code == "NETWORK_001"
        assert error.severity == ErrorSeverity.HIGH
        assert error.user_message == "Network connection failed. Check your internet connection."
        assert "Check your connection" in error.suggested_actions
    
    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from ApplicationError."""
        error = ConfigurationError(
            message="Missing API key",
            error_code="CONFIG_001",
            severity=ErrorSeverity.CRITICAL,
            user_message="Service configuration error. Please contact support.",
            suggested_actions=["Contact support", "Try again later"]
        )
        
        assert isinstance(error, ApplicationError)
        assert str(error) == "Missing API key"
        assert error.error_code == "CONFIG_001"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.user_message == "Service configuration error. Please contact support."
        assert "Contact support" in error.suggested_actions


class TestErrorContextCapture:
    """Test error context capture functionality."""
    
    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI Request object."""
        request = Mock()
        request.url.path = "/test/endpoint"
        request.method = "POST"
        request.headers = {"user-agent": "Test/1.0", "content-type": "application/json"}
        request.client.host = "127.0.0.1"
        return request
    
    def test_error_context_capture_creation(self):
        """Test ErrorContextCapture can be created."""
        capture = ErrorContextCapture()
        assert capture is not None
    
    @pytest.mark.asyncio
    async def test_capture_request_context(self, mock_request):
        """Test capturing request context from FastAPI request."""
        capture = ErrorContextCapture()
        session_id = "test-session-123"
        
        with patch('main.uuid.uuid4', return_value=Mock(hex='test-error-id')):
            with patch('main.datetime.datetime') as mock_datetime:
                mock_now = datetime.datetime(2024, 1, 1, 12, 0, 0)
                mock_datetime.now.return_value = mock_now
                
                context = await capture.capture_request_context(
                    request=mock_request,
                    session_id=session_id,
                    additional_data={"custom": "data"}
                )
        
        assert context.error_id == "test-error-id"
        assert context.timestamp == mock_now
        assert context.session_id == session_id
        assert context.endpoint == "/test/endpoint"
        assert context.user_agent == "Test/1.0"
        assert context.request_data["custom"] == "data"
        assert context.request_data["method"] == "POST"
        assert context.request_data["client_host"] == "127.0.0.1"
    
    @pytest.mark.asyncio
    async def test_capture_exception_context(self):
        """Test capturing exception context with stack trace."""
        capture = ErrorContextCapture()
        
        try:
            raise ValueError("Test exception")
        except Exception as e:
            with patch('main.uuid.uuid4', return_value=Mock(hex='test-error-id')):
                with patch('main.datetime.datetime') as mock_datetime:
                    mock_now = datetime.datetime(2024, 1, 1, 12, 0, 0)
                    mock_datetime.now.return_value = mock_now
                    
                    context = await capture.capture_exception_context(
                        exception=e,
                        session_id="test-session"
                    )
        
        assert context.error_id == "test-error-id"
        assert context.timestamp == mock_now
        assert context.session_id == "test-session"
        assert context.stack_trace is not None
        assert "ValueError: Test exception" in context.stack_trace
        assert "test_capture_exception_context" in context.stack_trace


class TestErrorMessageTranslator:
    """Test error message translation from technical to user-friendly messages."""
    
    def test_translator_creation(self):
        """Test ErrorMessageTranslator can be created."""
        translator = ErrorMessageTranslator()
        assert translator is not None
    
    def test_translate_file_validation_error(self):
        """Test translating file validation errors."""
        translator = ErrorMessageTranslator()
        
        error = FileSizeError("File size 52428800 bytes exceeds maximum allowed size 50000000 bytes")
        result = translator.translate_error(error)
        
        assert result.user_message != str(error)  # Should be different from technical message
        assert "too large" in result.user_message.lower()
        assert result.suggested_actions
        assert any("smaller file" in action.lower() for action in result.suggested_actions)
    
    def test_translate_session_error(self):
        """Test translating session errors."""
        translator = ErrorMessageTranslator()
        
        error = SessionError(
            message="Session not found in storage",
            error_code="SESSION_001",
            severity=ErrorSeverity.MEDIUM
        )
        result = translator.translate_error(error)
        
        assert result.user_message != str(error)
        assert "session" in result.user_message.lower()
        assert result.suggested_actions
        assert any("refresh" in action.lower() or "reload" in action.lower() for action in result.suggested_actions)
    
    def test_translate_llm_error(self):
        """Test translating LLM errors."""
        translator = ErrorMessageTranslator()
        
        error = LLMError(
            message="Google API rate limit exceeded: 429",
            error_code="LLM_001",
            severity=ErrorSeverity.HIGH
        )
        result = translator.translate_error(error)
        
        assert result.user_message != str(error)
        assert "busy" in result.user_message.lower() or "limit" in result.user_message.lower()
        assert result.suggested_actions
        assert any("wait" in action.lower() or "try again" in action.lower() for action in result.suggested_actions)
    
    def test_translate_network_error(self):
        """Test translating network errors."""
        translator = ErrorMessageTranslator()
        
        error = NetworkError(
            message="Connection timed out after 30 seconds",
            error_code="NETWORK_001",
            severity=ErrorSeverity.HIGH
        )
        result = translator.translate_error(error)
        
        assert result.user_message != str(error)
        assert "connection" in result.user_message.lower() or "network" in result.user_message.lower()
        assert result.suggested_actions
        assert any("connection" in action.lower() or "internet" in action.lower() for action in result.suggested_actions)
    
    def test_translate_unknown_error(self):
        """Test translating unknown/generic errors."""
        translator = ErrorMessageTranslator()
        
        error = Exception("Unknown error occurred")
        result = translator.translate_error(error)
        
        assert result.user_message != str(error)
        assert "unexpected error" in result.user_message.lower() or "something went wrong" in result.user_message.lower()
        assert result.suggested_actions
        assert any("try again" in action.lower() or "support" in action.lower() for action in result.suggested_actions)


class TestErrorHandler:
    """Test the main ErrorHandler class that orchestrates error processing."""
    
    @pytest.fixture
    def error_handler(self):
        """Create ErrorHandler instance for testing."""
        return ErrorHandler()
    
    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI Request object."""
        request = Mock()
        request.url.path = "/test/endpoint"
        request.method = "POST"
        request.headers = {"user-agent": "Test/1.0", "content-type": "application/json"}
        request.client.host = "127.0.0.1"
        return request
    
    def test_error_handler_creation(self, error_handler):
        """Test ErrorHandler can be created."""
        assert error_handler is not None
        assert hasattr(error_handler, 'context_capture')
        assert hasattr(error_handler, 'message_translator')
    
    @pytest.mark.asyncio
    async def test_process_error_with_request_context(self, error_handler, mock_request):
        """Test processing error with request context."""
        error = SessionError(
            message="Session expired",
            error_code="SESSION_001",
            severity=ErrorSeverity.MEDIUM
        )
        
        with patch.object(error_handler.context_capture, 'capture_request_context') as mock_capture:
            mock_context = ErrorContext(
                error_id="test-error-id",
                timestamp=datetime.datetime.now(),
                session_id="test-session",
                request_id=None,
                user_agent="Test/1.0",
                endpoint="/test/endpoint",
                stack_trace=None,
                request_data={}
            )
            mock_capture.return_value = mock_context
            
            result = await error_handler.process_error(
                error=error,
                request=mock_request,
                session_id="test-session"
            )
        
        assert isinstance(result, ErrorResult)
        assert result.error_code == "SESSION_001"
        assert result.severity == ErrorSeverity.MEDIUM
        assert result.category == ErrorCategory.SESSION
        assert result.context == mock_context
        assert result.technical_message == "Session expired"
        assert result.user_message != "Session expired"  # Should be translated
    
    @pytest.mark.asyncio
    async def test_process_error_without_request_context(self, error_handler):
        """Test processing error without request context."""
        error = LLMError(
            message="API error",
            error_code="LLM_001",
            severity=ErrorSeverity.HIGH
        )
        
        with patch.object(error_handler.context_capture, 'capture_exception_context') as mock_capture:
            mock_context = ErrorContext(
                error_id="test-error-id",
                timestamp=datetime.datetime.now(),
                session_id=None,
                request_id=None,
                user_agent=None,
                endpoint=None,
                stack_trace="mock stack trace",
                request_data={}
            )
            mock_capture.return_value = mock_context
            
            result = await error_handler.process_error(error=error)
        
        assert isinstance(result, ErrorResult)
        assert result.error_code == "LLM_001"
        assert result.severity == ErrorSeverity.HIGH
        assert result.category == ErrorCategory.LLM
        assert result.context == mock_context
    
    @pytest.mark.asyncio
    async def test_determine_error_category(self, error_handler):
        """Test error category determination from exception type."""
        # Test file validation error category
        file_error = FileSizeError("File too large")
        file_result = await error_handler.process_error(error=file_error)
        assert file_result.category == ErrorCategory.VALIDATION
        
        # Test session error category
        session_error = SessionError("Session error", "SESSION_001", ErrorSeverity.MEDIUM)
        session_result = await error_handler.process_error(error=session_error)
        assert session_result.category == ErrorCategory.SESSION
        
        # Test LLM error category
        llm_error = LLMError("LLM error", "LLM_001", ErrorSeverity.HIGH)
        llm_result = await error_handler.process_error(error=llm_error)
        assert llm_result.category == ErrorCategory.LLM
        
        # Test network error category
        network_error = NetworkError("Network error", "NETWORK_001", ErrorSeverity.HIGH)
        network_result = await error_handler.process_error(error=network_error)
        assert network_result.category == ErrorCategory.NETWORK
        
        # Test configuration error category
        config_error = ConfigurationError("Config error", "CONFIG_001", ErrorSeverity.CRITICAL)
        config_result = await error_handler.process_error(error=config_error)
        assert config_result.category == ErrorCategory.CONFIGURATION
        
        # Test unknown error category
        unknown_error = Exception("Unknown error")
        unknown_result = await error_handler.process_error(error=unknown_error)
        assert unknown_result.category == ErrorCategory.SYSTEM
    
    @pytest.mark.asyncio
    async def test_determine_recoverability(self, error_handler):
        """Test determining if errors are recoverable."""
        # Network errors should be recoverable
        network_error = NetworkError("Connection timeout", "NETWORK_001", ErrorSeverity.HIGH)
        network_result = await error_handler.process_error(error=network_error)
        assert network_result.recoverable is True
        assert network_result.retry_after is not None
        
        # LLM rate limit errors should be recoverable
        llm_error = LLMError("Rate limit", "LLM_001", ErrorSeverity.HIGH)
        llm_result = await error_handler.process_error(error=llm_error)
        assert llm_result.recoverable is True
        
        # Configuration errors should not be recoverable
        config_error = ConfigurationError("Missing API key", "CONFIG_001", ErrorSeverity.CRITICAL)
        config_result = await error_handler.process_error(error=config_error)
        assert config_result.recoverable is False
        assert config_result.retry_after is None
        
        # File validation errors should not be recoverable (user needs to change file)
        file_error = FileSizeError("File too large")
        file_result = await error_handler.process_error(error=file_error)
        assert file_result.recoverable is False