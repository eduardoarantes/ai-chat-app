"""
Unit tests for centralized error handling middleware and SSE error streaming.

Tests cover:
- FastAPI exception handlers for different error types
- Error middleware for request context capture
- Enhanced SSE error streaming with structured error data
- Error response formatting and status codes
- Middleware integration with existing endpoints
"""

import pytest

pytestmark = pytest.mark.error_handling

import asyncio
import datetime
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from error_handling.handlers import ErrorHandler, ErrorHandlingMiddleware, ErrorResponseHandler, SSEErrorStreamer

# Import the error handling components
from models.errors import (
    ApplicationError,
    ConfigurationError,
    ContentValidationError,
    ErrorCategory,
    ErrorContext,
    ErrorResult,
    ErrorSeverity,
    FileSizeError,
    LLMError,
    MimeTypeError,
    NetworkError,
    SecurityError,
    SessionError,
)


class TestErrorHandlingMiddleware:
    """Test the centralized error handling middleware."""

    @pytest.fixture
    def app(self):
        """Create a test FastAPI app."""
        return FastAPI()

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI Request object."""
        request = Mock(spec=Request)
        request.url.path = "/test/endpoint"
        request.method = "POST"
        request.headers = {"user-agent": "Test/1.0", "content-type": "application/json"}
        request.client.host = "127.0.0.1"
        request.session = {"session_id": "test-session-123"}
        return request

    @pytest.fixture
    def error_middleware(self):
        """Create ErrorHandlingMiddleware instance."""
        app = FastAPI()
        error_handler = ErrorHandler()
        return ErrorHandlingMiddleware(app, error_handler=error_handler)

    def test_middleware_creation(self, error_middleware):
        """Test ErrorHandlingMiddleware can be created."""
        assert error_middleware is not None
        assert hasattr(error_middleware, "error_handler")
        assert isinstance(error_middleware.error_handler, ErrorHandler)

    @pytest.mark.asyncio
    async def test_middleware_captures_application_errors(self, error_middleware, mock_request):
        """Test middleware captures and processes ApplicationError exceptions."""

        # Create a call_next function that raises an ApplicationError
        async def call_next(request):
            raise SessionError(
                message="Session expired",
                error_code="SESSION_001",
                severity=ErrorSeverity.MEDIUM,
                user_message="Your session has expired",
                suggested_actions=["Refresh the page"],
            )

        response = await error_middleware.dispatch(mock_request, call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400  # Bad Request for session errors

        # Parse response content
        response_data = json.loads(response.body.decode())
        assert response_data["error_code"] == "SESSION_001"
        assert response_data["user_message"] == "Your session has expired"
        assert response_data["suggested_actions"] == ["Refresh the page"]
        assert response_data["category"] == "session"
        assert response_data["severity"] == "medium"
        assert response_data["recoverable"] is True

    @pytest.mark.asyncio
    async def test_middleware_captures_file_validation_errors(self, error_middleware, mock_request):
        """Test middleware captures and processes file validation errors."""

        async def call_next(request):
            raise FileSizeError("File size 52428800 bytes exceeds maximum allowed size 50000000 bytes")

        response = await error_middleware.dispatch(mock_request, call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400

        response_data = json.loads(response.body.decode())
        assert response_data["category"] == "validation"
        assert "too large" in response_data["user_message"].lower()
        assert response_data["recoverable"] is False

    @pytest.mark.asyncio
    async def test_middleware_captures_network_errors(self, error_middleware, mock_request):
        """Test middleware captures and processes network errors."""

        async def call_next(request):
            raise NetworkError(message="Connection timeout", error_code="NETWORK_001", severity=ErrorSeverity.HIGH)

        response = await error_middleware.dispatch(mock_request, call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 503  # Service Unavailable

        response_data = json.loads(response.body.decode())
        assert response_data["error_code"] == "NETWORK_001"
        assert response_data["category"] == "network"
        assert response_data["severity"] == "high"
        assert response_data["recoverable"] is True
        assert response_data["retry_after"] == 30

    @pytest.mark.asyncio
    async def test_middleware_captures_llm_errors(self, error_middleware, mock_request):
        """Test middleware captures and processes LLM errors."""

        async def call_next(request):
            raise LLMError(message="Rate limit exceeded", error_code="LLM_001", severity=ErrorSeverity.HIGH)

        response = await error_middleware.dispatch(mock_request, call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 429  # Too Many Requests

        response_data = json.loads(response.body.decode())
        assert response_data["error_code"] == "LLM_001"
        assert response_data["category"] == "llm"
        assert response_data["recoverable"] is True
        assert response_data["retry_after"] == 60

    @pytest.mark.asyncio
    async def test_middleware_captures_configuration_errors(self, error_middleware, mock_request):
        """Test middleware captures and processes configuration errors."""

        async def call_next(request):
            raise ConfigurationError(message="Missing API key", error_code="CONFIG_001", severity=ErrorSeverity.CRITICAL)

        response = await error_middleware.dispatch(mock_request, call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500  # Internal Server Error

        response_data = json.loads(response.body.decode())
        assert response_data["error_code"] == "CONFIG_001"
        assert response_data["category"] == "configuration"
        assert response_data["severity"] == "critical"
        assert response_data["recoverable"] is False

    @pytest.mark.asyncio
    async def test_middleware_captures_generic_exceptions(self, error_middleware, mock_request):
        """Test middleware captures and processes generic exceptions."""

        async def call_next(request):
            raise ValueError("Something went wrong")

        response = await error_middleware.dispatch(mock_request, call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

        response_data = json.loads(response.body.decode())
        assert response_data["category"] == "system"
        assert "unexpected error" in response_data["user_message"].lower()
        assert response_data["recoverable"] is False

    @pytest.mark.asyncio
    async def test_middleware_passes_through_successful_requests(self, error_middleware, mock_request):
        """Test middleware passes through successful requests without modification."""
        expected_response = JSONResponse(content={"status": "success"})

        async def call_next(request):
            return expected_response

        response = await error_middleware.dispatch(mock_request, call_next)

        assert response is expected_response

    @pytest.mark.asyncio
    async def test_middleware_includes_error_context(self, error_middleware, mock_request):
        """Test middleware includes error context in responses."""

        async def call_next(request):
            raise SessionError("Session error", "SESSION_001", ErrorSeverity.MEDIUM)

        response = await error_middleware.dispatch(mock_request, call_next)

        response_data = json.loads(response.body.decode())
        assert "context" in response_data
        assert response_data["context"]["endpoint"] == "/test/endpoint"
        assert response_data["context"]["session_id"] == "test-session-123"
        assert response_data["context"]["user_agent"] == "Test/1.0"
        assert "error_id" in response_data["context"]
        assert "timestamp" in response_data["context"]


class TestSSEErrorStreamer:
    """Test the enhanced SSE error streaming functionality."""

    @pytest.fixture
    def sse_streamer(self):
        """Create SSEErrorStreamer instance."""
        return SSEErrorStreamer()

    def test_sse_streamer_creation(self, sse_streamer):
        """Test SSEErrorStreamer can be created."""
        assert sse_streamer is not None

    @pytest.mark.asyncio
    async def test_stream_error_result(self, sse_streamer):
        """Test streaming ErrorResult as SSE events."""
        error_context = ErrorContext(
            error_id="test-error-id",
            timestamp=datetime.datetime(2024, 1, 1, 12, 0, 0),
            session_id="test-session",
            request_id=None,
            user_agent="Test/1.0",
            endpoint="/stream",
            stack_trace=None,
            request_data={},
        )

        error_result = ErrorResult(
            error_code="TEST_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            technical_message="Technical error",
            user_message="User-friendly error message",
            suggested_actions=["Action 1", "Action 2"],
            context=error_context,
            recoverable=False,
            retry_after=None,
        )

        stream_generator = sse_streamer.stream_error_result(error_result)
        events = []
        async for event in stream_generator:
            events.append(event)

        # Check error event
        assert any("event: error" in event for event in events)
        error_data_event = next(event for event in events if "data: {" in event)
        assert "error_code" in error_data_event
        assert "user_message" in error_data_event
        assert "suggested_actions" in error_data_event

        # Check completion event
        assert "data: [DONE]\n\n" in events

    @pytest.mark.asyncio
    async def test_stream_validation_error(self, sse_streamer):
        """Test streaming validation errors with file context."""
        error = FileSizeError("File too large")

        stream_generator = sse_streamer.stream_validation_error(error=error, filename="test.pdf", session_id="test-session")
        events = []
        async for event in stream_generator:
            events.append(event)

        # Should include file context
        assert any("test.pdf" in event for event in events)
        assert any("event: validation_error" in event for event in events)
        assert "data: [DONE]\n\n" in events

    @pytest.mark.asyncio
    async def test_stream_recoverable_error_with_retry(self, sse_streamer):
        """Test streaming recoverable errors with retry information."""
        error = NetworkError(message="Connection timeout", error_code="NETWORK_001", severity=ErrorSeverity.HIGH)

        stream_generator = sse_streamer.stream_recoverable_error(error=error, retry_after=30, session_id="test-session")
        events = []
        async for event in stream_generator:
            events.append(event)

        # Should include retry information
        assert any("event: recoverable_error" in event for event in events)
        assert any("retry_after" in event for event in events)
        assert any('"retry_after": 30' in event for event in events)
        assert "data: [DONE]\n\n" in events

    @pytest.mark.asyncio
    async def test_stream_llm_error_with_suggestions(self, sse_streamer):
        """Test streaming LLM errors with specific suggestions."""
        error = LLMError(
            message="Rate limit exceeded",
            error_code="LLM_001",
            severity=ErrorSeverity.HIGH,
            user_message="AI service is busy",
            suggested_actions=["Wait and try again", "Use shorter prompt"],
        )

        stream_generator = sse_streamer.stream_llm_error(error=error, session_id="test-session")
        events = []
        async for event in stream_generator:
            events.append(event)

        # Should include LLM-specific context
        assert any("event: llm_error" in event for event in events)
        assert any("AI service is busy" in event for event in events)
        assert any("shorter prompt" in event for event in events)
        assert "data: [DONE]\n\n" in events


class TestErrorResponseHandler:
    """Test the error response formatting and status code determination."""

    @pytest.fixture
    def response_handler(self):
        """Create ErrorResponseHandler instance."""
        return ErrorResponseHandler()

    def test_response_handler_creation(self, response_handler):
        """Test ErrorResponseHandler can be created."""
        assert response_handler is not None

    def test_determine_http_status_code(self, response_handler):
        """Test HTTP status code determination for different error types."""
        # Validation errors -> 400 Bad Request
        validation_error = FileSizeError("File too large")
        assert response_handler.determine_http_status_code(validation_error) == 400

        # Session errors -> 400 Bad Request
        session_error = SessionError("Session expired", "SESSION_001", ErrorSeverity.MEDIUM)
        assert response_handler.determine_http_status_code(session_error) == 400

        # LLM rate limit errors -> 429 Too Many Requests
        llm_error = LLMError("Rate limit", "LLM_001", ErrorSeverity.HIGH)
        assert response_handler.determine_http_status_code(llm_error) == 429

        # Network errors -> 503 Service Unavailable
        network_error = NetworkError("Connection failed", "NETWORK_001", ErrorSeverity.HIGH)
        assert response_handler.determine_http_status_code(network_error) == 503

        # Configuration errors -> 500 Internal Server Error
        config_error = ConfigurationError("Missing API key", "CONFIG_001", ErrorSeverity.CRITICAL)
        assert response_handler.determine_http_status_code(config_error) == 500

        # Generic errors -> 500 Internal Server Error
        generic_error = Exception("Unknown error")
        assert response_handler.determine_http_status_code(generic_error) == 500

    def test_format_error_response(self, response_handler):
        """Test formatting ErrorResult into JSON response."""
        error_context = ErrorContext(
            error_id="test-error-id",
            timestamp=datetime.datetime(2024, 1, 1, 12, 0, 0),
            session_id="test-session",
            request_id=None,
            user_agent="Test/1.0",
            endpoint="/test",
            stack_trace=None,
            request_data={},
        )

        error_result = ErrorResult(
            error_code="TEST_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            technical_message="Technical error",
            user_message="User-friendly error",
            suggested_actions=["Action 1", "Action 2"],
            context=error_context,
            recoverable=True,
            retry_after=30,
        )

        response_data = response_handler.format_error_response(error_result)

        assert response_data["error_code"] == "TEST_001"
        assert response_data["user_message"] == "User-friendly error"
        assert response_data["suggested_actions"] == ["Action 1", "Action 2"]
        assert response_data["category"] == "validation"
        assert response_data["severity"] == "high"
        assert response_data["recoverable"] is True
        assert response_data["retry_after"] == 30

        # Context should be included but sanitized
        assert "context" in response_data
        assert response_data["context"]["error_id"] == "test-error-id"
        assert response_data["context"]["session_id"] == "test-session"
        assert response_data["context"]["endpoint"] == "/test"
        # Technical details should not be exposed
        assert "stack_trace" not in response_data["context"]
        assert "technical_message" not in response_data

    def test_sanitize_context_for_response(self, response_handler):
        """Test context sanitization for client responses."""
        context = ErrorContext(
            error_id="test-error-id",
            timestamp=datetime.datetime(2024, 1, 1, 12, 0, 0),
            session_id="test-session",
            request_id="req-123",
            user_agent="Test/1.0",
            endpoint="/test",
            stack_trace="Traceback (most recent call last):\n  File...",
            request_data={"password": "secret", "token": "abc123", "safe_data": "ok"},
        )

        sanitized = response_handler.sanitize_context_for_response(context)

        # Safe fields should be included
        assert sanitized["error_id"] == "test-error-id"
        assert sanitized["session_id"] == "test-session"
        assert sanitized["endpoint"] == "/test"

        # Sensitive/technical fields should be excluded
        assert "stack_trace" not in sanitized
        assert "request_data" not in sanitized or not sanitized.get("request_data")


class TestFastAPIErrorHandlers:
    """Test FastAPI exception handlers integration."""

    @pytest.fixture
    def app_with_handlers(self):
        """Create FastAPI app with error handlers configured."""
        app = FastAPI()
        create_error_handlers(app)
        return app

    def test_create_error_handlers_registers_handlers(self, app_with_handlers):
        """Test that create_error_handlers registers all expected exception handlers."""
        # Check that exception handlers are registered
        assert len(app_with_handlers.exception_handlers) > 0

        # Check for specific handler types
        handler_types = set(app_with_handlers.exception_handlers.keys())
        assert ApplicationError in handler_types or any(
            issubclass(handler_type, ApplicationError) for handler_type in handler_types if isinstance(handler_type, type)
        )

    @pytest.mark.asyncio
    async def test_setup_error_middleware_adds_middleware(self):
        """Test that setup_error_middleware adds the error handling middleware."""
        app = FastAPI()

        # Count initial middleware
        initial_middleware_count = len(app.user_middleware)

        setup_error_middleware(app)

        # Should have added one middleware
        assert len(app.user_middleware) == initial_middleware_count + 1

        # Check middleware type
        middleware_stack = app.user_middleware
        error_middleware = None
        for middleware in middleware_stack:
            if hasattr(middleware, "cls") and middleware.cls == ErrorHandlingMiddleware:
                error_middleware = middleware
                break

        assert error_middleware is not None
