"""Comprehensive error handling and user feedback system."""

import datetime
import json
import logging
import traceback
import uuid
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from models.errors import (
    ErrorSeverity, ErrorCategory, ErrorContext, ErrorResult,
    FileSizeError, MimeTypeError, SecurityError, ContentValidationError,
    SessionError, LLMError, NetworkError, ConfigurationError
)


class ErrorContextCapture:
    """Captures contextual information for error tracking and debugging."""
    
    async def capture_request_context(
        self,
        request: Optional[Request] = None,
        session_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """
        Capture context from a FastAPI request.
        
        Args:
            request: FastAPI Request object
            session_id: Current session ID
            additional_data: Additional context data
            
        Returns:
            ErrorContext: Captured context information
        """
        error_id = uuid.uuid4().hex
        timestamp = datetime.datetime.now()
        request_data = additional_data or {}
        
        if request:
            request_data.update({
                "method": request.method,
                "client_host": request.client.host if request.client else None,
                "content_type": request.headers.get("content-type"),
            })
            
            return ErrorContext(
                error_id=error_id,
                timestamp=timestamp,
                session_id=session_id,
                request_id=None,  # Could be extracted from headers if available
                user_agent=request.headers.get("user-agent"),
                endpoint=request.url.path,
                stack_trace=None,
                request_data=request_data
            )
        else:
            return ErrorContext(
                error_id=error_id,
                timestamp=timestamp,
                session_id=session_id,
                request_id=None,
                user_agent=None,
                endpoint=None,
                stack_trace=None,
                request_data=request_data
            )
    
    async def capture_exception_context(
        self,
        exception: Exception,
        session_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """
        Capture context from an exception with stack trace.
        
        Args:
            exception: The exception that occurred
            session_id: Current session ID
            additional_data: Additional context data
            
        Returns:
            ErrorContext: Captured context information with stack trace
        """
        error_id = uuid.uuid4().hex
        timestamp = datetime.datetime.now()
        stack_trace = traceback.format_exc()
        request_data = additional_data or {}
        
        return ErrorContext(
            error_id=error_id,
            timestamp=timestamp,
            session_id=session_id,
            request_id=None,
            user_agent=None,
            endpoint=None,
            stack_trace=stack_trace,
            request_data=request_data
        )


class ErrorMessageTranslator:
    """Translates technical error messages to user-friendly messages with suggested actions."""
    
    def __init__(self):
        """Initialize the translator with predefined message mappings."""
        self._translation_rules = {
            # File validation errors
            FileSizeError: {
                "user_message": "The file you selected is too large. Please choose a smaller file.",
                "suggested_actions": [
                    "Try compressing the file before uploading",
                    "Select a different file under 50MB",
                    "Contact support if you need to upload larger files"
                ],
                "severity": ErrorSeverity.MEDIUM,
                "category": ErrorCategory.VALIDATION
            },
            MimeTypeError: {
                "user_message": "This file type is not supported. Please select a supported file format.",
                "suggested_actions": [
                    "Check the list of supported file types",
                    "Convert your file to a supported format",
                    "Try uploading a different file"
                ],
                "severity": ErrorSeverity.MEDIUM,
                "category": ErrorCategory.VALIDATION
            },
            SecurityError: {
                "user_message": "This file appears to be unsafe and cannot be processed.",
                "suggested_actions": [
                    "Scan your file with antivirus software",
                    "Try uploading a different file",
                    "Contact support if you believe this is an error"
                ],
                "severity": ErrorSeverity.HIGH,
                "category": ErrorCategory.VALIDATION
            },
            ContentValidationError: {
                "user_message": "The file appears to be corrupted or in an invalid format.",
                "suggested_actions": [
                    "Try re-saving the file in the correct format",
                    "Upload a different version of the file",
                    "Ensure the file is not corrupted"
                ],
                "severity": ErrorSeverity.MEDIUM,
                "category": ErrorCategory.VALIDATION
            },
            
            # Application errors
            SessionError: {
                "user_message": "There was a problem with your session. Please try again.",
                "suggested_actions": [
                    "Refresh the page and try again",
                    "Clear your browser cache",
                    "Contact support if the problem persists"
                ],
                "severity": ErrorSeverity.MEDIUM,
                "category": ErrorCategory.SESSION
            },
            LLMError: {
                "user_message": "The AI service is temporarily unavailable. Please try again.",
                "suggested_actions": [
                    "Wait a moment and try your request again",
                    "Check your internet connection",
                    "Try with a shorter message or smaller file"
                ],
                "severity": ErrorSeverity.HIGH,
                "category": ErrorCategory.LLM
            },
            NetworkError: {
                "user_message": "Network connection issue. Please check your connection and try again.",
                "suggested_actions": [
                    "Check your internet connection",
                    "Try again in a few moments",
                    "Contact support if the problem continues"
                ],
                "severity": ErrorSeverity.MEDIUM,
                "category": ErrorCategory.NETWORK
            },
            ConfigurationError: {
                "user_message": "There's a configuration issue. Please contact support.",
                "suggested_actions": [
                    "Contact technical support",
                    "Try again later",
                    "Report this error with the error ID"
                ],
                "severity": ErrorSeverity.CRITICAL,
                "category": ErrorCategory.CONFIGURATION
            },
            
            # Generic errors
            Exception: {
                "user_message": "An unexpected error occurred. Please try again.",
                "suggested_actions": [
                    "Try your request again",
                    "Refresh the page if the problem persists",
                    "Contact support with the error ID if needed"
                ],
                "severity": ErrorSeverity.MEDIUM,
                "category": ErrorCategory.SYSTEM
            }
        }
    
    def translate_error(
        self, 
        exception: Exception, 
        context: ErrorContext,
        fallback_message: Optional[str] = None
    ) -> ErrorResult:
        """
        Translate a technical error to a user-friendly error result.
        
        Args:
            exception: The exception to translate
            context: Error context information
            fallback_message: Optional fallback message if no rule matches
            
        Returns:
            ErrorResult: User-friendly error result
        """
        # Get the most specific rule that matches
        rule = self._get_translation_rule(exception)
        
        # Generate error code
        error_code = self._generate_error_code(exception)
        
        # Extract technical message and sanitize it
        technical_message = self._sanitize_technical_message(str(exception))
        
        # Apply translation rule
        user_message = rule.get("user_message", fallback_message or technical_message)
        suggested_actions = rule.get("suggested_actions", [])
        severity = rule.get("severity", ErrorSeverity.MEDIUM)
        category = rule.get("category", ErrorCategory.SYSTEM)
        
        # Determine if error is recoverable
        recoverable = self._is_recoverable(exception)
        retry_after = self._get_retry_delay(exception) if recoverable else None
        
        return ErrorResult(
            error_code=error_code,
            severity=severity,
            category=category,
            technical_message=technical_message,
            user_message=user_message,
            suggested_actions=suggested_actions,
            context=context,
            recoverable=recoverable,
            retry_after=retry_after
        )
    
    def _get_translation_rule(self, exception: Exception) -> Dict[str, Any]:
        """Get the most specific translation rule for an exception."""
        # Try to find exact type match
        exception_type = type(exception)
        if exception_type in self._translation_rules:
            return self._translation_rules[exception_type]
        
        # Try to find parent class match
        for rule_type, rule in self._translation_rules.items():
            if isinstance(exception, rule_type):
                return rule
        
        # Fallback to generic exception rule
        return self._translation_rules.get(Exception, {})
    
    def _generate_error_code(self, exception: Exception) -> str:
        """Generate a unique error code based on exception type."""
        exception_name = type(exception).__name__
        timestamp = int(datetime.datetime.now().timestamp())
        return f"{exception_name}_{timestamp}"
    
    def _is_recoverable(self, exception: Exception) -> bool:
        """Determine if an error is recoverable (can be retried)."""
        recoverable_types = (NetworkError, LLMError)
        return isinstance(exception, recoverable_types)
    
    def _get_retry_delay(self, exception: Exception) -> Optional[int]:
        """Get suggested retry delay in seconds."""
        if isinstance(exception, NetworkError):
            return 5
        elif isinstance(exception, LLMError):
            return 10
        return None
    
    def _sanitize_technical_message(self, message: str) -> str:
        """
        Sanitize technical message to prevent logging of large content like base64 data.
        
        Args:
            message: Raw technical message from exception
            
        Returns:
            str: Sanitized message safe for logging
        """
        # Limit message length to prevent massive log entries
        max_length = 500
        
        # Check for base64-like patterns (mixed alphanumeric characters typical of base64)
        import re
        base64_pattern = re.compile(r'data:[^;]+;base64,[A-Za-z0-9+/]{50,}={0,2}|[A-Za-z0-9+/]{100,}={0,2}')
        
        if base64_pattern.search(message):
            # Replace base64 content with placeholder
            message = base64_pattern.sub('[BASE64_CONTENT_TRUNCATED]', message)
        
        # Check for very long continuous non-whitespace strings that might be file content
        # But avoid matching single repeated characters which are likely not file content
        long_string_pattern = re.compile(r'(?=.*[A-Za-z].*[A-Za-z].*[A-Za-z])\S{200,}')
        if long_string_pattern.search(message):
            message = long_string_pattern.sub('[LONG_CONTENT_TRUNCATED]', message)
        
        # Truncate if still too long
        if len(message) > max_length:
            message = message[:max_length] + "... [TRUNCATED]"
        
        return message


class ErrorHandler:
    """Main error handler that orchestrates error processing."""
    
    def __init__(self):
        self.context_capture = ErrorContextCapture()
        self.message_translator = ErrorMessageTranslator()
    
    async def handle_error(
        self,
        exception: Exception,
        request: Optional[Request] = None,
        session_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ErrorResult:
        """
        Comprehensive error handling pipeline.
        
        Args:
            exception: The exception to handle
            request: FastAPI request object
            session_id: Current session ID
            additional_context: Additional context data
            
        Returns:
            ErrorResult: Complete error handling result
        """
        try:
            # Capture error context
            if request:
                context = await self.context_capture.capture_request_context(
                    request, session_id, additional_context
                )
            else:
                context = await self.context_capture.capture_exception_context(
                    exception, session_id, additional_context
                )
            
            # Translate error to user-friendly format
            error_result = self.message_translator.translate_error(exception, context)
            
            # Log the error with context
            self._log_error(error_result)
            
            return error_result
            
        except Exception as handler_error:
            # If error handling itself fails, create minimal error result
            logging.error(f"Error handler failed: {handler_error}")
            return self._create_fallback_error_result(exception)
    
    def _log_error(self, error_result: ErrorResult) -> None:
        """Log error with appropriate level based on severity."""
        log_data = {
            "error_id": error_result.context.error_id,
            "error_code": error_result.error_code,
            "category": error_result.category.value,
            "severity": error_result.severity.value,
            "endpoint": error_result.context.endpoint,
            "session_id": error_result.context.session_id,
            "technical_message": error_result.technical_message
        }
        
        if error_result.severity == ErrorSeverity.CRITICAL:
            logging.critical(f"Critical error: {json.dumps(log_data)}")
        elif error_result.severity == ErrorSeverity.HIGH:
            logging.error(f"High severity error: {json.dumps(log_data)}")
        elif error_result.severity == ErrorSeverity.MEDIUM:
            logging.warning(f"Medium severity error: {json.dumps(log_data)}")
        else:
            logging.info(f"Low severity error: {json.dumps(log_data)}")
    
    def _create_fallback_error_result(self, exception: Exception) -> ErrorResult:
        """Create a minimal error result when error handling fails."""
        error_id = uuid.uuid4().hex
        timestamp = datetime.datetime.now()
        
        context = ErrorContext(
            error_id=error_id,
            timestamp=timestamp,
            session_id=None,
            request_id=None,
            user_agent=None,
            endpoint=None,
            stack_trace=traceback.format_exc(),
            request_data={}
        )
        
        return ErrorResult(
            error_code=f"FALLBACK_{error_id}",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM,
            technical_message=self.message_translator._sanitize_technical_message(str(exception)),
            user_message="A system error occurred. Please try again or contact support.",
            suggested_actions=["Try again", "Contact support"],
            context=context,
            recoverable=False,
            retry_after=None
        )


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for centralized error handling."""
    
    def __init__(self, app, error_handler: ErrorHandler):
        super().__init__(app)
        self.error_handler = error_handler
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process requests and handle any errors that occur.
        
        Args:
            request: FastAPI request
            call_next: Next middleware or endpoint
            
        Returns:
            Response: HTTP response
        """
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Extract session ID safely
            session_id = self._get_session_id_safely(request)
            
            # Handle the error
            error_result = await self.error_handler.handle_error(
                e, request, session_id
            )
            
            # Return appropriate HTTP response
            return self._create_error_response(error_result)
    
    def _get_session_id_safely(self, request: Request) -> Optional[str]:
        """Safely extract session ID from request."""
        try:
            # Try to get from session first
            if hasattr(request.state, 'session_id'):
                return request.state.session_id
            
            # Try to get from session middleware
            if hasattr(request, 'session') and 'session_id' in request.session:
                return request.session['session_id']
            
            # Try to get from query parameters as fallback
            return request.query_params.get('session_id')
            
        except Exception as e:
            logging.warning(f"Could not extract session ID: {e}")
            return None
    
    def _create_error_response(self, error_result: ErrorResult) -> JSONResponse:
        """Create appropriate HTTP response for error result."""
        status_code = self._get_status_code(error_result)
        
        response_data = {
            "error": True,
            "error_id": error_result.context.error_id,
            "error_code": error_result.error_code,
            "message": error_result.user_message,
            "suggested_actions": error_result.suggested_actions,
            "severity": error_result.severity.value,
            "category": error_result.category.value,
            "recoverable": error_result.recoverable
        }
        
        if error_result.retry_after:
            response_data["retry_after"] = error_result.retry_after
        
        return JSONResponse(
            status_code=status_code,
            content=response_data
        )
    
    def _get_status_code(self, error_result: ErrorResult) -> int:
        """Determine appropriate HTTP status code for error result."""
        if isinstance(error_result.context.request_data.get('original_exception'), HTTPException):
            return error_result.context.request_data['original_exception'].status_code
        
        # Map categories to status codes
        category_status_map = {
            ErrorCategory.VALIDATION: 400,
            ErrorCategory.SESSION: 401,
            ErrorCategory.NETWORK: 503,
            ErrorCategory.LLM: 503,
            ErrorCategory.CONFIGURATION: 500,
            ErrorCategory.SYSTEM: 500
        }
        
        return category_status_map.get(error_result.category, 500)


class SSEErrorStreamer:
    """Handles error streaming through Server-Sent Events."""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
    
    async def stream_error(
        self,
        exception: Exception,
        session_id: Optional[str] = None,
        request: Optional[Request] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream error information via SSE.
        
        Args:
            exception: Exception to stream
            session_id: Current session ID
            request: FastAPI request object
            
        Yields:
            str: SSE-formatted error messages
        """
        try:
            # Handle the error
            error_result = await self.error_handler.handle_error(
                exception, request, session_id
            )
            
            # Stream error event
            error_data = {
                "error_id": error_result.context.error_id,
                "error_code": error_result.error_code,
                "message": error_result.user_message,
                "suggested_actions": error_result.suggested_actions,
                "severity": error_result.severity.value,
                "category": error_result.category.value,
                "recoverable": error_result.recoverable
            }
            
            if error_result.retry_after:
                error_data["retry_after"] = error_result.retry_after
            
            yield f"event: error\n"
            yield f"data: {json.dumps(error_data)}\n\n"
            
        except Exception as streaming_error:
            # If streaming fails, send minimal error
            logging.error(f"Error streaming failed: {streaming_error}")
            yield f"event: error\n"
            yield f"data: {json.dumps({'message': 'An error occurred', 'error_id': uuid.uuid4().hex})}\n\n"


class ErrorResponseHandler:
    """Handles different types of error responses."""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
    
    async def handle_json_error(
        self,
        exception: Exception,
        request: Optional[Request] = None,
        session_id: Optional[str] = None
    ) -> JSONResponse:
        """Handle error with JSON response."""
        error_result = await self.error_handler.handle_error(
            exception, request, session_id
        )
        
        return JSONResponse(
            status_code=self._get_http_status_code(error_result),
            content={
                "error": True,
                "error_id": error_result.context.error_id,
                "message": error_result.user_message,
                "suggested_actions": error_result.suggested_actions,
                "severity": error_result.severity.value,
                "recoverable": error_result.recoverable
            }
        )
    
    def _get_http_status_code(self, error_result: ErrorResult) -> int:
        """Get appropriate HTTP status code for error result."""
        severity_status_map = {
            ErrorSeverity.CRITICAL: 500,
            ErrorSeverity.HIGH: 500,
            ErrorSeverity.MEDIUM: 400,
            ErrorSeverity.LOW: 400
        }
        
        category_status_map = {
            ErrorCategory.VALIDATION: 400,
            ErrorCategory.SESSION: 401,
            ErrorCategory.LLM: 503,
            ErrorCategory.NETWORK: 503,
            ErrorCategory.CONFIGURATION: 500,
            ErrorCategory.SYSTEM: 500
        }
        
        # Category takes precedence over severity
        return category_status_map.get(
            error_result.category,
            severity_status_map.get(error_result.severity, 500)
        )