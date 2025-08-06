"""Error models and exception hierarchy for the AI Foundation application."""

import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


# --- File Validation Exception Hierarchy ---

class FileValidationError(Exception):
    """Base exception for file validation errors."""
    pass


class FileSizeError(FileValidationError):
    """File exceeds maximum allowed size."""
    pass


class MimeTypeError(FileValidationError):
    """File MIME type not allowed."""
    pass


class SecurityError(FileValidationError):
    """File failed security validation."""
    pass


class ContentValidationError(FileValidationError):
    """File content validation failed."""
    pass


# --- Error Handling Enums ---

class ErrorSeverity(Enum):
    """Enumeration for error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Enumeration for error categories."""
    VALIDATION = "validation"
    SESSION = "session"
    LLM = "llm"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    SYSTEM = "system"


# --- Error Context and Result Models ---

@dataclass
class ErrorContext:
    """Captures contextual information about an error occurrence."""
    error_id: str
    timestamp: datetime.datetime
    session_id: Optional[str]
    request_id: Optional[str]
    user_agent: Optional[str]
    endpoint: Optional[str]
    stack_trace: Optional[str]
    request_data: Dict[str, Any]


@dataclass
class ErrorResult:
    """Complete error processing result with context and user-friendly messages."""
    error_code: str
    severity: ErrorSeverity
    category: ErrorCategory
    technical_message: str
    user_message: str
    suggested_actions: List[str]
    context: ErrorContext
    recoverable: bool
    retry_after: Optional[int]


# --- Application Exception Hierarchy ---

class ApplicationError(Exception):
    """Base exception for application errors with enhanced metadata."""
    
    def __init__(
        self, 
        message: str, 
        error_code: str, 
        severity: ErrorSeverity, 
        user_message: Optional[str] = None,
        suggested_actions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.severity = severity
        self.user_message = user_message or message
        self.suggested_actions = suggested_actions or []


class SessionError(ApplicationError):
    """Session management errors."""
    pass


class LLMError(ApplicationError):
    """Language model interaction errors."""
    pass


class NetworkError(ApplicationError):
    """Network and connectivity errors."""
    pass


class ConfigurationError(ApplicationError):
    """Configuration and environment errors."""
    pass