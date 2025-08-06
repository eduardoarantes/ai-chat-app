"""Validation models and enums for file processing."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


class ValidationStatus(Enum):
    """Enumeration for file validation statuses."""
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Result of file validation containing all validation metadata."""
    status: ValidationStatus
    file_hash: str
    mime_type: str
    file_size: int
    security_score: float
    validation_errors: List[str]
    metadata: Dict[str, Any]


@dataclass
class FileValidationConfig:
    """Configuration for file validation parameters."""
    max_file_size: int = 50 * 1024 * 1024  # 50MB default
    allowed_mime_types: Optional[set[str]] = None
    blocked_extensions: Optional[set[str]] = None
    enable_security_scanning: bool = True
    security_threshold: float = 0.8