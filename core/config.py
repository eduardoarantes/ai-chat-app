"""Core configuration and utility functions."""

import os
import logging
import secrets
from pathlib import Path
from typing import Optional, Set
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

from models.validation import FileValidationConfig
from validation.validators import FileValidator
from error_handling.handlers import ErrorHandler, ErrorHandlingMiddleware


# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# File validation configuration constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB default
ALLOWED_MIME_TYPES = {
    'image/jpeg', 'image/png', 'image/gif', 'image/webp',
    'application/pdf', 'text/plain', 'text/markdown',
    'text/csv', 'application/json', 'text/html'
}
BLOCKED_EXTENSIONS = {'.exe', '.bat', '.cmd', '.com', '.scr', '.jar'}
ENABLE_SECURITY_SCANNING = True
VALIDATION_CACHE_SIZE = 1000

# Chunking configuration constants
ENABLE_CHUNKING = False  # Disabled by default for backward compatibility
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 4000
ENABLE_SEMANTIC_BOUNDARIES = True
ENABLE_OVERLAP_OPTIMIZATION = True
ENABLE_METADATA_EXTRACTION = True

# Document persistence configuration constants
ENABLE_DOCUMENT_PERSISTENCE = True
MAX_DOCUMENTS_PER_SESSION = 50
MAX_TOTAL_DOCUMENT_SIZE_PER_SESSION = 500 * 1024 * 1024  # 500MB
SESSION_TTL_HOURS = 24  # Hours before session cleanup
DOCUMENT_TTL_HOURS = 72  # Hours before orphaned document cleanup
CLEANUP_INTERVAL_MINUTES = 60  # How often to run cleanup
MAX_SHARED_DOCUMENTS_PER_SESSION = 10
ENABLE_CROSS_SESSION_SHARING = False  # Disabled by default for security


class AppConfig:
    """Application configuration settings."""
    
    def __init__(self):
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", MAX_FILE_SIZE))
        self.allowed_mime_types = self._parse_mime_types(
            os.getenv("ALLOWED_MIME_TYPES", "")
        ) or ALLOWED_MIME_TYPES
        self.blocked_extensions = self._parse_extensions(
            os.getenv("BLOCKED_EXTENSIONS", "")
        ) or BLOCKED_EXTENSIONS
        self.enable_security_scanning = os.getenv("ENABLE_SECURITY_SCANNING", "true").lower() == "true"
        self.security_threshold = float(os.getenv("SECURITY_THRESHOLD", "0.8"))
        self.google_api_key = os.getenv("GOOGLE_API_KEY") 
        self.session_secret_key = os.getenv("SESSION_SECRET_KEY", secrets.token_urlsafe(32))
        self.session_max_age = int(os.getenv("SESSION_MAX_AGE", "86400"))  # 24 hours
        
        # Chunking configuration
        self.enable_chunking = os.getenv("ENABLE_CHUNKING", "false").lower() == "true"
        self.default_chunk_size = int(os.getenv("DEFAULT_CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
        self.default_chunk_overlap = int(os.getenv("DEFAULT_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
        self.min_chunk_size = int(os.getenv("MIN_CHUNK_SIZE", MIN_CHUNK_SIZE))
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", MAX_CHUNK_SIZE))
        self.enable_semantic_boundaries = os.getenv("ENABLE_SEMANTIC_BOUNDARIES", "true").lower() == "true"
        self.enable_overlap_optimization = os.getenv("ENABLE_OVERLAP_OPTIMIZATION", "true").lower() == "true"
        self.enable_metadata_extraction = os.getenv("ENABLE_METADATA_EXTRACTION", "true").lower() == "true"
        
        # Document persistence configuration
        self.enable_document_persistence = os.getenv("ENABLE_DOCUMENT_PERSISTENCE", "true").lower() == "true"
        self.max_documents_per_session = int(os.getenv("MAX_DOCUMENTS_PER_SESSION", MAX_DOCUMENTS_PER_SESSION))
        self.max_total_document_size_per_session = int(os.getenv("MAX_TOTAL_DOCUMENT_SIZE_PER_SESSION", MAX_TOTAL_DOCUMENT_SIZE_PER_SESSION))
        self.session_ttl_hours = int(os.getenv("SESSION_TTL_HOURS", SESSION_TTL_HOURS))
        self.document_ttl_hours = int(os.getenv("DOCUMENT_TTL_HOURS", DOCUMENT_TTL_HOURS))
        self.cleanup_interval_minutes = int(os.getenv("CLEANUP_INTERVAL_MINUTES", CLEANUP_INTERVAL_MINUTES))
        self.max_shared_documents_per_session = int(os.getenv("MAX_SHARED_DOCUMENTS_PER_SESSION", MAX_SHARED_DOCUMENTS_PER_SESSION))
        self.enable_cross_session_sharing = os.getenv("ENABLE_CROSS_SESSION_SHARING", "false").lower() == "true"
        
        # Validate required configuration
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    def _parse_mime_types(self, mime_types_str: str) -> Optional[Set[str]]:
        """Parse comma-separated MIME types from environment variable."""
        if not mime_types_str:
            return None
        return {mt.strip() for mt in mime_types_str.split(',') if mt.strip()}
    
    def _parse_extensions(self, extensions_str: str) -> Optional[Set[str]]:
        """Parse comma-separated file extensions from environment variable."""
        if not extensions_str:
            return None
        return {ext.strip() for ext in extensions_str.split(',') if ext.strip()}
    
    def get_file_validation_config(self) -> FileValidationConfig:
        """Get file validation configuration."""
        return FileValidationConfig(
            max_file_size=self.max_file_size,
            allowed_mime_types=self.allowed_mime_types,
            blocked_extensions=self.blocked_extensions,
            enable_security_scanning=self.enable_security_scanning,
            security_threshold=self.security_threshold
        )
    
    def get_chunking_config(self):
        """Get chunking configuration."""
        from models.chunking import ChunkingConfig, ChunkingStrategy
        
        return ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,  # Default strategy
            chunk_size=self.default_chunk_size,
            chunk_overlap=self.default_chunk_overlap,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
            enable_semantic_boundaries=self.enable_semantic_boundaries,
            enable_overlap_optimization=self.enable_overlap_optimization,
            enable_metadata_extraction=self.enable_metadata_extraction
        )
    
    def get_document_persistence_config(self) -> dict:
        """Get document persistence configuration."""
        return {
            "enable_document_persistence": self.enable_document_persistence,
            "max_documents_per_session": self.max_documents_per_session,
            "max_total_document_size_per_session": self.max_total_document_size_per_session,
            "session_ttl_hours": self.session_ttl_hours,
            "document_ttl_hours": self.document_ttl_hours,
            "cleanup_interval_minutes": self.cleanup_interval_minutes,
            "max_shared_documents_per_session": self.max_shared_documents_per_session,
            "enable_cross_session_sharing": self.enable_cross_session_sharing
        }


def create_fastapi_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    app = FastAPI(
        title="AI Foundation Chat",
        description="Real-time chat application with document processing",
        version="1.0.0"
    )
    
    return app


def setup_middleware(app: FastAPI, config: AppConfig) -> None:
    """Configure FastAPI middleware."""
    # Configure session middleware
    app.add_middleware(
        SessionMiddleware,
        secret_key=config.session_secret_key,
        max_age=config.session_max_age,
        same_site="lax",
        https_only=False  # Set to True in production with HTTPS
    )
    
    # Add error handling middleware
    error_handler = ErrorHandler()
    app.add_middleware(ErrorHandlingMiddleware, error_handler=error_handler)


def setup_static_files_and_templates(app: FastAPI) -> Jinja2Templates:
    """Configure static files and templates."""
    # Use relative paths for deployment compatibility
    base_dir = Path(__file__).parent.parent
    templates = Jinja2Templates(directory=str(base_dir / "templates"))
    
    app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")
    
    return templates


def create_file_validator(config: AppConfig) -> FileValidator:
    """Create file validator instance with configuration."""
    file_validation_config = config.get_file_validation_config()
    return FileValidator(file_validation_config)


def validate_environment() -> None:
    """Validate that all required environment variables are set."""
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


def setup_logging(level: str = "INFO") -> None:
    """Configure application logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            # Add file handler if needed
            # logging.FileHandler('app.log')
        ]
    )
    
    # Suppress some noisy loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)