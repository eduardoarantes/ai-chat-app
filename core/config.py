"""Core configuration and utility functions."""

import logging
import os
import secrets
from pathlib import Path
from typing import Optional, Set

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from error_handling.handlers import ErrorHandler, ErrorHandlingMiddleware
from models.validation import FileValidationConfig
from validation.validators import FileValidator

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# File validation configuration constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB default
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "text/html",
}
BLOCKED_EXTENSIONS = {".exe", ".bat", ".cmd", ".com", ".scr", ".jar"}
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

# Vector embeddings and semantic search configuration constants
ENABLE_VECTOR_SEARCH = True  # Enable vector embeddings and semantic search
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
ENABLE_GPU_ACCELERATION = True  # Use GPU if available
EMBEDDING_CACHE_SIZE = 10000  # Number of embeddings to cache
EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding generation
ENABLE_MODEL_QUANTIZATION = False  # Use quantized models
MODEL_CACHE_PATH = "./models/embeddings"
MAX_SEQUENCE_LENGTH = 512

# Vector database configuration
VECTOR_DB_PATH = "./data/vector_db"
COLLECTION_PREFIX = "documents"
ENABLE_VECTOR_DB_BACKUP = True
BACKUP_INTERVAL_HOURS = 24
MAX_BACKUP_FILES = 7
ENABLE_VECTOR_DB_COMPRESSION = True

# Semantic search configuration
DEFAULT_SEARCH_LIMIT = 10
MIN_SIMILARITY_THRESHOLD = 0.3
MAX_SIMILARITY_THRESHOLD = 1.0
ENABLE_QUERY_EXPANSION = True
ENABLE_RESULT_DEDUPLICATION = True
SEARCH_CACHE_SIZE = 1000
SEARCH_CACHE_TTL_MINUTES = 30
SIMILARITY_THRESHOLD_FOR_DEDUP = 0.9


class AppConfig:
    """Application configuration settings."""

    def __init__(self):
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", MAX_FILE_SIZE))
        self.allowed_mime_types = self._parse_mime_types(os.getenv("ALLOWED_MIME_TYPES", "")) or ALLOWED_MIME_TYPES
        self.blocked_extensions = self._parse_extensions(os.getenv("BLOCKED_EXTENSIONS", "")) or BLOCKED_EXTENSIONS
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
        self.max_total_document_size_per_session = int(
            os.getenv("MAX_TOTAL_DOCUMENT_SIZE_PER_SESSION", MAX_TOTAL_DOCUMENT_SIZE_PER_SESSION)
        )
        self.session_ttl_hours = int(os.getenv("SESSION_TTL_HOURS", SESSION_TTL_HOURS))
        self.document_ttl_hours = int(os.getenv("DOCUMENT_TTL_HOURS", DOCUMENT_TTL_HOURS))
        self.cleanup_interval_minutes = int(os.getenv("CLEANUP_INTERVAL_MINUTES", CLEANUP_INTERVAL_MINUTES))
        self.max_shared_documents_per_session = int(
            os.getenv("MAX_SHARED_DOCUMENTS_PER_SESSION", MAX_SHARED_DOCUMENTS_PER_SESSION)
        )
        self.enable_cross_session_sharing = os.getenv("ENABLE_CROSS_SESSION_SHARING", "false").lower() == "true"

        # Vector embeddings and semantic search configuration
        self.enable_vector_search = os.getenv("ENABLE_VECTOR_SEARCH", "true").lower() == "true"
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", EMBEDDING_MODEL_NAME)
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", EMBEDDING_DIMENSION))
        self.enable_gpu_acceleration = os.getenv("ENABLE_GPU_ACCELERATION", "true").lower() == "true"
        self.embedding_cache_size = int(os.getenv("EMBEDDING_CACHE_SIZE", EMBEDDING_CACHE_SIZE))
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", EMBEDDING_BATCH_SIZE))
        self.enable_model_quantization = os.getenv("ENABLE_MODEL_QUANTIZATION", "false").lower() == "true"
        self.model_cache_path = os.getenv("MODEL_CACHE_PATH", MODEL_CACHE_PATH)
        self.max_sequence_length = int(os.getenv("MAX_SEQUENCE_LENGTH", MAX_SEQUENCE_LENGTH))

        # Vector database configuration
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", VECTOR_DB_PATH)
        self.collection_prefix = os.getenv("COLLECTION_PREFIX", COLLECTION_PREFIX)
        self.enable_vector_db_backup = os.getenv("ENABLE_VECTOR_DB_BACKUP", "true").lower() == "true"
        self.backup_interval_hours = int(os.getenv("BACKUP_INTERVAL_HOURS", BACKUP_INTERVAL_HOURS))
        self.max_backup_files = int(os.getenv("MAX_BACKUP_FILES", MAX_BACKUP_FILES))
        self.enable_vector_db_compression = os.getenv("ENABLE_VECTOR_DB_COMPRESSION", "true").lower() == "true"

        # Semantic search configuration
        self.default_search_limit = int(os.getenv("DEFAULT_SEARCH_LIMIT", DEFAULT_SEARCH_LIMIT))
        self.min_similarity_threshold = float(os.getenv("MIN_SIMILARITY_THRESHOLD", MIN_SIMILARITY_THRESHOLD))
        self.max_similarity_threshold = float(os.getenv("MAX_SIMILARITY_THRESHOLD", MAX_SIMILARITY_THRESHOLD))
        self.enable_query_expansion = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
        self.enable_result_deduplication = os.getenv("ENABLE_RESULT_DEDUPLICATION", "true").lower() == "true"
        self.search_cache_size = int(os.getenv("SEARCH_CACHE_SIZE", SEARCH_CACHE_SIZE))
        self.search_cache_ttl_minutes = int(os.getenv("SEARCH_CACHE_TTL_MINUTES", SEARCH_CACHE_TTL_MINUTES))
        self.similarity_threshold_for_dedup = float(
            os.getenv("SIMILARITY_THRESHOLD_FOR_DEDUP", SIMILARITY_THRESHOLD_FOR_DEDUP)
        )

        # Validate required configuration
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

    def _parse_mime_types(self, mime_types_str: str) -> Optional[Set[str]]:
        """Parse comma-separated MIME types from environment variable."""
        if not mime_types_str:
            return None
        return {mt.strip() for mt in mime_types_str.split(",") if mt.strip()}

    def _parse_extensions(self, extensions_str: str) -> Optional[Set[str]]:
        """Parse comma-separated file extensions from environment variable."""
        if not extensions_str:
            return None
        return {ext.strip() for ext in extensions_str.split(",") if ext.strip()}

    def get_file_validation_config(self) -> FileValidationConfig:
        """Get file validation configuration."""
        return FileValidationConfig(
            max_file_size=self.max_file_size,
            allowed_mime_types=self.allowed_mime_types,
            blocked_extensions=self.blocked_extensions,
            enable_security_scanning=self.enable_security_scanning,
            security_threshold=self.security_threshold,
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
            enable_metadata_extraction=self.enable_metadata_extraction,
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
            "enable_cross_session_sharing": self.enable_cross_session_sharing,
        }

    def get_vector_search_config(self) -> dict:
        """Get vector embeddings and semantic search configuration."""
        return {
            "enable_vector_search": self.enable_vector_search,
            "embedding_model_name": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "enable_gpu_acceleration": self.enable_gpu_acceleration,
            "embedding_cache_size": self.embedding_cache_size,
            "embedding_batch_size": self.embedding_batch_size,
            "enable_model_quantization": self.enable_model_quantization,
            "model_cache_path": self.model_cache_path,
            "max_sequence_length": self.max_sequence_length,
        }

    def get_vector_database_config(self) -> dict:
        """Get vector database configuration."""
        return {
            "vector_db_path": self.vector_db_path,
            "collection_prefix": self.collection_prefix,
            "enable_vector_db_backup": self.enable_vector_db_backup,
            "backup_interval_hours": self.backup_interval_hours,
            "max_backup_files": self.max_backup_files,
            "enable_vector_db_compression": self.enable_vector_db_compression,
            "expected_dimension": self.embedding_dimension,
        }

    def get_semantic_search_config(self) -> dict:
        """Get semantic search configuration."""
        return {
            "default_search_limit": self.default_search_limit,
            "min_similarity_threshold": self.min_similarity_threshold,
            "max_similarity_threshold": self.max_similarity_threshold,
            "enable_query_expansion": self.enable_query_expansion,
            "enable_result_deduplication": self.enable_result_deduplication,
            "search_cache_size": self.search_cache_size,
            "search_cache_ttl_minutes": self.search_cache_ttl_minutes,
            "similarity_threshold_for_dedup": self.similarity_threshold_for_dedup,
        }


def create_fastapi_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    app = FastAPI(
        title="AI Foundation Chat", description="Real-time chat application with document processing", version="1.0.0"
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
        https_only=False,  # Set to True in production with HTTPS
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
    # Skip validation in test environments
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("CI"):
        return

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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            # Add file handler if needed
            # logging.FileHandler('app.log')
        ],
    )

    # Suppress some noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_config() -> AppConfig:
    """Get the global application configuration instance."""
    return AppConfig()
