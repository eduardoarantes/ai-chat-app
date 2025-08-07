"""File validation components and security scanning."""

import hashlib
import logging
import os
from typing import Optional

from fastapi import UploadFile

from models.errors import ContentValidationError, FileSizeError, MimeTypeError, SecurityError
from models.validation import FileValidationConfig, ValidationResult, ValidationStatus

# Import python-magic for MIME type detection
try:
    import magic

    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logging.warning("python-magic not available, using fallback MIME detection")


class SizeValidator:
    """Validates file sizes against configured limits."""

    def __init__(self, max_file_size: int):
        self.max_file_size = max_file_size

    def validate_size(self, file_size: int) -> None:
        """
        Validate file size against limits.

        Args:
            file_size: Size of the file in bytes

        Raises:
            FileSizeError: If file size is invalid
        """
        if file_size <= 0:
            raise FileSizeError("Cannot process empty file")

        if file_size > self.max_file_size:
            raise FileSizeError(f"File size {file_size} bytes exceeds maximum allowed size {self.max_file_size} bytes")


class MimeTypeDetector:
    """Detects MIME types using python-magic with fallback to extension-based detection."""

    def __init__(self):
        self.extension_mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".py": "text/plain",
            ".js": "text/plain",
            ".html": "text/html",
            ".css": "text/css",
            ".json": "application/json",
            ".xml": "text/xml",
            ".csv": "text/csv",
        }

    def detect_mime_type(self, content: bytes, filename: str) -> str:
        """
        Detect MIME type of file content.

        Args:
            content: File content as bytes
            filename: Name of the file

        Returns:
            str: Detected MIME type
        """
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_buffer(content, mime=True)
                logging.info(f"Detected MIME type using magic: {mime_type}")
                return mime_type
            except Exception as e:
                logging.warning(f"Magic MIME detection failed: {e}, falling back to extension")

        # Fallback to extension-based detection
        return self._get_mime_from_extension(filename)

    def _get_mime_from_extension(self, filename: str) -> str:
        """
        Get MIME type from file extension.

        Args:
            filename: Name of the file

        Returns:
            str: MIME type based on extension
        """
        if not filename:
            return "application/octet-stream"

        # Extract extension and normalize
        ext = os.path.splitext(filename.lower())[1]
        return self.extension_mime_map.get(ext, "application/octet-stream")


class SecurityScanner:
    """Scans files for security threats and assigns risk scores."""

    def __init__(self, security_threshold: float):
        self.security_threshold = security_threshold
        self.suspicious_extensions = {".exe", ".bat", ".cmd", ".com", ".scr", ".jar", ".vbs", ".ps1"}
        self.executable_signatures = [
            b"MZ",  # Windows PE executable
            b"\x7fELF",  # Linux ELF executable
            b"\xca\xfe\xba\xbe",  # Java class file
            b"PK\x03\x04",  # ZIP file (could contain executables)
        ]

    async def scan_file(self, content: bytes, mime_type: str, filename: str) -> float:
        """
        Scan file for security threats and return risk score.

        Args:
            content: File content as bytes
            mime_type: MIME type of the file
            filename: Name of the file

        Returns:
            float: Security score (0.0 = high risk, 1.0 = safe)
        """
        score = 1.0  # Start with safe assumption

        # Check file extension
        ext = os.path.splitext(filename.lower())[1]
        if ext in self.suspicious_extensions:
            score -= 0.5
            logging.warning(f"Suspicious extension detected: {ext}")

        # Check file signatures
        for signature in self.executable_signatures:
            if content.startswith(signature):
                score -= 0.4
                logging.warning(f"Executable signature detected: {signature}")
                break

        # Check file size (very large files are suspicious)
        file_size = len(content)
        if file_size > 100 * 1024 * 1024:  # 100MB
            score -= 0.2
            logging.warning(f"Very large file detected: {file_size} bytes")

        # Check MIME type vs extension consistency
        expected_mime = self._get_expected_mime_from_extension(filename)
        if expected_mime and expected_mime != mime_type:
            score -= 0.1
            logging.warning(f"MIME type mismatch: expected {expected_mime}, got {mime_type}")

        # Ensure score is within bounds
        return max(0.0, min(1.0, score))

    def _get_expected_mime_from_extension(self, filename: str) -> Optional[str]:
        """Get expected MIME type based on file extension."""
        ext = os.path.splitext(filename.lower())[1]
        mime_map = {
            ".txt": "text/plain",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".pdf": "application/pdf",
        }
        return mime_map.get(ext)


class ContentValidator:
    """Validates file content integrity and format consistency."""

    async def validate_content(self, content: bytes, mime_type: str, filename: str) -> bool:
        """
        Validate file content matches its declared type.

        Args:
            content: File content as bytes
            mime_type: MIME type of the file
            filename: Name of the file

        Returns:
            bool: True if content is valid

        Raises:
            ContentValidationError: If content validation fails
        """
        if mime_type.startswith("image/"):
            return await self._validate_image_content(content, mime_type)
        elif mime_type == "application/pdf":
            return await self._validate_pdf_content(content)
        elif mime_type.startswith("text/"):
            return await self._validate_text_content(content)
        else:
            # For unsupported types, just pass through
            return True

    async def _validate_image_content(self, content: bytes, mime_type: str) -> bool:
        """Validate image file content."""
        # Check for common image signatures
        if mime_type == "image/png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ContentValidationError("Invalid PNG file signature")
        elif mime_type == "image/jpeg" and not content.startswith(b"\xff\xd8"):
            raise ContentValidationError("Invalid JPEG file signature")
        elif mime_type == "image/gif" and not content.startswith(b"GIF"):
            raise ContentValidationError("Invalid GIF file signature")

        return True

    async def _validate_pdf_content(self, content: bytes) -> bool:
        """Validate PDF file content."""
        if not content.startswith(b"%PDF-"):
            raise ContentValidationError("Invalid PDF file signature")
        return True

    async def _validate_text_content(self, content: bytes) -> bool:
        """Validate text file content."""
        try:
            # Try to decode as UTF-8
            content.decode("utf-8")
            return True
        except UnicodeDecodeError:
            # Try other common encodings
            for encoding in ["latin-1", "cp1252"]:
                try:
                    content.decode(encoding)
                    return True
                except UnicodeDecodeError:
                    continue

            raise ContentValidationError("Text file contains invalid encoding")


class HashGenerator:
    """Generates cryptographic hashes for file deduplication."""

    def generate_hash(self, content: bytes) -> str:
        """
        Generate SHA256 hash of file content.

        Args:
            content: File content as bytes

        Returns:
            str: SHA256 hash as hexadecimal string
        """
        return hashlib.sha256(content).hexdigest()


class FileValidator:
    """Main file validation orchestrator."""

    def __init__(self, config: FileValidationConfig):
        self.config = config
        self.size_validator = SizeValidator(config.max_file_size)
        self.mime_detector = MimeTypeDetector()
        self.security_scanner = SecurityScanner(config.security_threshold)
        self.content_validator = ContentValidator()
        self.hash_generator = HashGenerator()

    async def validate_file(self, file: UploadFile) -> ValidationResult:
        """
        Comprehensive file validation pipeline.

        Args:
            file: FastAPI UploadFile object

        Returns:
            ValidationResult: Complete validation results
        """
        validation_errors = []
        metadata = {}

        try:
            # Read file content
            content = await file.read()
            file_size = len(content)

            # Generate file hash
            file_hash = self.hash_generator.generate_hash(content)

            # Detect MIME type
            mime_type = self.mime_detector.detect_mime_type(content, file.filename or "")

            # Store original position and reset file for potential future reads
            await file.seek(0)

            # Size validation
            try:
                self.size_validator.validate_size(file_size)
            except FileSizeError as e:
                validation_errors.append(str(e))

            # MIME type validation
            if self.config.allowed_mime_types and mime_type not in self.config.allowed_mime_types:
                validation_errors.append(f"MIME type '{mime_type}' not allowed")

            # Extension validation
            if self.config.blocked_extensions and file.filename:
                ext = os.path.splitext(file.filename.lower())[1]
                if ext in self.config.blocked_extensions:
                    validation_errors.append(f"File extension '{ext}' is blocked")

            # Security scanning
            security_score = 1.0
            if self.config.enable_security_scanning:
                security_score = await self.security_scanner.scan_file(content, mime_type, file.filename or "")
                if security_score < self.config.security_threshold:
                    validation_errors.append(
                        f"Security scan failed: score {security_score:.2f} below threshold {self.config.security_threshold}"
                    )

            # Content validation
            try:
                await self.content_validator.validate_content(content, mime_type, file.filename or "")
            except ContentValidationError as e:
                validation_errors.append(str(e))

            # Determine overall status
            if validation_errors:
                status = ValidationStatus.INVALID
            elif security_score < self.config.security_threshold:
                status = ValidationStatus.SUSPICIOUS
            else:
                status = ValidationStatus.VALID

            return ValidationResult(
                status=status,
                file_hash=file_hash,
                mime_type=mime_type,
                file_size=file_size,
                security_score=security_score,
                validation_errors=validation_errors,
                metadata=metadata,
            )

        except Exception as e:
            logging.error(f"File validation error: {e}")
            return ValidationResult(
                status=ValidationStatus.ERROR,
                file_hash="",
                mime_type="unknown",
                file_size=0,
                security_score=0.0,
                validation_errors=[f"Validation error: {str(e)}"],
                metadata={},
            )
