"""
Test suite for file validation and security framework.

This module tests the comprehensive file validation system including:
- File size validation
- MIME type detection
- Security scanning
- Content validation
- Hash generation and deduplication
- Custom exception handling
"""

import asyncio
import base64
import hashlib
from io import BytesIO
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from models.errors import ContentValidationError, FileSizeError, FileValidationError, MimeTypeError, SecurityError

# Import the validation components we'll implement
from models.validation import FileValidationConfig, ValidationResult, ValidationStatus
from tests.utils.fixtures import MockFileUpload
from validation.validators import (
    ContentValidator,
    FileValidator,
    HashGenerator,
    MimeTypeDetector,
    SecurityScanner,
    SizeValidator,
)


class TestFileValidationConfig:
    """Test file validation configuration."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = FileValidationConfig()
        assert config.max_file_size == 50 * 1024 * 1024  # 50MB
        assert config.allowed_mime_types is None
        assert config.blocked_extensions is None
        assert config.enable_security_scanning is True
        assert config.security_threshold == 0.8

    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        allowed_types = {"image/jpeg", "image/png", "text/plain"}
        blocked_exts = {".exe", ".bat"}

        config = FileValidationConfig(
            max_file_size=10 * 1024 * 1024,  # 10MB
            allowed_mime_types=allowed_types,
            blocked_extensions=blocked_exts,
            enable_security_scanning=False,
            security_threshold=0.5,
        )

        assert config.max_file_size == 10 * 1024 * 1024
        assert config.allowed_mime_types == allowed_types
        assert config.blocked_extensions == blocked_exts
        assert config.enable_security_scanning is False
        assert config.security_threshold == 0.5


class TestCustomExceptions:
    """Test custom exception hierarchy."""

    def test_file_validation_error_base(self):
        """Test base FileValidationError exception."""
        error = FileValidationError("Base validation error")
        assert str(error) == "Base validation error"
        assert isinstance(error, Exception)

    def test_file_size_error(self):
        """Test FileSizeError exception."""
        error = FileSizeError("File too large")
        assert str(error) == "File too large"
        assert isinstance(error, FileValidationError)
        assert isinstance(error, Exception)

    def test_mime_type_error(self):
        """Test MimeTypeError exception."""
        error = MimeTypeError("Invalid MIME type")
        assert str(error) == "Invalid MIME type"
        assert isinstance(error, FileValidationError)

    def test_security_error(self):
        """Test SecurityError exception."""
        error = SecurityError("Security scan failed")
        assert str(error) == "Security scan failed"
        assert isinstance(error, FileValidationError)

    def test_content_validation_error(self):
        """Test ContentValidationError exception."""
        error = ContentValidationError("Content validation failed")
        assert str(error) == "Content validation failed"
        assert isinstance(error, FileValidationError)


class TestValidationResult:
    """Test ValidationResult data class."""

    def test_validation_result_creation(self):
        """Test creating ValidationResult with all fields."""
        result = ValidationResult(
            status=ValidationStatus.VALID,
            file_hash="abc123",
            mime_type="image/png",
            file_size=1024,
            security_score=0.9,
            validation_errors=[],
            metadata={"width": 100, "height": 100},
        )

        assert result.status == ValidationStatus.VALID
        assert result.file_hash == "abc123"
        assert result.mime_type == "image/png"
        assert result.file_size == 1024
        assert result.security_score == 0.9
        assert result.validation_errors == []
        assert result.metadata == {"width": 100, "height": 100}

    def test_validation_result_with_errors(self):
        """Test ValidationResult with validation errors."""
        errors = ["File too large", "Invalid format"]
        result = ValidationResult(
            status=ValidationStatus.INVALID,
            file_hash="def456",
            mime_type="application/unknown",
            file_size=100 * 1024 * 1024,
            security_score=0.3,
            validation_errors=errors,
            metadata={},
        )

        assert result.status == ValidationStatus.INVALID
        assert result.validation_errors == errors
        assert result.security_score == 0.3


class TestSizeValidator:
    """Test file size validation component."""

    def test_size_validator_creation(self):
        """Test creating SizeValidator with max size."""
        max_size = 10 * 1024 * 1024  # 10MB
        validator = SizeValidator(max_size)
        assert validator.max_file_size == max_size

    def test_validate_size_success(self):
        """Test successful size validation."""
        validator = SizeValidator(10 * 1024 * 1024)  # 10MB
        file_content = b"x" * (5 * 1024 * 1024)  # 5MB

        # Should not raise exception
        validator.validate_size(len(file_content))

    def test_validate_size_failure(self):
        """Test size validation failure."""
        validator = SizeValidator(10 * 1024 * 1024)  # 10MB
        file_size = 15 * 1024 * 1024  # 15MB (exceeds limit)

        with pytest.raises(FileSizeError) as exc_info:
            validator.validate_size(file_size)

        assert "exceeds maximum allowed size" in str(exc_info.value)

    def test_validate_zero_size(self):
        """Test validation of zero-size file."""
        validator = SizeValidator(10 * 1024 * 1024)

        with pytest.raises(FileSizeError) as exc_info:
            validator.validate_size(0)

        assert "empty file" in str(exc_info.value).lower()

    def test_validate_negative_size(self):
        """Test validation of negative size (edge case)."""
        validator = SizeValidator(10 * 1024 * 1024)

        with pytest.raises(FileSizeError):
            validator.validate_size(-1)


class TestMimeTypeDetector:
    """Test MIME type detection component."""

    def test_mime_detector_creation(self):
        """Test creating MimeTypeDetector."""
        detector = MimeTypeDetector()
        assert detector is not None

    @patch("magic.from_buffer")
    def test_detect_mime_type_png(self, mock_magic):
        """Test MIME type detection for PNG image."""
        mock_magic.return_value = "image/png"
        detector = MimeTypeDetector()

        # PNG file signature
        png_content = b"\x89PNG\r\n\x1a\n" + b"x" * 100
        mime_type = detector.detect_mime_type(png_content, "test.png")

        assert mime_type == "image/png"
        mock_magic.assert_called_once_with(png_content, mime=True)

    @patch("magic.from_buffer")
    def test_detect_mime_type_pdf(self, mock_magic):
        """Test MIME type detection for PDF file."""
        mock_magic.return_value = "application/pdf"
        detector = MimeTypeDetector()

        # PDF file signature
        pdf_content = b"%PDF-1.4" + b"x" * 100
        mime_type = detector.detect_mime_type(pdf_content, "test.pdf")

        assert mime_type == "application/pdf"

    @patch("magic.from_buffer")
    def test_detect_mime_type_fallback(self, mock_magic):
        """Test MIME type detection with fallback to extension."""
        mock_magic.side_effect = Exception("Magic detection failed")
        detector = MimeTypeDetector()

        mime_type = detector.detect_mime_type(b"some content", "test.txt")

        # Should fallback to extension-based detection
        assert mime_type == "text/plain"

    def test_get_mime_from_extension(self):
        """Test getting MIME type from file extension."""
        detector = MimeTypeDetector()

        assert detector._get_mime_from_extension("test.jpg") == "image/jpeg"
        assert detector._get_mime_from_extension("test.png") == "image/png"
        assert detector._get_mime_from_extension("test.pdf") == "application/pdf"
        assert detector._get_mime_from_extension("test.txt") == "text/plain"
        assert detector._get_mime_from_extension("test.unknown") == "application/octet-stream"


class TestSecurityScanner:
    """Test security scanning component."""

    def test_security_scanner_creation(self):
        """Test creating SecurityScanner with threshold."""
        scanner = SecurityScanner(0.8)
        assert scanner.security_threshold == 0.8

    @pytest.mark.asyncio
    async def test_scan_file_safe(self):
        """Test scanning a safe file."""
        scanner = SecurityScanner(0.8)
        safe_content = b"This is a normal text file content."

        score = await scanner.scan_file(safe_content, "text/plain", "test.txt")

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score >= 0.8  # Should be considered safe

    @pytest.mark.asyncio
    async def test_scan_file_suspicious_extension(self):
        """Test scanning file with suspicious extension."""
        scanner = SecurityScanner(0.8)
        content = b"Some content"

        score = await scanner.scan_file(content, "application/octet-stream", "malware.exe")

        assert isinstance(score, float)
        assert score < 0.8  # Should be flagged as suspicious

    @pytest.mark.asyncio
    async def test_scan_file_large_size(self):
        """Test scanning very large file."""
        scanner = SecurityScanner(0.8)
        large_content = b"x" * (101 * 1024 * 1024)  # 101MB (exceeds 100MB threshold)

        score = await scanner.scan_file(large_content, "application/octet-stream", "large.bin")

        assert isinstance(score, float)
        # Large files should get lower scores
        assert score < 1.0

    @pytest.mark.asyncio
    async def test_scan_file_with_executable_signatures(self):
        """Test scanning file with executable signatures."""
        scanner = SecurityScanner(0.8)
        # MZ header (Windows executable)
        exe_content = b"MZ" + b"x" * 100

        score = await scanner.scan_file(exe_content, "application/octet-stream", "file.exe")

        assert score < 0.5  # Should be flagged as high risk


class TestContentValidator:
    """Test content validation component."""

    def test_content_validator_creation(self):
        """Test creating ContentValidator."""
        validator = ContentValidator()
        assert validator is not None

    @pytest.mark.asyncio
    async def test_validate_image_content_valid(self):
        """Test validating valid image content."""
        validator = ContentValidator()

        # Simple PNG content (1x1 pixel)
        png_content = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )

        result = await validator.validate_content(png_content, "image/png", "test.png")
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_image_content_invalid(self):
        """Test validating invalid image content."""
        validator = ContentValidator()

        # Invalid image content (just text)
        invalid_content = b"This is not an image"

        with pytest.raises(ContentValidationError):
            await validator.validate_content(invalid_content, "image/png", "test.png")

    @pytest.mark.asyncio
    async def test_validate_pdf_content_valid(self):
        """Test validating valid PDF content."""
        validator = ContentValidator()

        # Minimal PDF content
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\ntrailer\n<<>>\n%%EOF"

        result = await validator.validate_content(pdf_content, "application/pdf", "test.pdf")
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_text_content(self):
        """Test validating text content."""
        validator = ContentValidator()

        text_content = b"This is a normal text file content."

        result = await validator.validate_content(text_content, "text/plain", "test.txt")
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_unsupported_type(self):
        """Test validating unsupported file type."""
        validator = ContentValidator()

        content = b"Some binary content"

        # Should pass through unsupported types
        result = await validator.validate_content(content, "application/unknown", "test.unknown")
        assert result is True


class TestHashGenerator:
    """Test hash generation component."""

    def test_hash_generator_creation(self):
        """Test creating HashGenerator."""
        generator = HashGenerator()
        assert generator is not None

    def test_generate_hash_sha256(self):
        """Test generating SHA256 hash."""
        generator = HashGenerator()
        content = b"This is test content for hashing."

        hash_result = generator.generate_hash(content)

        # Verify it's a valid SHA256 hash
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 produces 64-character hex string

        # Verify reproducibility
        hash_result2 = generator.generate_hash(content)
        assert hash_result == hash_result2

    def test_generate_hash_different_content(self):
        """Test that different content produces different hashes."""
        generator = HashGenerator()
        content1 = b"Content 1"
        content2 = b"Content 2"

        hash1 = generator.generate_hash(content1)
        hash2 = generator.generate_hash(content2)

        assert hash1 != hash2

    def test_generate_hash_empty_content(self):
        """Test generating hash for empty content."""
        generator = HashGenerator()
        empty_content = b""

        hash_result = generator.generate_hash(empty_content)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
        # Hash of empty string should be consistent
        expected_hash = hashlib.sha256(empty_content).hexdigest()
        assert hash_result == expected_hash


class TestFileValidator:
    """Test main FileValidator orchestrator."""

    def test_file_validator_creation(self):
        """Test creating FileValidator with config."""
        config = FileValidationConfig()
        validator = FileValidator(config)

        assert validator.config == config
        assert isinstance(validator.size_validator, SizeValidator)
        assert isinstance(validator.mime_detector, MimeTypeDetector)
        assert isinstance(validator.security_scanner, SecurityScanner)
        assert isinstance(validator.content_validator, ContentValidator)
        assert isinstance(validator.hash_generator, HashGenerator)

    @pytest.mark.asyncio
    async def test_validate_file_success(self, mock_text_file):
        """Test successful file validation."""
        config = FileValidationConfig(
            max_file_size=10 * 1024 * 1024, allowed_mime_types={"text/plain"}, enable_security_scanning=True
        )
        validator = FileValidator(config)

        # Mock the file's read method
        mock_text_file.read = AsyncMock(return_value=b"This is test content.")

        result = await validator.validate_file(mock_text_file)

        assert isinstance(result, ValidationResult)
        assert result.status == ValidationStatus.VALID
        assert result.mime_type == "text/plain"
        assert result.file_size > 0
        assert isinstance(result.file_hash, str)
        assert len(result.file_hash) == 64
        assert result.security_score >= 0.8
        assert len(result.validation_errors) == 0

    @pytest.mark.asyncio
    async def test_validate_file_size_failure(self, mock_text_file):
        """Test file validation with size failure."""
        config = FileValidationConfig(max_file_size=10)  # Very small limit
        validator = FileValidator(config)

        # Mock large file content
        large_content = b"x" * 100  # Exceeds 10 byte limit
        mock_text_file.read = AsyncMock(return_value=large_content)

        result = await validator.validate_file(mock_text_file)

        assert result.status == ValidationStatus.INVALID
        assert len(result.validation_errors) > 0
        assert any("size" in error.lower() for error in result.validation_errors)

    @pytest.mark.asyncio
    async def test_validate_file_mime_type_restriction(self, mock_text_file):
        """Test file validation with MIME type restriction."""
        config = FileValidationConfig(allowed_mime_types={"image/png", "image/jpeg"})  # Only images allowed
        validator = FileValidator(config)

        mock_text_file.read = AsyncMock(return_value=b"This is text content.")
        mock_text_file.content_type = "text/plain"

        result = await validator.validate_file(mock_text_file)

        assert result.status == ValidationStatus.INVALID
        assert len(result.validation_errors) > 0
        assert any("mime type" in error.lower() for error in result.validation_errors)

    @pytest.mark.asyncio
    async def test_validate_file_security_failure(self, mock_text_file):
        """Test file validation with security failure."""
        config = FileValidationConfig(security_threshold=0.9, enable_security_scanning=True)  # Very high threshold
        validator = FileValidator(config)

        # Mock suspicious filename
        mock_text_file.filename = "malware.exe"
        mock_text_file.read = AsyncMock(return_value=b"MZ" + b"x" * 100)  # Executable signature

        result = await validator.validate_file(mock_text_file)

        assert result.status in [ValidationStatus.INVALID, ValidationStatus.SUSPICIOUS]
        assert result.security_score < 0.9

    @pytest.mark.asyncio
    async def test_validate_file_with_deduplication(self, mock_text_file):
        """Test file validation with hash-based deduplication check."""
        config = FileValidationConfig()
        validator = FileValidator(config)

        content = b"This is duplicate content."
        mock_text_file.read = AsyncMock(return_value=content)

        # First validation
        result1 = await validator.validate_file(mock_text_file)

        # Second validation with same content (simulate duplicate)
        mock_text_file.read = AsyncMock(return_value=content)
        result2 = await validator.validate_file(mock_text_file)

        # Both should have same hash
        assert result1.file_hash == result2.file_hash
        assert result1.status == ValidationStatus.VALID
        assert result2.status == ValidationStatus.VALID

    @pytest.mark.asyncio
    async def test_validate_file_disabled_security_scanning(self, mock_text_file):
        """Test file validation with security scanning disabled."""
        config = FileValidationConfig(enable_security_scanning=False)
        validator = FileValidator(config)

        mock_text_file.read = AsyncMock(return_value=b"Some content.")

        result = await validator.validate_file(mock_text_file)

        assert result.status == ValidationStatus.VALID
        # Security score should be neutral when scanning is disabled
        assert result.security_score == 1.0


class TestValidationIntegration:
    """Integration tests for the complete validation pipeline."""

    @pytest.mark.asyncio
    async def test_full_validation_pipeline_image(self, mock_image_file):
        """Test complete validation pipeline with image file."""
        config = FileValidationConfig(
            max_file_size=10 * 1024 * 1024, allowed_mime_types={"image/png", "image/jpeg"}, enable_security_scanning=True
        )
        validator = FileValidator(config)

        # Use actual PNG content
        png_content = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        mock_image_file.read = AsyncMock(return_value=png_content)

        result = await validator.validate_file(mock_image_file)

        assert result.status == ValidationStatus.VALID
        assert result.mime_type in ["image/png"]
        assert result.file_size == len(png_content)
        assert isinstance(result.file_hash, str)
        assert result.security_score >= 0.8
        assert len(result.validation_errors) == 0

    @pytest.mark.asyncio
    async def test_validation_pipeline_with_multiple_failures(self):
        """Test validation pipeline with multiple validation failures."""
        config = FileValidationConfig(
            max_file_size=100,  # Very small
            allowed_mime_types={"image/png"},  # Only PNG allowed
            blocked_extensions={".exe"},
            enable_security_scanning=True,
            security_threshold=0.9,
        )
        validator = FileValidator(config)

        # Create a problematic file
        problematic_file = MockFileUpload(
            filename="malware.exe",
            content=b"MZ" + b"x" * 200,  # Large executable content
            content_type="application/octet-stream",
        )
        problematic_file.read = AsyncMock(return_value=problematic_file.content)

        result = await validator.validate_file(problematic_file)

        assert result.status == ValidationStatus.INVALID
        assert len(result.validation_errors) > 1  # Multiple failures

        # Check for specific error types
        error_text = " ".join(result.validation_errors).lower()
        assert "size" in error_text
        assert any("mime" in error or "type" in error for error in result.validation_errors)

    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test validation error handling for corrupted files."""
        config = FileValidationConfig()
        validator = FileValidator(config)

        # Create a mock file that raises an exception during read
        corrupted_file = Mock()
        corrupted_file.filename = "corrupted.txt"
        corrupted_file.content_type = "text/plain"
        corrupted_file.read = AsyncMock(side_effect=Exception("File read error"))

        result = await validator.validate_file(corrupted_file)

        assert result.status == ValidationStatus.ERROR
        assert len(result.validation_errors) > 0
        assert any("error" in error.lower() for error in result.validation_errors)
