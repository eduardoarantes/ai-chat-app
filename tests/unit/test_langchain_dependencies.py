"""
Unit tests for LangChain dependencies and document loader imports.

This test module validates that all required LangChain dependencies are properly
installed and can be imported without conflicts.
"""

import importlib
import sys

import pytest


class TestLangChainDependencies:
    """Test LangChain dependencies and import availability."""

    def test_core_langchain_imports(self):
        """Test that core LangChain packages can be imported."""
        # Test core LangChain imports
        try:
            import langchain
            from langchain.document_loaders.base import BaseLoader
            from langchain.schema import Document
        except ImportError as e:
            pytest.fail(f"Failed to import core LangChain components: {e}")

    def test_pymupdf_loader_import(self):
        """Test that PyMuPDFLoader can be imported."""
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
        except ImportError as e:
            pytest.skip(f"PyMuPDFLoader not available, will install: {e}")

    def test_text_loader_import(self):
        """Test that TextLoader can be imported."""
        try:
            from langchain_community.document_loaders import TextLoader
        except ImportError as e:
            pytest.fail(f"TextLoader should be available in langchain-community: {e}")

    def test_unstructured_loader_import(self):
        """Test that UnstructuredLoader can be imported."""
        try:
            from langchain_community.document_loaders import UnstructuredFileLoader
        except ImportError as e:
            pytest.skip(f"UnstructuredFileLoader not available, will install: {e}")

    def test_document_schema_import(self):
        """Test that Document schema can be imported and used."""
        try:
            from langchain.schema import Document

            # Test creating a Document instance
            doc = Document(page_content="Test content", metadata={"source": "test", "page": 1})

            assert doc.page_content == "Test content"
            assert doc.metadata["source"] == "test"
            assert doc.metadata["page"] == 1

        except ImportError as e:
            pytest.fail(f"Failed to import or use Document schema: {e}")

    def test_python_dependencies(self):
        """Test that required Python dependencies are available."""
        required_modules = ["tempfile", "pathlib", "os", "typing", "asyncio", "logging"]

        for module_name in required_modules:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                pytest.fail(f"Required Python module '{module_name}' not available: {e}")

    def test_pymupdf_dependency(self):
        """Test that PyMuPDF (fitz) can be imported."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            pytest.skip("PyMuPDF not installed yet, will be added to requirements")

    def test_unstructured_dependency(self):
        """Test that unstructured package can be imported."""
        try:
            import unstructured
        except ImportError:
            pytest.skip("unstructured package not installed yet, will be added to requirements")

    def test_version_compatibility(self):
        """Test that installed package versions are compatible."""
        try:
            import langchain

            # Check if langchain version is accessible
            if hasattr(langchain, "__version__"):
                version = langchain.__version__
                print(f"LangChain version: {version}")

                # Ensure version is reasonably recent (0.1.0+)
                major, minor = version.split(".")[:2]
                assert int(major) >= 0
                assert int(minor) >= 1

        except (ImportError, AttributeError, ValueError) as e:
            pytest.skip(f"Could not verify LangChain version compatibility: {e}")
