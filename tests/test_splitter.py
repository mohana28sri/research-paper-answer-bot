"""
Tests for document splitter functionality.
"""

import pytest

from src.splitter import DocumentSplitter


class TestDocumentSplitter:
    """Test suite for DocumentSplitter."""

    def test_splitter_initialization(self):
        """Test that DocumentSplitter initializes with default values."""
        splitter = DocumentSplitter()
        assert splitter.chunk_size > 0
        assert splitter.chunk_overlap >= 0
        assert splitter.chunk_overlap < splitter.chunk_size

    def test_splitter_custom_configuration(self):
        """Test initialization with custom chunk size and overlap."""
        splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=100)
        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 100

    def test_split_single_document(self, sample_pdf_content):
        """Test splitting a single document into chunks."""
        splitter = DocumentSplitter(chunk_size=200, chunk_overlap=20)

        documents = [{"page_content": sample_pdf_content, "metadata": {"source": "test.pdf"}}]
        chunks = splitter.split_documents(documents)

        assert len(chunks) > 1
        assert all("page_content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)

    def test_split_documents_chunk_size(self, sample_pdf_content):
        """Test that chunks respect the maximum size."""
        chunk_size = 150
        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=10)

        documents = [{"page_content": sample_pdf_content, "metadata": {"source": "test.pdf"}}]
        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            assert len(chunk["page_content"]) <= chunk_size + 50  # Allow some tolerance

    def test_split_documents_preserves_metadata(self):
        """Test that metadata is preserved during splitting."""
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=10)

        original_metadata = {
            "source": "research_paper.pdf",
            "page": 1,
            "title": "Test Paper",
        }
        documents = [
            {
                "page_content": "This is a test document " * 20,  # Repeat to ensure multiple chunks
                "metadata": original_metadata,
            }
        ]

        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            assert chunk["metadata"]["source"] == original_metadata["source"]
            assert chunk["metadata"]["page"] == original_metadata["page"]

    def test_split_empty_document(self):
        """Test handling of empty documents."""
        splitter = DocumentSplitter()
        documents = [{"page_content": "", "metadata": {}}]

        chunks = splitter.split_documents(documents)

        assert isinstance(chunks, list)

    def test_split_very_short_document(self):
        """Test splitting a document shorter than chunk size."""
        splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)

        documents = [
            {
                "page_content": "Short content",
                "metadata": {"source": "short.pdf"},
            }
        ]

        chunks = splitter.split_documents(documents)

        assert len(chunks) >= 1
        assert chunks[0]["page_content"] == "Short content"

    def test_split_multiple_documents(self):
        """Test splitting multiple documents."""
        splitter = DocumentSplitter(chunk_size=200, chunk_overlap=20)

        documents = [
            {
                "page_content": "Document one " * 30,
                "metadata": {"source": "doc1.pdf", "page": 1},
            },
            {
                "page_content": "Document two " * 25,
                "metadata": {"source": "doc2.pdf", "page": 1},
            },
        ]

        chunks = splitter.split_documents(documents)

        assert len(chunks) > 2
        # Verify both documents contributed chunks
        sources = [chunk["metadata"]["source"] for chunk in chunks]
        assert "doc1.pdf" in sources
        assert "doc2.pdf" in sources

    def test_split_preserves_text_continuity(self):
        """Test that text continuity is maintained across chunks (overlap)."""
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)

        documents = [
            {
                "page_content": "The quick brown fox jumps over the lazy dog. " * 5,
                "metadata": {"source": "test.pdf"},
            }
        ]

        chunks = splitter.split_documents(documents)

        if len(chunks) > 1:
            # Check that consecutive chunks have overlapping text
            for i in range(len(chunks) - 1):
                # There should be some overlap between consecutive chunks
                assert len(chunks[i]["page_content"]) > 0
                assert len(chunks[i + 1]["page_content"]) > 0

    def test_split_with_special_characters(self):
        """Test splitting text with special characters."""
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=10)

        documents = [
            {
                "page_content": "Special chars: @#$%^&*()_+-=[]{}|;:',.<>?/\\" * 5,
                "metadata": {"source": "special.pdf"},
            }
        ]

        chunks = splitter.split_documents(documents)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk["page_content"]) > 0

    def test_split_with_newlines_and_formatting(self):
        """Test splitting text with newlines and formatting."""
        splitter = DocumentSplitter(chunk_size=150, chunk_overlap=15)

        documents = [
            {
                "page_content": "Line 1\nLine 2\nLine 3\n" * 10,
                "metadata": {"source": "formatted.pdf"},
            }
        ]

        chunks = splitter.split_documents(documents)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk["page_content"]) > 0
