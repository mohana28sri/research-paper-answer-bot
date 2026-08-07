"""
Tests for document loader functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, call

import pytest
from langchain_core.documents import Document

from src.loader import DocumentLoader, DocumentLoadError


def _make_config(monkeypatch, data_folder: str):
    """Return a real Config with DATA_FOLDER pointing at data_folder."""
    from src.config import Config

    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("CHROMA_DB_PATH", data_folder)
    monkeypatch.setenv("DATA_FOLDER", data_folder)
    return Config()


def _fake_doc(path: str, page: int = 0, content: str = "Test content") -> Document:
    """Build a Document that mimics what PyPDFLoader.load() returns."""
    return Document(page_content=content, metadata={"source": path, "page": page})


class TestDocumentLoader:
    """Test suite for DocumentLoader."""

    def test_loader_initialization(self, temp_data_dir, monkeypatch):
        """Test that DocumentLoader initializes correctly."""
        config = _make_config(monkeypatch, str(temp_data_dir))
        loader = DocumentLoader(config)
        assert loader.config is config
        assert loader._documents == []
        stats = loader.get_statistics()
        assert stats["total_pdfs"] == 0
        assert stats["total_pages"] == 0
        assert stats["successfully_loaded"] == 0
        assert stats["failed_files"] == []

    def test_load_single_pdf_success(self, temp_data_dir, monkeypatch):
        """Test successful loading of a single PDF via PyPDFLoader."""
        config = _make_config(monkeypatch, str(temp_data_dir))
        pdf_path = temp_data_dir / "test_paper.pdf"
        pdf_path.touch()

        fake_page = _fake_doc(str(pdf_path), page=0, content="Machine Learning content")

        with patch("src.loader.PyPDFLoader") as mock_loader_cls:
            mock_loader_cls.return_value.load.return_value = [fake_page]
            loader = DocumentLoader(config)
            documents = loader.load_single_document(str(pdf_path))

        assert len(documents) == 1
        assert "Machine Learning" in documents[0].page_content

    def test_load_single_pdf_file_not_found(self, temp_data_dir, monkeypatch):
        """Test that DocumentLoadError is raised when the PDF file does not exist."""
        config = _make_config(monkeypatch, str(temp_data_dir))
        loader = DocumentLoader(config)
        non_existent = temp_data_dir / "nonexistent.pdf"

        with pytest.raises(DocumentLoadError):
            loader.load_single_document(str(non_existent))

    def test_load_single_pdf_invalid_file_type(self, temp_data_dir, monkeypatch):
        """Test that DocumentLoadError is raised for non-PDF files."""
        config = _make_config(monkeypatch, str(temp_data_dir))
        txt_file = temp_data_dir / "test.txt"
        txt_file.write_text("This is a text file, not a PDF")

        loader = DocumentLoader(config)

        with pytest.raises(DocumentLoadError):
            loader.load_single_document(str(txt_file))

    def test_load_multiple_pdfs(self, temp_data_dir, monkeypatch):
        """Test that load_all_documents loads every PDF in the data folder."""
        config = _make_config(monkeypatch, str(temp_data_dir))

        pdf1 = temp_data_dir / "paper1.pdf"
        pdf2 = temp_data_dir / "paper2.pdf"
        pdf1.touch()
        pdf2.touch()

        doc1 = _fake_doc(str(pdf1), content="Content from paper 1")
        doc2 = _fake_doc(str(pdf2), content="Content from paper 2")

        def loader_factory(path):
            mock = Mock()
            mock.load.return_value = [doc1] if "paper1" in path else [doc2]
            return mock

        with patch("src.loader.PyPDFLoader", side_effect=loader_factory):
            loader = DocumentLoader(config)
            documents = loader.load_all_documents()

        assert len(documents) == 2
        contents = {d.page_content for d in documents}
        assert "Content from paper 1" in contents
        assert "Content from paper 2" in contents

    def test_load_all_documents_missing_folder(self, temp_data_dir, monkeypatch):
        """Test that DocumentLoadError is raised when the data folder does not exist."""
        missing = temp_data_dir / "does_not_exist"
        config = _make_config(monkeypatch, str(missing))
        loader = DocumentLoader(config)

        with pytest.raises(DocumentLoadError):
            loader.load_all_documents()

    def test_load_all_documents_empty_folder(self, temp_data_dir, monkeypatch):
        """Test that DocumentLoadError is raised when no PDFs are found."""
        config = _make_config(monkeypatch, str(temp_data_dir))
        loader = DocumentLoader(config)

        with pytest.raises(DocumentLoadError):
            loader.load_all_documents()

    def test_load_pdf_with_metadata(self, temp_data_dir, monkeypatch):
        """Test that loaded documents have source, paper_title, and page_number metadata."""
        config = _make_config(monkeypatch, str(temp_data_dir))
        pdf_path = temp_data_dir / "my_paper.pdf"
        pdf_path.touch()

        fake_page = _fake_doc(str(pdf_path), page=0)

        with patch("src.loader.PyPDFLoader") as mock_loader_cls:
            mock_loader_cls.return_value.load.return_value = [fake_page]
            loader = DocumentLoader(config)
            documents = loader.load_single_document(str(pdf_path))

        assert len(documents) == 1
        meta = documents[0].metadata
        assert meta["source"] == str(pdf_path)
        assert meta["paper_title"] == "my_paper"
        assert meta["page_number"] == 1  # page 0 + 1

    def test_load_pdf_exception_handling(self, temp_data_dir, monkeypatch):
        """Test that a PyPDFLoader failure is wrapped in DocumentLoadError."""
        config = _make_config(monkeypatch, str(temp_data_dir))
        pdf_path = temp_data_dir / "corrupted.pdf"
        pdf_path.touch()

        with patch("src.loader.PyPDFLoader") as mock_loader_cls:
            mock_loader_cls.return_value.load.side_effect = Exception("PDF corrupted")
            loader = DocumentLoader(config)

            with pytest.raises(DocumentLoadError):
                loader.load_single_document(str(pdf_path))
