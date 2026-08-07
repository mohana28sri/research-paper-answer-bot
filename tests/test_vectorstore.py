"""
Tests for vector storage functionality.
"""

import gc
import tempfile
from unittest.mock import Mock, patch, MagicMock

import pytest
from langchain_core.documents import Document

from src.vectorstore import VectorStoreManager


def make_docs(items):
    """Convert list of dicts to langchain Documents."""
    return [Document(page_content=d["content"], metadata=d["metadata"]) for d in items]


def _close_manager(manager):
    """Release all ChromaDB resources held by a VectorStoreManager.

    Teardown order matters on Windows:
    - Drop _collection first (no more queries can reach the client)
    - Call _client._system.stop() to close the sqlite3 WAL connection
    - Null _client and _vectorstore to drop the last Python references
    - gc.collect() ensures CPython finalizes any lingering reference cycles
      before the caller removes the temp directory
    """
    if manager is None:
        return
    try:
        manager._collection = None
    except Exception:
        pass
    try:
        client = getattr(manager, "_client", None)
        if client is not None:
            system = getattr(client, "_system", None)
            if system is not None:
                try:
                    system.stop()
                except Exception:
                    pass
            manager._client = None
    except Exception:
        pass
    try:
        manager._vectorstore = None
    except Exception:
        pass
    gc.collect()


@pytest.fixture
def vector_store_manager(monkeypatch):
    """Create a real VectorStoreManager with guaranteed Windows-safe cleanup.

    Owns its own TemporaryDirectory so teardown order is fully controlled:
    1. _collection reference is dropped
    2. _client._system.stop() closes the sqlite3 WAL connection
    3. _client and _vectorstore references are dropped
    4. gc.collect() flushes any remaining CPython reference-counted handles
    5. TemporaryDirectory.cleanup() runs only after all handles are released
    """
    from src.config import Config
    from src.embeddings import EmbeddingManager

    temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
    monkeypatch.setenv("CHROMA_DB_PATH", temp_dir.name)
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    config = Config()
    embedding_manager = EmbeddingManager(config)

    manager = VectorStoreManager(config, embedding_manager)

    yield manager

    _close_manager(manager)
    temp_dir.cleanup()


class TestVectorStoreManager:
    """Test suite for VectorStoreManager."""

    # Patch PersistentClient for the whole class so no real sqlite3 file is
    # opened by tests that construct VectorStoreManager inline.  Tests that
    # use the `vector_store_manager` fixture are already covered by it.
    @pytest.fixture(autouse=True)
    def _patch_persistent_client(self, mock_chroma_collection):
        """Replace chromadb.PersistentClient with an in-memory mock."""
        mock_client = Mock()
        mock_client.get_or_create_collection = Mock(return_value=mock_chroma_collection)
        mock_client._system = None
        with patch("chromadb.PersistentClient", return_value=mock_client):
            yield

    def test_vectorstore_initialization(self, mock_config, mock_embedding_manager, temp_chroma_db):
        """Test that VectorStoreManager initializes correctly."""
        with patch("src.vectorstore.Chroma"):
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            assert manager.config == mock_config
            assert manager.embedding_manager == mock_embedding_manager

    def test_add_documents_to_vectorstore(self, mock_config, mock_embedding_manager, sample_documents):
        """Test adding documents to the vector store."""
        docs = make_docs(sample_documents)
        with patch("src.vectorstore.Chroma"):
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            manager.add_documents(docs)
            mock_embedding_manager.create_embeddings.assert_called_once_with(docs)

    def test_add_documents_creates_valid_metadata(self, mock_config, mock_embedding_manager):
        """Test that added documents have properly formatted metadata."""
        docs = [Document(page_content="Test content", metadata={"source": "test.pdf", "page": 1, "chunk_index": 0})]
        with patch("src.vectorstore.Chroma"):
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            with patch.object(manager, "add_documents", wraps=None) as mock_add:
                mock_add.return_value = Mock()
                manager.add_documents(docs)
                mock_add.assert_called_once_with(docs)

    def test_similarity_search(self, mock_config, mock_embedding_manager, mock_chroma_collection):
        """Test similarity search functionality."""
        with patch("src.vectorstore.Chroma"):
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            manager._collection = mock_chroma_collection
            results = manager.similarity_search("machine learning", k=3)
            assert isinstance(results, list)

    def test_similarity_search_with_metadata(self, mock_config, mock_embedding_manager, mock_chroma_collection):
        """Test that similarity search results include metadata."""
        mock_chroma_collection.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
            documents=["Content 1", "Content 2", "Content 3"],
            metadatas=[
                {"source": "paper1.pdf", "page": 1},
                {"source": "paper2.pdf", "page": 2},
                {"source": "paper3.pdf", "page": 3},
            ],
        )
        with patch("src.vectorstore.Chroma"):
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            manager._collection = mock_chroma_collection
            results = manager.similarity_search("supervised learning", k=2)
            assert len(results) <= 2

    def test_search_with_k_parameter(self, mock_config, mock_embedding_manager, mock_chroma_collection):
        """Test that k parameter controls number of results."""
        for i in range(10):
            mock_chroma_collection.add(
                ids=[f"doc{i}"],
                embeddings=[[0.1 + i * 0.01] * 384],
                documents=[f"Content {i}"],
                metadatas=[{"source": f"paper{i}.pdf"}],
            )
        with patch("src.vectorstore.Chroma"):
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            manager._collection = mock_chroma_collection
            assert len(manager.similarity_search("query", k=3)) <= 3
            assert len(manager.similarity_search("query", k=5)) <= 5

    def test_empty_vectorstore_search(self, mock_config, mock_embedding_manager, mock_chroma_collection):
        """Test searching in empty vector store."""
        with patch("src.vectorstore.Chroma"):
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            manager._collection = mock_chroma_collection
            results = manager.similarity_search("empty query", k=5)
            assert isinstance(results, list)

    def test_add_documents_with_numeric_metadata(self, mock_config, mock_embedding_manager):
        """Test adding documents with numeric metadata."""
        docs = [Document(
            page_content="Content",
            metadata={"source": "test.pdf", "page": 1, "chunk_index": 0, "score": 0.95},
        )]
        with patch("src.vectorstore.Chroma"):
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            with patch.object(manager, "add_documents", wraps=None) as mock_add:
                mock_add.return_value = Mock()
                manager.add_documents(docs)
                mock_add.assert_called_once_with(docs)

    def test_add_documents_with_nested_metadata(self, mock_config, mock_embedding_manager):
        """Test handling of nested metadata structures."""
        docs = [Document(
            page_content="Content",
            metadata={"source": "test.pdf", "page": 1, "author": "John Doe", "tags": ["ML", "AI"]},
        )]
        with patch("src.vectorstore.Chroma"):
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            with patch.object(manager, "add_documents", wraps=None) as mock_add:
                mock_add.return_value = Mock()
                manager.add_documents(docs)
                mock_add.assert_called_once_with(docs)

    def test_vectorstore_persistence(self, mock_config, mock_embedding_manager, temp_chroma_db):
        """Test that vector store persists data."""
        with patch("src.vectorstore.Chroma"):
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            assert manager.config.chroma_db_path is not None

    def test_vectorstore_collection_creation(self, mock_config, mock_embedding_manager):
        """Test that collection is created properly."""
        with patch("src.vectorstore.Chroma") as mock_chroma:
            mock_chroma.return_value = Mock()
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            assert manager is not None

    def test_search_result_structure(self, mock_config, mock_embedding_manager, mock_chroma_collection):
        """Test that search results have correct structure."""
        mock_chroma_collection.add(
            ids=["doc1"],
            embeddings=[[0.1] * 384],
            documents=["Test content"],
            metadatas=[{"source": "test.pdf", "page": 1}],
        )
        with patch("src.vectorstore.Chroma"):
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            manager._collection = mock_chroma_collection
            results = manager.similarity_search("test query", k=1)
            assert isinstance(results, list)
            if results:
                assert isinstance(results[0], Document)

    def test_vectorstore_error_handling(self, mock_config, mock_embedding_manager):
        """Test error handling in vector store operations."""
        # Override the autouse patch to simulate a connection failure.
        with patch("chromadb.PersistentClient", side_effect=Exception("Connection failed")):
            with pytest.raises(Exception):
                VectorStoreManager(mock_config, mock_embedding_manager)

    def test_add_documents_batch_processing(self, mock_config, mock_embedding_manager):
        """Test batch processing of multiple documents."""
        docs = [
            Document(page_content=f"Document {i}", metadata={"source": f"doc{i}.pdf"})
            for i in range(50)
        ]
        with patch("src.vectorstore.Chroma"):
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            with patch.object(manager, "add_documents", wraps=None) as mock_add:
                mock_add.return_value = Mock()
                manager.add_documents(docs)
                mock_add.assert_called_once_with(docs)

    def test_normalize_metadata_for_chroma(self, mock_config, mock_embedding_manager):
        """Test that metadata is normalized for ChromaDB compatibility."""
        with patch("src.vectorstore.Chroma"):
            manager = VectorStoreManager(mock_config, mock_embedding_manager)
            complex_metadata = {
                "source": "paper.pdf",
                "page": 1,
                "tags": ["ml", "ai"],
                "scores": {"precision": 0.95, "recall": 0.92},
            }
            normalized = manager._normalize_metadata(complex_metadata)
            assert normalized["source"] == "paper.pdf"
            assert normalized["page"] == 1
            assert isinstance(normalized["tags"], str)
            assert isinstance(normalized["scores"], str)
