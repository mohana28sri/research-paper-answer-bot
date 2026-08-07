"""
Shared pytest fixtures and configuration for all tests.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest
from dotenv import load_dotenv


# Load test environment
load_dotenv()


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_chroma_db():
    """Create a temporary directory for ChromaDB.

    Uses ignore_cleanup_errors=True (Python 3.10+) so that any residual
    sqlite3 file locks on Windows do not cause the test suite to error
    during teardown.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config(temp_chroma_db, monkeypatch):
    """Create a mock Config instance with test values."""
    from src.config import Config

    # Set environment variables for testing
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
    monkeypatch.setenv("CHAT_MODEL", "meta/llama-3.1-8b-instruct")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("CHROMA_DB_PATH", str(temp_chroma_db))
    monkeypatch.setenv("DATA_FOLDER", "./data")

    return Config()


@pytest.fixture
def mock_llm():
    """Create a mock LLM instance."""
    llm = Mock()
    llm.invoke = Mock(return_value="Test answer from LLM")
    llm.temperature = 0.0
    return llm


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings instance."""
    embeddings = Mock()

    def embed_text(text):
        """Generate a deterministic mock embedding."""
        # Return a fixed-size embedding (384 dimensions for MiniLM)
        return [float(hash(text) % 100) / 100 for _ in range(384)]

    def embed_documents(texts):
        """Generate embeddings for multiple documents."""
        return [embed_text(text) for text in texts]

    def embed_query(query):
        """Generate embedding for a query."""
        return embed_text(query)

    embeddings.embed_documents = Mock(side_effect=embed_documents)
    embeddings.embed_query = Mock(side_effect=embed_query)

    return embeddings


@pytest.fixture
def sample_pdf_content():
    """Return sample PDF-like text content."""
    return """
    Machine Learning Fundamentals
    
    Abstract: This paper presents fundamental concepts in machine learning including
    supervised learning, unsupervised learning, and reinforcement learning.
    
    1. Introduction
    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    
    2. Supervised Learning
    Supervised learning involves training models on labeled datasets where the
    desired output is known.
    
    3. Unsupervised Learning
    Unsupervised learning finds patterns in unlabeled data without predefined outputs.
    
    4. Reinforcement Learning
    Reinforcement learning uses agents that learn through interaction with environments.
    
    Conclusion: Machine learning has revolutionized many fields of technology.
    """


@pytest.fixture
def sample_documents():
    """Return sample document chunks."""
    return [
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {
                "source": "test_paper.pdf",
                "page": 1,
                "chunk_index": 0,
            },
        },
        {
            "content": "Supervised learning trains on labeled datasets.",
            "metadata": {
                "source": "test_paper.pdf",
                "page": 2,
                "chunk_index": 1,
            },
        },
        {
            "content": "Unsupervised learning finds patterns in unlabeled data.",
            "metadata": {
                "source": "test_paper.pdf",
                "page": 3,
                "chunk_index": 2,
            },
        },
    ]


@pytest.fixture
def mock_chroma_collection():
    """Create a mock Chroma collection."""
    collection = Mock()

    # Store documents in memory for testing
    stored_documents = []

    def add_mock(ids, embeddings, documents, metadatas):
        """Mock add method."""
        for i, doc_id in enumerate(ids):
            stored_documents.append({
                "id": doc_id,
                "embedding": embeddings[i] if embeddings else None,
                "document": documents[i] if documents else None,
                "metadata": metadatas[i] if metadatas else None,
            })
        return ids

    def query_mock(query_embeddings, n_results=3):
        """Mock query method."""
        if not stored_documents:
            return {
                "ids": [[]],
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]],
            }

        # Simple mock: return all stored documents (in production, this would use similarity)
        result_count = min(n_results, len(stored_documents))
        return {
            "ids": [[doc["id"] for doc in stored_documents[:result_count]]],
            "distances": [[0.1] * result_count],
            "documents": [[doc["document"] for doc in stored_documents[:result_count]]],
            "metadatas": [[doc["metadata"] for doc in stored_documents[:result_count]]],
        }

    collection.add = Mock(side_effect=add_mock)
    collection.query = Mock(side_effect=query_mock)
    collection.count = Mock(return_value=len(stored_documents))

    return collection


@pytest.fixture
def mock_embedding_manager():
    """Create a mock EmbeddingManager instance."""
    from src.embeddings import EmbeddingManager
    manager = Mock(spec=EmbeddingManager)

    def _embed_docs(documents):
        return [[0.1] * 384 for _ in documents]

    def _embed_query(query):
        return [0.1] * 384

    manager.create_embeddings = Mock(side_effect=_embed_docs)
    manager.embed_query = Mock(side_effect=_embed_query)
    manager.model = Mock()
    return manager


@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    """Create a mock Chroma client."""
    client = Mock()
    client.get_or_create_collection = Mock(return_value=mock_chroma_collection)
    return client


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Cleanup temporary files after each test."""
    yield
    # Cleanup code here if needed


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client for API tests."""
    from fastapi.testclient import TestClient
    return TestClient
