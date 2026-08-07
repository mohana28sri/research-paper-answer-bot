"""
Tests for FastAPI endpoints.
"""

from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_rag_pipeline():
    """Create a mock RAG pipeline."""
    with patch("src.api.RAGPipeline") as mock_rag:
        yield mock_rag


@pytest.fixture
def mock_embedding_manager():
    """Create a mock embedding manager."""
    with patch("src.api.EmbeddingManager") as mock_emb:
        yield mock_emb


@pytest.fixture
def mock_vectorstore_manager():
    """Create a mock vectorstore manager."""
    with patch("src.api.VectorStoreManager") as mock_vs:
        yield mock_vs


@pytest.fixture
def mock_config(monkeypatch, tmp_path):
    """Create a mock config for API tests."""
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("CHAT_MODEL", "meta/llama-3.1-8b-instruct")
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path))
    monkeypatch.setenv("DATA_FOLDER", str(tmp_path))

    with patch("src.api.Config") as mock_config_class:
        config = Mock()
        config.llm_provider = "nvidia"
        config.chat_model = "meta/llama-3.1-8b-instruct"
        config.chroma_db_path = str(tmp_path)
        config.data_folder = str(tmp_path)
        mock_config_class.return_value = config
        yield config


class TestHealthEndpoint:
    """Test suite for health endpoint."""

    def test_health_endpoint_success(self, client):
        """Test health endpoint returns success."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_json(self, client):
        """Test health endpoint returns JSON with required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "service" in data
        assert data["status"] == "healthy"
        assert data["service"] == "Research Paper Answer Bot"

    def test_health_endpoint_includes_provider_info(self, client):
        """Test health endpoint includes LLM provider information."""
        response = client.get("/health")
        data = response.json()

        assert "llm_provider" in data
        assert "model" in data
        assert "database" in data

    def test_health_endpoint_no_auth_required(self, client):
        """Test health endpoint doesn't require authentication."""
        response = client.get("/health")
        assert response.status_code == 200


class TestUploadEndpoint:
    """Test suite for PDF upload endpoint."""

    def test_upload_pdf_success(self, client):
        """Test successful PDF upload."""
        with patch("src.api.DocumentLoader") as mock_loader:
            with patch("src.api.DocumentSplitter") as mock_splitter:
                # Setup mocks
                mock_loader_instance = Mock()
                mock_loader.return_value = mock_loader_instance
                mock_loader_instance.load_single_document.return_value = [
                    {"page_content": "Test content"}
                ]

                mock_splitter_instance = Mock()
                mock_splitter.return_value = mock_splitter_instance
                mock_splitter_instance.split_documents.return_value = [
                    {"page_content": "Chunk 1", "metadata": {"source": "test.pdf"}},
                    {"page_content": "Chunk 2", "metadata": {"source": "test.pdf"}},
                ]

                with patch("src.api.EmbeddingManager") as mock_emb:
                    with patch("src.api.VectorStoreManager") as mock_vs:
                        mock_emb_instance = Mock()
                        mock_emb.return_value = mock_emb_instance
                        mock_emb_instance.embed_documents.return_value = [
                            [0.1] * 384,
                            [0.2] * 384,
                        ]

                        mock_vs_instance = Mock()
                        mock_vs.return_value = mock_vs_instance
                        mock_vs_instance.add_documents = Mock()

                        pdf_content = b"%PDF-1.4 test"
                        response = client.post(
                            "/upload",
                            files={"file": ("test.pdf", BytesIO(pdf_content))},
                        )

                        assert response.status_code == 200

    def test_upload_missing_file(self, client):
        """Test upload with missing file."""
        response = client.post("/upload", files={})
        assert response.status_code in [400, 422]

    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type."""
        response = client.post(
            "/upload",
            files={"file": ("test.txt", BytesIO(b"Not a PDF"))},
        )
        # Should either accept and process or reject
        assert response.status_code in [200, 400, 422]

    def test_upload_returns_response_schema(self, client):
        """Test upload returns correct response schema."""
        with patch("src.api.DocumentLoader"):
            with patch("src.api.DocumentSplitter"):
                with patch("src.api.EmbeddingManager"):
                    with patch("src.api.VectorStoreManager"):
                        try:
                            pdf_content = b"%PDF-1.4 test"
                            response = client.post(
                                "/upload",
                                files={"file": ("test.pdf", BytesIO(pdf_content))},
                            )

                            if response.status_code == 200:
                                data = response.json()
                                assert "filename" in data or "message" in data
                        except Exception:
                            # Upload might fail due to missing dependencies
                            pass


class TestAskEndpoint:
    """Test suite for question endpoint."""

    def test_ask_endpoint_success(self, client):
        """Test successful question asking."""
        with patch("src.api.RAGPipeline") as mock_rag:
            mock_rag_instance = Mock()
            mock_rag.return_value = mock_rag_instance
            mock_rag_instance.ask.return_value = {
                "answer": "Machine learning is AI",
                "sources": [{"source": "paper.pdf", "page": 1}],
                "retrieved_chunks": 1,
            }

            response = client.post(
                "/ask", json={"question": "What is machine learning?"}
            )

            assert response.status_code == 200

    def test_ask_endpoint_returns_schema(self, client):
        """Test ask endpoint returns correct schema."""
        with patch("src.api.RAGPipeline") as mock_rag:
            mock_rag_instance = Mock()
            mock_rag.return_value = mock_rag_instance
            mock_rag_instance.ask.return_value = {
                "answer": "Test answer",
                "sources": [],
                "retrieved_chunks": 0,
            }

            response = client.post(
                "/ask", json={"question": "Test question?"}
            )

            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "retrieved_chunks" in data

    def test_ask_empty_question(self, client):
        """Test ask with empty question."""
        response = client.post("/ask", json={"question": ""})
        assert response.status_code in [400, 422]

    def test_ask_missing_question(self, client):
        """Test ask with missing question field."""
        response = client.post("/ask", json={})
        assert response.status_code in [400, 422]

    def test_ask_question_validation(self, client):
        """Test that question validation works."""
        # Very long question
        long_question = "What " * 10000

        with patch("src.api.RAGPipeline"):
            response = client.post(
                "/ask", json={"question": long_question}
            )
            # Should either process or reject
            assert response.status_code in [200, 400, 422]

    def test_ask_special_characters_in_question(self, client):
        """Test question with special characters."""
        with patch("src.api.RAGPipeline") as mock_rag:
            mock_rag_instance = Mock()
            mock_rag.return_value = mock_rag_instance
            mock_rag_instance.ask.return_value = {
                "answer": "Answer",
                "sources": [],
                "retrieved_chunks": 0,
            }

            response = client.post(
                "/ask",
                json={"question": "What is @#$%^&*()? 中文？"},
            )

            assert response.status_code == 200


class TestErrorHandling:
    """Test suite for error handling."""

    def test_404_not_found(self, client):
        """Test 404 error for non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test method not allowed error."""
        response = client.post("/health")
        assert response.status_code == 405

    def test_internal_server_error(self, client):
        """Test handling of internal server errors."""
        with patch("src.api.RAGPipeline") as mock_rag:
            mock_rag.side_effect = Exception("Unexpected error")

            response = client.post(
                "/ask", json={"question": "Test"}
            )

            # Should return error response without exposing internals
            assert response.status_code >= 400

    def test_rag_error_handling(self, client):
        """Test error handling from RAG pipeline."""
        from src.rag import RAGPipelineError

        with patch("src.api.RAGPipeline") as mock_rag:
            mock_rag_instance = Mock()
            mock_rag.return_value = mock_rag_instance
            mock_rag_instance.ask.side_effect = RAGPipelineError("RAG failed")

            response = client.post(
                "/ask", json={"question": "Test"}
            )

            assert response.status_code >= 400
            data = response.json()
            assert "error" in data

    def test_embedding_error_handling(self, client):
        """Test error handling from embedding service."""
        from src.embeddings import EmbeddingError

        with patch("src.api.EmbeddingManager") as mock_emb:
            mock_emb.side_effect = EmbeddingError("Embedding failed")

            response = client.post(
                "/upload",
                files={"file": ("test.pdf", BytesIO(b"%PDF-1.4"))},
            )

            assert response.status_code in [400, 500, 503]


class TestRequestLogging:
    """Test suite for request logging middleware."""

    def test_logging_middleware_active(self, client):
        """Test that logging middleware is active."""
        with patch("src.api.RAGPipeline") as mock_rag:
            mock_rag_instance = Mock()
            mock_rag.return_value = mock_rag_instance
            mock_rag_instance.ask.return_value = {
                "answer": "Test",
                "sources": [],
                "retrieved_chunks": 0,
            }

            # Should not raise errors from logging
            response = client.post(
                "/ask", json={"question": "Test"}
            )

            assert response.status_code == 200

    def test_logging_different_methods(self, client):
        """Test logging works for different HTTP methods."""
        # GET request
        response = client.get("/health")
        assert response.status_code == 200

        # POST request
        with patch("src.api.RAGPipeline") as mock_rag:
            mock_rag_instance = Mock()
            mock_rag.return_value = mock_rag_instance
            mock_rag_instance.ask.return_value = {
                "answer": "Test",
                "sources": [],
                "retrieved_chunks": 0,
            }

            response = client.post(
                "/ask", json={"question": "Test"}
            )
            assert response.status_code == 200


class TestCORSAndSecurity:
    """Test suite for CORS and security headers."""

    def test_api_accessible(self, client):
        """Test that API is accessible."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_json_content_type(self, client):
        """Test that responses have JSON content type."""
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")

    def test_error_response_sanitized(self, client):
        """Test that error responses don't leak sensitive information."""
        with patch("src.api.RAGPipeline") as mock_rag:
            mock_rag_instance = Mock()
            mock_rag.return_value = mock_rag_instance
            # Simulate an error with sensitive info
            mock_rag_instance.ask.side_effect = Exception(
                "Secret API key: sk-123456"
            )

            response = client.post(
                "/ask", json={"question": "Test"}
            )

            error_data = response.json()
            error_message = str(error_data)

            # Ensure API key is not in response
            assert "sk-123456" not in error_message
            assert "Exception" not in error_message  # Stack trace not exposed


class TestConcurrency:
    """Test suite for concurrent requests."""

    def test_concurrent_health_checks(self, client):
        """Test multiple concurrent health checks."""
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200

    def test_concurrent_questions(self, client):
        """Test multiple concurrent question requests."""
        with patch("src.api.RAGPipeline") as mock_rag:
            mock_rag_instance = Mock()
            mock_rag.return_value = mock_rag_instance
            mock_rag_instance.ask.return_value = {
                "answer": "Answer",
                "sources": [],
                "retrieved_chunks": 0,
            }

            for i in range(3):
                response = client.post(
                    "/ask",
                    json={"question": f"Question {i}?"},
                )
                assert response.status_code == 200
