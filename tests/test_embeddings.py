"""
Tests for embedding generation functionality.
"""

from unittest.mock import Mock, patch, MagicMock

import pytest

from src.embeddings import EmbeddingManager


class TestEmbeddingManager:
    """Test suite for EmbeddingManager."""

    def test_embedding_manager_initialization(self, mock_config):
        """Test that EmbeddingManager initializes correctly."""
        with patch("src.embeddings.HuggingFaceEmbeddings"):
            manager = EmbeddingManager(mock_config)
            assert manager.config == mock_config

    def test_embedding_manager_loads_embeddings(self, mock_config):
        """Test that EmbeddingManager loads the embeddings model."""
        with patch("src.embeddings.HuggingFaceEmbeddings") as mock_hf:
            mock_instance = Mock()
            mock_hf.return_value = mock_instance

            manager = EmbeddingManager(mock_config)
            assert manager.embeddings is not None

    def test_embed_text_single_document(self, mock_config, mock_embeddings):
        """Test embedding a single text document."""
        with patch("src.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings):
            manager = EmbeddingManager(mock_config)

            text = "Machine learning is a subset of artificial intelligence."
            embedding = manager.embed_text(text)

            assert isinstance(embedding, list)
            assert len(embedding) == 384  # MiniLM produces 384-dim embeddings
            assert all(isinstance(x, float) for x in embedding)

    def test_embed_documents_batch(self, mock_config, mock_embeddings):
        """Test embedding multiple documents."""
        with patch("src.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings):
            manager = EmbeddingManager(mock_config)

            documents = [
                "First document about machine learning",
                "Second document about deep learning",
                "Third document about neural networks",
            ]

            embeddings = manager.embed_documents(documents)

            assert len(embeddings) == len(documents)
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(len(emb) == 384 for emb in embeddings)

    def test_embed_query(self, mock_config, mock_embeddings):
        """Test embedding a query."""
        with patch("src.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings):
            manager = EmbeddingManager(mock_config)

            query = "What is machine learning?"
            embedding = manager.embed_query(query)

            assert isinstance(embedding, list)
            assert len(embedding) == 384

    def test_embed_empty_text(self, mock_config):
        """Test handling of empty text."""
        with patch("src.embeddings.HuggingFaceEmbeddings") as mock_hf:
            mock_instance = Mock()
            mock_instance.embed_documents.return_value = [[0.0] * 384]
            mock_hf.return_value = mock_instance

            manager = EmbeddingManager(mock_config)
            embedding = manager.embed_documents([""])

            assert isinstance(embedding, list)

    def test_embed_very_long_text(self, mock_config, mock_embeddings):
        """Test embedding very long text."""
        with patch("src.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings):
            manager = EmbeddingManager(mock_config)

            long_text = "word " * 10000  # Very long text
            embedding = manager.embed_text(long_text)

            assert isinstance(embedding, list)
            assert len(embedding) == 384

    def test_embed_special_characters(self, mock_config, mock_embeddings):
        """Test embedding text with special characters."""
        with patch("src.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings):
            manager = EmbeddingManager(mock_config)

            special_text = "Text with @#$%^&*() special chars! 测试中文 العربية"
            embedding = manager.embed_text(special_text)

            assert isinstance(embedding, list)
            assert len(embedding) == 384

    def test_embed_multilingual_text(self, mock_config, mock_embeddings):
        """Test embedding text in multiple languages."""
        with patch("src.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings):
            manager = EmbeddingManager(mock_config)

            texts = [
                "English text about AI",
                "Texte français sur l'IA",
                "中文文本关于人工智能",
            ]

            embeddings = manager.embed_documents(texts)

            assert len(embeddings) == 3
            assert all(len(emb) == 384 for emb in embeddings)

    def test_embedding_consistency(self, mock_config, mock_embeddings):
        """Test that same text produces same embedding."""
        with patch("src.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings):
            manager = EmbeddingManager(mock_config)

            text = "Consistent text for testing"
            embedding1 = manager.embed_text(text)
            embedding2 = manager.embed_text(text)

            assert embedding1 == embedding2

    def test_embedding_different_for_different_texts(self, mock_config, mock_embeddings):
        """Test that different texts produce different embeddings."""
        with patch("src.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings):
            manager = EmbeddingManager(mock_config)

            text1 = "First text for comparison"
            text2 = "Completely different text"

            embedding1 = manager.embed_text(text1)
            embedding2 = manager.embed_text(text2)

            assert embedding1 != embedding2

    def test_embedding_output_format(self, mock_config, mock_embeddings):
        """Test that embeddings have correct output format."""
        with patch("src.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings):
            manager = EmbeddingManager(mock_config)

            embedding = manager.embed_text("Test text")

            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding)

    def test_embedding_manager_error_handling(self, mock_config):
        """Test error handling in EmbeddingManager."""
        with patch("src.embeddings.HuggingFaceEmbeddings") as mock_hf:
            mock_hf.side_effect = Exception("Model loading failed")

            with pytest.raises(Exception):
                EmbeddingManager(mock_config)

    def test_embed_documents_normalization(self, mock_config, mock_embeddings):
        """Test that embeddings are normalized properly."""
        with patch("src.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings):
            manager = EmbeddingManager(mock_config)

            documents = ["Doc 1", "Doc 2", "Doc 3"]
            embeddings = manager.embed_documents(documents)

            # Check format: should be list of lists
            assert isinstance(embeddings, list)
            assert all(isinstance(e, list) for e in embeddings)
            assert all(len(e) == 384 for e in embeddings)
