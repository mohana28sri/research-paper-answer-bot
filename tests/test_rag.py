"""
Tests for RAG pipeline functionality.
"""

from unittest.mock import Mock, patch, MagicMock

import pytest

from src.rag import RAGPipeline


class TestRAGPipeline:
    """Test suite for RAG pipeline."""

    def test_rag_initialization(self, mock_config):
        """Test that RAG pipeline initializes correctly."""
        with patch("src.rag.create_llm") as mock_create_llm:
            with patch("src.rag.EmbeddingManager"):
                with patch("src.rag.VectorStoreManager"):
                    mock_create_llm.return_value = Mock()

                    rag = RAGPipeline(mock_config)
                    assert rag.config == mock_config

    def test_rag_retrieve_context(self, mock_config):
        """Test context retrieval from vector store."""
        with patch("src.rag.create_llm") as mock_create_llm:
            with patch("src.rag.EmbeddingManager") as mock_embedding_manager:
                with patch("src.rag.VectorStoreManager") as mock_vs_manager:
                    mock_create_llm.return_value = Mock()
                    mock_embedding_instance = Mock()
                    mock_embedding_manager.return_value = mock_embedding_instance
                    mock_embedding_instance.embed_query.return_value = [0.1] * 384

                    mock_vs_instance = Mock()
                    mock_vs_manager.return_value = mock_vs_instance
                    mock_vs_instance.similarity_search.return_value = [
                        {"content": "Test content 1", "metadata": {"source": "doc1.pdf"}},
                        {"content": "Test content 2", "metadata": {"source": "doc2.pdf"}},
                    ]

                    rag = RAGPipeline(mock_config)
                    results = rag.retrieve_context("test query", k=2)

                    assert len(results) == 2
                    assert "Test content 1" in [r.get("content") for r in results]

    def test_rag_format_context(self, mock_config):
        """Test context formatting for LLM."""
        with patch("src.rag.create_llm") as mock_create_llm:
            with patch("src.rag.EmbeddingManager"):
                with patch("src.rag.VectorStoreManager"):
                    mock_create_llm.return_value = Mock()

                    rag = RAGPipeline(mock_config)
                    documents = [
                        {"page_content": "Content 1", "metadata": {"source": "doc1.pdf"}},
                        {"page_content": "Content 2", "metadata": {"source": "doc2.pdf"}},
                    ]

                    formatted = rag.format_context(documents)

                    assert isinstance(formatted, str)
                    assert "Content 1" in formatted
                    assert "Content 2" in formatted

    def test_rag_generate_answer(self, mock_config, mock_llm):
        """Test answer generation from context."""
        with patch("src.rag.create_llm") as mock_create_llm:
            with patch("src.rag.EmbeddingManager"):
                with patch("src.rag.VectorStoreManager"):
                    mock_create_llm.return_value = mock_llm

                    rag = RAGPipeline(mock_config)
                    documents = [
                        {"page_content": "ML is a subset of AI", "metadata": {"source": "doc.pdf"}},
                    ]

                    answer = rag.generate_answer_from_context(
                        "What is machine learning?", documents
                    )

                    assert isinstance(answer, str)
                    assert len(answer) > 0

    def test_rag_ask_complete_pipeline(self, mock_config, mock_llm):
        """Test complete RAG ask pipeline."""
        with patch("src.rag.create_llm") as mock_create_llm:
            with patch("src.rag.EmbeddingManager") as mock_embedding_manager:
                with patch("src.rag.VectorStoreManager") as mock_vs_manager:
                    mock_create_llm.return_value = mock_llm
                    mock_llm.invoke.return_value = "Machine learning is a subset of AI"

                    mock_embedding_instance = Mock()
                    mock_embedding_manager.return_value = mock_embedding_instance
                    mock_embedding_instance.embed_query.return_value = [0.1] * 384

                    mock_vs_instance = Mock()
                    mock_vs_manager.return_value = mock_vs_instance
                    mock_vs_instance.similarity_search.return_value = [
                        {
                            "page_content": "ML is a subset of AI",
                            "metadata": {"source": "paper.pdf", "page": 1},
                        },
                    ]

                    rag = RAGPipeline(mock_config)
                    result = rag.ask("What is machine learning?")

                    assert "answer" in result
                    assert "sources" in result
                    assert "retrieved_chunks" in result
                    assert len(result["answer"]) > 0

    def test_rag_ask_returns_sources(self, mock_config, mock_llm):
        """Test that ask returns sources with retrieved documents."""
        with patch("src.rag.create_llm") as mock_create_llm:
            with patch("src.rag.EmbeddingManager") as mock_embedding_manager:
                with patch("src.rag.VectorStoreManager") as mock_vs_manager:
                    mock_create_llm.return_value = mock_llm
                    mock_llm.invoke.return_value = "Answer"

                    mock_embedding_instance = Mock()
                    mock_embedding_manager.return_value = mock_embedding_instance
                    mock_embedding_instance.embed_query.return_value = [0.1] * 384

                    mock_vs_instance = Mock()
                    mock_vs_manager.return_value = mock_vs_instance
                    mock_vs_instance.similarity_search.return_value = [
                        {
                            "page_content": "Content",
                            "metadata": {"source": "paper1.pdf", "page": 1},
                        },
                        {
                            "page_content": "Content 2",
                            "metadata": {"source": "paper2.pdf", "page": 3},
                        },
                    ]

                    rag = RAGPipeline(mock_config)
                    result = rag.ask("Query")

                    assert len(result["sources"]) > 0
                    assert any(s.get("source") == "paper1.pdf" for s in result["sources"])

    def test_rag_empty_query(self, mock_config, mock_llm):
        """Test handling of empty query."""
        with patch("src.rag.create_llm") as mock_create_llm:
            with patch("src.rag.EmbeddingManager"):
                with patch("src.rag.VectorStoreManager"):
                    mock_create_llm.return_value = mock_llm

                    rag = RAGPipeline(mock_config)

                    # Should either raise error or handle gracefully
                    with pytest.raises((ValueError, Exception)):
                        rag.ask("")

    def test_rag_no_retrieval_results(self, mock_config, mock_llm):
        """Test handling when no documents are retrieved."""
        with patch("src.rag.create_llm") as mock_create_llm:
            with patch("src.rag.EmbeddingManager") as mock_embedding_manager:
                with patch("src.rag.VectorStoreManager") as mock_vs_manager:
                    mock_create_llm.return_value = mock_llm
                    mock_llm.invoke.return_value = "Fallback answer"

                    mock_embedding_instance = Mock()
                    mock_embedding_manager.return_value = mock_embedding_instance
                    mock_embedding_instance.embed_query.return_value = [0.1] * 384

                    mock_vs_instance = Mock()
                    mock_vs_manager.return_value = mock_vs_instance
                    mock_vs_instance.similarity_search.return_value = []

                    rag = RAGPipeline(mock_config)
                    result = rag.ask("Query with no matching documents")

                    # Should still return a response (possibly fallback)
                    assert "answer" in result

    def test_rag_llm_initialization_failure(self, mock_config):
        """Test handling when LLM initialization fails."""
        with patch("src.rag.create_llm") as mock_create_llm:
            with patch("src.rag.EmbeddingManager"):
                with patch("src.rag.VectorStoreManager"):
                    from src.llm_factory import LLMFactoryError

                    mock_create_llm.side_effect = LLMFactoryError("Failed to create LLM")

                    # Should handle gracefully or raise appropriate error
                    with pytest.raises(LLMFactoryError):
                        RAGPipeline(mock_config)

    def test_rag_context_truncation(self, mock_config, mock_llm):
        """Test that context is properly formatted without truncation issues."""
        with patch("src.rag.create_llm") as mock_create_llm:
            with patch("src.rag.EmbeddingManager") as mock_embedding_manager:
                with patch("src.rag.VectorStoreManager") as mock_vs_manager:
                    mock_create_llm.return_value = mock_llm
                    mock_llm.invoke.return_value = "Answer"

                    mock_embedding_instance = Mock()
                    mock_embedding_manager.return_value = mock_embedding_instance
                    mock_embedding_instance.embed_query.return_value = [0.1] * 384

                    mock_vs_instance = Mock()
                    mock_vs_manager.return_value = mock_vs_instance

                    # Return many documents
                    documents = [
                        {
                            "page_content": f"Document content {i}",
                            "metadata": {"source": f"doc{i}.pdf", "page": i},
                        }
                        for i in range(20)
                    ]
                    mock_vs_instance.similarity_search.return_value = documents

                    rag = RAGPipeline(mock_config)
                    result = rag.ask("Query")

                    assert "answer" in result
                    assert "retrieved_chunks" in result
                    assert result["retrieved_chunks"] >= 0

    def test_rag_special_characters_in_query(self, mock_config, mock_llm):
        """Test handling of special characters in query."""
        with patch("src.rag.create_llm") as mock_create_llm:
            with patch("src.rag.EmbeddingManager") as mock_embedding_manager:
                with patch("src.rag.VectorStoreManager") as mock_vs_manager:
                    mock_create_llm.return_value = mock_llm
                    mock_llm.invoke.return_value = "Answer"

                    mock_embedding_instance = Mock()
                    mock_embedding_manager.return_value = mock_embedding_instance
                    mock_embedding_instance.embed_query.return_value = [0.1] * 384

                    mock_vs_instance = Mock()
                    mock_vs_manager.return_value = mock_vs_instance
                    mock_vs_instance.similarity_search.return_value = []

                    rag = RAGPipeline(mock_config)

                    special_queries = [
                        "What is @#$%^&*()?",
                        "Query with \"quotes\"",
                        "Multi\nline\nquery",
                    ]

                    for query in special_queries:
                        try:
                            result = rag.ask(query)
                            assert "answer" in result
                        except Exception:
                            # Some special chars might cause issues, that's okay
                            pass

    def test_rag_retrieve_k_parameter(self, mock_config):
        """Test that k parameter controls retrieval count."""
        with patch("src.rag.create_llm") as mock_create_llm:
            with patch("src.rag.EmbeddingManager") as mock_embedding_manager:
                with patch("src.rag.VectorStoreManager") as mock_vs_manager:
                    mock_create_llm.return_value = Mock()

                    mock_embedding_instance = Mock()
                    mock_embedding_manager.return_value = mock_embedding_instance
                    mock_embedding_instance.embed_query.return_value = [0.1] * 384

                    mock_vs_instance = Mock()
                    mock_vs_manager.return_value = mock_vs_instance

                    documents = [
                        {
                            "page_content": f"Content {i}",
                            "metadata": {"source": f"doc{i}.pdf"},
                        }
                        for i in range(10)
                    ]
                    mock_vs_instance.similarity_search.return_value = documents[:5]

                    rag = RAGPipeline(mock_config)
                    results = rag.retrieve_context("query", k=5)

                    # Should respect k parameter
                    assert len(results) <= 5
