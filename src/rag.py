"""Retrieval-Augmented Generation pipeline for the Research Paper Answer Bot.

This module coordinates retrieval from ChromaDB, context formatting, and
answer generation using ChatOpenAI.
"""

import logging
import re
from typing import Any, Dict, List

from langchain_core.documents import Document

from src.config import Config
from src.llm_factory import LLMFactoryError, create_llm
from src.vectorstore import VectorStoreManager, VectorStoreError
from src.embeddings import EmbeddingManager


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class RAGPipelineError(Exception):
    """Raised when the RAG pipeline encounters an error."""


class RAGPipeline:
    """Coordinate retrieval and answer generation for research paper queries."""

    def __init__(self, config, vector_store_manager=None):
        """Initialize the pipeline with configuration and vector store access."""
        self.config = config

        if vector_store_manager is None:
            embedding_manager = EmbeddingManager(config)
            vector_store_manager = VectorStoreManager(config, embedding_manager)

        self.vector_store_manager = vector_store_manager
        self.llm = create_llm(self.config)

        logger.info(
            "RAG pipeline initialized with provider: %s and model: %s",
            self.config.llm_provider,
            self.config.chat_model,
        )

    @staticmethod
    def _to_document(item: Any) -> Document:
        """Normalise a dict or pass through an existing Document."""
        if isinstance(item, Document):
            return item
        if isinstance(item, dict):
            return Document(
                page_content=item.get("page_content", item.get("content", "")),
                metadata=dict(item.get("metadata") or {}),
            )
        return item

    def retrieve_context(self, query: str, k: int = 3) -> List[Any]:
        """Retrieve relevant document chunks for a user query."""
        if not query or not query.strip():
            raise RAGPipelineError("Query cannot be empty")

        if k <= 0:
            raise RAGPipelineError("k must be greater than zero")

        try:
            documents = self.vector_store_manager.similarity_search(query=query, k=k)
            logger.info("Retrieved %s document(s) for query", len(documents))
            return documents
        except VectorStoreError as exc:
            raise RAGPipelineError(f"Failed to retrieve context: {exc}") from exc

    def format_context(self, documents: List[Any]) -> str:
        """Format retrieved documents into a readable context string."""
        if not documents:
            return ""

        formatted_parts: List[str] = []
        for raw in documents:
            doc = self._to_document(raw)
            paper_title = doc.metadata.get("paper_title", "Unknown")
            page_number = doc.metadata.get("page_number", "Unknown")
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip()

            section = (
                f"Source: {paper_title}\n"
                f"Page: {page_number}\n"
                f"Source Path: {source}\n"
                f"Content: {content}"
            )
            formatted_parts.append(section)

        return "\n\n".join(formatted_parts)

    def _build_fallback_answer(self, query: str, documents: List[Any]) -> str:
        """Create a lightweight answer from retrieved context when the LLM is unavailable."""
        if not documents:
            return "I could not find this information in the provided research papers."

        query_terms = [term.lower() for term in re.findall(r"\b\w+\b", query) if len(term) > 2]
        best_sentence = ""
        best_score = -1
        paper_title = "Unknown"
        page_number = "Unknown"

        for raw in documents:
            doc = self._to_document(raw)
            paper_title = str(doc.metadata.get("paper_title", "Unknown"))
            page_number = str(doc.metadata.get("page_number", "Unknown"))
            content = doc.page_content.strip()
            sentences = re.split(r"(?<=[.!?])\s+", content)

            for sentence in sentences:
                if not sentence.strip():
                    continue
                score = sum(1 for term in query_terms if term in sentence.lower())
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()

        if not best_sentence:
            best_sentence = documents[0].page_content.strip()

        if not best_sentence:
            return "I could not find this information in the provided research papers."

        return (
            f"Based on the available context from {paper_title} (page {page_number}), "
            f"{best_sentence}"
        )

    def generate_answer_from_context(self, query: str, documents: List[Any]) -> str:
        """Generate an answer from already retrieved documents."""
        if not query or not query.strip():
            raise RAGPipelineError("Query cannot be empty")

        if not documents:
            return "I could not find this information in the provided research papers."

        try:
            context = self.format_context(documents)
            if not context.strip():
                return "I could not find this information in the provided research papers."

            prompt = (
                "You are a research paper assistant.\n\n"
                "Answer questions only using the provided research paper context.\n"
                "Do not use outside knowledge.\n"
                "If information is missing, clearly say: "
                "'I could not find this information in the provided research papers.'\n\n"
                "Keep the answer concise and explain clearly.\n"
                "Always mention the paper title and page number used for the answer.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{query}\n\n"
                "Answer:"
            )

            if self.llm is None:
                return self._build_fallback_answer(query, documents)

            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            logger.info("Generated answer for query")
            return answer.strip()
        except Exception as exc:
            logger.warning("LLM generation failed, using context-based fallback: %s", exc)
            return self._build_fallback_answer(query, documents)

    def generate_answer(self, query: str) -> str:
        """Generate an answer by retrieving context and then answering."""
        if not query or not query.strip():
            raise RAGPipelineError("Query cannot be empty")

        try:
            documents = self.retrieve_context(query=query, k=self.config.top_k_results)
            return self.generate_answer_from_context(query=query, documents=documents)
        except VectorStoreError as exc:
            raise RAGPipelineError(f"Vector database failure: {exc}") from exc
        except Exception as exc:
            raise RAGPipelineError(f"Failed to generate answer: {exc}") from exc

    def ask(self, query: str) -> Dict[str, Any]:
        """Return the answer, sources, and retrieved chunk count for a query."""
        if not query or not query.strip():
            raise RAGPipelineError("Query cannot be empty")

        try:
            documents = self.retrieve_context(query=query, k=self.config.top_k_results)
            answer = self.generate_answer_from_context(query=query, documents=documents)

            sources = []
            for raw in documents:
                doc = self._to_document(raw)
                sources.append(
                    {
                        "paper_title": doc.metadata.get("paper_title", "Unknown"),
                        "page_number": doc.metadata.get("page_number", "Unknown"),
                        "source": doc.metadata.get("source", "Unknown"),
                    }
                )

            return {
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": len(documents),
            }
        except VectorStoreError as exc:
            raise RAGPipelineError(f"Vector database failure: {exc}") from exc
        except Exception as exc:
            raise RAGPipelineError(f"Failed to process query: {exc}") from exc
