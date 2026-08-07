"""Embedding management for the Research Paper Answer Bot.

This module wraps LangChain Hugging Face embedding support and exposes a
simple interface for converting document chunks and user queries into vector
representations suitable for ChromaDB retrieval.
"""

import logging
from typing import List, Any, Dict

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Config


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class EmbeddingManager:
    """Manage embedding generation for document chunks and user queries."""

    def __init__(self, config: Config) -> None:
        """Initialize embedding manager."""

        self.config = config
        self.model_name = config.embedding_model

        self._model: HuggingFaceEmbeddings | None = None

        self._initialize_model()

        # Compatibility attribute expected by tests
        self.embeddings = self._model

    def _initialize_model(self) -> None:
        """Create and cache Hugging Face embedding model."""

        try:
            self._model = HuggingFaceEmbeddings(
                model_name=self.model_name
            )

            logger.info(
                "Embedding model loaded: %s",
                self.model_name
            )

        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc

    def create_embeddings(
        self,
        documents: List[Document]
    ) -> List[List[float]]:
        """Create embeddings for LangChain documents."""

        if not documents:
            raise EmbeddingError(
                "Cannot embed an empty document list"
            )

        if not all(
            isinstance(doc, Document)
            for doc in documents
        ):
            raise EmbeddingError(
                "All documents must be Document instances"
            )

        if self._model is None:
            raise EmbeddingError(
                "Embedding model is not initialized"
            )

        try:
            embeddings = self._model.embed_documents(
                [
                    doc.page_content
                    for doc in documents
                ]
            )

            logger.info(
                "Generated embeddings for %s documents",
                len(documents)
            )

            return embeddings

        except Exception as exc:
            raise EmbeddingError(
                f"Failed to generate embeddings: {exc}"
            ) from exc


    def embed_query(
        self,
        query: str
    ) -> List[float]:
        """Create embedding vector for a query."""

        if not query or not query.strip():
            raise EmbeddingError(
                "Query cannot be empty"
            )

        if self._model is None:
            raise EmbeddingError(
                "Embedding model is not initialized"
            )

        try:
            embedding = self._model.embed_query(query)

            logger.info(
                "Generated query embedding"
            )

            return embedding

        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed query: {exc}"
            ) from exc


    # -------------------------------
    # Compatibility methods for tests
    # -------------------------------

    def embed_text(
        self,
        text: str
    ) -> List[float]:
        """
        Alias for embed_query().
        Used for single text embedding.
        """

        return self.embed_query(text)


    def embed_documents(
        self,
        documents
    ) -> List[List[float]]:
        """
        Alias for document embedding.

        Supports:
        - List[Document]
        - List[str]
        """

        if not documents:
            raise EmbeddingError(
                "Cannot embed empty documents"
            )

        if isinstance(documents[0], Document):
            return self.create_embeddings(documents)

        if self._model is None:
            raise EmbeddingError(
                "Embedding model is not initialized"
            )

        try:
            return self._model.embed_documents(
                documents
            )

        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed documents: {exc}"
            ) from exc


    @property
    def model(self) -> HuggingFaceEmbeddings | None:
        """Return embedding model."""

        return self._model


    def get_model_info(self) -> Dict[str, Any]:
        """Return embedding model metadata."""

        return {
            "model_name": self.model_name,
            "provider": "huggingface",
            "class": self.__class__.__name__,
        }