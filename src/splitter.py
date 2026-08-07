"""
Document splitter module for the Research Paper Answer Bot.

This module handles splitting long documents into smaller chunks while
preserving metadata using LangChain RecursiveCharacterTextSplitter.
"""

import logging
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

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


class DocumentSplitError(Exception):
    """Raised when document splitting fails."""


class DocumentSplitter:
    """
    Splits documents into smaller chunks while preserving metadata.
    """

    def __init__(
        self,
        config: Config = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> None:
        """
        Initialize DocumentSplitter.

        Supports:

        DocumentSplitter(config)

        or

        DocumentSplitter(chunk_size=500, chunk_overlap=50)
        """

        self.config = config

        if config is not None:
            self.chunk_size = config.chunk_size
            self.chunk_overlap = config.chunk_overlap
        else:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        self._statistics: Dict[str, Any] = {
            "total_input_documents": 0,
            "total_output_chunks": 0,
            "average_chunk_size": 0.0,
            "chunk_size_used": self.chunk_size,
            "chunk_overlap_used": self.chunk_overlap,
        }

        logger.info(
            "DocumentSplitter initialized: chunk_size=%s overlap=%s",
            self.chunk_size,
            self.chunk_overlap
        )

    @staticmethod
    def _to_document(item: Any) -> Document:
        """
        Convert a dict or pass through an existing Document.

        Args:
            item: A LangChain Document or a dict with 'page_content' and
                  optional 'metadata' keys.

        Returns:
            A LangChain Document.

        Raises:
            DocumentSplitError: If item is neither a Document nor a valid dict.
        """
        if isinstance(item, Document):
            return item
        if isinstance(item, dict):
            if "page_content" not in item:
                raise DocumentSplitError(
                    "Dict items must contain a 'page_content' key"
                )
            logger.info("Converted dictionary input to LangChain Document")
            return Document(
                page_content=item["page_content"],
                metadata=dict(item.get("metadata") or {}),
            )
        raise DocumentSplitError(
            f"Unsupported document type: {type(item).__name__}. "
            "Expected a LangChain Document or a dict with 'page_content'."
        )

    def split_documents(
        self,
        documents: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Split documents into chunks and preserve metadata.

        Accepts a mixed list of LangChain Document objects and/or dicts of
        the form ``{"page_content": str, "metadata": dict}``.  Dicts are
        converted to Document objects internally before splitting.

        Args:
            documents: List of LangChain Document objects and/or dicts.

        Returns:
            List of dicts with 'page_content' and 'metadata' keys.

        Raises:
            DocumentSplitError: If input is invalid or splitting fails.
        """

        if not documents:
            raise DocumentSplitError(
                "Cannot split empty document list"
            )

        # Normalise: convert any dicts to Document objects
        normalised: List[Document] = [self._to_document(d) for d in documents]

        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    " ",
                    ""
                ]
            )

            chunks: List[Document] = splitter.split_documents(normalised)

            # Add unique chunk id
            for index, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = index

            # Update statistics
            self._statistics["total_input_documents"] = len(documents)
            self._statistics["total_output_chunks"] = len(chunks)

            if chunks:
                total_size = sum(
                    len(chunk.page_content)
                    for chunk in chunks
                )

                self._statistics["average_chunk_size"] = (
                    total_size / len(chunks)
                )

            logger.info(
                "Document split completed: %s documents -> %s chunks",
                len(documents),
                len(chunks)
            )

            # Return as dicts so callers can use either Document or dict access
            return [
                {"page_content": c.page_content, "metadata": c.metadata}
                for c in chunks
            ]

        except DocumentSplitError:
            raise
        except Exception as exc:
            raise DocumentSplitError(
                f"Failed to split documents: {exc}"
            ) from exc

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return splitting statistics.
        """

        return self._statistics.copy()

    def reset(self) -> None:
        """
        Reset statistics.
        """

        self._statistics = {
            "total_input_documents": 0,
            "total_output_chunks": 0,
            "average_chunk_size": 0.0,
            "chunk_size_used": self.chunk_size,
            "chunk_overlap_used": self.chunk_overlap,
        }

        logger.info("DocumentSplitter statistics reset")

    def __repr__(self) -> str:
        return (
            f"DocumentSplitter("
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )