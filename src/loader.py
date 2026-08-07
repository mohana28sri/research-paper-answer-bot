"""
Document loader module for the Research Paper Answer Bot.

This module handles loading PDF research papers from the configured data folder,
extracting text and metadata from each page, and returning LangChain Document
objects with preserved source information.

Usage:
    from src.config import Config
    from src.loader import DocumentLoader
    
    config = Config()
    loader = DocumentLoader(config)
    documents = loader.load_all_documents()
    stats = loader.get_statistics()
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from src.config import Config, ConfigurationError


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class DocumentLoadError(Exception):
    """
    Raised when a document cannot be loaded or processed.
    """
    pass


class DocumentLoader:
    """
    Loads PDF research papers and extracts text with preserved metadata.
    
    This class handles loading multiple PDF files from a configured directory,
    extracting text and metadata from each page, and returning LangChain
    Document objects. It provides statistics tracking and error handling for
    corrupted or missing files.
    
    Attributes:
        config (Config): Application configuration object.
        _documents (List[Document]): Cached list of loaded documents.
        _statistics (Dict[str, Any]): Statistics about loaded documents.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the DocumentLoader.
        
        Args:
            config (Config): Configuration object containing data folder path.
        
        Raises:
            ConfigurationError: If data folder path is not configured.
        """
        self.config = config
        self._documents: List[Document] = []
        self._statistics: Dict[str, Any] = {
            "total_pdfs": 0,
            "total_pages": 0,
            "successfully_loaded": 0,
            "failed_files": [],
        }
        logger.info("DocumentLoader initialized")

    def load_all_documents(self) -> List[Document]:
        """
        Load all PDF documents from the configured data folder.
        
        Iterates through all PDF files in the data folder, loads them,
        and preserves metadata for each page. Handles errors gracefully
        and continues loading remaining files if one fails.
        
        Returns:
            List[Document]: List of LangChain Document objects with metadata.
        
        Raises:
            DocumentLoadError: If data folder doesn't exist or is empty.
        """
        data_folder = Path(self.config.data_folder)
        
        # Validate data folder
        if not data_folder.exists():
            error_msg = f"Data folder not found: {data_folder}"
            logger.error(error_msg)
            raise DocumentLoadError(error_msg)
        
        # Find all PDF files
        pdf_files = list(data_folder.glob("*.pdf"))
        
        if not pdf_files:
            error_msg = f"No PDF files found in: {data_folder}"
            logger.warning(error_msg)
            raise DocumentLoadError(error_msg)
        
        self._statistics["total_pdfs"] = len(pdf_files)
        logger.info(f"Found {len(pdf_files)} PDF file(s) in {data_folder}")
        
        # Load each PDF file
        for pdf_path in pdf_files:
            self._load_single_pdf(pdf_path)
        
        # Log final statistics
        logger.info(
            f"Loading complete: {self._statistics['successfully_loaded']} "
            f"PDFs loaded, {self._statistics['total_pages']} total pages"
        )
        
        if self._statistics["failed_files"]:
            logger.warning(
                f"Failed to load {len(self._statistics['failed_files'])} file(s): "
                f"{', '.join(self._statistics['failed_files'])}"
            )
        
        return self._documents

    def load_single_document(self, pdf_path: str) -> List[Document]:
        """
        Load a single PDF document by file path.
        
        Args:
            pdf_path (str): Full or relative path to the PDF file.
        
        Returns:
            List[Document]: List of Document objects for the PDF's pages.
        
        Raises:
            DocumentLoadError: If the file doesn't exist or cannot be loaded.
        """
        pdf_path_obj = Path(pdf_path)
        
        if not pdf_path_obj.exists():
            error_msg = f"PDF file not found: {pdf_path}"
            logger.error(error_msg)
            raise DocumentLoadError(error_msg)
        
        if pdf_path_obj.suffix.lower() != ".pdf":
            error_msg = f"File is not a PDF: {pdf_path}"
            logger.error(error_msg)
            raise DocumentLoadError(error_msg)
        
        logger.info(f"Loading document: {pdf_path_obj.name}")
        
        try:
            loader = PyPDFLoader(str(pdf_path_obj))
            documents = loader.load()
            
            # Enrich metadata for each page
            paper_title = pdf_path_obj.stem
            enriched_docs = []
            
            for i, doc in enumerate(documents):
                doc.metadata["source"] = str(pdf_path_obj)
                doc.metadata["paper_title"] = paper_title
                doc.metadata["page_number"] = doc.metadata.get("page", i) + 1
                enriched_docs.append(doc)
            
            logger.info(
                f"Successfully loaded {len(enriched_docs)} page(s) from {pdf_path_obj.name}"
            )
            
            return enriched_docs
        
        except Exception as e:
            error_msg = f"Failed to load {pdf_path_obj.name}: {str(e)}"
            logger.error(error_msg)
            raise DocumentLoadError(error_msg) from e

    def _load_single_pdf(self, pdf_path: Path) -> None:
        """
        Internal method to load a single PDF and update statistics.
        
        Loads a PDF file, adds documents to the internal cache,
        and updates loading statistics. Errors are logged but do not
        prevent loading of other files.
        
        Args:
            pdf_path (Path): Path object pointing to the PDF file.
        """
        try:
            documents = self.load_single_document(str(pdf_path))
            self._documents.extend(documents)
            self._statistics["successfully_loaded"] += 1
            self._statistics["total_pages"] += len(documents)
        
        except DocumentLoadError as e:
            logger.error(f"Error loading {pdf_path.name}: {str(e)}")
            self._statistics["failed_files"].append(pdf_path.name)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve statistics about loaded documents.
        
        Returns a dictionary containing counts of total PDFs found,
        successfully loaded PDFs, total pages, and any files that failed
        to load.
        
        Returns:
            Dict[str, Any]: Dictionary with the following keys:
                - total_pdfs: Total number of PDF files found
                - successfully_loaded: Number of PDFs successfully loaded
                - total_pages: Total number of pages extracted
                - failed_files: List of PDF filenames that failed to load
        """
        return self._statistics.copy()

    def reset(self) -> None:
        """
        Clear cached documents and reset statistics.
        
        Useful for reloading documents or starting fresh.
        """
        self._documents = []
        self._statistics = {
            "total_pdfs": 0,
            "total_pages": 0,
            "successfully_loaded": 0,
            "failed_files": [],
        }
        logger.info("DocumentLoader reset")

    def __len__(self) -> int:
        """
        Return the number of documents currently loaded.
        
        Returns:
            int: Number of Document objects in cache.
        """
        return len(self._documents)

    def __repr__(self) -> str:
        """
        Return string representation of the DocumentLoader.
        
        Returns:
            str: Readable representation including document count.
        """
        return (
            f"DocumentLoader(documents={len(self._documents)}, "
            f"pages={self._statistics['total_pages']})"
        )
