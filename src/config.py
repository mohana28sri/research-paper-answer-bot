"""
Configuration management for the Research Paper Answer Bot.

This module loads and validates environment variables from a .env file
and provides a centralized configuration object for the entire application.
All application constants are defined here for easy access and modification.

Usage:
    from src.config import Config
    config = Config()
    api_key = config.openai_api_key
    embedding_model = config.embedding_model
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """
    Raised when required configuration variables are missing or invalid.
    """
    pass


class Config:
    """
    Centralized configuration manager for the Research Paper Answer Bot.
    
    Loads environment variables from a .env file and provides validated
    access to all application configuration values. Ensures all required
    environment variables are present at initialization.
    
    Attributes:
        openai_api_key (str): OpenAI API key for GPT-4 Mini access.
        embedding_model (str): Name of the embedding model from Sentence Transformers.
        chat_model (str): OpenAI model identifier for chat completions.
        chroma_db_path (str): Filesystem path for ChromaDB vector store persistence.
        data_folder (str): Directory path containing research paper PDFs.
        chunk_size (int): Number of characters per document chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
        top_k_results (int): Number of top similar documents to retrieve.
    """

    def __init__(self, env_file: Optional[str] = None) -> None:
        """
        Initialize the configuration manager.
        
        Loads environment variables from a .env file (or the specified path)
        and validates that all required configuration variables are present.
        
        Args:
            env_file (Optional[str]): Path to the .env file. If None, looks for
                .env in the current working directory or parent directories.
        
        Raises:
            ConfigurationError: If any required environment variable is missing.
        """
        self._load_environment(env_file)
        self._validate_required_variables()

    @staticmethod
    def _load_environment(env_file: Optional[str] = None) -> None:
        """
        Load environment variables from a .env file.
        
        Attempts to load from a specified file, or searches for .env in
        common locations (current directory, project root).
        
        Args:
            env_file (Optional[str]): Explicit path to .env file.
        """
        if env_file:
            if not Path(env_file).exists():
                raise ConfigurationError(
                    f"Environment file not found: {env_file}"
                )
            load_dotenv(env_file)
        else:
            # Search for .env in current directory and parent directories
            env_path = Path(".env")
            if not env_path.exists():
                env_path = Path(__file__).parent.parent / ".env"
            
            if env_path.exists():
                load_dotenv(env_path)

    def _validate_required_variables(self) -> None:
        """
        Validate that all required environment variables are present.
        
        Checks:
        - LLM_PROVIDER is set to a supported value
        - NVIDIA_API_KEY is set (if provider is NVIDIA)
        - CHROMA_DB_PATH is configured
        
        Raises:
            ConfigurationError: If any required variable is missing or invalid.
        """
        provider = self.llm_provider.lower()

        if not provider:
            raise ConfigurationError(
                "LLM_PROVIDER is missing. Please set it in your .env file. "
                "Supported values: nvidia, openai, google, gemini, ollama"
            )

        supported_providers = {"nvidia", "openai", "google", "gemini", "ollama"}
        if provider not in supported_providers:
            raise ConfigurationError(
                f"LLM_PROVIDER '{provider}' is not supported. "
                f"Supported values: {', '.join(sorted(supported_providers))}"
            )

        if not os.getenv("CHROMA_DB_PATH"):
            raise ConfigurationError(
                "CHROMA_DB_PATH is missing. Please set it in your .env file."
            )

        if provider == "nvidia":
            if not os.getenv("NVIDIA_API_KEY"):
                raise ConfigurationError(
                    "NVIDIA_API_KEY is missing. Please set it in your .env file."
                )
        elif provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ConfigurationError(
                    "OPENAI_API_KEY is missing. Please set it in your .env file."
                )
        elif provider in {"google", "gemini"}:
            if not os.getenv("GOOGLE_API_KEY"):
                raise ConfigurationError(
                    "GOOGLE_API_KEY is missing. Please set it in your .env file."
                )
        elif provider == "groq":
            if not os.getenv("GROQ_API_KEY"):
                raise ConfigurationError(
                    "GROQ_API_KEY is missing. Please set it in your .env file."
                )

    @property
    def nvidia_api_key(self) -> str:
        """Retrieve the NVIDIA API key."""
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "NVIDIA_API_KEY is not set. Please add it to your .env file."
            )
        return api_key

    @property
    def openai_api_key(self) -> str:
        """
        Retrieve the OpenAI API key.
        
        Returns:
            str: The OpenAI API key for authentication.
        
        Raises:
            ConfigurationError: If OPENAI_API_KEY is not set.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Please add it to your .env file."
            )
        return api_key

    @property
    def google_api_key(self) -> str:
        """Retrieve the Google API key."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "GOOGLE_API_KEY is not set. Please add it to your .env file."
            )
        return api_key

    @property
    def groq_api_key(self) -> str:
        """Retrieve the Groq API key."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set. Please add it to your .env file."
            )
        return api_key

    @property
    def ollama_base_url(self) -> str:
        """Retrieve the Ollama server base URL."""
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def llm_provider(self) -> str:
        """Retrieve the configured LLM provider."""
        return os.getenv("LLM_PROVIDER", "nvidia").strip().lower()

    @property
    def embedding_model(self) -> str:
        """
        Retrieve the embedding model name.
        
        Returns:
            str: Sentence Transformers model identifier.
        """
        return os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    @property
    def chat_model(self) -> str:
        """
        Retrieve the chat model name.
        
        Returns:
            str: Chat model identifier for the selected provider.
        """
        return os.getenv("CHAT_MODEL", "meta/llama-3.1-8b-instruct")

    @property
    def chroma_db_path(self) -> str:
        """
        Retrieve the ChromaDB storage path.
        
        Returns:
            str: Filesystem path for vector database persistence.
        """
        return os.getenv("CHROMA_DB_PATH", "./chroma_db")

    @property
    def data_folder(self) -> str:
        """
        Retrieve the research papers data folder path.
        
        Returns:
            str: Filesystem path to the directory containing PDF files.
        """
        return os.getenv("DATA_FOLDER", "./data")

    @property
    def chunk_size(self) -> int:
        """
        Retrieve the document chunk size.
        
        Returns:
            int: Number of characters per chunk for document splitting.
        """
        try:
            return int(os.getenv("CHUNK_SIZE", "1000"))
        except ValueError:
            raise ConfigurationError(
                "CHUNK_SIZE must be a valid integer. "
                f"Got: {os.getenv('CHUNK_SIZE')}"
            )

    @property
    def chunk_overlap(self) -> int:
        """
        Retrieve the document chunk overlap.
        
        Returns:
            int: Number of overlapping characters between consecutive chunks.
        """
        try:
            return int(os.getenv("CHUNK_OVERLAP", "200"))
        except ValueError:
            raise ConfigurationError(
                "CHUNK_OVERLAP must be a valid integer. "
                f"Got: {os.getenv('CHUNK_OVERLAP')}"
            )

    @property
    def top_k_results(self) -> int:
        """
        Retrieve the number of top results to retrieve.
        
        Returns:
            int: Number of similar documents to return from retrieval.
        """
        try:
            return int(os.getenv("TOP_K_RESULTS", "3"))
        except ValueError:
            raise ConfigurationError(
                "TOP_K_RESULTS must be a valid integer. "
                f"Got: {os.getenv('TOP_K_RESULTS')}"
            )

    def to_dict(self) -> dict:
        """
        Convert configuration to a dictionary.
        
        Returns a dictionary of all configuration values for easy
        serialization or logging purposes.
        
        Returns:
            dict: Dictionary containing all configuration key-value pairs.
        """
        return {
            "openai_api_key": "***" + self.openai_api_key[-4:],  # Masked for security
            "google_api_key": "***" + self.google_api_key[-4:] if self.llm_provider == "google" else None,
            "groq_api_key": "***" + self.groq_api_key[-4:] if self.llm_provider == "groq" else None,
            "llm_provider": self.llm_provider,
            "embedding_model": self.embedding_model,
            "chat_model": self.chat_model,
            "chroma_db_path": self.chroma_db_path,
            "data_folder": self.data_folder,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k_results": self.top_k_results,
        }

    def __repr__(self) -> str:
        """
        Return a string representation of the configuration.
        
        Returns:
            str: Readable representation of configuration (API key masked).
        """
        config_dict = self.to_dict()
        return f"Config({config_dict})"
