"""Factory module for creating configured LLM clients."""

import logging
from typing import Any, Callable, Dict

from langchain_openai import ChatOpenAI

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover - optional dependency or import-time failure
    ChatGoogleGenerativeAI = None

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - optional dependency
    ChatGroq = None

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


class LLMFactoryError(Exception):
    """Raised when the requested LLM provider is unsupported or misconfigured."""


class LLMFactory:
    """Factory for creating provider-specific LangChain chat models."""

    @staticmethod
    def create(config: Config) -> Any:
        """Create an LLM client for the configured provider."""
        provider = (getattr(config, "llm_provider", "nvidia") or "nvidia").strip().lower()
        registry: Dict[str, Callable[[Config], Any]] = {
            "nvidia": LLMFactory._create_nvidia,
            "openai": LLMFactory._create_openai,
            "google": LLMFactory._create_gemini,
            "gemini": LLMFactory._create_gemini,
            "ollama": LLMFactory._create_ollama,
        }

        factory = registry.get(provider)
        if factory is None:
            raise LLMFactoryError(
                f"Unsupported LLM provider '{provider}'. Use 'nvidia', 'openai', 'google', 'gemini', or 'ollama'."
            )

        return factory(config)

    @staticmethod
    def _create_nvidia(config: Config) -> Any:
        try:
            api_key = config.nvidia_api_key
        except Exception as exc:  # pragma: no cover - configuration missing
            raise LLMFactoryError(
                "NVIDIA_API_KEY is missing. Please set it in your environment or .env file."
            ) from exc

        logger.info("Initializing NVIDIA NIM LLM")
        return ChatOpenAI(
            model=config.chat_model,
            temperature=0.0,
            openai_api_key=api_key,
            openai_api_base="https://integrate.api.nvidia.com/v1",
        )

    @staticmethod
    def _create_openai(config: Config) -> Any:
        api_key = config.openai_api_key
        logger.info("Initializing OpenAI LLM")
        return ChatOpenAI(model=config.chat_model, temperature=0.0, api_key=api_key)

    @staticmethod
    def _create_gemini(config: Config) -> Any:
        if ChatGoogleGenerativeAI is None:
            raise LLMFactoryError(
                "langchain-google-genai is not installed. Install it to use the Gemini provider."
            )

        api_key = config.google_api_key
        logger.info("Initializing Google Gemini LLM")
        return ChatGoogleGenerativeAI(model=config.chat_model, google_api_key=api_key)

    @staticmethod
    def _create_ollama(config: Config) -> Any:
        logger.info("Initializing Ollama LLM")
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMFactoryError(
                "langchain-ollama is not installed. Install it to use the Ollama provider."
            ) from exc

        return ChatOllama(model=config.chat_model, base_url=config.ollama_base_url)


def create_llm(config: Config) -> Any:
    """Create an LLM client for the configured provider."""
    return LLMFactory.create(config)


def get_llm(config: Config) -> Any:
    """Backward-compatible alias for create_llm."""
    return create_llm(config)
