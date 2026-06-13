"""Ollama LLM Provider."""

import logging
import os

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class OllamaLLMProvider:
    """Provider for Ollama LLM instances via OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    DEFAULT_API_KEY = "ollama"

    @staticmethod
    def create(config: dict) -> ChatOpenAI:
        """
        Create an Ollama LLM instance.

        Args:
            config: Configuration dictionary

        Returns:
            ChatOpenAI instance configured for Ollama

        Environment Variables:
            - OLLAMA_BASE_URL (optional)
            - OLLAMA_API_KEY (optional)
        """
        base_url = config.pop("base_url", None) or os.getenv("OLLAMA_BASE_URL")
        api_key = config.pop("api_key", None) or os.getenv("OLLAMA_API_KEY")

        params = {
            "base_url": base_url or OllamaLLMProvider.DEFAULT_BASE_URL,
            # Some OpenAI-compatible clients still expect a non-empty api_key field.
            "api_key": api_key or OllamaLLMProvider.DEFAULT_API_KEY,
            **config,
        }

        logger.info(f"Creating Ollama ChatOpenAI with model: {params.get('model', 'default')}")
        logger.debug(f"Ollama parameters: {list(params.keys())}")

        return ChatOpenAI(**params)
