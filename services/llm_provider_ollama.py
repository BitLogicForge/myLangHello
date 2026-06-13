"""Ollama LLM Provider."""

import logging
import os
from typing import cast

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# MARK: Ollama Provider
class OllamaLLMProvider:
    """Provider for Ollama LLM instances via OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL: str = "http://localhost:11434/v1"
    DEFAULT_API_KEY: str = "ollama"

    @staticmethod
    def create(config: dict[str, object]) -> ChatOpenAI:
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
        base_url = cast(str | None, config.pop("base_url", None) or os.getenv("OLLAMA_BASE_URL"))
        api_key = cast(str | None, config.pop("api_key", None) or os.getenv("OLLAMA_API_KEY"))

        params = {
            "base_url": base_url or OllamaLLMProvider.DEFAULT_BASE_URL,
            # Some OpenAI-compatible clients still expect a non-empty api_key field.
            "api_key": api_key or OllamaLLMProvider.DEFAULT_API_KEY,
            **config,
        }

        logger.info(f"Creating Ollama ChatOpenAI with model: {params.get('model', 'default')}")
        logger.debug(f"Ollama parameters: {list(params.keys())}")

        return ChatOpenAI(**params)  # pyright: ignore[reportArgumentType]
