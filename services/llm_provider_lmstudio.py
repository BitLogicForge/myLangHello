"""LM Studio LLM Provider."""

import logging
import os

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LMStudioLLMProvider:
    """Provider for LM Studio LLM instances."""

    DEFAULT_BASE_URL = "http://localhost:1234/v1"
    DEFAULT_API_KEY = "lm-studio"

    @staticmethod
    def create(config: dict) -> ChatOpenAI:
        """
        Create an LM Studio LLM instance.

        Args:
            config: Configuration dictionary

        Returns:
            ChatOpenAI instance configured for LM Studio

        Environment Variables:
            - LMSTUDIO_BASE_URL (optional)
            - LMSTUDIO_API_KEY (optional, usually not needed)
        """
        base_url = config.pop("base_url", None) or os.getenv("LMSTUDIO_BASE_URL")
        api_key = config.pop("api_key", None) or os.getenv("LMSTUDIO_API_KEY")

        params = {
            "base_url": base_url or LMStudioLLMProvider.DEFAULT_BASE_URL,
            # Some OpenAI-compatible clients still expect a non-empty api_key field.
            "api_key": api_key or LMStudioLLMProvider.DEFAULT_API_KEY,
            **config,
        }

        logger.info(f"Creating LM Studio ChatOpenAI with model: {params.get('model', 'default')}")
        logger.debug(f"LM Studio parameters: {list(params.keys())}")

        return ChatOpenAI(**params)
