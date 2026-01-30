"""OpenAI LLM Provider."""

import logging
import os

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class OpenAILLMProvider:
    """Provider for OpenAI LLM instances."""

    @staticmethod
    def create(config: dict) -> ChatOpenAI:
        """
        Create an OpenAI LLM instance.

        Args:
            config: Configuration dictionary

        Returns:
            ChatOpenAI instance

        Environment Variables:
            - OPENAI_API_KEY (required)
            - OPENAI_ORGANIZATION (optional)
            - OPENAI_BASE_URL (optional)
        """
        # Get API key and organization from environment if not in config
        api_key = config.pop("api_key", None) or os.getenv("OPENAI_API_KEY")
        organization = config.pop("organization", None) or os.getenv("OPENAI_ORGANIZATION")
        base_url = config.pop("base_url", None) or os.getenv("OPENAI_BASE_URL")

        # Validate required parameters
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment or config")

        # Build parameters
        params = {
            "api_key": api_key,
            **config,
        }

        if organization:
            params["organization"] = organization

        if base_url:
            params["base_url"] = base_url

        logger.info(f"Creating ChatOpenAI with model: {params.get('model', 'default')}")
        logger.debug(f"OpenAI parameters: {list(params.keys())}")

        return ChatOpenAI(**params)
