"""Azure OpenAI LLM Provider."""

import os
import logging
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)


class AzureLLMProvider:
    """Provider for Azure OpenAI LLM instances."""

    @staticmethod
    def create(config: dict) -> AzureChatOpenAI:
        """
        Create an Azure OpenAI LLM instance.

        Args:
            config: Configuration dictionary

        Returns:
            AzureChatOpenAI instance

        Environment Variables:
            - AZURE_OPENAI_API_KEY (required)
            - AZURE_OPENAI_ENDPOINT (required)
            - AZURE_OPENAI_API_VERSION (optional, default: 2024-02-15-preview)
            - AZURE_OPENAI_DEPLOYMENT_NAME (required if not in config)
        """
        # Get API key and endpoint from environment if not in config
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        # Get deployment name (REQUIRED for Azure)
        deployment_name = (
            config.pop("deployment_name", None)
            or config.pop("azure_deployment", None)
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        )

        # Remove 'model' from config as Azure uses deployment_name instead
        config.pop("model", None)

        # Validate required parameters
        if not api_key:
            logger.warning("AZURE_OPENAI_API_KEY not found in environment or config")

        if not azure_endpoint:
            logger.warning("AZURE_OPENAI_ENDPOINT not found in environment or config")

        if not deployment_name:
            logger.error("AZURE_OPENAI_DEPLOYMENT_NAME is required but not found!")
            raise ValueError(
                "Azure OpenAI requires deployment_name. "
                "Set it in config or AZURE_OPENAI_DEPLOYMENT_NAME environment variable"
            )

        # Build parameters
        params = {
            "api_key": api_key,
            "azure_endpoint": azure_endpoint,
            "api_version": api_version,
            "deployment_name": deployment_name,
            **config,
        }

        logger.info(f"Creating AzureChatOpenAI with deployment: {deployment_name}")
        logger.debug(f"Azure parameters: {list(params.keys())}")

        return AzureChatOpenAI(**params)
