"""LLM Factory - Creates LLM instances based on configuration."""

import os
import json
import logging
from typing import Optional, Any
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM instances based on provider configuration."""

    @staticmethod
    def load_config(config_path: str = "llm_config.json") -> dict:
        """
        Load LLM configuration from JSON file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded LLM configuration from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                "provider": "azure",
                "azure": {"model": "gpt-4", "temperature": 0.7},
                "openai": {"model": "gpt-4", "temperature": 0.7},
                "common_params": {},
            }
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {e}")
            raise

    @staticmethod
    def create_llm(
        config_path: str = "llm_config.json",
        provider: Optional[str] = None,
        **override_params: Any,
    ) -> BaseChatModel:
        """
        Create an LLM instance based on configuration.

        Args:
            config_path: Path to the configuration file
            provider: Override the provider from config (azure, openai)
            **override_params: Override any configuration parameters

        Returns:
            LLM instance (AzureChatOpenAI or ChatOpenAI)

        Environment Variables:
            For Azure:
                - AZURE_OPENAI_API_KEY (required)
                - AZURE_OPENAI_ENDPOINT (required)
                - AZURE_OPENAI_API_VERSION (optional)

            For OpenAI:
                - OPENAI_API_KEY (required)
                - OPENAI_ORGANIZATION (optional)
        """
        # Load configuration
        config = LLMFactory.load_config(config_path)

        # Determine provider
        selected_provider = provider or config.get("provider", "azure")
        logger.info(f"Creating LLM with provider: {selected_provider}")

        # Get provider-specific config
        provider_config = config.get(selected_provider, {}).copy()
        common_params = config.get("common_params", {})

        # Merge common params with provider params
        provider_config.update(common_params)

        # Apply overrides
        provider_config.update(override_params)

        # Remove None values
        provider_config = {k: v for k, v in provider_config.items() if v is not None}

        # Create the appropriate LLM
        if selected_provider == "azure":
            return LLMFactory._create_azure_llm(provider_config)
        elif selected_provider == "openai":
            return LLMFactory._create_openai_llm(provider_config)
        else:
            raise ValueError(f"Unsupported provider: {selected_provider}")

    @staticmethod
    def _create_azure_llm(config: dict) -> AzureChatOpenAI:
        """
        Create an Azure OpenAI LLM instance.

        Args:
            config: Configuration dictionary

        Returns:
            AzureChatOpenAI instance
        """
        # Get API key and endpoint from environment if not in config
        api_key = config.pop("api_key", None) or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = config.pop("azure_endpoint", None) or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = config.get("api_version", None) or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
        )

        if not api_key:
            logger.warning("AZURE_OPENAI_API_KEY not found in environment or config")

        if not azure_endpoint:
            logger.warning("AZURE_OPENAI_ENDPOINT not found in environment or config")

        # Build parameters
        params = {
            "api_key": api_key,
            "azure_endpoint": azure_endpoint,
            "api_version": api_version,
            **config,
        }

        logger.info(f"Creating AzureChatOpenAI with model: {params.get('model', 'default')}")
        logger.debug(f"Azure parameters: {list(params.keys())}")

        return AzureChatOpenAI(**params)

    @staticmethod
    def _create_openai_llm(config: dict) -> ChatOpenAI:
        """
        Create an OpenAI LLM instance.

        Args:
            config: Configuration dictionary

        Returns:
            ChatOpenAI instance
        """
        # Get API key and organization from environment if not in config
        api_key = config.pop("api_key", None) or os.getenv("OPENAI_API_KEY")
        organization = config.pop("organization", None) or os.getenv("OPENAI_ORGANIZATION")
        base_url = config.pop("base_url", None) or os.getenv("OPENAI_BASE_URL")

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

    @staticmethod
    def get_available_providers() -> list[str]:
        """
        Get list of available LLM providers.

        Returns:
            List of provider names
        """
        return ["azure", "openai"]
