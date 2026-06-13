"""LLM Factory - Creates LLM instances based on configuration."""

import logging
from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel

from config import Config

from .llm_provider_azure import AzureLLMProvider
from .llm_provider_lmstudio import LMStudioLLMProvider
from .llm_provider_ollama import OllamaLLMProvider
from .llm_provider_openai import OpenAILLMProvider

logger = logging.getLogger(__name__)


# MARK: LLM Factory
class LLMFactory:
    """Factory for creating LLM instances based on provider configuration."""

    SUPPORTED_PROVIDERS: tuple[str, ...] = ("azure", "lmstudio", "openai", "ollama")

    @staticmethod
    def create_llm() -> BaseChatModel:
        """
        Create an LLM instance based on configuration.

        Returns:
            LLM instance (AzureChatOpenAI or ChatOpenAI)
        """
        # Load configuration
        config_obj = Config()
        config = config_obj.get_all() or {}

        # Determine provider from config
        selected_provider = cast(str, config.get("provider", "openai"))
        logger.info(f"Creating LLM with provider: {selected_provider}")

        # Get provider-specific config
        provider_dict = cast(dict[str, object], config.get(selected_provider, {}))
        provider_config = provider_dict.copy()
        common_params = cast(dict[str, object], config.get("common_params", {}))

        # Merge common params with provider params
        provider_config.update(common_params)

        # Remove None values
        provider_config = {k: v for k, v in provider_config.items() if v is not None}

        # Create the appropriate LLM
        if selected_provider == "azure":
            return AzureLLMProvider.create(provider_config)
        elif selected_provider == "lmstudio":
            return LMStudioLLMProvider.create(provider_config)
        elif selected_provider == "ollama":
            return OllamaLLMProvider.create(provider_config)
        elif selected_provider == "openai":
            return OpenAILLMProvider.create(provider_config)
        else:
            supported = ", ".join(LLMFactory.SUPPORTED_PROVIDERS)
            raise ValueError(
                f"Unsupported provider: {selected_provider}. Supported providers: {supported}"
            )
