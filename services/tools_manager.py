"""Tools Manager - Handles tool registration and configuration."""

import logging
from langchain_core.language_models.chat_models import BaseChatModel

from tools import (
    calculator,
    weather,
    read_file,
    write_file,
    http_get,
    list_dir,
    system_info,
    random_joke,
    current_date,
    loan_calculator,
    currency_converter,
    joke_format,
)

logger = logging.getLogger(__name__)


class ToolsManager:
    """Manages tool registration and configuration."""

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the tools manager.

        Args:
            llm: Language model for toolkit
        """
        self.llm = llm
        logger.debug("ToolsManager initializing")
        self.tools = self._register_tools()
        logger.info(f"ToolsManager initialized with {len(self.tools)} tools")

    def _register_tools(self) -> list:
        """Register and configure all tools."""
        logger.debug("Registering tools...")

        tools_list = [
            calculator,
            weather,
            read_file,
            write_file,
            http_get,
            list_dir,
            system_info,
            random_joke,
            current_date,
            loan_calculator,
            currency_converter,
            joke_format,
        ]
        logger.info(f"Registered {len(tools_list)} utility tools")
        return tools_list

    def get_tools(self) -> list:
        """Get all registered tools."""
        return self.tools
