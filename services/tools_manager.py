"""Tools Manager - Handles tool registration and configuration."""

import logging

from tools import (
    calculator,
    weather,
    read_file,
    write_file,
    http_get,
    random_joke,
    current_date,
    loan_calculator,
    currency_converter,
    joke_format,
)

logger = logging.getLogger(__name__)


class ToolsManager:
    """Manages tool registration and configuration."""

    _tools = None

    @classmethod
    def get_tools(cls) -> list:
        """Get all registered tools."""
        if cls._tools is None:
            cls._tools = cls._register_tools()
        return cls._tools

    @classmethod
    def _register_tools(cls) -> list:
        """Register and configure all tools."""
        logger.debug("Registering tools...")

        tools_list = [
            calculator,
            weather,
            read_file,
            write_file,
            http_get,
            random_joke,
            current_date,
            loan_calculator,
            currency_converter,
            joke_format,
        ]
        logger.info(f"Registered {len(tools_list)} utility tools")
        return tools_list
