"""Tools Manager - Handles tool registration and configuration."""

import logging
from typing import List, Optional

from langchain_core.tools import BaseTool

from tools import (
    calculator,
    currency_converter,
    current_date,
    http_get,
    joke_format,
    loan_calculator,
    random_joke,
    read_file,
    weather,
    write_file,
)

logger = logging.getLogger(__name__)


class ToolsManager:
    """Manages tool registration and configuration."""

    _tools: Optional[List[BaseTool]] = None

    @classmethod
    def get_tools(cls, tool_names: Optional[List[str]] = None) -> List[BaseTool]:
        """Get registered tools, optionally filtered by name."""
        if cls._tools is None:
            cls._tools = cls._register_tools()

        if tool_names:
            return [t for t in cls._tools if t.name in tool_names]
        return cls._tools

    @classmethod
    def _register_tools(cls) -> List[BaseTool]:
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

    @classmethod
    def reload_tools(cls) -> List[BaseTool]:
        """Force reload of all tools."""
        cls._tools = None
        return cls.get_tools()
