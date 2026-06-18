"""Tools Manager - Handles tool registration and configuration."""

import logging

from langchain_core.tools import BaseTool

from tools import (
    calculator,
    city_to_coordinates,
    currency_converter,
    current_date,
    financial_metrics_calculator,
    http_get,
    joke_format,
    loan_calculator,
    market_data,
    portfolio_add_stock,
    portfolio_analyze,
    portfolio_create,
    random_joke,
    read_file,
    risk_assessment,
    stock_comparison,
    stock_price_query,
    technical_analysis,
    weather,
    write_file,
)

logger = logging.getLogger(__name__)


# MARK: Tools Manager
class ToolsManager:
    """Manages tool registration and configuration."""

    _tools: list[BaseTool] | None = None

    @classmethod
    def get_tools(cls, tool_names: list[str] | None = None) -> list[BaseTool]:
        """Get registered tools, optionally filtered by name."""
        if cls._tools is None:
            cls._tools = cls._register_tools()

        if tool_names:
            return [t for t in cls._tools if t.name in tool_names]
        return cls._tools

    # MARK: Tool Registration
    @classmethod
    def _register_tools(cls) -> list[BaseTool]:
        """Register and configure all tools."""
        logger.debug("Registering tools...")

        tools_list = [
            # Stock Analysis Tools
            stock_price_query,
            financial_metrics_calculator,
            portfolio_create,
            portfolio_add_stock,
            portfolio_analyze,
            technical_analysis,
            risk_assessment,
            market_data,
            stock_comparison,

            # General Utility Tools
            calculator,
            current_date,
            http_get,
            read_file,
            write_file,

            # Demo Tools (kept for entertainment)
            weather,
            random_joke,
            joke_format,
            loan_calculator,
            currency_converter,
            city_to_coordinates,
        ]
        logger.info(f"Registered {len(tools_list)} tools (stock analysis + utilities)")
        return tools_list

    @classmethod
    def reload_tools(cls) -> list[BaseTool]:
        """Force reload of all tools."""
        cls._tools = None
        return cls.get_tools()
