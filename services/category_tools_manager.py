"""Advanced Tools Manager with Category-Based Tool Selection."""

from typing import Dict, List
from langchain_core.tools import BaseTool
from tools import (
    # Stock Data Tools
    stock_price_query,
    market_data,
    stock_comparison,

    # Analysis Tools
    financial_metrics_calculator,
    technical_analysis,
    risk_assessment,

    # Portfolio Tools
    portfolio_create,
    portfolio_add_stock,
    portfolio_analyze,

    # Utility Tools
    calculator,
    current_date,
    http_get,
]


class CategoryToolsManager:
    """Manager that organizes tools by explicit categories."""

    # Explicit category mapping
    TOOL_CATEGORIES: Dict[str, List[BaseTool]] = {
        "stock_data": [
            stock_price_query,
            market_data,
            stock_comparison,
        ],
        "analysis": [
            financial_metrics_calculator,
            technical_analysis,
            risk_assessment,
        ],
        "portfolio": [
            portfolio_create,
            portfolio_add_stock,
            portfolio_analyze,
        ],
        "utilities": [
            calculator,
            current_date,
            http_get,
        ],
    }

    @classmethod
    def get_tools_by_category(cls, categories: List[str]) -> List[BaseTool]:
        """
        Get tools filtered by specific categories.

        Usage:
        # For research sessions
        research_tools = get_tools_by_category(["stock_data", "analysis"])

        # For portfolio management
        portfolio_tools = get_tools_by_category(["portfolio", "analysis"])

        # For quick queries
        quick_tools = get_tools_by_category(["stock_data", "utilities"])
        """
        selected_tools = []
        for category in categories:
            if category in cls.TOOL_CATEGORIES:
                selected_tools.extend(cls.TOOL_CATEGORIES[category])

        return selected_tools

    @classmethod
    def get_category_summary(cls) -> Dict[str, int]:
        """Get summary of tools per category."""
        return {
            category: len(tools)
            for category, tools in cls.TOOL_CATEGORIES.items()
        }


# Example usage in agent initialization:
"""
# Instead of loading all 20 tools, load by category:

# Research Session - Load only analysis tools
research_tools = CategoryToolsManager.get_tools_by_category(["stock_data", "analysis"])
# Returns: 6 tools instead of 20

# Portfolio Session - Load portfolio management tools
portfolio_tools = CategoryToolsManager.get_tools_by_category(["portfolio", "analysis"])
# Returns: 6 tools instead of 20

# Trading Session - Load all trading-related tools
trading_tools = CategoryToolsManager.get_tools_by_category(["stock_data", "analysis", "portfolio"])
# Returns: 9 tools instead of 20

# This dramatically improves tool selection accuracy!
"""

# Example for session-based tool selection:
SESSION_CONFIGS = {
    "research": ["stock_data", "analysis"],
    "portfolio_management": ["portfolio", "analysis"],
    "quick_queries": ["stock_data", "utilities"],
    "full_analysis": ["stock_data", "analysis", "portfolio", "utilities"],
    "risk_management": ["analysis", "portfolio"],
}


async def initialize_session_agent(session_type: str):
    """Initialize agent with tools appropriate for the session type."""
    from services.agent_factory import AgentFactory

    # Get tools based on session category
    categories = SESSION_CONFIGS.get(session_type, ["stock_data"])
    tools = CategoryToolsManager.get_tools_by_category(categories)

    print(f"🔧 Loaded {len(tools)} tools for '{session_type}' session")

    # Create agent with filtered tools
    # agent = AgentFactory(llm=llm, tools=tools, system_prompt=prompt)
    # return agent


# Example usage:
"""
# Research session - 6 tools, very focused
await initialize_session_agent("research")
# AI has easier time choosing between 6 tools vs 20

# Portfolio session - 6 tools, very focused
await initialize_session_agent("portfolio_management")

# Full analysis - 15 tools, comprehensive
await initialize_session_agent("full_analysis")
"""