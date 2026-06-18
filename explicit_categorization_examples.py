"""Example of explicit categorization through enhanced descriptions."""

# METHOD 1: Add category prefix to descriptions
@tool
async def stock_price_query(symbol: str, include_details: bool = True) -> str:
    """[Stock Data] Get current stock price and detailed market data for a given symbol."""
    #          ↑ CATEGORY PREFIX
    pass

@tool
async def financial_metrics_calculator(symbol: str, metrics: list[str] | None = None) -> str:
    """[Financial Analysis] Calculate comprehensive financial metrics and ratios for stock analysis."""
    #                ↑ CATEGORY PREFIX
    pass

@tool
async def portfolio_create(portfolio_name: str, initial_capital: float) -> str:
    """[Portfolio Management] Create a new investment portfolio with specified initial capital."""
    #              ↑ CATEGORY PREFIX
    pass

# METHOD 2: Use system prompt to teach categories
CATEGORIZED_SYSTEM_PROMPT = """
You are a stock analysis assistant with access to categorized tools:

STOCK DATA TOOLS:
- stock_price_query: Get current prices and market data
- market_data: Get market indices and sector performance

FINANCIAL ANALYSIS TOOLS:
- financial_metrics_calculator: Analyze fundamentals and ratios
- technical_analysis: Perform technical indicator analysis

RISK ANALYSIS TOOLS:
- risk_assessment: Evaluate portfolio and stock risk
- stock_comparison: Compare stocks across risk metrics

PORTFOLIO MANAGEMENT TOOLS:
- portfolio_create: Create new portfolios
- portfolio_add_stock: Add positions to portfolios
- portfolio_analyze: Analyze portfolio performance

Choose tools based on the user's question category.
"""

# METHOD 3: Add metadata field to tools (requires custom tool class)
class CategorizedTool(BaseTool):
    """Tool with explicit category metadata."""

    category: str = "General"  # Explicit category field

    def _run(self, *args, **kwargs):
        pass

stock_price_query.category = "Stock Data"
financial_metrics_calculator.category = "Financial Analysis"
portfolio_create.category = "Portfolio Management"