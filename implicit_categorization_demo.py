"""Demo showing how AI implicitly learns tool categories."""

# Example of how AI "sees" your tools:

ai_tool_understanding = {
    "stock_price_query": {
        "name": "stock_price_query",
        "description": "Get current stock price and detailed market data for a given symbol.",
        "ai_inferred_category": "Stock Data Retrieval",
        "keywords": ["stock", "price", "symbol", "market data"],
        "use_cases": ["What's AAPL price?", "Show me Tesla stock"]
    },
    "financial_metrics_calculator": {
        "name": "financial_metrics_calculator",
        "description": "Calculate comprehensive financial metrics and ratios for stock analysis.",
        "ai_inferred_category": "Financial Analysis",
        "keywords": ["financial", "metrics", "ratios", "analysis"],
        "use_cases": ["Analyze fundamentals", "Show P/E ratio"]
    },
    "risk_assessment": {
        "name": "risk_assessment",
        "description": "Comprehensive risk assessment for portfolios or individual stocks.",
        "ai_inferred_category": "Risk Analysis",
        "keywords": ["risk", "assessment", "portfolio", "volatility"],
        "use_cases": ["What's the risk?", "Analyze downside risk"]
    },
    "portfolio_create": {
        "name": "portfolio_create",
        "description": "Create a new investment portfolio with specified initial capital.",
        "ai_inferred_category": "Portfolio Management",
        "keywords": ["portfolio", "create", "investment", "capital"],
        "use_cases": ["Create portfolio", "Start investing"]
    },
    "calculator": {
        "name": "calculator",
        "description": "Evaluate a math expression safely and return the result as a string.",
        "ai_inferred_category": "General Utilities",
        "keywords": ["calculator", "math", "expression", "calculate"],
        "use_cases": ["Calculate ROI", "Math operations"]
    }
}

# AI builds mental model like this:
"""
AI's Internal Categorization Process:
1. Read tool name → "portfolio_create" → suggests portfolio category
2. Read tool description → "Create a new investment portfolio" → confirms portfolio
3. Read parameter names → "portfolio_name", "initial_capital" → confirms portfolio
4. Build mental map: portfolio_create + portfolio_add_stock + portfolio_analyze = "Portfolio Tools"

This happens automatically during model initialization!
"""