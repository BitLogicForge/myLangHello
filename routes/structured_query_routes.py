"""Structured query routes for defined JSON output responses."""

import logging
import json
import time
import re
from typing import Any, Dict, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import Config
from models.structured_models import (
    StockAnalysisResponse,
    PortfolioAnalysisResponse,
    MarketOverviewResponse,
    StockComparisonResponse,
    InvestmentRecommendation
)
from services import AgentExecutionSettings, AgentRunner, SupportsAStream
from utils import prepare_messages_with_history, is_str_dict, is_list, is_tuple

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/structured", tags=["Structured Query"])


# MARK: State Management (shared with agent_routes)
agent_executor: Optional[SupportsAStream] = None
agent_loaded_state: bool = False
config = Config()


def set_agent_executor(executor: Optional[SupportsAStream], loaded: bool) -> None:
    """Set the agent executor for structured query handling."""
    global agent_executor, agent_loaded_state
    agent_executor = executor
    agent_loaded_state = loaded


# MARK: Request/Response Models
class StructuredQueryRequest(BaseModel):
    """Request model for structured queries."""
    question: str = Field(..., description="Question to ask the agent")
    query_type: str = Field(
        ...,
        description="Type of structured output required",
        regex="^(stock_analysis|portfolio_analysis|market_overview|stock_comparison|investment_recommendation)$"
    )
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the query")


class StructuredQueryResponse(BaseModel):
    """Generic structured query response wrapper."""
    query_type: str = Field(..., description="Type of query processed")
    success: bool = Field(..., description="Whether the query was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Structured response data")
    error: Optional[str] = Field(None, description="Error message if unsuccessful")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# MARK: Structured Query Endpoint
@router.post("/query", response_model=StructuredQueryResponse)
async def structured_query(request: StructuredQueryRequest):
    """
    Process agent query and return structured JSON output.

    Supports multiple query types:
    - stock_analysis: Comprehensive stock analysis with metrics
    - portfolio_analysis: Portfolio performance and risk analysis
    - market_overview: Market indices and sector performance
    - stock_comparison: Comparative analysis of multiple stocks
    - investment_recommendation: Investment recommendations with reasoning
    """
    if not agent_loaded_state or agent_executor is None:
        logger.warning("Structured query attempted but agent not loaded")
        raise HTTPException(status_code=503, detail="Agent not loaded")

    start_time = time.time()

    try:
        logger.info(f"Processing structured query (type: {request.query_type})")
        logger.debug(f"Question: {request.question[:100]}...")

        # Prepare messages
        messages = prepare_messages_with_history(request.question, None)

        # Create enhanced prompt for structured output
        structured_prompt = _create_structured_prompt(request.query_type, request.parameters)

        # Combine user question with structured prompt
        enhanced_messages = messages + [{"role": "system", "content": structured_prompt}]

        # Execute agent
        runner = AgentRunner(agent_executor, AgentExecutionSettings.from_config(config))
        response = await runner.run({"messages": enhanced_messages})

        # Extract response content
        response_text = _extract_response_content(response)

        # Parse into structured format
        structured_data = await _parse_structured_response(
            response_text,
            request.query_type,
            request.parameters
        )

        processing_time = time.time() - start_time

        logger.info(f"Structured query completed in {processing_time:.2f}s")

        return StructuredQueryResponse(
            query_type=request.query_type,
            success=True,
            data=structured_data,
            processing_time=processing_time
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Structured query error: {e}", exc_info=True)

        return StructuredQueryResponse(
            query_type=request.query_type,
            success=False,
            error=str(e),
            processing_time=processing_time
        )


# MARK: Prompt Engineering for Structured Output
def _create_structured_prompt(query_type: str, parameters: Optional[Dict[str, Any]]) -> str:
    """Create enhanced system prompt for structured output."""

    base_instructions = """
You are a financial analysis AI that MUST respond with structured JSON data.
Your response should be ONLY valid JSON - no markdown formatting, no additional text.

Focus on accuracy and use the tools available to gather real data.
"""

    type_specific_instructions = {
        "stock_analysis": """
For stock analysis, provide JSON with this exact structure:
{
    "basic_info": {
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "current_price": 175.50,
        "change": 2.30,
        "change_percent": 1.33,
        "sector": "Technology"
    },
    "financial_metrics": {
        "pe_ratio": 28.5,
        "eps": 6.15,
        "market_cap": "2.8T",
        "dividend_yield": 0.5,
        "beta": 1.2,
        "debt_to_equity": 1.5
    },
    "technical_indicators": {
        "rsi": 55.0,
        "macd": 0.8,
        "sma_20": 173.2,
        "sma_50": 170.8,
        "signal": "bullish"
    },
    "risk_metrics": {
        "risk_level": "moderate",
        "volatility": 2.5,
        "beta": 1.2,
        "var_95": 4.2,
        "max_drawdown": 8.5
    },
    "recommendation": "buy",
    "confidence": 0.75,
    "reasoning": "Strong technical indicators and solid fundamentals...",
    "key_factors": ["Strong earnings growth", "Positive technical trend", "Moderate risk"]
}
""",

        "portfolio_analysis": """
For portfolio analysis, provide JSON with this exact structure:
{
    "portfolio_id": "PORT_1234",
    "portfolio_name": "Growth Portfolio",
    "positions": [
        {
            "symbol": "AAPL",
            "shares": 50,
            "average_cost": 170.0,
            "current_price": 175.5,
            "current_value": 8775.0,
            "profit_loss": 275.0,
            "profit_loss_percent": 3.24
        }
    ],
    "summary": {
        "total_value": 25000.0,
        "total_cost": 24500.0,
        "total_return": 500.0,
        "total_return_percent": 2.04,
        "position_count": 4
    },
    "risk_analysis": {
        "overall_risk": "moderate",
        "portfolio_beta": 1.1,
        "concentration_risk": "low",
        "sector_diversification": "good"
    },
    "recommendations": ["Consider diversifying into healthcare", "Rebalance technology exposure"],
    "performance_rating": "good"
}
""",

        "market_overview": """
For market overview, provide JSON with this exact structure:
{
    "major_indices": [
        {"name": "S&P 500", "value": 5234.18, "change": 45.23, "change_percent": 0.87},
        {"name": "NASDAQ", "value": 16439.22, "change": 183.02, "change_percent": 1.12}
    ],
    "sector_performance": [
        {"sector": "Technology", "performance": 1.45, "trend": "outperforming"},
        {"sector": "Healthcare", "performance": 0.82, "trend": "performing"}
    ],
    "market_sentiment": "bullish",
    "volatility_index": 14.25,
    "trend_analysis": "Markets showing upward momentum with technology leading...",
    "key_insights": ["Strong tech earnings", "Low volatility environment"]
}
""",

        "stock_comparison": """
For stock comparison, provide JSON with this exact structure:
{
    "comparison_type": "technical",
    "stocks_compared": ["AAPL", "MSFT", "GOOGL"],
    "scores": [
        {
            "symbol": "AAPL",
            "overall_score": 85,
            "financial_score": 80,
            "technical_score": 90,
            "risk_score": 75,
            "ranking": 1
        }
    ],
    "winner": "AAPL",
    "reasoning": "Apple shows strongest technical indicators...",
    "key_differences": ["Apple has better momentum", "Microsoft more conservative"]
}
""",

        "investment_recommendation": """
For investment recommendations, provide JSON with this exact structure:
{
    "symbol": "AAPL",
    "action": "buy",
    "confidence": 0.8,
    "target_price": 195.0,
    "stop_loss": 165.0,
    "time_horizon": "6-12 months",
    "position_size": "5-10% of portfolio",
    "reasoning": "Strong fundamentals and positive technical trend...",
    "risks": ["Market volatility", "Competition risks"],
    "catalysts": ["New product launch", "Earnings growth"]
}
"""
    }

    specific_instructions = type_specific_instructions.get(query_type, "")

    parameter_hints = ""
    if parameters:
        parameter_hints = f"\n\nAdditional parameters to consider: {json.dumps(parameters, indent=2)}"

    return f"""{base_instructions}

{specific_instructions}
{parameter_hints}

IMPORTANT: Your response must be ONLY valid JSON. Do not include any explanatory text outside the JSON structure.
Ensure all numeric values are proper numbers (not strings).
Ensure all required fields are included.
"""


# MARK: Response Content Extraction
def _extract_response_content(response: Any) -> str:
    """Extract text content from agent response."""

    if isinstance(response, dict):
        messages_list = response.get("messages")
        if is_list(messages_list) and len(messages_list) > 0:
            final_message = messages_list[-1]
            content = getattr(final_message, "content", None)
            if content is not None:
                return str(content)
            elif is_tuple(final_message) and len(final_message) > 1:
                return str(final_message[1])
            else:
                return str(final_message)
        else:
            return str(response.get("output", response))
    elif response is not None:
        return str(response)
    else:
        return "No response generated"


# MARK: Structured Response Parsing
async def _parse_structured_response(
    response_text: str,
    query_type: str,
    parameters: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Parse agent response into structured format."""

    try:
        # Clean up the response text
        cleaned_text = _clean_json_response(response_text)

        # Parse JSON
        parsed_data = json.loads(cleaned_text)

        # Validate against expected schema
        validated_data = await _validate_structured_data(parsed_data, query_type, parameters)

        return validated_data

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        # Fallback: create structured response from raw text
        return _create_fallback_response(response_text, query_type)

    except Exception as e:
        logger.error(f"Structured parsing error: {e}")
        return _create_fallback_response(response_text, query_type)


def _clean_json_response(response_text: str) -> str:
    """Clean response text to extract pure JSON."""

    # Remove markdown code blocks
    cleaned = re.sub(r'```json\s*', '', response_text)
    cleaned = re.sub(r'```\s*', '', cleaned)

    # Remove any text before the first {
    cleaned = re.sub(r'^[^{]*', '', cleaned)

    # Remove any text after the last }
    cleaned = re.sub(r'[^}]*$', '', cleaned)

    return cleaned.strip()


async def _validate_structured_data(
    parsed_data: Dict[str, Any],
    query_type: str,
    parameters: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Validate and enhance structured data."""

    # Add timestamp if not present
    if "timestamp" not in parsed_data:
        parsed_data["timestamp"] = datetime.now().isoformat()

    # Add any missing required fields with defaults
    if query_type == "stock_analysis":
        if "basic_info" not in parsed_data:
            raise ValueError("Missing required field: basic_info")
        if "recommendation" not in parsed_data:
            parsed_data["recommendation"] = "hold"
        if "confidence" not in parsed_data:
            parsed_data["confidence"] = 0.5

    elif query_type == "portfolio_analysis":
        if "portfolio_id" not in parsed_data:
            parsed_data["portfolio_id"] = "UNKNOWN"
        if "positions" not in parsed_data:
            parsed_data["positions"] = []

    return parsed_data


def _create_fallback_response(response_text: str, query_type: str) -> Dict[str, Any]:
    """Create fallback response when parsing fails."""

    return {
        "raw_response": response_text,
        "query_type": query_type,
        "timestamp": datetime.now().isoformat(),
        "note": "Structured parsing failed, returning raw response"
    }


# MARK: Dedicated Query Type Endpoints
@router.post("/stock-analysis", response_model=StockAnalysisResponse)
async def stock_analysis(question: str, symbol: Optional[str] = None):
    """Dedicated endpoint for stock analysis with predefined structure."""
    request = StructuredQueryRequest(
        question=question,
        query_type="stock_analysis",
        parameters={"symbol": symbol} if symbol else None
    )

    response = await structured_query(request)

    if not response.success or response.data is None:
        raise HTTPException(status_code=500, detail=response.error)

    return StockAnalysisResponse(**response.data)


@router.post("/portfolio-analysis", response_model=PortfolioAnalysisResponse)
async def portfolio_analysis(question: str, portfolio_id: Optional[str] = None):
    """Dedicated endpoint for portfolio analysis with predefined structure."""
    request = StructuredQueryRequest(
        question=question,
        query_type="portfolio_analysis",
        parameters={"portfolio_id": portfolio_id} if portfolio_id else None
    )

    response = await structured_query(request)

    if not response.success or response.data is None:
        raise HTTPException(status_code=500, detail=response.error)

    return PortfolioAnalysisResponse(**response.data)


@router.post("/market-overview", response_model=MarketOverviewResponse)
async def market_overview(question: str):
    """Dedicated endpoint for market overview with predefined structure."""
    request = StructuredQueryRequest(
        question=question,
        query_type="market_overview"
    )

    response = await structured_query(request)

    if not response.success or response.data is None:
        raise HTTPException(status_code=500, detail=response.error)

    return MarketOverviewResponse(**response.data)


@router.post("/stock-comparison", response_model=StockComparisonResponse)
async def stock_comparison(question: str, symbols: Optional[str] = None):
    """Dedicated endpoint for stock comparison with predefined structure."""
    request = StructuredQueryRequest(
        question=question,
        query_type="stock_comparison",
        parameters={"symbols": symbols} if symbols else None
    )

    response = await structured_query(request)

    if not response.success or response.data is None:
        raise HTTPException(status_code=500, detail=response.error)

    return StockComparisonResponse(**response.data)


@router.post("/investment-recommendation", response_model=InvestmentRecommendation)
async def investment_recommendation(question: str, symbol: Optional[str] = None):
    """Dedicated endpoint for investment recommendations with predefined structure."""
    request = StructuredQueryRequest(
        question=question,
        query_type="investment_recommendation",
        parameters={"symbol": symbol} if symbol else None
    )

    response = await structured_query(request)

    if not response.success or response.data is None:
        raise HTTPException(status_code=500, detail=response.error)

    return InvestmentRecommendation(**response.data)