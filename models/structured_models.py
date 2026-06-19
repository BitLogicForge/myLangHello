"""Structured output models for financial analysis queries."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


# MARK: Stock Analysis Schemas
class StockBasicInfo(BaseModel):
    """Basic stock information."""
    symbol: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    current_price: float = Field(..., description="Current stock price")
    change: float = Field(..., description="Price change")
    change_percent: float = Field(..., description="Percentage change")
    sector: str = Field(..., description="Industry sector")


class FinancialMetrics(BaseModel):
    """Financial metrics and ratios."""
    pe_ratio: Optional[float] = Field(None, description="Price-to-earnings ratio")
    eps: Optional[float] = Field(None, description="Earnings per share")
    market_cap: Optional[str] = Field(None, description="Market capitalization")
    dividend_yield: Optional[float] = Field(None, description="Dividend yield percentage")
    beta: Optional[float] = Field(None, description="Beta coefficient")
    debt_to_equity: Optional[float] = Field(None, description="Debt-to-equity ratio")


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""
    rsi: Optional[float] = Field(None, description="RSI indicator")
    macd: Optional[float] = Field(None, description="MACD value")
    sma_20: Optional[float] = Field(None, description="20-day simple moving average")
    sma_50: Optional[float] = Field(None, description="50-day simple moving average")
    signal: Optional[str] = Field(None, description="Trading signal (bullish/bearish/neutral)")


class RiskMetrics(BaseModel):
    """Risk assessment metrics."""
    risk_level: str = Field(..., description="Risk category (low/moderate/high)")
    volatility: float = Field(..., description="Volatility percentage")
    beta: Optional[float] = Field(None, description="Market correlation beta")
    var_95: Optional[float] = Field(None, description="Value at Risk at 95% confidence")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown percentage")


class StockAnalysisResponse(BaseModel):
    """Comprehensive stock analysis response."""
    basic_info: StockBasicInfo
    financial_metrics: Optional[FinancialMetrics] = None
    technical_indicators: Optional[TechnicalIndicators] = None
    risk_metrics: Optional[RiskMetrics] = None
    recommendation: str = Field(..., description="Investment recommendation (buy/sell/hold)")
    confidence: float = Field(..., description="Confidence level 0-1", ge=0, le=1)
    reasoning: str = Field(..., description="Detailed reasoning behind recommendation")
    key_factors: List[str] = Field(default_factory=list, description="Key factors influencing decision")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# MARK: Portfolio Analysis Schemas
class PortfolioPosition(BaseModel):
    """Individual portfolio position."""
    symbol: str = Field(..., description="Stock symbol")
    shares: float = Field(..., description="Number of shares")
    average_cost: float = Field(..., description="Average cost per share")
    current_price: float = Field(..., description="Current market price")
    current_value: float = Field(..., description="Current position value")
    profit_loss: float = Field(..., description="Profit/loss amount")
    profit_loss_percent: float = Field(..., description="Profit/loss percentage")


class PortfolioSummary(BaseModel):
    """Portfolio summary metrics."""
    total_value: float = Field(..., description="Total portfolio value")
    total_cost: float = Field(..., description="Total cost basis")
    total_return: float = Field(..., description="Total return amount")
    total_return_percent: float = Field(..., description="Total return percentage")
    position_count: int = Field(..., description="Number of positions")


class PortfolioRiskAnalysis(BaseModel):
    """Portfolio risk analysis."""
    overall_risk: str = Field(..., description="Overall risk level")
    portfolio_beta: float = Field(..., description="Portfolio beta")
    concentration_risk: str = Field(..., description="Concentration risk assessment")
    sector_diversification: str = Field(..., description="Sector diversification quality")


class PortfolioAnalysisResponse(BaseModel):
    """Comprehensive portfolio analysis response."""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    portfolio_name: str = Field(..., description="Portfolio name")
    positions: List[PortfolioPosition]
    summary: PortfolioSummary
    risk_analysis: PortfolioRiskAnalysis
    recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")
    performance_rating: str = Field(..., description="Overall performance rating")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# MARK: Market Analysis Schemas
class MarketIndex(BaseModel):
    """Market index data."""
    name: str = Field(..., description="Index name")
    value: float = Field(..., description="Current index value")
    change: float = Field(..., description="Daily change")
    change_percent: float = Field(..., description="Daily percentage change")


class SectorPerformance(BaseModel):
    """Sector performance data."""
    sector: str = Field(..., description="Sector name")
    performance: float = Field(..., description="Performance percentage")
    trend: str = Field(..., description="Trend direction (outperforming/underperforming)")


class MarketOverviewResponse(BaseModel):
    """Market overview analysis response."""
    major_indices: List[MarketIndex]
    sector_performance: List[SectorPerformance]
    market_sentiment: str = Field(..., description="Overall market sentiment")
    volatility_index: Optional[float] = Field(None, description="VIX value")
    trend_analysis: str = Field(..., description="Market trend analysis")
key_insights: List[str] = Field(default_factory=list, description="Key market insights")
timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# MARK: Comparison Analysis Schemas
class StockComparisonScore(BaseModel):
    """Stock comparison score."""
    symbol: str = Field(..., description="Stock symbol")
    overall_score: float = Field(..., description="Overall score 0-100")
    financial_score: Optional[float] = Field(None, description="Financial metrics score")
    technical_score: Optional[float] = Field(None, description="Technical indicators score")
    risk_score: Optional[float] = Field(None, description="Risk profile score")
    ranking: int = Field(..., description="Ranking among compared stocks")


class StockComparisonResponse(BaseModel):
    """Stock comparison analysis response."""
    comparison_type: str = Field(..., description="Type of comparison performed")
    stocks_compared: List[str] = Field(..., description="Symbols that were compared")
    scores: List[StockComparisonScore]
    winner: str = Field(..., description="Recommended stock symbol")
    reasoning: str = Field(..., description="Detailed comparison reasoning")
    key_differences: List[str] = Field(default_factory=list, description="Key differences found")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# MARK: Investment Recommendation Schemas
class InvestmentRecommendation(BaseModel):
    """Investment recommendation response."""
    symbol: str = Field(..., description="Stock symbol")
    action: str = Field(..., description="Recommended action (buy/sell/hold)")
    confidence: float = Field(..., description="Confidence level 0-1", ge=0, le=1)
    target_price: Optional[float] = Field(None, description="Target price")
    stop_loss: Optional[float] = Field(None, description="Recommended stop-loss price")
    time_horizon: str = Field(..., description="Investment time horizon")
    position_size: Optional[str] = Field(None, description="Recommended position size")
    reasoning: str = Field(..., description="Detailed recommendation reasoning")
    risks: List[str] = Field(default_factory=list, description="Key risks to consider")
    catalysts: List[str] = Field(default_factory=list, description="Positive catalysts")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())