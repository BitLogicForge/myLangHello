# Structured Query System - Complete Guide

## 🎯 Overview

The Structured Query System transforms your AI agent from free-form text responses to **predictable, validated JSON outputs**. This is essential for:

- **API Integrations**: Frontend applications can parse responses reliably
- **Data Processing**: Automated systems can extract specific fields
- **Type Safety**: Pydantic models ensure data structure consistency
- **Testing**: Predictable response formats for automated testing
- **Monitoring**: Easy logging and analytics on structured data

## 🚀 Quick Start

### Start the API Server

```bash
# Run the FastAPI server
python api.py

# Server will be available at:
# http://localhost:8000
# http://localhost:8000/docs (Interactive API documentation)
```

### Test Structured Queries

```bash
# Install httpx for HTTP testing
pip install httpx

# Run the demo
python examples/structured_query_demo.py
```

## 📡 Available Endpoints

### 1. General Structured Query

**Endpoint:** `POST /structured/query`

**Usage:**
```bash
curl -X POST "http://localhost:8000/structured/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Analyze Apple (AAPL) stock comprehensively",
    "query_type": "stock_analysis",
    "parameters": {"symbol": "AAPL"}
  }'
```

**Query Types:**
- `stock_analysis` - Comprehensive stock analysis
- `portfolio_analysis` - Portfolio performance and risk
- `market_overview` - Market indices and sector performance
- `stock_comparison` - Multi-stock comparative analysis
- `investment_recommendation` - Investment recommendations

### 2. Stock Analysis Endpoint

**Endpoint:** `POST /structured/stock-analysis`

**Response Schema:**
```json
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
  "key_factors": [
    "Strong earnings growth",
    "Positive technical trend",
    "Moderate risk"
  ],
  "timestamp": "2026-06-18T21:30:00"
}
```

### 3. Portfolio Analysis Endpoint

**Endpoint:** `POST /structured/portfolio-analysis`

**Response Schema:**
```json
{
  "portfolio_id": "PORT_1234",
  "portfolio_name": "Growth Portfolio",
  "positions": [
    {
      "symbol": "AAPL",
      "shares": 50.0,
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
  "recommendations": [
    "Consider diversifying into healthcare",
    "Rebalance technology exposure"
  ],
  "performance_rating": "good",
  "timestamp": "2026-06-18T21:30:00"
}
```

### 4. Market Overview Endpoint

**Endpoint:** `POST /structured/market-overview`

**Response Schema:**
```json
{
  "major_indices": [
    {
      "name": "S&P 500",
      "value": 5234.18,
      "change": 45.23,
      "change_percent": 0.87
    },
    {
      "name": "NASDAQ",
      "value": 16439.22,
      "change": 183.02,
      "change_percent": 1.12
    }
  ],
  "sector_performance": [
    {
      "sector": "Technology",
      "performance": 1.45,
      "trend": "outperforming"
    },
    {
      "sector": "Healthcare",
      "performance": 0.82,
      "trend": "performing"
    }
  ],
  "market_sentiment": "bullish",
  "volatility_index": 14.25,
  "trend_analysis": "Markets showing upward momentum...",
  "key_insights": [
    "Strong tech earnings",
    "Low volatility environment"
  ],
  "timestamp": "2026-06-18T21:30:00"
}
```

## 🛠️ Implementation Details

### Architecture

```
User Request → API Endpoint → Enhanced Prompt → Agent → JSON Parser → Structured Response
                ↓
        Query Type Detection
        Prompt Engineering
        Schema Validation
        Error Handling
```

### Key Components

**1. Structured Models (`models/structured_models.py`)**
- Pydantic models for each query type
- Type safety and validation
- Clear field descriptions

**2. Enhanced Prompts (`routes/structured_query_routes.py`)**
- System prompts that enforce JSON output
- Schema templates for each query type
- Fallback handling

**3. Response Processing**
- JSON extraction from agent responses
- Schema validation
- Error recovery

### Prompt Engineering

The system uses enhanced prompts that include:

1. **Base Instructions**: "You MUST respond with structured JSON data"
2. **Schema Templates**: Exact JSON structure examples
3. **Field Requirements**: Required vs optional fields
4. **Validation Rules**: Data types and constraints
5. **Error Recovery**: Fallback handling for parsing failures

## 🎯 Use Cases

### 1. Frontend Integration

```javascript
// React/JavaScript example
async function getStockAnalysis(symbol) {
  const response = await fetch('/structured/stock-analysis', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      question: `Analyze ${symbol} stock`,
      symbol: symbol
    })
  });

  const data = await response.json();

  // Now you can access specific fields:
  updateUI({
    price: data.basic_info.current_price,
    recommendation: data.recommendation,
    confidence: data.confidence
  });
}
```

### 2. Backend Data Processing

```python
# Python backend integration
async def process_investment_signals():
    async with httpx.AsyncClient() as client:
        # Get multiple analyses
        symbols = ["AAPL", "MSFT", "GOOGL"]

        analyses = []
        for symbol in symbols:
            response = await client.post(
                f"{BASE_URL}/structured/stock-analysis",
                params={"question": f"Analyze {symbol}", "symbol": symbol}
            )
            analyses.append(response.json())

        # Process structured data
        buy_signals = [
            analysis for analysis in analyses
            if analysis["recommendation"] == "buy" and
            analysis["confidence"] > 0.7
        ]

        return buy_signals
```

### 3. Automated Trading Systems

```python
# Trading bot integration
def execute_trading_strategy():
    analysis = get_structured_analysis("NVDA")

    if analysis["recommendation"] == "buy" and analysis["confidence"] > 0.8:
        # Execute trade based on structured data
        entry_price = analysis["basic_info"]["current_price"]
        stop_loss = analysis.get("target_price", entry_price * 0.95)

        execute_order(
            symbol="NVDA",
            action="BUY",
            quantity=calculate_position_size(analysis["risk_metrics"]),
            stop_loss=stop_loss
        )
```

## 📊 Response Format Comparison

### Traditional Agent Response
```json
{
  "output": "Based on my analysis, Apple stock is currently trading at $175.50 with a positive recommendation. The technical indicators show...",
  "session_id": "abc123"
}
```

### Structured Agent Response
```json
{
  "basic_info": {
    "current_price": 175.50,
    "symbol": "AAPL"
  },
  "recommendation": "buy",
  "confidence": 0.75,
  "risk_level": "moderate"
}
```

## 🔧 Advanced Configuration

### Custom Query Types

To add new structured query types:

1. **Define Schema** (`models/structured_models.py`):
```python
class CustomAnalysisResponse(BaseModel):
    field_name: str = Field(..., description="Field description")
    numeric_value: float = Field(..., ge=0, le=100)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
```

2. **Add Prompt Template** (`routes/structured_query_routes.py`):
```python
def _create_structured_prompt(query_type: str, parameters: Optional[Dict[str, Any]]) -> str:
    if query_type == "custom_analysis":
        return '''
        Provide JSON with this structure:
        {
            "field_name": "value",
            "numeric_value": 85,
            "timestamp": "2026-06-18T21:30:00"
        }
        '''
```

3. **Add Endpoint**:
```python
@router.post("/custom-analysis", response_model=CustomAnalysisResponse)
async def custom_analysis(question: str):
    # Implementation
```

### Error Handling

The system includes multiple layers of error handling:

1. **Agent Errors**: Network issues, timeout handling
2. **JSON Parsing**: Invalid JSON format recovery
3. **Schema Validation**: Missing required fields
4. **Fallback Responses**: Raw text when parsing fails

## 🧪 Testing

### Manual Testing

```bash
# Test individual endpoints
curl -X POST "http://localhost:8000/structured/stock-analysis" \
  -H "Content-Type: application/json" \
  -d '{"question": "Analyze AAPL", "symbol": "AAPL"}'
```

### Automated Testing

```bash
# Run the comprehensive demo
python examples/structured_query_demo.py
```

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation and testing.

## 🚨 Troubleshooting

### Common Issues

**1. "Agent not loaded" Error**
```bash
# Make sure the API server is running
python api.py

# Check if agent initialized successfully
# Look for "✅ Agent loaded successfully" in startup logs
```

**2. JSON Parsing Errors**
```bash
# Check agent logs for response format issues
# The system includes fallback handling for parsing failures

# Test with simpler questions first
curl -X POST "http://localhost:8000/structured/stock-analysis" \
  -d '{"question": "What is AAPL stock price?", "symbol": "AAPL"}'
```

**3. Missing Fields in Response**
```bash
# Check logs for validation errors
# The system includes defaults for optional fields
# Required fields will cause validation errors if missing
```

## 💡 Best Practices

### 1. Query Design
- Be specific about required information
- Use appropriate query types for your needs
- Provide context when available

### 2. Error Handling
- Always check `success` field in responses
- Handle `error` field gracefully
- Implement retry logic for network issues

### 3. Performance
- Use dedicated endpoints when possible
- Cache frequently requested analyses
- Monitor processing times

### 4. Integration
- Validate response schemas in your code
- Use TypeScript/Pydantic models for type safety
- Implement proper error handling in frontend

## 🎯 Benefits Summary

✅ **Predictable Structure**: Always get the same response format
✅ **Type Safety**: Pydantic validation ensures data quality
✅ **Easy Integration**: Simple parsing for frontend applications
✅ **Better Testing**: Reliable response formats for automation
✅ **Monitoring**: Easy logging and analytics
✅ **Documentation**: Clear API contracts

The structured query system transforms your AI agent from a conversational tool into a **production-ready API service** with reliable, predictable outputs! 🚀