# Quick Start: First 3 Stock Tools Implementation Guide

This guide provides ready-to-implement stock tools to replace the demo tools and align the project with its stock chatbot backend purpose.

## Tool 1: Stock Price Query (Replace `weather`)

```python
# Add to tools.py

import httpx
from typing import Optional

class StockPriceInput(BaseModel):
    symbol: str = Field(
        ..., 
        description="Stock ticker symbol (e.g. 'AAPL', 'GOOGL', 'MSFT')"
    )
    timeframe: str = Field(
        default="current",
        description="Timeframe: 'current' for latest price, 'historical' for recent data"
    )

@tool(args_schema=StockPriceInput)
async def stock_price_query(symbol: str, timeframe: str = "current") -> str:
    """Get stock price information for given symbol using Alpha Vantage API."""
    try:
        # Get API key from environment
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            return "Error: ALPHA_VANTAGE_API_KEY not found in environment"
        
        symbol = symbol.strip().upper()
        
        if timeframe == "current":
            # Get current quote
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=10.0)
            
            if resp.status_code != 200:
                return f"Error: API returned status {resp.status_code}"
            
            data = resp.json()
            
            # Parse Alpha Vantage response
            if "Global Quote" in data:
                quote = data["Global Quote"]
                price = quote.get("05. price", "N/A")
                change = quote.get("09. change", "N/A") 
                change_percent = quote.get("10. change percent", "N/A")
                volume = quote.get("06. volume", "N/A")
                
                return f"""📊 {symbol} Stock Quote:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Price:    ${price}
Change:           ${change} ({change_percent})
Volume:           {volume}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ INVESTMENT DISCLAIMER: This information is for educational purposes only."""
            else:
                return f"Error: Could not retrieve data for {symbol}"
                
        elif timeframe == "historical":
            # Get recent historical data
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=10.0)
            
            if resp.status_code != 200:
                return f"Error: API returned status {resp.status_code}"
                
            data = resp.json()
            
            if "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                recent_dates = list(time_series.keys())[:5]  # Last 5 days
                
                result = f"📈 {symbol} Recent Prices:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                for date in recent_dates:
                    day_data = time_series[date]
                    close = day_data.get("4. close", "N/A")
                    volume = day_data.get("5. volume", "N/A")
                    result += f"{date}: ${close} | Vol: {volume}\n"
                
                result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                result += "\n⚠️ INVESTMENT DISCLAIMER: This information is for educational purposes only."
                return result
            else:
                return f"Error: Could not retrieve historical data for {symbol}"
                
    except Exception as e:
        return f"Error retrieving stock data: {str(e)}"
```

## Tool 2: Investment Return Calculator (Replace `loan_calculator`)

```python
# Add to tools.py

class InvestmentReturnInput(BaseModel):
    principal: float = Field(
        ..., gt=0, 
        description="Initial investment amount in USD"
    )
    annual_return: float = Field(
        ..., 
        description="Expected annual return rate as percentage (e.g. 7.5 for 7.5%)"
    )
    years: int = Field(
        ..., gt=0, 
        description="Investment time horizon in years"
    )
    compound_frequency: str = Field(
        default="annually",
        description="Compounding frequency: 'annually', 'monthly', 'daily'"
    )

@tool(args_schema=InvestmentReturnInput)
async def investment_return_calculator(
    principal: float, 
    annual_return: float, 
    years: int,
    compound_frequency: str = "annually"
) -> str:
    """Calculate investment growth with compound interest over time."""
    try:
        # Validate inputs
        if principal <= 0 or annual_return < 0 or years <= 0:
            return "Error: Principal and years must be positive, return must be non-negative"
        
        # Convert annual rate to decimal
        annual_rate = annual_return / 100
        
        # Determine compounding periods per year
        frequency_map = {
            "annually": 1,
            "quarterly": 4, 
            "monthly": 12,
            "daily": 365
        }
        
        n = frequency_map.get(compound_frequency.lower(), 1)
        rate_per_period = annual_rate / n
        total_periods = n * years
        
        # Calculate compound interest: A = P(1 + r/n)^(nt)
        if annual_rate == 0:
            final_amount = principal
        else:
            final_amount = principal * (1 + rate_per_period) ** total_periods
        
        total_return = final_amount - principal
        total_return_percent = (total_return / principal) * 100
        
        # Calculate simple interest for comparison
        simple_interest_final = principal * (1 + annual_rate * years)
        
        return f"""💰 Investment Growth Calculator:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Initial Investment:  ${principal:,.2f}
Expected Return:     {annual_return}%
Time Horizon:        {years} years
Compounding:        {compound_frequency}

Final Amount:        ${final_amount:,.2f}
Total Profit:        ${total_return:,.2f}
Total Return:        {total_return_percent:.2f}%

*Simple Interest:    ${simple_interest_final:,.2f}
*Compound Benefit:   ${final_amount - simple_interest_final:,.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ INVESTMENT DISCLAIMER: 
Past performance does not guarantee future results. 
This calculator is for educational purposes only."""

    except ValueError:
        return "Error: Invalid input format. Please check your numbers."
    except Exception as e:
        return f"Error calculating investment returns: {str(e)}"
```

## Tool 3: Stock Comparison Tool (New Tool)

```python
# Add to tools.py

class StockComparisonInput(BaseModel):
    symbols: list[str] = Field(
        ..., 
        min_items=2, 
        max_items=5,
        description="List of stock symbols to compare (e.g. ['AAPL', 'MSFT', 'GOOGL'])"
    )
    metric: str = Field(
        default="price",
        description="Comparison metric: 'price', 'market_cap', 'volume', 'pe_ratio'"
    )

@tool(args_schema=StockComparisonInput)
async def stock_comparison_tool(symbols: list[str], metric: str = "price") -> str:
    """Compare multiple stocks across different financial metrics."""
    try:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            return "Error: ALPHA_VANTAGE_API_KEY not found in environment"
        
        symbols = [s.strip().upper() for s in symbols]
        
        result = f"📊 Stock Comparison ({metric.upper()}):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        stock_data = []
        
        for symbol in symbols:
            try:
                # Get quote for each symbol
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=10.0)
                
                if resp.status_code == 200 and "Global Quote" in resp.json():
                    quote = resp.json()["Global Quote"]
                    
                    if metric == "price":
                        value = quote.get("05. price", "N/A")
                        stock_data.append((symbol, value))
                    elif metric == "volume":
                        value = quote.get("06. volume", "N/A") 
                        stock_data.append((symbol, value))
                    elif metric == "change_percent":
                        value = quote.get("10. change percent", "N/A")
                        stock_data.append((symbol, value))
                    else:
                        stock_data.append((symbol, "Metric not available"))
                        
                else:
                    stock_data.append((symbol, "Error"))
                    
            except Exception:
                stock_data.append((symbol, "Error"))
        
        # Sort and display results
        if metric in ["price", "volume"]:
            # Try to sort numerically
            try:
                stock_data.sort(key=lambda x: float(x[1]) if x[1] not in ["N/A", "Error"] else 0, reverse=True)
            except (ValueError, TypeError):
                pass
        
        for symbol, value in stock_data:
            result += f"{symbol:8s}: {value}\n"
        
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        result += "\n⚠️ INVESTMENT DISCLAIMER: This information is for educational purposes only."
        
        return result
        
    except Exception as e:
        return f"Error comparing stocks: {str(e)}"
```

## Environment Setup

Add these to your `.env` file:

```env
# Alpha Vantage API (Free tier: 25 calls/day)
# Get your free API key at: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

## Updated Database Schema

Add to `table_info.json`:

```json
{
  "stocks": {
    "description": "Stock master data and company information for market analysis",
    "usage_notes": "Core reference table for all stock-related queries and joins",
    "columns": [
      {
        "name": "symbol", 
        "desc": "Stock ticker symbol (primary key) - e.g. 'AAPL', 'MSFT'"
      },
      {
        "name": "company_name",
        "desc": "Full company name for display purposes"
      },
      {
        "name": "sector",
        "desc": "Industry sector (Technology, Healthcare, Finance, etc.)"
      },
      {
        "name": "market_cap",
        "desc": "Market capitalization in billions"
      },
      {
        "name": "ipo_date",
        "desc": "Initial public offering date for historical analysis"
      }
    ],
    "common_queries": [
      "Get tech stocks: SELECT * FROM stocks WHERE sector = 'Technology'",
      "Market cap leaders: SELECT TOP 10 * FROM stocks ORDER BY market_cap DESC"
    ]
  },
  "stock_prices": {
    "description": "Historical and current stock price data for performance analysis", 
    "usage_notes": "Use date range filtering for time series analysis and performance calculations",
    "columns": [
      {
        "name": "symbol",
        "desc": "Stock ticker symbol (foreign key to stocks table)"
      },
      {
        "name": "price_date",
        "desc": "Date of the price data - use for time series analysis"
      },
      {
        "name": "open_price",
        "desc": "Opening price for the trading day"
      },
      {
        "name": "high_price", 
        "desc": "Highest price during the trading day"
      },
      {
        "name": "low_price",
        "desc": "Lowest price during the trading day"
      },
      {
        "name": "close_price",
        "desc": "Closing price (most commonly used for analysis)"
      },
      {
        "name": "volume",
        "desc": "Trading volume for the day"
      }
    ],
    "common_queries": [
      "Recent prices: SELECT TOP 30 * FROM stock_prices WHERE symbol = 'AAPL' ORDER BY price_date DESC",
      "Price range: SELECT * FROM stock_prices WHERE symbol = 'MSFT' AND price_date BETWEEN '2024-01-01' AND '2024-12-31'",
      "Daily change: SELECT close_price - open_price as daily_change FROM stock_prices WHERE symbol = 'GOOGL'"
    ]
  }
}
```

## Testing Your New Tools

```python
# Test the tools in examples/stock_example.py

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from main import AgentApp

load_dotenv()

async def test_stock_tools():
    """Test the new stock analysis tools"""
    
    app = AgentApp()
    
    # Test 1: Stock price query
    await app.run(
        "What's the current stock price of Apple (AAPL)?"
    )
    
    # Test 2: Investment calculator
    await app.run(
        "If I invest $10,000 with a 7% annual return for 10 years, how much will I have?"
    )
    
    # Test 3: Stock comparison
    await app.run(
        "Compare the current prices of Microsoft (MSFT), Google (GOOGL), and Apple (AAPL)"
    )

if __name__ == "__main__":
    asyncio.run(test_stock_tools())
```

## Quick Implementation Steps

1. **Add API Key**: Get free Alpha Vantage API key and add to `.env`
2. **Update Tools**: Copy the 3 tool implementations above into `tools.py`
3. **Update Schema**: Add stock tables to `table_info.json`
4. **Remove Old Tools**: Comment out or remove `weather`, `loan_calculator`, `random_joke` tools
5. **Update System Prompt**: Change persona from "rude assistant" to "financial advisor"
6. **Test**: Run the test script to verify everything works

You'll have a functioning stock analysis chatbot backend in under 2 hours! 🚀