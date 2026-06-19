"""Structured Query Demo - Demonstrating structured JSON output from agent queries.

This shows how to get defined, predictable JSON responses from the AI agent
instead of free-form text responses.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("Warning: httpx not available. Install with: pip install httpx")

API_BASE_URL = "http://localhost:8000"


async def test_general_structured_query():
    """Test the general structured query endpoint."""
    print("🎯 Testing General Structured Query")
    print("=" * 50)

    if not HTTPX_AVAILABLE:
        print("❌ httpx not available, skipping HTTP test")
        return

    try:
        async with httpx.AsyncClient() as client:
            # Test stock analysis
            request_data = {
                "question": "Analyze Apple (AAPL) stock comprehensively",
                "query_type": "stock_analysis",
                "parameters": {"symbol": "AAPL"}
            }

            print(f"📤 Request: {request_data['question']}")
            print(f"🔧 Query Type: {request_data['query_type']}")

            response = await client.post(
                f"{API_BASE_URL}/structured/query",
                json=request_data,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {result['success']}")
                print(f"⏱️  Processing Time: {result['processing_time']:.2f}s")

                if result['data']:
                    print(f"📊 Structured Data:")
                    print(json.dumps(result['data'], indent=2))
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Detail: {response.text}")

    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("💡 Make sure the API server is running: python api.py")


async def test_stock_analysis_endpoint():
    """Test the dedicated stock analysis endpoint."""
    print("\n📊 Testing Stock Analysis Endpoint")
    print("=" * 50)

    if not HTTPX_AVAILABLE:
        print("❌ httpx not available, skipping HTTP test")
        return

    try:
        async with httpx.AsyncClient() as client:
            request_data = {
                "question": "Should I buy Tesla (TSLA) stock?",
                "symbol": "TSLA"
            }

            print(f"📤 Question: {request_data['question']}")
            print(f"🔍 Symbol: {request_data['symbol']}")

            response = await client.post(
                f"{API_BASE_URL}/structured/stock-analysis",
                params=request_data,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Stock Analysis Received:")
                print(f"   Symbol: {result.get('basic_info', {}).get('symbol')}")
                print(f"   Company: {result.get('basic_info', {}).get('company_name')}")
                print(f"   Price: ${result.get('basic_info', {}).get('current_price', 0):.2f}")
                print(f"   Recommendation: {result.get('recommendation')}")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
            else:
                print(f"❌ Error: {response.status_code}")

    except Exception as e:
        print(f"❌ Connection error: {e}")


async def test_portfolio_analysis_endpoint():
    """Test the portfolio analysis endpoint."""
    print("\n💼 Testing Portfolio Analysis Endpoint")
    print("=" * 50)

    if not HTTPX_AVAILABLE:
        print("❌ httpx not available, skipping HTTP test")
        return

    try:
        async with httpx.AsyncClient() as client:
            request_data = {
                "question": "Analyze my portfolio performance and risk",
                "portfolio_id": "PORT_1234"
            }

            print(f"📤 Question: {request_data['question']}")
            print(f"📁 Portfolio ID: {request_data['portfolio_id']}")

            response = await client.post(
                f"{API_BASE_URL}/structured/portfolio-analysis",
                params=request_data,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Portfolio Analysis Received:")
                print(f"   Portfolio: {result.get('portfolio_name')}")
                print(f"   Total Value: ${result.get('summary', {}).get('total_value', 0):,.2f}")
                print(f"   Return: {result.get('summary', {}).get('total_return_percent', 0):.2f}%")
                print(f"   Risk Level: {result.get('risk_analysis', {}).get('overall_risk')}")
            else:
                print(f"❌ Error: {response.status_code}")

    except Exception as e:
        print(f"❌ Connection error: {e}")


async def test_market_overview_endpoint():
    """Test the market overview endpoint."""
    print("\n🌍 Testing Market Overview Endpoint")
    print("=" * 50)

    if not HTTPX_AVAILABLE:
        print("❌ httpx not available, skipping HTTP test")
        return

    try:
        async with httpx.AsyncClient() as client:
            request_data = {
                "question": "What's the current market overview and major indices performance?"
            }

            print(f"📤 Question: {request_data['question']}")

            response = await client.post(
                f"{API_BASE_URL}/structured/market-overview",
                params=request_data,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Market Overview Received:")
                print(f"   Market Sentiment: {result.get('market_sentiment')}")
                print(f"   Trend Analysis: {result.get('trend_analysis', '')[:60]}...")

                indices = result.get('major_indices', [])
                if indices:
                    print(f"   Major Indices ({len(indices)}):")
                    for index in indices[:3]:
                        print(f"      {index.get('name')}: {index.get('change_percent', 0):+.2f}%")
            else:
                print(f"❌ Error: {response.status_code}")

    except Exception as e:
        print(f"❌ Connection error: {e}")


async def test_stock_comparison_endpoint():
    """Test the stock comparison endpoint."""
    print("\n📊 Testing Stock Comparison Endpoint")
    print("=" * 50)

    if not HTTPX_AVAILABLE:
        print("❌ httpx not available, skipping HTTP test")
        return

    try:
        async with httpx.AsyncClient() as client:
            request_data = {
                "question": "Compare Apple, Microsoft, and Google for investment",
                "symbols": "AAPL,MSFT,GOOGL"
            }

            print(f"📤 Question: {request_data['question']}")
            print(f"🔍 Symbols: {request_data['symbols']}")

            response = await client.post(
                f"{API_BASE_URL}/structured/stock-comparison",
                params=request_data,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Stock Comparison Received:")
                print(f"   Comparison Type: {result.get('comparison_type')}")
                print(f"   Winner: {result.get('winner')}")
                print(f"   Stocks Compared: {result.get('stocks_compared')}")

                scores = result.get('scores', [])
                if scores:
                    print(f"   Rankings:")
                    for score in scores:
                        print(f"      {score.get('ranking')}. {score.get('symbol')}: {score.get('overall_score')}/100")
            else:
                print(f"❌ Error: {response.status_code}")

    except Exception as e:
        print(f"❌ Connection error: {e}")


async def test_investment_recommendation_endpoint():
    """Test the investment recommendation endpoint."""
    print("\n💡 Testing Investment Recommendation Endpoint")
    print("=" * 50)

    if not HTTPX_AVAILABLE:
        print("❌ httpx not available, skipping HTTP test")
        return

    try:
        async with httpx.AsyncClient() as client:
            request_data = {
                "question": "Should I invest in NVIDIA for long-term growth?",
                "symbol": "NVDA"
            }

            print(f"📤 Question: {request_data['question']}")
            print(f"🔍 Symbol: {request_data['symbol']}")

            response = await client.post(
                f"{API_BASE_URL}/structured/investment-recommendation",
                params=request_data,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Investment Recommendation Received:")
                print(f"   Action: {result.get('action')}")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
                print(f"   Target Price: ${result.get('target_price', 0):.2f}")
                print(f"   Stop Loss: ${result.get('stop_loss', 0):.2f}")
                print(f"   Time Horizon: {result.get('time_horizon')}")
                print(f"   Position Size: {result.get('position_size')}")
            else:
                print(f"❌ Error: {response.status_code}")

    except Exception as e:
        print(f"❌ Connection error: {e}")


async def test_direct_agent_call():
    """Test structured output using direct agent calls (no HTTP)."""
    print("\n🤖 Testing Direct Agent Structured Output")
    print("=" * 50)

    try:
        from main import AgentApp

        app = AgentApp()

        # Test with structured output prompt
        structured_question = """
        Analyze Meta (META) stock and provide your response in this exact JSON format:

        {
            "basic_info": {
                "symbol": "META",
                "company_name": "Meta Platforms Inc.",
                "current_price": 505.25,
                "change": 17.50,
                "change_percent": 3.59,
                "sector": "Technology"
            },
            "recommendation": "buy",
            "confidence": 0.75,
            "reasoning": "Strong technical indicators and positive momentum...",
            "key_factors": ["Strong earnings", "Technical breakout", "Moderate risk"]
        }

        Provide ONLY the JSON response, no additional text.
        """

        print("📤 Running structured query...")
        response = await app.run(structured_question)

        print(f"📤 Agent Response:")
        print(f"{response}")

    except Exception as e:
        print(f"❌ Agent error: {e}")


async def main():
    """Run all structured query tests."""
    print("🚀 Structured Query Demo")
    print("=" * 70)
    print("Testing different structured output endpoints with your stock agent")
    print("=" * 70)

    # Test all endpoints
    await test_general_structured_query()
    await test_stock_analysis_endpoint()
    await test_portfolio_analysis_endpoint()
    await test_market_overview_endpoint()
    await test_stock_comparison_endpoint()
    await test_investment_recommendation_endpoint()

    # Test direct agent call
    await test_direct_agent_call()

    print("\n🎉 Structured Query Demo Complete!")
    print("=" * 70)
    print("✅ Demonstrated:")
    print("• General structured query endpoint")
    print("• Stock analysis with predefined JSON schema")
    print("• Portfolio analysis with structured metrics")
    print("• Market overview with standardized indices")
    print("• Stock comparison with scoring system")
    print("• Investment recommendations with action items")
    print("• Direct agent calls with structured output prompts")
    print("\n💡 All endpoints return predictable, validated JSON responses")


if __name__ == "__main__":
    print("🎯 Structured Query Testing Suite")
    print("This demo tests the structured output endpoints")
    print("Make sure the API server is running: python api.py")
    print("=" * 70)

    asyncio.run(main())