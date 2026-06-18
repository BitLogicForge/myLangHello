"""Advanced Stock Analysis Workflow Demo - Demonstrating sophisticated multi-step AI agent capabilities.

This script showcases the REAL power of AI agents:
- Multi-step reasoning and tool chaining
- State management across conversations
- Complex workflow orchestration
- Context-aware decision making
- Tool composition (using tool outputs as inputs)
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from main import AgentApp

load_dotenv()


async def demo_portfolio_workflow():
    """Demonstrate complete portfolio management workflow with tool chaining."""

    print("🎯 Advanced Portfolio Management Workflow")
    print("=" * 70)

    app = AgentApp()
    conversation_history = []

    # Step 1: Create portfolio
    print("\n📊 Step 1: Creating Portfolio")
    print("-" * 40)

    result = await app.run(
        "Create a conservative investment portfolio called 'Retirement 2030' "
        "with $50,000 initial capital"
    )
    conversation_history.append({
        "question": "Create portfolio",
        "answer": result
    })

    # Extract portfolio ID from response (agent will need to remember this)
    await asyncio.sleep(2)

    # Step 2: Multi-step stock screening and selection
    print("\n🔍 Step 2: Stock Screening Workflow")
    print("-" * 40)

    result = await app.run(
        "I want to build a diversified portfolio. First, show me the top performing "
        "stocks in technology, healthcare, and finance sectors today. Then analyze "
        "which ones would be good for conservative long-term investing based on "
        "their risk metrics and financial health."
    )
    conversation_history.append({
        "question": "Stock screening",
        "answer": result
    })

    await asyncio.sleep(3)

    # Step 3: Adding multiple positions with reasoning
    print("\n💼 Step 3: Strategic Position Building")
    print("-" * 40)

    result = await app.run(
        "Based on the analysis, add these positions to my Retirement 2030 portfolio:\n"
        "1. 50 shares of Apple (AAPL) at $175\n"
        "2. 30 shares of Johnson & Johnson (JNJ) at $162\n"
        "3. 40 shares of Visa (V) at $280\n"
        "For each addition, explain why it fits our conservative strategy."
    )
    conversation_history.append({
        "question": "Adding positions",
        "answer": result
    })

    await asyncio.sleep(3)

    # Step 4: Multi-dimensional portfolio analysis
    print("\n📈 Step 4: Comprehensive Portfolio Analysis")
    print("-" * 40)

    result = await app.run(
        "Analyze my Retirement 2030 portfolio comprehensively:\n"
        "1. What's the current performance and total return?\n"
        "2. How well diversified am I across sectors?\n"
        "3. What's my risk level and downside exposure?\n"
        "4. Compare my portfolio's volatility to the S&P 500\n"
        "Provide specific recommendations for optimization."
    )
    conversation_history.append({
        "question": "Portfolio analysis",
        "answer": result
    })

    await asyncio.sleep(3)

    # Step 5: Risk-adjusted decision making
    print("\n🎯 Step 5: Risk Assessment and Hedging Strategy")
    print("-" * 40)

    result = await app.run(
        "Perform a detailed risk assessment on my Retirement 2030 portfolio. "
        "Calculate:\n"
        "- Concentration risk (am I too exposed to any single position?)\n"
        "- Sector risk (do I have enough defensive positions?)\n"
        "- Market correlation risk (how will I perform in a downturn?)\n"
        "Then suggest specific actions to reduce high-risk exposures."
    )
    conversation_history.append({
        "question": "Risk assessment",
        "answer": result
    })

    await asyncio.sleep(3)

    # Step 6: Portfolio optimization workflow
    print("\n⚡ Step 6: Portfolio Optimization Recommendations")
    print("-" * 40)

    result = await app.run(
        "Based on all the analysis, provide specific optimization recommendations:\n"
        "1. Which positions should I increase or decrease?\n"
        "2. What defensive stocks should I add to reduce volatility?\n"
        "3. Should I rebalance any positions that have drifted?\n"
        "4. What's the optimal allocation for a conservative 7-year timeline?\n"
        "Prioritize recommendations by impact and feasibility."
    )
    conversation_history.append({
        "question": "Optimization",
        "answer": result
    })

    print("\n🎉 Portfolio Workflow Complete!")
    print("=" * 70)
    print(f"Total conversation steps: {len(conversation_history)}")
    print("Demonstrated: Tool chaining, state management, multi-step reasoning")


async def demo_investment_research_workflow():
    """Demonstrate comprehensive investment research process with tool composition."""

    print("\n🔬 Advanced Investment Research Workflow")
    print("=" * 70)

    app = AgentApp()

    # Research Question: "Should I invest in NVIDIA for the long term?"

    print("\n📚 Research Question: Long-term NVIDIA Investment Analysis")
    print("-" * 50)

    # Phase 1: Initial screening
    print("\n🔍 Phase 1: Initial Screening")
    print("-" * 30)

    result = await app.run(
        "I'm researching NVIDIA (NVDA) for long-term investment. Start with:\n"
        "1. Current stock price and recent performance\n"
        "2. How does it compare to sector peers (AMD, Intel)?\n"
        "3. What's the current market sentiment for semiconductor stocks?"
    )

    await asyncio.sleep(3)

    # Phase 2: Deep technical analysis
    print("\n📈 Phase 2: Technical Analysis")
    print("-" * 30)

    result = await app.run(
        "Now perform deep technical analysis on NVIDIA:\n"
        "1. All key technical indicators (RSI, MACD, Bollinger Bands, SMAs)\n"
        "2. Current trend analysis and price targets\n"
        "3. Support and resistance levels\n"
        "4. What do the technical indicators suggest about entry timing?"
    )

    await asyncio.sleep(3)

    # Phase 3: Fundamental analysis
    print("\n💰 Phase 3: Fundamental Analysis")
    print(-30)

    result = await app.run(
        "Analyze NVIDIA's fundamentals:\n"
        "1. All key financial metrics (P/E, earnings growth, margins)\n"
        "2. How do these metrics compare to historical averages?\n"
        "3. What do the fundamentals say about stock valuation?\n"
        "4. Are there any red flags in the financial health?"
    )

    await asyncio.sleep(3)

    # Phase 4: Risk assessment
    print("\n🎯 Phase 4: Risk Assessment")
    print("-" * 30)

    result = await app.run(
        "Assess the risks of investing in NVIDIA:\n"
        "1. Complete risk profile (volatility, beta, downside risk)\n"
        "2. Sector-specific risks for semiconductor industry\n"
        "3. Concentration risk if I add this to a tech-heavy portfolio\n"
        "4. What position size would be appropriate for moderate risk tolerance?"
    )

    await asyncio.sleep(3)

    # Phase 5: Investment decision synthesis
    print("\n🤔 Phase 5: Investment Decision Synthesis")
    print("-" * 30)

    result = await app.run(
        "Synthesize all the research into an investment recommendation:\n"
        "1. Should I buy NVIDIA for long-term investment? Yes/No/Maybe\n"
        "2. Top 3 reasons supporting your decision\n"
        "3. Top 3 concerns or risks\n"
        "4. If yes, what's the optimal entry strategy and position sizing?\n"
        "5. What specific indicators would change your recommendation?\n"
        "Provide a clear, actionable investment thesis."
    )

    print("\n🎉 Investment Research Complete!")
    print("=" * 70)


async def demo_multi_stock_comparison_workflow():
    """Demonstrate complex multi-stock decision making with comparative analysis."""

    print("\n🔄 Multi-Stock Comparison and Selection Workflow")
    print("=" * 70)

    app = AgentApp()

    print("\n💡 Scenario: Choosing between top tech stocks for $10,000 investment")
    print("-" * 60)

    # Stage 1: Candidate identification
    print("\n🔍 Stage 1: Candidate Identification")
    print("-" * 35)

    result = await app.run(
        "I have $10,000 to invest in one tech stock. Help me choose between:\n"
        "Apple (AAPL), Microsoft (MSFT), Google (GOOGL), NVIDIA (NVDA), and Meta (META).\n"
        "First, give me a quick comparison of current prices and recent performance."
    )

    await asyncio.sleep(2)

    # Stage 2: Multi-dimensional comparison
    print("\n📊 Stage 2: Multi-Dimensional Analysis")
    print("-" * 35)

    result = await app.run(
        "Now compare these 5 stocks across all dimensions:\n"
        "1. Financial metrics and valuation\n"
        "2. Technical indicators and momentum\n"
        "3. Risk profiles and volatility\n"
        "4. Growth prospects and competitive position\n"
        "Create a scoring matrix and rank them."
    )

    await asyncio.sleep(3)

    # Stage 3: Context-aware recommendation
    print("\n🎯 Stage 3: Context-Aware Selection")
    print("-" * 35)

    result = await app.run(
        "Based on the analysis, recommend the best stock for my situation:\n"
        "- Investment timeline: 3-5 years\n"
        "- Risk tolerance: Moderate\n"
        "- Goal: Growth with some downside protection\n"
        "- Existing portfolio: Heavy in index funds, no individual tech stocks\n"
        "Explain your reasoning and suggest 2 backup options."
    )

    await asyncio.sleep(2)

    # Stage 4: Action plan
    print("\n📋 Stage 4: Investment Action Plan")
    print(-35)

    result = await app.run(
        "Create a detailed action plan for investing in your recommended stock:\n"
        "1. Optimal entry timing based on technical indicators\n"
        "2. Position sizing for $10,000 investment\n"
        "3. Stop-loss level to limit downside\n"
        "4. Take-profit targets for 1-year and 3-year horizons\n"
        "5. What indicators would make you exit the position?"
    )

    print("\n🎉 Multi-Stock Selection Complete!")
    print("=" * 70)


async def demo_sector_rotation_workflow():
    """Demonstrate advanced sector analysis and portfolio rebalancing strategy."""

    print("\n🌐 Sector Rotation and Portfolio Rebalancing Workflow")
    print("=" * 70)

    app = AgentApp()

    print("\n💡 Scenario: Quarterly portfolio review and sector allocation adjustment")
    print("-" * 70)

    # Step 1: Market and sector analysis
    print("\n🌍 Step 1: Market and Sector Analysis")
    print("-" * 40)

    result = await app.run(
        "Perform a comprehensive market analysis:\n"
        "1. Current market indices performance and trend\n"
        "2. Sector performance ranking (best to worst)\n"
        "3. Which sectors are showing strength and which are weakening?\n"
        "4. What macro trends are driving sector performance?"
    )

    await asyncio.sleep(2)

    # Step 2: Portfolio vs Market comparison
    print("\n📊 Step 2: Portfolio Sector Analysis")
    print("-" * 40)

    # First create a sample portfolio
    await app.run(
        "Create a portfolio called 'Growth Portfolio' with $25,000 and add:\n"
        "25 shares of AAPL at $175, 20 shares of MSFT at $378, "
        "15 shares of TSLA at $248, 30 shares of NVDA at $875"
    )

    await asyncio.sleep(2)

    result = await app.run(
        "Analyze my Growth Portfolio's sector allocation:\n"
        "1. What's my current sector breakdown?\n"
        "2. How does this compare to market sector weights?\n"
        "3. Am I overexposed or underexposed to any sectors?\n"
        "4. What are the risks of my current sector allocation?"
    )

    await asyncio.sleep(3)

    # Step 3: Sector rotation strategy
    print("\n🔄 Step 3: Sector Rotation Strategy")
    print("-" * 40)

    result = await app.run(
        "Based on the market analysis, develop a sector rotation strategy:\n"
        "1. Which sectors should I increase exposure to?\n"
        "2. Which sectors should I reduce?\n"
        "3. Suggest specific stocks to add from strong sectors\n"
        "4. Which existing positions should I reduce or exit?\n"
        "Prioritize by potential impact and risk reduction."
    )

    await asyncio.sleep(3)

    # Step 4: Rebalancing execution plan
    print("\n⚖️ Step 4: Portfolio Rebalancing Plan")
    print("-" * 40)

    result = await app.run(
        "Create a specific rebalancing action plan:\n"
        "1. Exact positions to add and why\n"
        "2. Exact positions to reduce and why\n"
        "3. Target allocation percentages\n"
        "4. Implementation timeline (immediate vs gradual)\n"
        "5. What indicators would trigger further rebalancing?"
    )

    print("\n🎉 Sector Rotation Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    print("🚀 Advanced Stock Analysis Workflows")
    print("Demonstrating multi-step reasoning, tool chaining, and state management")
    print("=" * 80)

    import argparse

    parser = argparse.ArgumentParser(description="Advanced Stock Analysis Workflows")
    parser.add_argument(
        "--workflow",
        choices=["portfolio", "research", "comparison", "sector"],
        default="portfolio",
        help="Advanced workflow type to demonstrate"
    )

    args = parser.parse_args()
    workflow_choice = str(args.workflow)

    if workflow_choice == "portfolio":
        asyncio.run(demo_portfolio_workflow())
    elif workflow_choice == "research":
        asyncio.run(demo_investment_research_workflow())
    elif workflow_choice == "comparison":
        asyncio.run(demo_multi_stock_comparison_workflow())
    elif workflow_choice == "sector":
        asyncio.run(demo_sector_rotation_workflow())