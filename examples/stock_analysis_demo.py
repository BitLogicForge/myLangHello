"""Stock Analysis Demo - Demonstrating the new stock chatbot backend capabilities.

This script shows how the stock analysis tools work together in chains to provide
comprehensive investment insights and portfolio management.
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from main import AgentApp

load_dotenv()


async def demo_stock_analysis():
    """Demonstrate comprehensive stock analysis workflow."""

    print("🎯 Stock Analysis Chatbot Backend Demo")
    print("=" * 50)

    app = AgentApp()

    # Demo 1: Individual Stock Analysis (Tool Chaining)
    print("\n📊 Demo 1: Individual Stock Analysis")
    print("-" * 30)

    result: dict[str, object] | None = await app.run(
        "What's the current stock price of Apple (AAPL) and can you analyze its financial metrics?"
    )

    print("\n⏱️ Waiting 2 seconds before next demo...")
    await asyncio.sleep(2)

    # Demo 2: Technical Analysis
    print("\n📈 Demo 2: Technical Analysis")
    print("-" * 30)

    _ = await app.run(
        "Can you perform a technical analysis on Tesla (TSLA) with RSI, MACD, and Bollinger Bands?"
    )

    print("\n⏱️ Waiting 2 seconds before next demo...")
    await asyncio.sleep(2)

    # Demo 3: Risk Assessment
    print("\n🎯 Demo 3: Risk Assessment")
    print("-" * 30)

    _ = await app.run(
        "What's the risk profile of NVIDIA (NVDA)? Analyze its volatility and downside risk."
    )

    print("\n⏱️ Waiting 2 seconds before next demo...")
    await asyncio.sleep(2)

    # Demo 4: Stock Comparison
    print("\n📊 Demo 4: Stock Comparison")
    print("-" * 30)

    _ = await app.run(
        "Compare Apple (AAPL), Microsoft (MSFT), and Google (GOOGL) across financial metrics"
    )

    print("\n⏱️ Waiting 2 seconds before next demo...")
    await asyncio.sleep(2)

    # Demo 5: Market Overview
    print("\n🌍 Demo 5: Market Overview")
    print("-" * 30)

    _ = await app.run(
        "What's the current market overview? Show me major indices and sector performance."
    )

    print("\n⏱️ Waiting 2 seconds before next demo...")
    await asyncio.sleep(2)

    # Demo 6: Portfolio Creation and Management (Multi-step Chain)
    print("\n💼 Demo 6: Portfolio Management Chain")
    print("-" * 30)

    # Step 1: Create portfolio
    _ = await app.run(
        "Create a new investment portfolio called 'Tech Growth' with $10,000 initial capital"
    )

    print("\n⏱️ Waiting 2 seconds before adding stocks...")
    await asyncio.sleep(2)

    # Step 2: Add stocks to portfolio
    _ = await app.run(
        "Add 50 shares of Apple (AAPL) at $175 to the most recent portfolio"
    )

    print("\n⏱️ Waiting 2 seconds before adding more stocks...")
    await asyncio.sleep(2)

    _ = await app.run(
        "Add 20 shares of Microsoft (MSFT) at $378 to the portfolio"
    )

    print("\n⏱️ Waiting 2 seconds before analysis...")
    await asyncio.sleep(2)

    # Step 3: Analyze portfolio
    _ = await app.run(
        "Analyze the portfolio performance and provide recommendations"
    )

    print("\n⏱️ Waiting 2 seconds before risk assessment...")
    await asyncio.sleep(2)

    # Step 4: Risk assessment
    _ = await app.run(
        "Assess the risk level of my portfolio"
    )

    print("\n🎉 Demo Complete!")
    print("=" * 50)


async def demo_conversation_flow():
    """Demonstrate natural conversation flow for investment discussion."""

    print("\n🗣️ Investment Conversation Flow Demo")
    print("=" * 50)

    app = AgentApp()

    conversation = [
        "I'm interested in technology stocks for long-term investment. What do you recommend?",
        "Can you compare Apple and Microsoft for me?",
        "What about the risk levels of these stocks?",
        "If I wanted to create a balanced portfolio with $25,000, how should I allocate it?",
    ]

    for i, question in enumerate(conversation, 1):
        print(f"\n❓ Question {i}: {question}")
        print("-" * 40)
        _ = await app.run(question)

        if i < len(conversation):
            print("\n⏱️ Waiting 3 seconds before follow-up...")
            await asyncio.sleep(3)

    print("\n🎉 Conversation Demo Complete!")


async def quick_tools_demo():
    """Quick demonstration of individual tool capabilities."""

    print("\n🛠️ Quick Tools Demo")
    print("=" * 50)

    app = AgentApp()

    quick_demos = [
        "Get me a quick stock price quote for Amazon (AMZN)",
        "Show me the top market movers today",
        "What's the performance of major market indices?",
        "Compare the technical indicators for Meta (META) and Tesla (TSLA)",
    ]

    for i, question in enumerate(quick_demos, 1):
        print(f"\n⚡ Quick Demo {i}: {question}")
        print("-" * 40)
        _ = await app.run(question)

        if i < len(quick_demos):
            await asyncio.sleep(2)

    print("\n🎉 Quick Demo Complete!")


if __name__ == "__main__":
    print("🚀 Stock Analysis Chatbot Backend - Interactive Demo")
    print("This demo showcases the comprehensive stock analysis capabilities")
    print("=" * 70)

    # Choose which demo to run
    import argparse

    parser = argparse.ArgumentParser(description="Stock Analysis Demo")
    parser.add_argument(
        "--demo",
        choices=["full", "conversation", "quick"],
        default="full",
        help="Demo type to run"
    )

    args: argparse.Namespace = parser.parse_args()

    demo_choice: str = str(args.demo)

    if demo_choice == "full":
        asyncio.run(demo_stock_analysis())
    elif demo_choice == "conversation":
        asyncio.run(demo_conversation_flow())
    elif demo_choice == "quick":
        asyncio.run(quick_tools_demo())