"""Interactive Conversation Demo - Showing multi-turn context and conversational AI.

This demonstrates the chatbot's ability to:
- Maintain conversation context across multiple turns
- Build on previous responses
- Handle follow-up questions
- Remember user preferences and portfolio state
- Provide natural, conversational financial guidance
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from main import AgentApp

load_dotenv()


async def demo_natural_investment_conversation():
    """Demonstrate natural, multi-turn investment conversation."""

    print("💬 Natural Investment Conversation Demo")
    print("=" * 70)

    app = AgentApp()

    conversation = [
        # Opening - User expresses interest
        "Hi! I'm interested in starting to invest in individual stocks. "
        "I have $15,000 to invest and I'm looking at tech stocks. What do you think?",

        # Follow-up - User seeking clarification
        "You mentioned risk management. Can you explain what that means for "
        "someone like me who's new to individual stock investing?",

        # Action request - User wants to build portfolio
        "That makes sense. Let's start building a diversified portfolio. "
        "Create a portfolio called 'My First Stocks' with the full $15,000.",

        # Step-by-step building
        "Good, it's created. Now let's add some positions. I want to start with "
        "50 shares of Apple at $175. Why is Apple a good starting point?",

        # Follow-up question
        "That's helpful. What's the current risk level of my portfolio so far?",

        # Adding more positions with reasoning
        "I want to add Microsoft too. What's a good number of shares to add at $378 "
        "to keep my portfolio balanced, and why?",

        # Portfolio review
        "Now that I have both Apple and Microsoft, how does my portfolio look? "
        "Am I diversified enough?",

        # Risk concern
        "I'm a bit worried about having everything in tech. Should I add something "
        "from a different sector? What would you recommend?",

        # Learning moment
        "That's interesting about Johnson & Johnson. Before I add it, can you show me "
        "a comparison of the risk metrics between my current tech stocks and J&J?",

        # Final portfolio decision
        "Based on all that we've discussed, what should my final portfolio allocation be "
        "to balance growth and safety for a 5-year investment timeline?",

        # Exit planning and future monitoring
        "Great! Now that we've built this portfolio, what indicators should I watch "
        "to know when to buy more or sell positions? How often should I review this?"
    ]

    for i, message in enumerate(conversation, 1):
        print(f"\n🗣️  Turn {i}: {message[:60]}...")
        print("-" * 60)

        result = await app.run(message)

        # Show conversation progression
        if i < len(conversation):
            print(f"\n⏳ Continuing conversation in 2 seconds...")
            await asyncio.sleep(2)

    print("\n🎉 Natural Conversation Complete!")
    print("=" * 70)
    print("This demo showcased:")
    print("• Multi-turn context maintenance")
    print("• Follow-up question handling")
    print("• Progressive portfolio building")
    print("• Educational investment guidance")
    print("• Long-term planning and monitoring")


async def demo_problem_solving_conversation():
    """Demonstrate problem-solving conversation around investment challenges."""

    print("\n🔧 Investment Problem-Solving Conversation")
    print("=" * 70)

    app = AgentApp()

    conversation = [
        # Problem statement
        "I have a problem. My portfolio is down 12% this year and I'm not sure what to do. "
        "Can you help me analyze what's wrong?",

        # User provides more context
        "My portfolio consists of Tesla (TSLA), NVIDIA (NVDA), and Meta (META). "
        "I bought them all last year when tech was doing really well.",

        # Diagnostic phase
        "Can you analyze each of these stocks individually to see which ones are "
        "dragging down my performance?",

        # Solution exploration
        "Based on that analysis, what should I do? Should I sell some of these stocks "
        "or hold onto them? What are the pros and cons?",

        # Risk management focus
        "I'm concerned about protecting my remaining capital. What defensive strategies "
        "can I use to limit further losses?",

        # Recovery planning
        "If I want to recover my losses over the next 2 years, what allocation changes "
        "would you recommend? Should I add any safer stocks?",

        # Action planning
        "Let's create a specific recovery plan. What should I sell, what should I buy, "
        "and what should trigger these decisions?",

        # Monitoring and adjustment
        "How will I know if my recovery plan is working? What metrics should I track "
        "and how often should I review my progress?"
    ]

    for i, message in enumerate(conversation, 1):
        print(f"\n🔍 Problem-Solving Turn {i}")
        print("-" * 40)
        print(f"User: {message[:80]}...")

        result = await app.run(message)

        if i < len(conversation):
            print(f"\n⏳ Continuing analysis in 2 seconds...")
            await asyncio.sleep(2)

    print("\n✅ Problem-Solving Conversation Complete!")
    print("=" * 70)


async def demo_learning_conversation():
    """Demonstrate educational conversation about investment concepts."""

    print("\n📚 Investment Learning Conversation")
    print("=" * 70)

    app = AgentApp()

    learning_modules = [
        # Concept introduction
        "I want to understand technical analysis better. Can you explain what "
        "technical indicators are and how they differ from fundamental analysis?",

        # Practical demonstration
        "That's helpful. Can you show me practical examples of the most important "
        "technical indicators by analyzing Apple stock using each one?",

        # Interpretation learning
        "Looking at those Apple technical indicators, how would I know if it's a "
        "good time to buy or sell? Walk me through the decision process.",

        # Risk management learning
        "Now I want to learn about position sizing. If I have $10,000 and moderate "
        "risk tolerance, how should I allocate it across different stocks?",

        # Portfolio construction learning
        "Let's practice building a diversified portfolio. Show me how to construct "
        "a balanced portfolio using stocks from different sectors.",

        # Advanced concepts
        "I've heard about beta and correlation. Can you explain these concepts "
        "and show me how they affect portfolio risk?",

        # Practical application
        "Let's put it all together. If I wanted to build a low-risk portfolio "
        "with $20,000, what would be the optimal allocation and why?"
    ]

    for i, question in enumerate(learning_modules, 1):
        print(f"\n📖 Learning Module {i}")
        print("-" * 30)
        print(f"Topic: {question[:60]}...")

        result = await app.run(question)

        if i < len(learning_modules):
            print(f"\n⏳ Moving to next concept in 2 seconds...")
            await asyncio.sleep(2)

    print("\n🎓 Learning Conversation Complete!")
    print("=" * 70)
    print("This demo showcased:")
    print("• Progressive learning from basics to advanced")
    print("• Practical demonstrations of concepts")
    print("• Real-world application of knowledge")
    print("• Interactive educational guidance")


if __name__ == "__main__":
    print("🗣️  Interactive Conversation Demos")
    print("Demonstrating natural language understanding and context management")
    print("=" * 80)

    import argparse

    parser = argparse.ArgumentParser(description="Interactive Conversation Demos")
    parser.add_argument(
        "--conversation",
        choices=["natural", "problem-solving", "learning"],
        default="natural",
        help="Type of conversation to demonstrate"
    )

    args = parser.parse_args()
    conversation_type = str(args.conversation)

    if conversation_type == "natural":
        asyncio.run(demo_natural_investment_conversation())
    elif conversation_type == "problem-solving":
        asyncio.run(demo_problem_solving_conversation())
    elif conversation_type == "learning":
        asyncio.run(demo_learning_conversation())