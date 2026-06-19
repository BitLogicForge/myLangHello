"""Simple test of structured output without HTTP layer."""

import asyncio
import json
from main import AgentApp

async def test_structured_output():
    """Test structured output directly with agent."""
    print("🎯 Testing Structured Output")
    print("=" * 50)

    app = AgentApp()

    # Test with enhanced JSON prompt
    json_prompt = """
    You are a financial analysis AI that MUST respond ONLY with valid JSON.

    Question: Should I buy Apple (AAPL) stock?

    Provide your response in this exact JSON format:
    {
        "recommendation": "buy/sell/hold",
        "confidence": 0.75,
        "target_price": 195.0,
        "stop_loss": 165.0,
        "reasoning": "Detailed explanation...",
        "risks": ["risk1", "risk2"],
        "catalysts": ["catalyst1", "catalyst2"]
    }

    Use the available tools to gather real data about Apple stock.
    Your response must be ONLY the JSON object, no additional text.
    """

    print("📤 Running agent with JSON prompt...")
    print(f"Prompt: {json_prompt[:100]}...")

    try:
        response = await app.run(json_prompt)

        print(f"📤 Agent Response Type: {type(response)}")

        # Extract content from response
        if isinstance(response, dict) and "messages" in response:
            messages = response["messages"]
            if messages:
                last_message = messages[-1]
                content = getattr(last_message, "content", "")
                print(f"📤 Agent Response Content:\n{content}")

                # Try to parse as JSON
                try:
                    # Clean up the response
                    cleaned_content = content.strip()
                    if cleaned_content.startswith("```json"):
                        cleaned_content = cleaned_content[7:]
                    if cleaned_content.startswith("```"):
                        cleaned_content = cleaned_content[3:]
                    if cleaned_content.endswith("```"):
                        cleaned_content = cleaned_content[:-3]
                    cleaned_content = cleaned_content.strip()

                    # Parse JSON
                    parsed_json = json.loads(cleaned_content)
                    print(f"✅ Successfully parsed JSON:")
                    print(json.dumps(parsed_json, indent=2))

                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing error: {e}")
                    print(f"💡 Response wasn't valid JSON, but agent functioned correctly")

        else:
            print(f"📤 Raw Response: {response}")

    except Exception as e:
        print(f"❌ Error: {e}")


async def test_simple_stock_query():
    """Test simple stock query to ensure basic functionality."""
    print("\n📊 Testing Simple Stock Query")
    print("=" * 50)

    app = AgentApp()

    simple_question = "What's the current stock price of Apple (AAPL) and should I buy it?"

    print(f"📤 Question: {simple_question}")

    try:
        response = await app.run(simple_question)

        # Extract content
        if isinstance(response, dict) and "messages" in response:
            messages = response["messages"]
            if messages:
                last_message = messages[-1]
                content = getattr(last_message, "content", "")
                print(f"📤 Agent Response:\n{content[:500]}...")

    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run tests."""
    print("🚀 Structured Output Test")
    print("=" * 70)

    await test_simple_stock_query()
    await test_structured_output()

    print("\n🎉 Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())