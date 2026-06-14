"""Example demonstrating how to run the main agent application."""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from main import AgentApp

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))


_ = load_dotenv()


async def main() -> None:
    """Main entry point."""
    print("Hello, Function Calling Agent!")

    # Example question
    question = (
        # "tell me weather in poznan today, and what date is today, and weather in london"
        # "tell me their coordinetses"
        # "calculate 2+666*7, and convert 100 USD to EUR"
        # "list first 5 countries on letter B and their codes from db"
        # "then check weather for each country treating them as city"
        # "write it to file weather.txt"
        # "what is my name? do i have sibilings?"
        # "calculate loan for amount 25000 USD, term 5 years, interest rate 4.5 and convert to EUR"
        # "calculate loan payment for amount 25000 USD, term 5,7,8,10 years, interest rate 4.5"
        # "tell me 2 jokes, and format it"
        # "check avaiable views in db, plus i want 2 jokes , but funny ones"
        "check avaiable views in db"
    )

    # Optional: Test with conversation history
    history: list[tuple[str, str]] | None = None
    # Uncomment to test with history:
    history = [
        ("user", "Hello, my name is John and i have sister Jane."),
        ("assistant", "Hi John! How can I help you today?"),
        ("user", "I have a question"),
        ("assistant", "Sure, I'd be happy to help. What's your question?"),
    ]

    app = AgentApp()
    _ = await app.run(question=question, history=history)


if __name__ == "__main__":
    asyncio.run(main())
