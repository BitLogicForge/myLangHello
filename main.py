"""Refactored Agent Application with Single Responsibility Principle."""

import asyncio
import logging
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv

from config import Config
from services import (
    AgentConfigurator,
    AgentExecutionSettings,
    AgentRunner,
    StreamingOutputFormatter,
)
from utils import prepare_messages_with_history, setup_logging

# Load environment variables
load_dotenv()

# Configure logging
app_config = Config()
setup_logging(debug=bool(app_config.get("agent.debug", False)))
logger = logging.getLogger(__name__)


class AgentApp:
    """Main application orchestrator - coordinates all components."""

    def __init__(self):
        """Initialize the agent application."""
        logger.info("Initializing AgentApp...")

        self.config = app_config
        self.execution_settings = AgentExecutionSettings.from_config(self.config)

        # Create configurator and build agent
        configurator = AgentConfigurator()
        self.agent_executor = configurator.build_agent()
        self.agent_runner = AgentRunner(self.agent_executor, self.execution_settings)
        self.output_formatter = StreamingOutputFormatter()

        logger.info("✅ AgentApp initialized successfully")

    async def run(
        self, question: str, history: Optional[List[Tuple[str, str]]] = None
    ) -> Optional[dict]:
        """
        Run the agent with a question and optional conversation history.

        Args:
            question: User question/input
            history: Optional conversation history as list of (role, content) tuples
                    Example: [("user", "Hello"), ("assistant", "Hi!"), ...]

        Returns:
            Agent response dictionary
        """
        logger.info("Running agent with question...")
        logger.debug(f"Question: {question[:100]}...")

        # Prepare messages using shared utility
        messages = prepare_messages_with_history(question, history)
        if history:
            logger.info(f"Including {len(history)} history messages + current question")

        agent_input: dict[str, Any] = {"messages": messages}

        try:
            self.output_formatter.print_header()

            final_response = await self.agent_runner.run(
                agent_input,
                on_event=self.output_formatter.print_event,
            )

            logger.info("✅ Agent completed successfully")
            self.output_formatter.print_footer()

            return final_response
        except Exception as e:
            logger.error(f"❌ Agent execution failed: {str(e)}")
            raise


async def main() -> None:
    """Main entry point."""
    print("Hello, Function Calling Agent!")

    # Example question
    question = (
        "tell me weather in poznan today, and what date is today, and weather in london"
        "tell me their coordinetses"
        # "calculate 2+666*7, and convert 100 USD to EUR"
        # "list first 5 countries on letter B and their codes from db"
        # "then check weather for each country treating them as city"
        # "write it to file weather.txt"
        # "what is my name? do i have sibilings?"
        # "calculate loan for amount 25000 USD, term 5 years, interest rate 4.5 and convert to EUR"
        # "calculate loan payment for amount 25000 USD, term 5,7,8,10 years, interest rate 4.5"
        # "tell me 2 jokes, and format it"
        # "check avaiable views in db, plus i want 2 jokes , but funny ones"
        # "check avaiable views in db"
    )

    # Optional: Test with conversation history
    history: Optional[List[Tuple[str, str]]] = None
    # Uncomment to test with history:
    history = [
        ("user", "Hello, my name is John and i have sister Jane."),
        ("assistant", "Hi John! How can I help you today?"),
        ("user", "I have a question"),
        ("assistant", "Sure, I'd be happy to help. What's your question?"),
    ]

    app = AgentApp()
    await app.run(question=question, history=history)


if __name__ == "__main__":
    asyncio.run(main())
