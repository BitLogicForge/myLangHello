"""Refactored Agent Application with Single Responsibility Principle."""

import logging
from typing import Optional, List, Tuple
from dotenv import load_dotenv

from services import AgentConfigurator, StreamingOutputFormatter
from services.logging_config import setup_logging

# Load environment variables
load_dotenv()

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


class AgentApp:
    """Main application orchestrator - coordinates all components."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        llm_config_path: str = "llm_config.json",
        llm_provider: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the agent application.

        Args:
            model: LLM model name
            temperature: LLM temperature setting
            llm_config_path: Path to LLM configuration file (default: llm_config.json)
            llm_provider: Override LLM provider (azure or openai). If None, uses config file.
            **kwargs: Additional configuration options passed to AgentConfigurator
        """
        logger.info("Initializing AgentApp...")

        # Create configurator and build agent
        configurator = AgentConfigurator(
            model=model,
            temperature=temperature,
            llm_config_path=llm_config_path,
            llm_provider=llm_provider,
            **kwargs,
        )
        self.agent_executor = configurator.build_agent()
        self.output_formatter = StreamingOutputFormatter()

        logger.info("✅ AgentApp initialized successfully")

    def run(self, question: str, history: Optional[List[Tuple[str, str]]] = None) -> Optional[dict]:
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

        # Build messages with history
        messages: List[Tuple[str, str]] = []
        if history:
            messages.extend(history)
            logger.info(f"Including {len(history)} history messages")
        messages.append(("user", question))

        try:
            self.output_formatter.print_header()

            # Stream the agent execution to see thoughts in real-time
            step_count = 0
            final_response: Optional[dict] = None
            tool_timings: dict[str, float] = {}

            for event in self.agent_executor.stream(
                {"messages": messages}, config={"recursion_limit": 15}  # type: ignore
            ):
                step_count += 1
                self.output_formatter.print_event(event, step_count, tool_timings)
                final_response = event

            logger.info("✅ Agent completed successfully")
            self.output_formatter.print_footer()

            return final_response
        except Exception as e:
            logger.error(f"❌ Agent execution failed: {str(e)}")
            raise


def main():
    """Main entry point."""
    print("Hello, Function Calling Agent!")

    # Example question
    question = (
        # "tell me weather in poznan today, and what date is today, and weather in london"
        # "calculate loan for amount 25000 USD, term 5 years, interest rate 4.5 and convert to EUR"
        # "calculate loan payment for amount 25000 USD, term 5,7,8,10 years, interest rate 4.5"
        "tell me a joke, and format it"
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

    app = AgentApp(llm_provider="openai")
    app.run(question=question, history=history)


if __name__ == "__main__":
    main()
