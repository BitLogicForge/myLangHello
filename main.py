"""Refactored Agent Application with Single Responsibility Principle."""

import logging

from dotenv import load_dotenv

from config import Config
from services import (
    AgentConfigurator,
    AgentExecutionSettings,
    AgentRunner,
    StreamingOutputFormatter,
    SupportsAStream,
)
from utils import prepare_messages_with_history, setup_logging

# Load environment variables
_ = load_dotenv()

# Configure logging
app_config = Config()
setup_logging(debug=bool(app_config.get("agent.debug", False)))
logger = logging.getLogger(__name__)


class AgentApp:
    """Main application orchestrator - coordinates all components."""

    config: Config
    execution_settings: AgentExecutionSettings
    agent_executor: SupportsAStream
    agent_runner: AgentRunner
    output_formatter: StreamingOutputFormatter

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
        self, question: str, history: list[tuple[str, str]] | None = None
    ) -> dict[str, object] | None:
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

        agent_input: dict[str, object] = {"messages": messages}

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
