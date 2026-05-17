"""Refactored Agent Application with Single Responsibility Principle."""

import logging
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv

from config import Config
from services import (
    AgentConfigurator,
    AgentExecutionSettings,
    AgentRunner,
    DiscussionOrchestrator,
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

        configurator = AgentConfigurator()
        self.agent_executor = configurator.build_agent()
        self.llm = configurator.llm
        self.system_prompt = configurator.system_prompt
        self.tools = getattr(configurator, "tools", None)

        if self.llm is None or self.system_prompt is None:
            raise RuntimeError("Discussion mode requires configured llm and system prompt")

        self.agent_runner = AgentRunner(self.agent_executor, self.execution_settings)
        self.discussion_orchestrator = DiscussionOrchestrator(self.llm, self.system_prompt)
        self.output_formatter = StreamingOutputFormatter()

        logger.info("AgentApp initialized successfully")

    def run(self, question: str, history: Optional[List[Tuple[str, str]]] = None) -> Optional[dict]:
        """
        Run the standard tool-using agent with a question and optional conversation history.

        Args:
            question: User question/input
            history: Optional conversation history as list of (role, content) tuples

        Returns:
            Agent response dictionary
        """
        logger.info("Running agent with question...")
        logger.debug("Question: %s...", question[:100])

        messages = prepare_messages_with_history(question, history)
        if history:
            logger.info("Including %s history messages + current question", len(history))

        agent_input: dict[str, Any] = {"messages": messages}

        try:
            self.output_formatter.print_header()

            final_response = self.agent_runner.run(
                agent_input,
                on_event=self.output_formatter.print_event,
            )

            logger.info("Agent completed successfully")
            self.output_formatter.print_footer()

            return final_response
        except Exception:
            logger.exception("Agent execution failed")
            raise

    def run_discussion(
        self,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None,
        rounds: int = 2,
    ) -> dict[str, Any]:
        """Run the fun multi-agent discussion mode."""
        logger.info("Running discussion mode with %s rounds", rounds)
        self.output_formatter.print_discussion_header(rounds)
        result = self.discussion_orchestrator.run(
            question=question,
            history=history,
            rounds=rounds,
        )
        transcript = result.get("transcript", [])
        if isinstance(transcript, list):
            for turn in transcript:
                self.output_formatter.print_discussion_turn(
                    round_number=turn.round_number,
                    speaker=turn.speaker,
                    role=turn.role,
                    content=turn.content,
                )
        self.output_formatter.print_discussion_summary(str(result.get("output", "")))
        self.output_formatter.print_footer()
        return result


def main() -> None:
    """Main entry point."""
    print("Hello, Function Calling Agent!")

    question = (
        "I am preparing a short demo for teammates and I want it to feel more advanced. "
    )

    history: Optional[List[Tuple[str, str]]] = [
        # ("user", "I am preparing a short demo for teammates and I want it to feel more advanced."),
        # (
        #     "assistant",
        #     "That makes sense. We should focus on features that are easy to explain and visually clear.",
        # ),
        # (
        #     "user",
        #     "I do not want a huge refactor. I only have around 2 days and I want something fun but believable.",
        # ),
        # (
        #     "assistant",
        #     "Understood. We should balance wow factor, implementation speed, and demo reliability.",
        # ),
    ]

    app = AgentApp()
    app.run_discussion(question=question, history=history, rounds=2)


if __name__ == "__main__":
    main()
