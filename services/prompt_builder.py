"""Prompt Builder - Creates and configures prompt templates."""

import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import Config
from utils import read_text_file

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds and configures prompt templates for the agent."""

    def __init__(self) -> None:
        """Initialize the prompt builder."""
        config = Config()
        self.system_prompt_path = config.get("system_prompt_path")
        logger.debug(f"PromptBuilder initialized with prompt file: {self.system_prompt_path}")
        self._load_system_prompt()

    def _load_system_prompt(self) -> None:
        """Load system prompt from file with error handling."""
        try:
            self.system_prompt = read_text_file(self.system_prompt_path)
            logger.info(f"System prompt loaded ({len(self.system_prompt)} chars)")
        except FileNotFoundError:
            logger.error(f"System prompt file not found: {self.system_prompt_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load system prompt from {self.system_prompt_path}: {e}")
            raise

    def build_prompt(self) -> ChatPromptTemplate:
        """
        Build the complete prompt template.

        Returns:
            Configured ChatPromptTemplate
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        logger.info("Prompt template built successfully")
        return prompt
