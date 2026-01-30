"""Prompt Builder - Creates and configures prompt templates."""

import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import Config
from .file_utils import read_text_file

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds and configures prompt templates for the agent."""

    def __init__(self):
        """Initialize the prompt builder."""
        config = Config()
        self.system_prompt_path = config.get("system_prompt_path", "messages/system_prompt.txt")
        logger.debug(f"PromptBuilder initialized with prompt file: {self.system_prompt_path}")
        self.system_prompt = read_text_file(self.system_prompt_path)
        logger.info(f"System prompt prepared ({len(self.system_prompt)} chars)")

    def build_prompt(self) -> ChatPromptTemplate:
        """
        Build the complete prompt template.

        Returns:
            Configured ChatPromptTemplate
        """
        system_prompt = read_text_file(self.system_prompt_path)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        logger.info("Prompt template built successfully")
        return prompt
