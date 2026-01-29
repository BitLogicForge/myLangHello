"""Prompt Builder - Creates and configures prompt templates."""

import json
import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Optional

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds and configures prompt templates for the agent."""

    def __init__(self, system_prompt_path: str = "messages/system_prompt.txt"):
        """
        Initialize the prompt builder.

        Args:
            system_prompt_path: Path to system prompt file
        """
        self.system_prompt_path = system_prompt_path
        logger.debug(f"PromptBuilder initialized with prompt file: {system_prompt_path}")

    def build_prompt(self, custom_schema: Optional[dict] = None) -> ChatPromptTemplate:
        """
        Build the complete prompt template.

        Args:
            custom_schema: Optional custom schema information

        Returns:
            Configured ChatPromptTemplate
        """
        logger.debug("Building prompt template...")
        system_prompt = self._load_system_prompt()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        if custom_schema:
            logger.debug(f"Adding custom schema with {len(custom_schema)} tables")
            schema_context = self._format_schema_context(custom_schema)
            prompt = prompt.partial(schema_context=schema_context)

        logger.info("Prompt template built successfully")
        return prompt

    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        logger.debug(f"Loading system prompt from {self.system_prompt_path}")
        try:
            with open(self.system_prompt_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"System prompt loaded ({len(content)} chars)")
                return content
        except FileNotFoundError:
            logger.error(f"System prompt file not found: {self.system_prompt_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")
            raise

    @staticmethod
    def _format_schema_context(schema: dict) -> str:
        """
        Format schema information for the prompt as JSON.

        Args:
            schema: Schema dictionary with structure:
                   {table_name: {description: str, columns: {col_name: col_desc}}}

        Returns:
            Formatted schema context as JSON string with explanation
        """
        schema_json = json.dumps(schema, indent=2)
        return (
            "Database Schema (JSON format):\n"
            "Each table has a 'description' and 'columns' dictionary.\n"
            "Column names map to their descriptions.\n\n"
            f"{schema_json}"
        )
