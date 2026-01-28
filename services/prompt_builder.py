"""Prompt Builder - Creates and configures prompt templates."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Optional


class PromptBuilder:
    """Builds and configures prompt templates for the agent."""

    def __init__(self, system_prompt_path: str = "messages/system_prompt.txt"):
        """
        Initialize the prompt builder.

        Args:
            system_prompt_path: Path to system prompt file
        """
        self.system_prompt_path = system_prompt_path

    def build_prompt(self, custom_schema: Optional[dict] = None) -> ChatPromptTemplate:
        """
        Build the complete prompt template.

        Args:
            custom_schema: Optional custom schema information

        Returns:
            Configured ChatPromptTemplate
        """
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
            schema_context = self._format_schema_context(custom_schema)
            return prompt.partial(schema_context=schema_context)

        return prompt

    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        with open(self.system_prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _format_schema_context(schema: dict) -> str:
        """
        Format schema information for the prompt.

        Args:
            schema: Schema dictionary

        Returns:
            Formatted schema context string
        """
        return "\n".join([f"- {table}: {desc}" for table, desc in schema.items()])
