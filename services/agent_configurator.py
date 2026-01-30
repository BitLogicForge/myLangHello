"""Agent configuration and initialization service."""

import logging
from typing import Optional, Any

from config import Config
from .tools_manager import ToolsManager
from .prompt_builder import PromptBuilder
from .agent_factory import AgentFactory
from .llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class AgentConfigurator:
    """Handles agent initialization and component setup."""

    def __init__(self) -> None:
        """Initialize the agent configurator."""
        self.config = Config()

        # Component storage
        self.llm: Optional[Any] = None
        self.tools_manager: Optional[ToolsManager] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.system_prompt: Optional[Any] = None
        self.agent_factory: Optional[AgentFactory] = None

    def build_agent(self) -> Any:
        """
        Build the complete agent executor by initializing all components in order.

        Returns:
            Agent executor instance
        """
        logger.info("Building agent...")

        # Initialize components in correct order
        logger.info("Initializing LLM")
        self.llm = LLMFactory.create_llm()

        logger.info("Setting up tools manager...")
        self.tools = ToolsManager().get_tools()

        logger.info("Building system prompt...")

        self.system_prompt = PromptBuilder().system_prompt

        self.setup_agent_factory()

        # Create and return executor
        if self.agent_factory is None:
            raise RuntimeError("Agent factory creation failed")

        agent_executor = self.agent_factory.create_executor()
        logger.info("✅ Agent built successfully")
        return agent_executor

    def setup_agent_factory(self) -> AgentFactory:
        """
        Create the agent factory.

        Returns:
            AgentFactory instance
        """
        if self.llm is None:
            raise RuntimeError("LLM must be initialized. ")
        if self.tools is None:
            raise RuntimeError("Tools  must be initialized. ")
        if self.system_prompt is None:
            raise RuntimeError("System prompt must be initialized. ")

        logger.info("Creating agent factory...")

        self.agent_factory = AgentFactory(
            llm=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )
        return self.agent_factory
