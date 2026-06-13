"""Agent configuration and initialization service."""

from collections.abc import AsyncIterable
import logging
from typing import Callable, Protocol, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from config import Config

from .agent_factory import AgentFactory
from .llm_factory import LLMFactory
from .prompt_builder import PromptBuilder
from .tools_manager import ToolsManager

logger = logging.getLogger(__name__)


# MARK: Protocols
class SupportsAStream(Protocol):
    """Protocol for objects supporting asynchronous event streaming."""

    @property
    def astream(self) -> Callable[..., AsyncIterable[dict[str, object]]]:
        ...


# MARK: Agent Configurator
class AgentConfigurator:
    """Handles agent initialization and component setup."""

    config: Config
    llm: BaseChatModel | None
    tools_manager: ToolsManager | None
    tools: list[BaseTool] | None
    prompt_builder: PromptBuilder | None
    system_prompt: str | None
    agent_factory: AgentFactory | None

    def __init__(self) -> None:
        """Initialize the agent configurator."""
        self.config = Config()

        # Component storage
        self.llm = None
        self.tools_manager = None
        self.tools = None
        self.prompt_builder = None
        self.system_prompt = None
        self.agent_factory = None

    def build_agent(self) -> SupportsAStream:
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

        _ = self.setup_agent_factory()

        # Create and return executor
        if self.agent_factory is None:
            raise RuntimeError("Agent factory creation failed")

        agent_executor = cast(SupportsAStream, self.agent_factory.create_db_agent())
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
