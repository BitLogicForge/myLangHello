"""Agent Factory - Creates and configures the agent and executor."""

import logging
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating and configuring agents."""

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: list,
        prompt: ChatPromptTemplate,
        verbose: bool = True,
        max_iterations: int = 10,
        max_execution_time: int = 60,
    ):
        """
        Initialize the agent factory.

        Args:
            llm: Language model instance
            tools: List of tools for the agent
            prompt: Prompt template
            verbose: Enable verbose output
            max_iterations: Maximum agent iterations
            max_execution_time: Maximum execution time in seconds
        """
        self.llm = llm
        self.tools = tools
        self.prompt = prompt
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time
        logger.info(
            f"AgentFactory initialized with {len(tools)} tools, "
            f"max_iterations={max_iterations}, max_execution_time={max_execution_time}s"
        )

    def create_agent(self):
        """Create the agent."""
        logger.debug("Creating React agent with LangGraph...")

        # Extract system message from prompt template for state_modifier
        system_message = None
        if hasattr(self.prompt, "messages") and len(self.prompt.messages) > 0:
            # Get the first message if it's a system message
            first_message = self.prompt.messages[0]
            if hasattr(first_message, "prompt") and hasattr(first_message.prompt, "template"):
                system_message = first_message.prompt.template

        # Create agent with LangGraph
        agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            state_modifier=system_message,
        )
        logger.info("Agent created successfully with LangGraph")
        return agent

    def create_executor(self):
        """
        Create the agent executor (returns LangGraph agent).

        Returns:
            LangGraph CompiledStateGraph (agent)
        """
        logger.debug("Creating agent executor...")
        agent = self.create_agent()
        logger.info("Agent executor created successfully")
        return agent
