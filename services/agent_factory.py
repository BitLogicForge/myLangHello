"""Agent Factory - Creates and configures the agent using LangGraph."""

import logging
from typing import Optional, Any
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from config import Config

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating and configuring LangGraph ReAct agents."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list,
        system_prompt: str,
        checkpointer: Optional[Any] = None,
    ):
        """
        Initialize the agent factory with LangGraph.

        Args:
            llm: Language model instance
            tools: List of tools for the agent
            system_prompt: System prompt text to guide agent behavior
            checkpointer: Optional memory checkpointer for conversation persistence
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer

        # Read agent settings from config
        config = Config()
        self.verbose = config.get("agent.verbose", True)
        self.max_iterations = config.get("agent.max_iterations", 10)
        self.max_execution_time = config.get("agent.max_execution_time", 60)

        logger.info(
            f"AgentFactory initialized with {len(tools)} tools using LangGraph ReAct agent, "
            f"recursion_limit={self.max_iterations}"
        )

    def create_agent(self):
        """
        Create a LangGraph ReAct agent.

        Returns:
            LangGraph agent graph that can be invoked
        """
        logger.debug("Creating LangGraph ReAct agent...")

        # Create the ReAct agent with LangGraph

        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.checkpointer,
        )
        return agent

    def create_executor(self):
        """
        Create the agent executor (LangGraph agent).

        Returns:
            LangGraph agent (CompiledGraph) that can be invoked

        Note:
            The LangGraph agent is invoked differently than AgentExecutor:
            - Use: agent.invoke({"messages": [("user", "your question")]})
            - With memory: agent.invoke(
                {"messages": [...]},
                config={"configurable": {"thread_id": "1"}}
              )
        """
        logger.debug("Creating LangGraph agent executor...")
        agent = self.create_agent()
        logger.info("Agent executor (LangGraph) created successfully")
        return agent

    def invoke(self, input_text: str, config: Optional[RunnableConfig] = None):
        """
        Convenience method to invoke the agent with a simple text input.

        Args:
            input_text: The user's input message
            config: Optional configuration (e.g., for thread_id in memory)

        Returns:
            Agent response
        """
        agent = self.create_executor()

        # Configure recursion limit
        if config is None:
            config = {"recursion_limit": self.max_iterations}
        else:
            # Merge with provided config
            config = {**config, "recursion_limit": self.max_iterations}

        # Invoke with messages format (standard LangGraph format)
        result = agent.invoke({"messages": [("user", input_text)]}, config=config)  # type: ignore

        return result
