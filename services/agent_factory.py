"""Agent Factory - Creates and configures the agent using LangGraph."""

import logging
from typing import Optional, Any
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel


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

        logger.info(f"AgentFactory initialized with {len(tools)} tools using LangGraph ReAct agent")

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
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
        )
        return agent
