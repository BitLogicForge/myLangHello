"""Agent Factory - Creates and configures the agent using LangGraph."""

import logging
from typing import Optional, Any
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating and configuring LangGraph ReAct agents."""

    def __init__(
        self,
        llm: AzureChatOpenAI,
        tools: list,
        prompt: ChatPromptTemplate,
        verbose: bool = True,
        max_iterations: int = 10,
        max_execution_time: int = 60,
        checkpointer: Optional[Any] = None,
    ):
        """
        Initialize the agent factory with LangGraph.

        Args:
            llm: Language model instance
            tools: List of tools for the agent
            prompt: Prompt template
            verbose: Enable verbose output (for backward compatibility)
            max_iterations: Maximum agent iterations (Note: LangGraph uses recursion_limit)
            max_execution_time: Maximum execution time in seconds (for backward compatibility)
            checkpointer: Optional memory checkpointer for conversation persistence
        """
        self.llm = llm
        self.tools = tools
        self.prompt = prompt
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time
        self.checkpointer = checkpointer
        logger.info(
            f"AgentFactory initialized with {len(tools)} tools using LangGraph ReAct agent, "
            f"recursion_limit={max_iterations}"
        )

    def create_agent(self):
        """
        Create a LangGraph ReAct agent.

        Returns:
            LangGraph agent graph that can be invoked
        """
        logger.debug("Creating LangGraph ReAct agent...")

        # Create the ReAct agent with the updated langchain.agents API
        # Moved from langgraph.prebuilt to langchain.agents
        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.checkpointer,  # Enable memory if checkpointer provided
        )

        logger.info("LangGraph ReAct agent created successfully")
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
