"""Agent Factory - Creates and configures the agent and executor."""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

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
        """Create the agent using OpenAI tool calling."""
        logger.debug("Creating agent with OpenAI tool calling...")

        # Bind tools to the LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        # Create the agent runnable
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
            }
            | self.prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )

        logger.info("Agent created successfully with OpenAI tool calling")
        return agent

    def create_executor(self) -> AgentExecutor:
        """
        Create the agent executor.

        Returns:
            Configured AgentExecutor
        """
        logger.debug("Creating agent executor...")
        agent = self.create_agent()

        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=self.max_iterations,
            max_execution_time=self.max_execution_time,
        )
        logger.info("Agent executor created successfully")
        return executor
