"""Agent Factory - Creates and configures the agent and executor."""

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


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

    def create_agent(self):
        """Create the agent."""
        return create_openai_functions_agent(self.llm, self.tools, self.prompt)

    def create_executor(self) -> AgentExecutor:
        """
        Create the agent executor.

        Returns:
            Configured AgentExecutor
        """
        agent = self.create_agent()

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=self.max_iterations,
            max_execution_time=self.max_execution_time,
        )
