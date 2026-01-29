"""Example: Using LangGraph ReAct Agent - State-of-the-Art Approach

This example demonstrates how to use the new LangGraph-based agent
which provides better performance, built-in memory, and streaming support.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from services import DatabaseManager, ToolsManager, PromptBuilder, AgentFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def main():
    """Run the LangGraph ReAct agent example."""
    logger.info("Starting LangGraph ReAct Agent Example...")

    # Initialize components
    db_uri = os.getenv(
        "DATABASE_URI", "mssql+pyodbc://server/database?driver=ODBC+Driver+17+for+SQL+Server"
    )

    # Setup database
    db_manager = DatabaseManager(db_uri)
    db = db_manager.get_database()

    # Setup LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)

    # Setup tools
    tools_manager = ToolsManager(db=db, llm=llm, enable_sql_tool=True)
    tools = tools_manager.get_tools()

    # Setup prompt
    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build_prompt()

    # Create LangGraph agent
    agent_factory = AgentFactory(
        llm=llm,
        tools=tools,
        prompt=prompt,
        max_iterations=15,
    )

    # Get the agent (this is a CompiledGraph from LangGraph)
    agent = agent_factory.create_executor()

    logger.info("\n" + "=" * 60)
    logger.info("LangGraph ReAct Agent Ready!")
    logger.info("=" * 60 + "\n")

    # Example 1: Simple query
    print("\n--- Example 1: Simple Calculation ---")
    result = agent.invoke(
        {"messages": [("user", "What is 25 * 17 + 42?")]}, config={"recursion_limit": 15}
    )
    print(f"Result: {result['messages'][-1].content}")

    # Example 2: Tool usage
    print("\n--- Example 2: Weather Check ---")
    result = agent.invoke(
        {"messages": [("user", "What's the weather in New York?")]}, config={"recursion_limit": 15}
    )
    print(f"Result: {result['messages'][-1].content}")

    # Example 3: File operations
    print("\n--- Example 3: List Directory ---")
    result = agent.invoke(
        {"messages": [("user", "List the files in the current directory")]},
        config={"recursion_limit": 15},
    )
    print(f"Result: {result['messages'][-1].content}")

    # Example 4: With memory (thread-based conversation)
    print("\n--- Example 4: Conversation with Memory ---")
    thread_id = "conversation-1"

    # First message
    result = agent.invoke(
        {"messages": [("user", "My name is Alice and I live in Boston")]},
        config={"recursion_limit": 15, "configurable": {"thread_id": thread_id}},
    )
    print(f"Agent: {result['messages'][-1].content}")

    # Follow-up message (agent should remember context)
    result = agent.invoke(
        {"messages": [("user", "What's my name and where do I live?")]},
        config={"recursion_limit": 15, "configurable": {"thread_id": thread_id}},
    )
    print(f"Agent: {result['messages'][-1].content}")

    logger.info("\n" + "=" * 60)
    logger.info("Examples completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
