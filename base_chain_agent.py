from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType

from base_chain_tools import (
    calculator_tool,
    weather_tool,
    read_file_tool,
    write_file_tool,
    http_get_tool,
    list_dir_tool,
    system_info_tool,
    random_joke_tool,
)

load_dotenv()


# Small wrapper functions exposed as LangChain tools. They keep simple string
# signatures so agents can call them easily.
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result as a string."""
    return calculator_tool(expression)


@tool
def weather(city: str) -> str:
    """Return a fake weather report for the given city."""
    return weather_tool(city)


@tool
def read_file(path: str) -> str:
    """Read a file and return its contents or an error message."""
    return read_file_tool(path)


@tool
def write_file(spec: str) -> str:
    """Write content to a file. Spec: first line is path, rest is content."""
    return write_file_tool(spec)


@tool
def http_get(url: str) -> str:
    """Perform an HTTP GET and return a short summary/result."""
    return http_get_tool(url)


@tool
def list_dir(path: str = ".") -> str:
    """List files in a directory and return a newline-separated listing."""
    return list_dir_tool(path)


@tool
def system_info(query: str = "") -> str:
    """Return basic system information. The query parameter is ignored."""
    return system_info_tool()


@tool
def random_joke(query: str = "") -> str:
    """Return a small, harmless random joke. The query parameter is ignored."""
    return random_joke_tool()


def main():
    print("Hello, state-of-the-art agent!")
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

    # Register tools
    tools = [
        calculator,
        weather,
        read_file,
        write_file,
        http_get,
        list_dir,
        system_info,
        random_joke,
    ]

    # Initialize a zero-shot agent that can use tools
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Example question demonstrating multiple utilities
    question = (
        "What is 5 + 7? Also, what's the weather in Paris? "
        "Show me a joke and list the current directory."
    )

    # The agent will decide which tool(s) to use automatically
    response = agent.run(question)

    print("\nFinal Agent Response:")
    print(response)


if __name__ == "__main__":
    main()
