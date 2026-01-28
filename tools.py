from langchain_core.tools import tool
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

# Define tools with structured schemas for function calling


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
