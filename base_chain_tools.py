from dotenv import load_dotenv

import os
import requests
import datetime
import platform
import random

load_dotenv()


# Basic utility tools for agents or manual use
def calculator_tool(expression: str) -> str:
    """Evaluate a simple math expression and return the result as a string.

    This uses a restricted eval environment to avoid exposing builtins.
    """
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def weather_tool(city: str) -> str:
    """Return a fake weather report for the given city (placeholder)."""
    return f"The weather in {city} is sunny and 25°C."


def read_file_tool(path: str, encoding: str = "utf-8") -> str:
    """Read a file and return its contents or an error message."""
    try:
        if not os.path.exists(path):
            return f"Error: file not found: {path}"
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def write_file_tool(spec: str, encoding: str = "utf-8") -> str:
    """Write content to a file.

    The `spec` argument should be two parts separated by the first newline: the
    target path on the first line, and the content to write on subsequent lines.
    This text-based interface keeps the tool signature simple for agents.
    """
    try:
        if "\n" not in spec:
            return "Error: write spec must include a path line, then a newline, then content"
        path, content = spec.split("\n", 1)
        # Ensure directory exists
        dirpath = os.path.dirname(path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def http_get_tool(url: str, timeout: int = 5) -> str:
    """Perform a simple HTTP GET and return a short summary/result."""
    try:
        resp = requests.get(url, timeout=timeout)
        summary = f"Status: {resp.status_code}; Length: {len(resp.content)}"
        # Return start of text for small inline previews when possible
        try:
            text_preview = resp.text[:1000]
            return summary + "\n" + text_preview
        except Exception:
            return summary
    except Exception as e:
        return f"HTTP GET error: {e}"


def list_dir_tool(path: str = ".") -> str:
    """List files in a directory and return a newline-separated listing."""
    try:
        if not os.path.exists(path):
            return f"Error: path not found: {path}"
        items = os.listdir(path)
        return "\n".join(items)
    except Exception as e:
        return f"Error listing directory: {e}"


def datetime_tool() -> str:
    """Return current date/time information in ISO format (UTC)."""
    now = datetime.datetime.utcnow()
    return now.isoformat() + "Z"


def system_info_tool() -> str:
    """Return basic system information."""
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def random_joke_tool() -> str:
    """Return a small, harmless random joke."""
    jokes = [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "I told my computer I needed a break, and it said 'No problem — I'll go to sleep.'",
        "There are 10 kinds of people: those who understand binary and those who don't.",
    ]
    return random.choice(jokes)


def main():
    print("Hello, agent with tools!")
    # Demonstrate the utilities in a simple way
    print(calculator_tool("5+7"))
    print(weather_tool("Paris"))
    print("Listing current dir:")
    print(list_dir_tool("."))


if __name__ == "__main__":
    main()
