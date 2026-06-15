"""Output formatter for agent execution streaming events."""

import re
import time
from typing import Callable, cast

from colorama import Fore, Style
from colorama import init as colorama_init

colorama_init(autoreset=True)


# MARK: Content Processor
class ContentProcessor:
    """Handles content transformation and sanitization."""

    @staticmethod
    def truncate_content(content: str, max_length: int = 500) -> str:
        """Truncate content to max length with ellipsis."""
        if len(content) > max_length:
            return f"{content[:max_length]}... (truncated)"
        return content

    @staticmethod
    def clean_sql_error(error_msg: str) -> str:
        """Clean up SQL error messages to be more readable."""
        # Extract the actual SQL Server error message
        # Look for pattern: [ErrorCode] [Driver info][Server info]Message. (ErrorNum)
        match = re.search(r"\[SQL Server\](.*?)\.\s*\(\d+\)", error_msg)
        if match:
            clean_error = match.group(1).strip()

            # Extract the SQL query if present
            sql_match = re.search(r"\[SQL:\s*(.*?)\]", error_msg, re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1).strip()
                return f"❌ SQL Error: {clean_error}\n   Query: {sql_query}"
            return f"❌ SQL Error: {clean_error}"

        # Fallback: just remove the background link and technical details
        error_msg = re.sub(r"\(Background on this error at:.*?\)", "", error_msg)
        error_msg = re.sub(r"\(pyodbc\.\w+\)", "", error_msg)
        error_msg = re.sub(r"\('[\w\d]+',\s*", "", error_msg)
        error_msg = error_msg.replace('"', "'").strip()

        return error_msg


# MARK: Execution Monitor
class ExecutionMonitor:
    """Tracks execution metrics and timings."""

    def __init__(self) -> None:
        self.tool_timings: dict[str, float] = {}

    def start_tool_timing(self, tool_call_id: str) -> None:
        """Record the start time of a tool execution."""
        self.tool_timings[tool_call_id] = time.time()

    def get_tool_execution_time(self, tool_call_id: str) -> float | None:
        """Get the execution time for a tool call."""
        if tool_call_id in self.tool_timings:
            return time.time() - self.tool_timings[tool_call_id]
        return None

    def format_execution_time(self, tool_call_id: str) -> str:
        """Format execution time as a string."""
        exec_time = self.get_tool_execution_time(tool_call_id)
        if exec_time is not None:
            return f" ⏱️  {exec_time:.3f}s"
        return ""


# MARK: Message Formatter
class MessageFormatter:
    """Pure formatting logic for different message types."""

    content_processor: ContentProcessor
    execution_monitor: ExecutionMonitor

    def __init__(
        self, content_processor: ContentProcessor, execution_monitor: ExecutionMonitor
    ):
        self.content_processor = content_processor
        self.execution_monitor = execution_monitor

    def format_human_message(self, msg: object) -> list[str]:
        """Format human/user message with color."""
        content = getattr(msg, "content", "")
        return [Fore.BLUE + Style.BRIGHT + f"👤 User: {content}"]

    def format_ai_message(self, msg: object) -> list[str]:
        """Format AI message with optional tool calls and color."""
        lines: list[str] = []

        content = getattr(msg, "content", "")
        if content:
            lines.append(Fore.MAGENTA + Style.BRIGHT + f"🤖 AI: {content}")

        # Handle tool calls
        tool_calls = cast(list[object] | None, getattr(msg, "tool_calls", None))
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    tc_dict: dict[str, object] = cast(dict[str, object], tool_call)
                    tool_call_id = str(tc_dict.get("id", "unknown"))
                    tool_name = str(tc_dict.get("name", "unknown"))
                    lines.append(
                        Fore.YELLOW + Style.BRIGHT + f"🔧 Calling Tool: {tool_name}"
                    )
                    lines.append(Fore.YELLOW + f"   Args: {tc_dict.get('args', {})}")
                    self.execution_monitor.start_tool_timing(tool_call_id)

        return lines

    def format_tool_message(self, msg: object) -> list[str]:
        """Format tool result message with color."""
        tool_name = getattr(msg, "name", "unknown")
        tool_call_id = cast(str | None, getattr(msg, "tool_call_id", None))

        # Get execution time string
        exec_time_str = ""
        if tool_call_id:
            exec_time_str = self.execution_monitor.format_execution_time(tool_call_id)

        content = str(getattr(msg, "content", ""))

        # Clean up SQL database errors
        if tool_name == "sql_db_query" and "Error:" in content:
            content = self.content_processor.clean_sql_error(content)

        content = self.content_processor.truncate_content(content)

        return [
            Fore.WHITE + Style.BRIGHT + f"⚙️  Tool Result ({tool_name}){exec_time_str}:",
            Fore.WHITE + f"   {content}",
        ]

    def format_default_message(self, msg: object) -> list[str]:
        """Format unknown message type with color."""
        content = getattr(msg, "content", "") if hasattr(msg, "content") else str(msg)
        return [Fore.WHITE + Style.DIM + f"💬 {content}"]


# MARK: Stream Renderer
class StreamRenderer:
    """Handles output display to console."""

    @staticmethod
    def print_header() -> None:
        """Print execution trace header with color."""
        print("\n" + Fore.CYAN + Style.BRIGHT + "=" * 80)
        print(Fore.YELLOW + Style.BRIGHT + "AGENT EXECUTION TRACE (STREAMING)")
        print(Fore.CYAN + Style.BRIGHT + "=" * 80)

    @staticmethod
    def print_footer() -> None:
        """Print execution completion footer with color."""
        print("\n" + Fore.CYAN + Style.BRIGHT + "=" * 80)
        print(Fore.GREEN + Style.BRIGHT + "EXECUTION COMPLETE")
        print(Fore.CYAN + Style.BRIGHT + "=" * 80)

    @staticmethod
    def print_step_header(step_count: int, node_name: str) -> None:
        """Print step header."""
        print(Fore.CYAN + Style.BRIGHT + f"\n--- Step {step_count}: {node_name} ---")

    @staticmethod
    def print_lines(lines: list[str]) -> None:
        """Print multiple lines."""
        for line in lines:
            print(line)


# MARK: Output Formatter
class StreamingOutputFormatter:
    """Coordinates formatting and display of agent execution events."""

    content_processor: ContentProcessor
    execution_monitor: ExecutionMonitor
    message_formatter: MessageFormatter
    renderer: StreamRenderer

    def __init__(self):
        self.content_processor = ContentProcessor()
        self.execution_monitor = ExecutionMonitor()
        self.message_formatter = MessageFormatter(
            self.content_processor, self.execution_monitor
        )
        self.renderer = StreamRenderer()

    def print_header(self) -> None:
        """Print execution trace header."""
        self.renderer.print_header()

    def print_footer(self) -> None:
        """Print execution completion footer."""
        self.renderer.print_footer()

    def print_event(
        self,
        event: dict[str, object],
        step_count: int,
        _tool_timings: dict[str, float] | None = None,
    ) -> None:
        """
        Print a single streaming event from the agent.

        Args:
            event: Event dictionary from agent executor stream
            step_count: Current step number
            tool_timings: (Deprecated) Use internal execution monitor instead
        """
        # Message type handlers dispatch dictionary
        handlers: dict[str, Callable[[object], list[str]]] = {
            "human": self.message_formatter.format_human_message,
            "user": self.message_formatter.format_human_message,
            "ai": self.message_formatter.format_ai_message,
            "assistant": self.message_formatter.format_ai_message,
            "system": self.message_formatter.format_ai_message,
            "tool": self.message_formatter.format_tool_message,
            "tool_result": self.message_formatter.format_tool_message,
        }

        for node_name, node_data in event.items():
            self.renderer.print_step_header(step_count, node_name)

            if isinstance(node_data, dict):
                node_dict: dict[str, object] = cast(dict[str, object], node_data)
                if "messages" not in node_dict:
                    continue

                messages = cast(list[object] | None, node_dict.get("messages"))
                if isinstance(messages, list):
                    for msg in messages:
                        msg_type: str = str(getattr(msg, "type", "default"))

                        # Get handler or use default
                        handler = handlers.get(
                            msg_type, self.message_formatter.format_default_message
                        )

                        # Print all lines returned by handler
                        self.renderer.print_lines(handler(msg))
