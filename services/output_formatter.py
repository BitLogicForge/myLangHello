"""Output formatter for agent execution streaming events."""

import time
from typing import Any, Callable, Dict, List

from colorama import Fore, Style
from colorama import init as colorama_init

colorama_init(autoreset=True)


class StreamingOutputFormatter:
    """Formats and displays agent execution events in real-time."""

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
    def _format_human_message(msg: Any) -> List[str]:
        """Format human/user message with color."""
        return [Fore.BLUE + Style.BRIGHT + f"👤 User: {msg.content}"]

    @staticmethod
    def _format_ai_message(msg: Any, tool_timings: Dict[str, float]) -> List[str]:
        """Format AI message with optional tool calls and color."""
        lines = []

        if msg.content:
            lines.append(Fore.MAGENTA + Style.BRIGHT + f"🤖 AI: {msg.content}")

        # Handle tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_call_id = tool_call.get("id", "unknown")
                tool_name = tool_call.get("name", "unknown")
                lines.append(Fore.YELLOW + Style.BRIGHT + f"🔧 Calling Tool: {tool_name}")
                lines.append(Fore.YELLOW + f"   Args: {tool_call.get('args', {})}")
                tool_timings[tool_call_id] = time.time()

        return lines

    @staticmethod
    def _format_tool_message(msg: Any, tool_timings: Dict[str, float]) -> List[str]:
        """Format tool result message with color."""
        tool_name = getattr(msg, "name", "unknown")
        tool_call_id = getattr(msg, "tool_call_id", None)

        # Calculate execution time
        exec_time_str = ""
        if tool_call_id and tool_call_id in tool_timings:
            exec_time = time.time() - tool_timings[tool_call_id]
            exec_time_str = f" ⏱️  {exec_time:.3f}s"

        content = str(msg.content)
        if len(content) > 500:
            content = f"{content[:500]}... (truncated)"

        return [
            Fore.WHITE + Style.BRIGHT + f"⚙️  Tool Result ({tool_name}){exec_time_str}:",
            Fore.WHITE + f"   {content}",
        ]

    @staticmethod
    def _format_default_message(msg: Any) -> List[str]:
        """Format unknown message type with color."""
        content = msg.content if hasattr(msg, "content") else str(msg)
        return [Fore.WHITE + Style.DIM + f"💬 {content}"]

    @classmethod
    def print_event(cls, event: dict, step_count: int, tool_timings: Dict[str, float]) -> None:
        """
        Print a single streaming event from the agent.

        Args:
            event: Event dictionary from agent executor stream
            step_count: Current step number
            tool_timings: Dictionary tracking tool execution start times
        """
        # Message type handlers dispatch dictionary
        handlers: Dict[str, Callable[[Any], List[str]]] = {
            "human": lambda msg: cls._format_human_message(msg),
            "user": lambda msg: cls._format_human_message(msg),
            "ai": lambda msg: cls._format_ai_message(msg, tool_timings),
            "assistant": lambda msg: cls._format_ai_message(msg, tool_timings),
            "system": lambda msg: cls._format_ai_message(msg, tool_timings),
            "tool": lambda msg: cls._format_tool_message(msg, tool_timings),
            "tool_result": lambda msg: cls._format_tool_message(msg, tool_timings),
        }

        for node_name, node_data in event.items():
            print(Fore.CYAN + Style.BRIGHT + f"\n--- Step {step_count}: {node_name} ---")

            if "messages" not in node_data:
                continue

            for msg in node_data["messages"]:
                msg_type: str = getattr(msg, "type", "default")

                # Get handler or use default
                handler = handlers.get(msg_type, lambda m: cls._format_default_message(m))

                # Print all lines returned by handler
                for line in handler(msg):
                    print(line)
