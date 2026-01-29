"""Output formatter for agent execution streaming events."""

import time
from typing import Dict


class StreamingOutputFormatter:
    """Formats and displays agent execution events in real-time."""

    @staticmethod
    def print_header() -> None:
        """Print execution trace header."""
        print("\n" + "=" * 80)
        print("AGENT EXECUTION TRACE (STREAMING)")
        print("=" * 80)

    @staticmethod
    def print_footer() -> None:
        """Print execution completion footer."""
        print("\n" + "=" * 80)
        print("EXECUTION COMPLETE")
        print("=" * 80)

    @staticmethod
    def print_event(event: dict, step_count: int, tool_timings: Dict[str, float]) -> None:
        """
        Print a single streaming event from the agent.

        Args:
            event: Event dictionary from agent executor stream
            step_count: Current step number
            tool_timings: Dictionary tracking tool execution start times
        """
        for node_name, node_data in event.items():
            print(f"\n--- Step {step_count}: {node_name} ---")

            if "messages" in node_data:
                for msg in node_data["messages"]:
                    # Human/User message
                    if hasattr(msg, "type") and msg.type == "human":
                        print(f"👤 User: {msg.content}")

                    # AI message with potential tool calls
                    elif hasattr(msg, "type") and msg.type == "ai":
                        if msg.content:
                            print(f"🤖 AI: {msg.content}")

                        # Check for tool calls
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_call_id = tool_call.get("id", "unknown")
                                tool_name = tool_call.get("name", "unknown")
                                print(f"🔧 Calling Tool: {tool_name}")
                                print(f"   Args: {tool_call.get('args', {})}")
                                # Record start time
                                tool_timings[tool_call_id] = time.time()

                    # Tool message (tool response)
                    elif hasattr(msg, "type") and msg.type == "tool":
                        tool_name = getattr(msg, "name", "unknown")
                        tool_call_id = getattr(msg, "tool_call_id", None)

                        # Calculate execution time if we have a start time
                        exec_time_str = ""
                        if tool_call_id and tool_call_id in tool_timings:
                            exec_time = time.time() - tool_timings[tool_call_id]
                            exec_time_str = f" ⏱️  {exec_time:.3f}s"

                        print(f"⚙️  Tool Result ({tool_name}){exec_time_str}:")
                        content = str(msg.content)
                        if len(content) > 500:
                            print(f"   {content[:500]}... (truncated)")
                        else:
                            print(f"   {content}")

                    # Other message types
                    else:
                        content = msg.content if hasattr(msg, "content") else str(msg)
                        print(f"💬 {content}")
