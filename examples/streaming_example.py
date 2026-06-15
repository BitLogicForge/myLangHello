"""Example demonstrating how to stream responses in real-time.

This includes:
1. Direct LLM streaming (token-by-token).
2. Agent execution streaming (step-by-step and tool-by-tool).
"""

import asyncio
import sys
from pathlib import Path
from typing import cast
from dotenv import load_dotenv
from utils import is_str_dict, is_list

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from services.llm_factory import LLMFactory
from main import AgentApp

_ = load_dotenv()


async def stream_llm_tokens():
    """Demonstrates streaming individual tokens directly from the LLM."""
    print("\n--- 1. Streaming Tokens Directly from LLM ---")

    try:
        # Create LLM instance
        llm = LLMFactory.create_llm()

        prompt = "Write a short 3-line poem about vegetables."
        print(f"Prompt: {prompt}\n")
        print("Response: ", end="", flush=True)

        # Use astream to yield token chunks
        async for chunk in llm.astream(prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            print(content, end="", flush=True)
        print("\n" + "-" * 40)

    except Exception as e:
        print(f"\n❌ LLM Streaming Error: {e}")


async def stream_agent_steps():
    """Demonstrates streaming steps and tool calls from the Agent Executor."""
    print("\n--- 2. Streaming Agent Execution Steps ---")

    try:
        # Initialize AgentApp
        app = AgentApp()

        # We query the weather tool and the calculator tool to see agent steps
        question = "What is the weather in Poznan, and what is 152 * 4?"
        print(f"Question: {question}\n")

        # Access the raw LangGraph agent executor
        agent_executor = app.agent_executor

        # Stream events from the agent graph
        # Run with astream_events or astream
        async for event in agent_executor.astream({"messages": [("user", question)]}):
            # Each event represents updates from a node in the agent graph (e.g. 'agent' or 'tools')
            for node_name, node_output in event.items():
                print(f"\n📍 Node: [{node_name}]")
                node_output_dict = cast(dict[str, object], node_output)
                messages = node_output_dict.get("messages", [])
                if is_list(messages):
                    for msg in messages:
                        # If it's a tool call request
                        tool_calls_obj: object = getattr(msg, "tool_calls", None)
                        if is_list(tool_calls_obj) and tool_calls_obj:
                            for tool_call in tool_calls_obj:
                                if is_str_dict(tool_call):
                                    print(
                                        f"   🔧 Tool Call: {tool_call.get('name')}({tool_call.get('args')})"
                                    )
                        # If it's standard text output
                        else:
                            content_obj: object = getattr(msg, "content", None)
                            if isinstance(content_obj, str) and content_obj:
                                # Print preview of the content
                                preview = content_obj.strip().replace("\n", " ")
                                if len(preview) > 100:
                                    preview = preview[:100] + "..."
                                print(f"   💬 Message: {preview}")
        print("\n" + "-" * 40)

    except Exception as e:
        print(f"\n❌ Agent Streaming Error: {e}")


async def main():
    # Run direct LLM streaming
    await stream_llm_tokens()

    # Run step-by-step agent streaming
    await stream_agent_steps()


if __name__ == "__main__":
    asyncio.run(main())
