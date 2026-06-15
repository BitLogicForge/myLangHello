# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""Chainlit application for conversational LangGraph agent interface."""

import chainlit as cl
from main import AgentApp
from langchain_core.messages import HumanMessage
from typing import cast
from utils import is_str_dict, is_list


# MARK: Chat Start
@cl.on_chat_start
async def start():
    """Fires when a new chat session is established in Chainlit."""
    try:
        # Initialize the shared AgentApp instance
        app = AgentApp()
        cl.user_session.set("agent_app", app)

        _ = await cl.Message(
            content="👋 Welcome to your LangGraph Chatbot! Ask me anything, and I'll use my tools (calculator, weather, database, etc.) to help you."
        ).send()
    except Exception as e:
        _ = await cl.Message(content=f"❌ Error initializing Agent: {e}").send()


# MARK: Message Handler
@cl.on_message
async def main(message: cl.Message):
    """Fires when a user sends a message in the chat."""
    agent_app = cast(AgentApp, cl.user_session.get("agent_app"))
    if not agent_app:
        _ = await cl.Message(content="❌ Agent session is not active.").send()
        return

    # Create a message to accumulate final text
    final_message = cl.Message(content="")

    # Track active steps to update them dynamically
    active_steps: dict[str, cl.Step] = {}

    try:
        # Stream events from LangGraph agent executor
        async for event in agent_app.agent_executor.astream(
            {"messages": [HumanMessage(content=message.content)]}
        ):
            if not is_str_dict(event):
                continue

            for node_name, node_output in event.items():
                if not is_str_dict(node_output):
                    continue

                messages = node_output.get("messages", [])
                if not is_list(messages):
                    continue

                for msg in messages:
                    # 1. Handle tool execution starts (agent requesting a tool)
                    tool_calls = getattr(msg, "tool_calls", None)
                    if is_list(tool_calls) and tool_calls:
                        for tool_call in tool_calls:
                            if not is_str_dict(tool_call):
                                continue

                            call_id = tool_call.get("id")
                            if not isinstance(call_id, str):
                                continue

                            step = cl.Step(
                                name=str(tool_call.get("name", "tool")), type="tool"
                            )
                            step.input = str(tool_call.get("args", ""))
                            _ = await step.send()
                            active_steps[call_id] = step
                        continue

                    # 2. Handle tool outputs (tools node finished executing)
                    if node_name == "tools" and getattr(msg, "tool_call_id", None):
                        tool_call_id_val = getattr(msg, "tool_call_id", None)
                        if not isinstance(tool_call_id_val, str):
                            continue

                        call_id = tool_call_id_val
                        msg_content = getattr(msg, "content", "")

                        if call_id in active_steps:
                            step = active_steps[call_id]
                            step.output = str(msg_content)
                            _ = await step.update()
                            del active_steps[call_id]
                        else:
                            # Fallback if step wasn't captured in call phase
                            msg_name = getattr(msg, "name", "Tool Output")
                            step = cl.Step(
                                name=str(msg_name) if msg_name else "Tool Output",
                                type="tool",
                            )
                            step.output = str(msg_content)
                            _ = await step.send()
                        continue

                    # 3. Accumulate final text responses from the agent
                    msg_content = cast(object, getattr(msg, "content", None))
                    msg_tool_calls = getattr(msg, "tool_calls", None)
                    msg_type = getattr(msg, "type", None)

                    if msg_content and not msg_tool_calls:
                        # Avoid adding raw tool feedback back as text message
                        if node_name != "tools" and msg_type == "ai":
                            # Update the UI incrementally
                            _ = await final_message.stream_token(str(msg_content))

        # Finalize the message stream
        _ = await final_message.send()

    except Exception as e:
        _ = await cl.Message(content=f"❌ Error during execution: {e}").send()
