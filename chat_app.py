"""Chainlit application for conversational LangGraph agent interface."""

import chainlit as cl
from main import AgentApp
from langchain_core.messages import HumanMessage
from typing import Dict, cast


@cl.on_chat_start
async def start():
    """Fires when a new chat session is established in Chainlit."""
    try:
        # Initialize the shared AgentApp instance
        app = AgentApp()
        cl.user_session.set("agent_app", app)
        
        await cl.Message(
            content="👋 Welcome to your LangGraph Chatbot! Ask me anything, and I'll use my tools (calculator, weather, database, etc.) to help you."
        ).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error initializing Agent: {e}").send()


@cl.on_message
async def main(message: cl.Message):
    """Fires when a user sends a message in the chat."""
    agent_app = cast(AgentApp, cl.user_session.get("agent_app"))
    if not agent_app:
        await cl.Message(content="❌ Agent session is not active.").send()
        return

    # Create a message to accumulate final text
    final_message = cl.Message(content="")
    
    # Track active steps to update them dynamically
    active_steps: Dict[str, cl.Step] = {}
    
    try:
        # Stream events from LangGraph agent executor
        async for event in agent_app.agent_executor.astream({"messages": [HumanMessage(content=message.content)]}):
            for node_name, node_output in event.items():
                messages = node_output.get("messages", [])
                
                for msg in messages:
                    # 1. Handle tool execution starts (agent requesting a tool)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            call_id = tool_call.get("id")
                            if call_id:
                                step = cl.Step(name=tool_call["name"], type="tool")
                                step.input = str(tool_call["args"])
                                await step.send()
                                active_steps[call_id] = step
                    
                    # 2. Handle tool outputs (tools node finished executing)
                    elif node_name == "tools" and hasattr(msg, "tool_call_id") and msg.tool_call_id:
                        call_id = msg.tool_call_id
                        if call_id in active_steps:
                            step = active_steps[call_id]
                            step.output = str(msg.content)
                            await step.update()
                            del active_steps[call_id]
                        else:
                            # Fallback if step wasn't captured in call phase
                            step = cl.Step(name=msg.name or "Tool Output", type="tool")
                            step.output = str(msg.content)
                            await step.send()
                            
                    # 3. Accumulate final text responses from the agent
                    elif hasattr(msg, "content") and msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                        # Avoid adding raw tool feedback back as text message
                        if node_name != "tools" and msg.type == "ai":
                            final_message.content += msg.content
                            # Update the UI incrementally
                            await final_message.stream_token(msg.content)

        # Finalize the message stream
        await final_message.send()
        
    except Exception as e:
        await cl.Message(content=f"❌ Error during execution: {e}").send()
