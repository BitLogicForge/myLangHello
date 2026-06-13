"""Example demonstrating LangGraph's Human-in-the-loop (Interrupt and Approval) pattern.

This script shows how to:
1. Set up a state graph that interrupts before executing sensitive tools.
2. Run the agent and capture the interrupt state.
3. Simulate user approval to resume and complete execution.
"""

import asyncio
import sys
from pathlib import Path
from typing import TypedDict, Annotated, cast
from collections.abc import Sequence
from dotenv import load_dotenv

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from utils import is_str_dict, is_list
from services.llm_factory import LLMFactory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.message import add_messages  # pyright: ignore[reportMissingTypeStubs]
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

_ = load_dotenv()


# 1. Define the Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 2. Define a sensitive tool that requires human approval
@tool
def delete_file_tool(filename: str) -> str:
    """Deletes a file from the system. (Requires high privilege)."""
    return f"Successfully deleted file: {filename}."


async def main():
    print("🤖 Initializing Human-in-the-Loop Example...")
    try:
        # Create LLM
        llm = LLMFactory.create_llm()
        
        # Bind tool to the model
        tools = [delete_file_tool]
        model_with_tools = llm.bind_tools(tools)
        
        # 3. Define the node functions
        def call_model(state: AgentState):
            messages = state["messages"]
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}
        
        tool_node = ToolNode(tools)
        
        # Define the conditional routing logic
        def route_after_model(state: AgentState):
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
            return END
        
        # 4. Construct the LangGraph StateGraph
        workflow = StateGraph(AgentState)
        
        _ = workflow.add_node("agent", call_model)  # pyright: ignore[reportUnknownMemberType]
        _ = workflow.add_node("tools", tool_node)  # pyright: ignore[reportUnknownMemberType]
        
        _ = workflow.add_edge(START, "agent")
        _ = workflow.add_conditional_edges("agent", route_after_model, ["tools", END])
        _ = workflow.add_edge("tools", "agent")
        
        # 5. Compile the graph with memory and interrupt_before
        # This tells the graph to pause execution right before running the 'tools' node
        memory = MemorySaver()
        graph = workflow.compile(  # pyright: ignore[reportUnknownMemberType]
            checkpointer=memory,
            interrupt_before=["tools"]
        )
        
        # 6. First execution run (Thread config is required for checkpointers)
        thread_config: RunnableConfig = {"configurable": {"thread_id": "user_session_1"}}
        question = "Please delete the file named sensitive_data.csv"
        
        print(f"\n💬 User: {question}")
        print("⏳ Running agent...")
        
        # Start the graph execution
        async for event in graph.astream(  # pyright: ignore[reportUnknownMemberType]
            {"messages": [HumanMessage(content=question)]}, 
            config=thread_config,
            stream_mode="values"
        ):
            event_dict = cast(dict[str, object], event)
            messages = event_dict.get("messages", [])
            if is_list(messages) and messages:
                last_msg = messages[-1]
                content_obj: object = getattr(last_msg, "content", None)
                if isinstance(content_obj, str) and content_obj:
                    print(f"🤖 Agent: {content_obj}")
                
        # 7. Check if graph execution was interrupted
        state = await graph.aget_state(thread_config)
        
        if state.next:
            print(f"\n⚠️  [PAUSED] Graph execution interrupted before node: {state.next}")
            
            # Retrieve the pending tool call information
            values = state.values
            if is_str_dict(values):
                messages = values.get("messages", [])
                if is_list(messages) and messages:
                    last_message = messages[-1]
                    tool_calls_obj: object = getattr(last_message, "tool_calls", None)
                    if is_list(tool_calls_obj) and tool_calls_obj:
                        for tool_call in tool_calls_obj:
                            if is_str_dict(tool_call):
                                name = tool_call.get("name")
                                args = tool_call.get("args")
                                print(f"👉 Action Requested: {name}({args})")
            
            # Simulate human response/decision
            print("\n👤 Human Action: [Approving action...]")
            
            # To resume, we simply execute the graph again with input=None (keeping the thread config)
            print("⏳ Resuming execution with approval...")
            async for event in graph.astream(  # pyright: ignore[reportUnknownMemberType]
                None,  # Passing None tells LangGraph to resume from the interrupted state
                config=thread_config,
                stream_mode="values"
            ):
                event_dict = cast(dict[str, object], event)
                messages = event_dict.get("messages", [])
                if is_list(messages) and messages:
                    last_msg = messages[-1]
                    resumed_content_obj: object = getattr(last_msg, "content", None)
                    if isinstance(resumed_content_obj, str) and resumed_content_obj:
                        print(f"Resumed Response: {resumed_content_obj}")
                        
        # Check final state
        final_state = await graph.aget_state(thread_config)
        print(f"\n🏁 Finished! Next nodes to execute: {final_state.next}")
        
    except Exception as e:
        print(f"\n❌ Error running Human-in-the-Loop example: {e}")


if __name__ == "__main__":
    asyncio.run(main())
