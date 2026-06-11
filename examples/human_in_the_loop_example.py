"""Example demonstrating LangGraph's Human-in-the-loop (Interrupt and Approval) pattern.

This script shows how to:
1. Set up a state graph that interrupts before executing sensitive tools.
2. Run the agent and capture the interrupt state.
3. Simulate user approval to resume and complete execution.
"""

import asyncio
import sys
from pathlib import Path
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from services.llm_factory import LLMFactory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

load_dotenv()


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
        
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", route_after_model, ["tools", END])
        workflow.add_edge("tools", "agent")
        
        # 5. Compile the graph with memory and interrupt_before
        # This tells the graph to pause execution right before running the 'tools' node
        memory = MemorySaver()
        graph = workflow.compile(
            checkpointer=memory,
            interrupt_before=["tools"]
        )
        
        # 6. First execution run (Thread config is required for checkpointers)
        thread_config: RunnableConfig = {"configurable": {"thread_id": "user_session_1"}}
        question = "Please delete the file named sensitive_data.csv"
        
        print(f"\n💬 User: {question}")
        print("⏳ Running agent...")
        
        # Start the graph execution
        async for event in graph.astream(
            {"messages": [HumanMessage(content=question)]}, 
            config=thread_config,
            stream_mode="values"
        ):
            last_msg = event["messages"][-1]
            if last_msg.content:
                print(f"🤖 Agent: {last_msg.content}")
                
        # 7. Check if graph execution was interrupted
        state = await graph.aget_state(thread_config)
        
        if state.next:
            print(f"\n⚠️  [PAUSED] Graph execution interrupted before node: {state.next}")
            
            # Retrieve the pending tool call information
            last_message = state.values["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    print(f"👉 Action Requested: {tool_call['name']}({tool_call['args']})")
            
            # Simulate human response/decision
            print("\n👤 Human Action: [Approving action...]")
            
            # To resume, we simply execute the graph again with input=None (keeping the thread config)
            print("⏳ Resuming execution with approval...")
            async for event in graph.astream(
                None,  # Passing None tells LangGraph to resume from the interrupted state
                config=thread_config,
                stream_mode="values"
            ):
                last_msg = event["messages"][-1]
                if last_msg.content:
                    print(f"Resumed Response: {last_msg.content}")
                    
        # Check final state
        final_state = await graph.aget_state(thread_config)
        print(f"\n🏁 Finished! Next nodes to execute: {final_state.next}")
        
    except Exception as e:
        print(f"\n❌ Error running Human-in-the-Loop example: {e}")


if __name__ == "__main__":
    asyncio.run(main())
