"""Example demonstrating Multi-Agent Collaboration in LangGraph.

This script shows how to:
1. Define a state graph with a Supervisor/Router agent.
2. Route queries dynamically to specialized Researcher and Writer agents.
3. Share execution state across agents to construct a unified response.
"""

import asyncio
import sys
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, cast
from dotenv import load_dotenv

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from services.llm_factory import LLMFactory
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import some tools from the workspace
from tools import weather, calculator
from langgraph.prebuilt import ToolNode

load_dotenv()


# 1. Define the Shared State
class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: str


async def main():
    print("🤖 Initializing Multi-Agent Collaboration Example...")
    try:
        # Create LLM
        llm = LLMFactory.create_llm()
        
        # 2. Define specialized agent behaviors
        # Researcher Agent (uses tools to gather facts)
        research_tools = [weather, calculator]
        research_model = llm.bind_tools(research_tools)
        
        def researcher_agent(state: MultiAgentState):
            print("🔬 [Researcher Agent]: Gathering facts and running calculations...")
            messages = state["messages"]
            response = research_model.invoke(messages)
            return {
                "messages": [response],
                "next_agent": "tools" if response.tool_calls else "supervisor"
            }
            
        # Writer Agent (performs creative drafting/writing)
        def writer_agent(state: MultiAgentState):
            print("✍️  [Writer Agent]: Creating polished narrative draft...")
            messages = list(state["messages"])
            
            writer_prompt = (
                "You are an expert copywriter. Take the raw factual information provided "
                "by the Researcher and draft a friendly, professional summary. Do not "
                "change any calculated figures or facts."
            )
            messages.append(HumanMessage(content=writer_prompt))
            response = llm.invoke(messages)
            return {
                "messages": [response],
                "next_agent": "supervisor"
            }
            
        # Supervisor Agent (orchestrates the workflow)
        def supervisor_agent(state: MultiAgentState):
            print("👑 [Supervisor Agent]: Routing query to the correct expert...")
            messages = list(state["messages"])
            
            # Formulate the routing prompt
            route_prompt = (
                "You are the team lead supervisor. Decide what to do next based on history. "
                "Your choices are:\n"
                "- If the user needs to find facts, calculate math, or check weather, reply with ONLY the word: RESEARCHER\n"
                "- If the facts are gathered but we need a polished written response, reply with ONLY the word: WRITER\n"
                "- If the task is completed and we have a final creative report, reply with ONLY the word: FINISH\n\n"
                "Do not reply with any other text."
            )
            messages.append(HumanMessage(content=route_prompt))
            decision = str(llm.invoke(messages).content).strip().upper()
            
            if "RESEARCHER" in decision:
                next_step = "researcher"
            elif "WRITER" in decision:
                next_step = "writer"
            else:
                next_step = "finish"
                
            print(f"   Route Decision: -> {next_step.upper()}")
            return {"next_agent": next_step}

        # 3. Construct the StateGraph
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes
        workflow.add_node("supervisor", supervisor_agent)
        workflow.add_node("researcher", researcher_agent)
        workflow.add_node("writer", writer_agent)
        workflow.add_node("tools", ToolNode(research_tools))
        
        # Add routing edges
        workflow.add_edge(START, "supervisor")
        
        # Supervisor routes to researcher, writer, or ends
        def route_supervisor(state: MultiAgentState):
            decision = state["next_agent"]
            if decision == "researcher":
                return "researcher"
            elif decision == "writer":
                return "writer"
            return END
            
        workflow.add_conditional_edges(
            "supervisor", 
            route_supervisor, 
            {"researcher": "researcher", "writer": "writer", END: END}
        )
        
        # Researcher routes to tools or back to supervisor
        def route_researcher(state: MultiAgentState):
            return state["next_agent"]
            
        workflow.add_conditional_edges(
            "researcher",
            route_researcher,
            {"tools": "tools", "supervisor": "supervisor"}
        )
        
        # Tools always route back to researcher to evaluate findings
        workflow.add_edge("tools", "researcher")
        
        # Writer routes back to supervisor for final check
        workflow.add_edge("writer", "supervisor")
        
        # Compile
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)
        
        # 4. Run the Multi-Agent setup
        thread_config: RunnableConfig = {"configurable": {"thread_id": "multi_agent_session_1"}}
        question = "What is the weather in Poznan? Give me a funny weather report about it."
        
        print(f"\nUser Question: {question}\n")
        
        result = cast(MultiAgentState, await graph.ainvoke(
            {"messages": [HumanMessage(content=question)], "next_agent": "supervisor"},
            config=thread_config
        ))
        
        print("\n🏁 Final Multi-Agent Output:")
        print("-" * 50)
        # Find the last message that isn't the supervisor prompt message
        for msg in reversed(result["messages"]):
            if msg.content and str(msg.content).strip().upper() not in ["RESEARCHER", "WRITER", "FINISH"]:
                print(msg.content)
                break
        print("-" * 50)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
