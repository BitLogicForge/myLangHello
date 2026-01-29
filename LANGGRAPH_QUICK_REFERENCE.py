"""
LangGraph ReAct Agent - Quick Reference
========================================

BASIC USAGE:
-----------
from services import AgentFactory

# Create agent
agent_factory = AgentFactory(llm=llm, tools=tools, prompt=prompt)
agent = agent_factory.create_executor()

# Simple invocation
result = agent.invoke(
    {"messages": [("user", "What is 2+2?")]},
    config={"recursion_limit": 15}
)
answer = result["messages"][-1].content


WITH MEMORY:
-----------
from langgraph.checkpoint.memory import MemorySaver

# Create agent with memory
agent_factory = AgentFactory(
    llm=llm,
    tools=tools,
    prompt=prompt,
    checkpointer=MemorySaver()  # Add memory
)
agent = agent_factory.create_executor()

# Use with thread ID for conversation continuity
result = agent.invoke(
    {"messages": [("user", "My name is Alice")]},
    config={"configurable": {"thread_id": "user-123"}}
)

# Continue conversation (remembers context)
result = agent.invoke(
    {"messages": [("user", "What's my name?")]},
    config={"configurable": {"thread_id": "user-123"}}
)


STREAMING:
----------
# Stream responses token by token
for chunk in agent.stream(
    {"messages": [("user", "Tell me a story")]},
    config={"recursion_limit": 15}
):
    if "messages" in chunk:
        print(chunk["messages"][-1].content)


CONVENIENCE METHOD:
------------------
# Use the built-in convenience method
result = agent_factory.invoke("What is the weather?")
answer = result["messages"][-1].content


KEY DIFFERENCES FROM OLD API:
-----------------------------
OLD (AgentExecutor):
  result = agent.invoke({"input": "question"})
  answer = result["output"]

NEW (LangGraph):
  result = agent.invoke({"messages": [("user", "question")]})
  answer = result["messages"][-1].content


CONFIGURATION:
--------------
config = {
    "recursion_limit": 15,  # Max iterations
    "configurable": {
        "thread_id": "user-123"  # For memory
    }
}


ADVANCED FEATURES:
-----------------
1. Graph Visualization:
   agent.get_graph().print_ascii()

2. Interrupt & Resume:
   Use checkpointer to save state and resume later

3. Parallel Tool Execution:
   Automatically optimized by LangGraph

4. Custom State Modifiers:
   Pass functions to state_modifier parameter
"""
