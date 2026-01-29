# LangGraph Migration Guide

## Overview

The agent has been upgraded from the traditional `AgentExecutor` to **LangGraph's `create_react_agent`**, providing a state-of-the-art (SOTA) implementation with better performance, built-in memory, and advanced features.

## What Changed

### Old Approach (AgentExecutor)

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent

agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)
result = executor.invoke({"input": "your question"})
```

### New Approach (LangGraph)

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=system_message,
    checkpointer=checkpointer  # Optional: for memory
)
result = agent.invoke(
    {"messages": [("user", "your question")]},
    config={"recursion_limit": 10}
)
```

## Key Improvements

### 1. **Graph-Based Execution**

- LangGraph uses a state machine graph architecture
- Better control flow and reasoning
- More predictable behavior

### 2. **Built-in Memory Support**

```python
# Add memory with checkpointer
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=system_message,
    checkpointer=checkpointer
)

# Use with thread-based conversations
result = agent.invoke(
    {"messages": [("user", "Remember this")]},
    config={"configurable": {"thread_id": "user-123"}}
)
```

### 3. **Native Streaming Support**

```python
# Stream agent responses token-by-token
for chunk in agent.stream(
    {"messages": [("user", "Tell me a story")]},
    config={"recursion_limit": 15}
):
    print(chunk)
```

### 4. **Better Error Handling**

- Automatic retry logic
- Graceful failure handling
- Better debugging with graph visualization

## Migration Steps

### Step 1: Install LangGraph

```bash
pip install langgraph
```

### Step 2: Update AgentFactory

The `AgentFactory` class has been updated to use `create_react_agent`. No changes needed to your existing code if you're using the factory pattern.

### Step 3: Update Invocation (if using directly)

**Old way:**

```python
result = agent.invoke({"input": "What is 2+2?"})
answer = result["output"]
```

**New way:**

```python
result = agent.invoke({"messages": [("user", "What is 2+2?")]})
answer = result["messages"][-1].content
```

### Step 4: Add Memory (Optional)

```python
from langgraph.checkpoint.memory import MemorySaver

agent_factory = AgentFactory(
    llm=llm,
    tools=tools,
    prompt=prompt,
    checkpointer=MemorySaver()  # Add this for memory
)
```

## API Comparison

| Feature        | AgentExecutor                 | LangGraph ReAct             |
| -------------- | ----------------------------- | --------------------------- |
| Architecture   | Linear chain                  | State graph                 |
| Memory         | External (ChatMessageHistory) | Built-in (checkpointer)     |
| Streaming      | Limited                       | Native support              |
| Debugging      | Verbose logs                  | Graph visualization         |
| Performance    | Good                          | Better                      |
| Max Iterations | `max_iterations`              | `recursion_limit` in config |

## Example Usage

See `example_langgraph_agent.py` for complete examples including:

- Basic tool usage
- Multi-step reasoning
- Conversation with memory
- Streaming responses

## Backwards Compatibility

The `AgentFactory` maintains backward compatibility:

- `create_executor()` still works
- Returns a LangGraph agent (not AgentExecutor)
- Old invocation patterns work via wrapper methods

## Advanced Features

### 1. Custom State Modification

```python
def custom_state_modifier(state):
    # Modify state before each step
    return state

agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=custom_state_modifier
)
```

### 2. Interrupt and Resume

```python
# Agent can be interrupted and resumed
config = {"configurable": {"thread_id": "session-1"}}
try:
    result = agent.invoke({"messages": [...]}, config)
except KeyboardInterrupt:
    # Later, resume from checkpoint
    result = agent.invoke({"messages": [...]}, config)
```

### 3. Parallel Tool Execution

LangGraph automatically optimizes parallel tool calls when possible.

## Troubleshooting

### Issue: Import Error

```
ModuleNotFoundError: No module named 'langgraph'
```

**Solution:** Run `pip install langgraph`

### Issue: Different Response Format

**Solution:** Update response parsing:

```python
# Old
answer = result["output"]

# New
answer = result["messages"][-1].content
```

### Issue: Memory Not Working

**Solution:** Make sure to:

1. Add checkpointer to agent creation
2. Include thread_id in config
3. Use same thread_id for conversation continuity

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ReAct Agent Guide](https://langchain-ai.github.io/langgraph/tutorials/react-agent/)
- [Migration Examples](./example_langgraph_agent.py)

## Performance Tips

1. **Use recursion_limit wisely:** Set based on task complexity (10-20 for most tasks)
2. **Enable streaming:** For better UX in interactive applications
3. **Use checkpointers:** Only when memory is needed (adds overhead)
4. **Monitor graph execution:** Use built-in debugging tools
