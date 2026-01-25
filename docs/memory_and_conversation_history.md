# Memory & Conversation History Guide

Comprehensive guide to managing conversation context and memory in LangChain agents.

---

## Table of Contents

1. [Overview](#overview)
2. [Memory Types](#memory-types)
3. [Implementation Patterns](#implementation-patterns)
4. [Persistence Strategies](#persistence-strategies)
5. [Context Window Management](#context-window-management)
6. [Best Practices](#best-practices)
7. [Production Examples](#production-examples)

---

## Overview

### Why Memory Matters

**Problem:** LLMs are stateless - they don't remember previous conversations without explicit context.

**Solution:** Memory systems that:

- Store conversation history
- Provide relevant context to the agent
- Manage token limits efficiently
- Persist across sessions

**Use Cases:**

- Multi-turn conversations
- Customer support chatbots
- Personal assistants
- Long-running tasks

---

## Memory Types

### 1. ConversationBufferMemory

**What It Does:** Stores all messages in a list

**Pros:**

- ✅ Simple implementation
- ✅ Perfect recall of entire conversation
- ✅ No data loss

**Cons:**

- ❌ Grows indefinitely
- ❌ Can exceed token limits
- ❌ Expensive with long conversations

**Implementation:**

```python
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(temperature=0, model="gpt-4")

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,  # Return as list of messages, not string
    output_key="output"     # Which key to store in memory
)

# Create prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # Memory goes here
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent with memory
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Use it
response1 = agent_executor.invoke({"input": "My name is John"})
response2 = agent_executor.invoke({"input": "What's my name?"})
# Agent will remember: "Your name is John"
```

---

### 2. ConversationBufferWindowMemory

**What It Does:** Keeps only the last N messages

**Pros:**

- ✅ Controlled memory size
- ✅ Recent context always available
- ✅ Predictable token usage

**Cons:**

- ❌ Forgets older information
- ❌ May lose important early context

**Implementation:**

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=5,  # Keep last 5 interactions (10 messages: 5 user + 5 assistant)
    memory_key="chat_history",
    return_messages=True
)

# Rest is the same as ConversationBufferMemory
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)
```

**When to Use:**

- Fixed token budget
- Recent context most important
- Long conversations

---

### 3. ConversationSummaryMemory

**What It Does:** Summarizes conversation history using LLM

**Pros:**

- ✅ Compact representation
- ✅ Keeps key information from entire conversation
- ✅ Scales to very long conversations

**Cons:**

- ❌ Costs tokens to create summaries
- ❌ May lose details in summarization
- ❌ Slower (extra LLM calls)

**Implementation:**

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=llm,  # LLM used to create summaries
    memory_key="chat_history",
    return_messages=True
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)
```

**When to Use:**

- Very long conversations
- Important to retain context from entire history
- Token budget is critical

---

### 4. ConversationSummaryBufferMemory

**What It Does:** Hybrid - keeps recent messages verbatim, summarizes older ones

**Pros:**

- ✅ Best of both worlds
- ✅ Recent messages in full detail
- ✅ Older context summarized
- ✅ Controlled token usage

**Cons:**

- ❌ More complex
- ❌ Summarization costs

**Implementation:**

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000,  # When to start summarizing
    memory_key="chat_history",
    return_messages=True
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)
```

**When to Use:**

- Production applications
- Balance between detail and efficiency
- Most versatile option

---

### 5. ConversationTokenBufferMemory

**What It Does:** Keeps messages within a token limit

**Pros:**

- ✅ Precise token control
- ✅ No risk of exceeding context window
- ✅ Automatic truncation

**Cons:**

- ❌ May cut off mid-conversation
- ❌ No intelligent summarization

**Implementation:**

```python
from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(
    llm=llm,  # For token counting
    max_token_limit=1000,
    memory_key="chat_history",
    return_messages=True
)
```

---

## Memory Comparison Matrix

| Memory Type        | Token Control    | Context Quality    | Cost            | Complexity     | Best For               |
| ------------------ | ---------------- | ------------------ | --------------- | -------------- | ---------------------- |
| **Buffer**         | ❌ None          | ⭐⭐⭐⭐⭐ Perfect | 💰 High (grows) | ⭐ Simple      | Short conversations    |
| **Window**         | ⭐⭐ Fixed size  | ⭐⭐⭐ Recent only | 💰💰 Medium     | ⭐ Simple      | Recent context matters |
| **Summary**        | ⭐⭐⭐⭐ Compact | ⭐⭐⭐ Good        | 💰💰 Medium     | ⭐⭐⭐ Complex | Long conversations     |
| **Summary Buffer** | ⭐⭐⭐⭐ Smart   | ⭐⭐⭐⭐ Excellent | 💰💰 Medium     | ⭐⭐⭐ Complex | **Production**         |
| **Token Buffer**   | ⭐⭐⭐⭐⭐ Exact | ⭐⭐ Recent        | 💰 Low          | ⭐⭐ Medium    | Strict limits          |

---

## Implementation Patterns

### Pattern 1: Session-Based Memory (Web Apps)

**Use Case:** Different users, different conversations

```python
# utils/memory.py
from langchain.memory import ConversationSummaryBufferMemory
from typing import Dict

class MemoryManager:
    """Manage memory per session/user."""

    def __init__(self, llm):
        self.llm = llm
        self.sessions: Dict[str, ConversationSummaryBufferMemory] = {}

    def get_memory(self, session_id: str) -> ConversationSummaryBufferMemory:
        """Get or create memory for a session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=2000,
                memory_key="chat_history",
                return_messages=True
            )
        return self.sessions[session_id]

    def clear_memory(self, session_id: str):
        """Clear memory for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

# Usage
memory_manager = MemoryManager(llm)

# In your endpoint
def chat(session_id: str, message: str):
    memory = memory_manager.get_memory(session_id)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)
    response = agent_executor.invoke({"input": message})
    return response["output"]
```

---

### Pattern 2: Custom Memory with Filtering

**Use Case:** Store only specific types of messages

```python
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class FilteredMemory(ConversationBufferMemory):
    """Memory that filters out tool/system messages."""

    def save_context(self, inputs: dict, outputs: dict):
        """Save only user and assistant messages."""
        # Filter inputs
        if "input" in inputs:
            super().save_context(
                {"input": inputs["input"]},
                {"output": outputs.get("output", "")}
            )

# Usage
memory = FilteredMemory(
    memory_key="chat_history",
    return_messages=True
)
```

---

### Pattern 3: Memory with Metadata

**Use Case:** Track additional context (user info, timestamps)

```python
from langchain.memory import ConversationBufferMemory
from datetime import datetime

class MetadataMemory(ConversationBufferMemory):
    """Memory with metadata tracking."""

    def __init__(self, user_id: str, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.session_start = datetime.now()

    def save_context(self, inputs: dict, outputs: dict):
        # Add timestamp to messages
        timestamp = datetime.now().isoformat()

        # Store with metadata
        inputs_with_meta = {
            **inputs,
            "metadata": {
                "user_id": self.user_id,
                "timestamp": timestamp
            }
        }
        super().save_context(inputs_with_meta, outputs)

# Usage
memory = MetadataMemory(
    user_id="user_123",
    memory_key="chat_history",
    return_messages=True
)
```

---

## Persistence Strategies

### 1. File-Based Persistence

**Simple approach for small-scale applications**

```python
import json
from pathlib import Path
from langchain.memory import ConversationBufferMemory

class FilePersistedMemory:
    """Memory persisted to JSON file."""

    def __init__(self, session_id: str, memory_dir: str = "memories"):
        self.session_id = session_id
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.file_path = self.memory_dir / f"{session_id}.json"

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self._load()

    def _load(self):
        """Load memory from file."""
        if self.file_path.exists():
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                # Restore messages
                for msg in data.get("messages", []):
                    if msg["type"] == "human":
                        self.memory.chat_memory.add_user_message(msg["content"])
                    else:
                        self.memory.chat_memory.add_ai_message(msg["content"])

    def save(self):
        """Save memory to file."""
        messages = []
        for msg in self.memory.chat_memory.messages:
            messages.append({
                "type": "human" if msg.type == "human" else "ai",
                "content": msg.content
            })

        with open(self.file_path, 'w') as f:
            json.dump({"messages": messages}, f, indent=2)

    def get_memory(self):
        return self.memory

# Usage
memory_wrapper = FilePersistedMemory(session_id="user_123")
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory_wrapper.get_memory()
)

response = agent_executor.invoke({"input": "Hello"})
memory_wrapper.save()  # Persist after each interaction
```

---

### 2. Redis-Based Persistence

**Production-ready, fast, distributed**

```python
import redis
import json
from langchain.memory import ConversationBufferMemory

class RedisMemory:
    """Memory persisted to Redis."""

    def __init__(self, session_id: str, redis_url: str = "redis://localhost:6379"):
        self.session_id = session_id
        self.redis_client = redis.from_url(redis_url)
        self.key = f"chat_memory:{session_id}"

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self._load()

    def _load(self):
        """Load memory from Redis."""
        data = self.redis_client.get(self.key)
        if data:
            messages = json.loads(data)
            for msg in messages:
                if msg["type"] == "human":
                    self.memory.chat_memory.add_user_message(msg["content"])
                else:
                    self.memory.chat_memory.add_ai_message(msg["content"])

    def save(self):
        """Save memory to Redis."""
        messages = []
        for msg in self.memory.chat_memory.messages:
            messages.append({
                "type": "human" if msg.type == "human" else "ai",
                "content": msg.content
            })

        # Save with expiration (e.g., 24 hours)
        self.redis_client.setex(
            self.key,
            86400,  # 24 hours in seconds
            json.dumps(messages)
        )

    def clear(self):
        """Clear memory from Redis."""
        self.redis_client.delete(self.key)
        self.memory.clear()

# Usage
redis_memory = RedisMemory(session_id="user_123")
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=redis_memory.memory
)

response = agent_executor.invoke({"input": "Hello"})
redis_memory.save()
```

---

### 3. Database Persistence (PostgreSQL)

**Full-featured persistence with querying capabilities**

```python
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = 'chat_messages'

    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    message_type = Column(String)  # 'human' or 'ai'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DatabaseMemory:
    """Memory persisted to PostgreSQL."""

    def __init__(self, session_id: str, db_url: str):
        self.session_id = session_id
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db_session = Session()

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self._load()

    def _load(self):
        """Load last 20 messages from database."""
        messages = (
            self.db_session.query(ChatMessage)
            .filter(ChatMessage.session_id == self.session_id)
            .order_by(ChatMessage.timestamp.desc())
            .limit(20)
            .all()
        )

        for msg in reversed(messages):
            if msg.message_type == "human":
                self.memory.chat_memory.add_user_message(msg.content)
            else:
                self.memory.chat_memory.add_ai_message(msg.content)

    def save(self):
        """Save new messages to database."""
        # Get messages not yet saved
        for msg in self.memory.chat_memory.messages:
            message = ChatMessage(
                id=f"{self.session_id}_{datetime.utcnow().timestamp()}",
                session_id=self.session_id,
                message_type="human" if msg.type == "human" else "ai",
                content=msg.content
            )
            self.db_session.merge(message)

        self.db_session.commit()

# Usage
db_memory = DatabaseMemory(
    session_id="user_123",
    db_url="postgresql://user:pass@localhost/chatdb"
)
```

---

## Context Window Management

### Token Counting

```python
from langchain.callbacks import get_openai_callback

def chat_with_token_tracking(agent_executor, message: str):
    """Track token usage per conversation."""

    with get_openai_callback() as cb:
        response = agent_executor.invoke({"input": message})

        print(f"Tokens used: {cb.total_tokens}")
        print(f"Prompt tokens: {cb.prompt_tokens}")
        print(f"Completion tokens: {cb.completion_tokens}")
        print(f"Cost: ${cb.total_cost:.4f}")

    return response
```

### Smart Truncation

```python
from langchain_core.messages import trim_messages

def create_memory_with_smart_truncation(llm):
    """Memory that intelligently truncates old messages."""

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Wrap with trimming
    def get_trimmed_messages():
        messages = memory.chat_memory.messages
        # Keep system message + last 10 messages
        return trim_messages(
            messages,
            max_tokens=2000,
            strategy="last",
            token_counter=llm,
        )

    return memory, get_trimmed_messages
```

---

## Best Practices

### 1. Choose Right Memory Type

```python
# For prototyping
memory = ConversationBufferMemory()

# For production
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000
)
```

### 2. Always Persist in Production

```python
# DON'T: Lose data on server restart
memory = ConversationBufferMemory()

# DO: Persist to Redis/DB
memory = RedisMemory(session_id=user_id)
```

### 3. Set Memory Limits

```python
# DON'T: Unlimited growth
memory = ConversationBufferMemory()

# DO: Control size
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000
)
```

### 4. Clear Memory When Appropriate

```python
# New conversation
if user_says_reset:
    memory.clear()
    redis_memory.clear()
```

### 5. Monitor Memory Size

```python
def check_memory_size(memory):
    """Log memory statistics."""
    messages = memory.chat_memory.messages
    print(f"Messages: {len(messages)}")

    total_chars = sum(len(msg.content) for msg in messages)
    print(f"Total characters: {total_chars}")
```

---

## Production Example

**Complete production-ready implementation**

```python
# memory/manager.py
from typing import Dict, Optional
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
import redis
import json

class ProductionMemoryManager:
    """Production-ready memory management."""

    def __init__(
        self,
        llm: ChatOpenAI,
        redis_url: str = "redis://localhost:6379",
        max_token_limit: int = 2000,
        session_ttl: int = 86400  # 24 hours
    ):
        self.llm = llm
        self.redis_client = redis.from_url(redis_url)
        self.max_token_limit = max_token_limit
        self.session_ttl = session_ttl
        self._local_cache: Dict[str, ConversationSummaryBufferMemory] = {}

    def get_memory(self, session_id: str) -> ConversationSummaryBufferMemory:
        """Get memory for a session (with caching)."""

        # Check local cache first
        if session_id in self._local_cache:
            return self._local_cache[session_id]

        # Create new memory
        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=self.max_token_limit,
            memory_key="chat_history",
            return_messages=True
        )

        # Load from Redis
        self._load_from_redis(session_id, memory)

        # Cache locally
        self._local_cache[session_id] = memory

        return memory

    def save_memory(self, session_id: str):
        """Persist memory to Redis."""
        if session_id not in self._local_cache:
            return

        memory = self._local_cache[session_id]
        messages = []

        for msg in memory.chat_memory.messages:
            messages.append({
                "type": msg.type,
                "content": msg.content
            })

        # Save to Redis with expiration
        key = f"chat_memory:{session_id}"
        self.redis_client.setex(
            key,
            self.session_ttl,
            json.dumps(messages)
        )

    def _load_from_redis(self, session_id: str, memory: ConversationSummaryBufferMemory):
        """Load memory from Redis."""
        key = f"chat_memory:{session_id}"
        data = self.redis_client.get(key)

        if data:
            messages = json.loads(data)
            for msg in messages:
                if msg["type"] == "human":
                    memory.chat_memory.add_user_message(msg["content"])
                elif msg["type"] == "ai":
                    memory.chat_memory.add_ai_message(msg["content"])

    def clear_memory(self, session_id: str):
        """Clear memory for a session."""
        # Clear local cache
        if session_id in self._local_cache:
            del self._local_cache[session_id]

        # Clear Redis
        key = f"chat_memory:{session_id}"
        self.redis_client.delete(key)

    def cleanup_expired(self):
        """Remove expired sessions from local cache."""
        # Redis handles TTL automatically
        # Just clear local cache periodically
        self._local_cache.clear()

# Usage in FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
llm = ChatOpenAI(model="gpt-4")
memory_manager = ProductionMemoryManager(llm)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        memory = memory_manager.get_memory(request.session_id)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory
        )

        response = agent_executor.invoke({"input": request.message})

        # Save after successful response
        memory_manager.save_memory(request.session_id)

        return {"response": response["output"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    memory_manager.clear_memory(session_id)
    return {"status": "cleared"}
```

---

_Last Updated: January 25, 2026_
