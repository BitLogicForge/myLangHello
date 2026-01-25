# Streaming LangChain Responses: FastAPI → React

## Table of Contents

- [Streaming LangChain Responses: FastAPI → React](#streaming-langchain-responses-fastapi--react)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Option A: Simple Way - Using LangServe (RECOMMENDED)](#option-a-simple-way---using-langserve-recommended)
    - [Backend Setup (LangServe)](#backend-setup-langserve)
      - [Install Dependencies](#install-dependencies)
      - [FastAPI Server (20 lines!)](#fastapi-server-20-lines)
    - [User Authentication \& Session Management](#user-authentication--session-management)
      - [How to Distinguish Users in LangServe](#how-to-distinguish-users-in-langserve)
    - [Load Balancing Setup (Not Automatic!)](#load-balancing-setup-not-automatic)
      - [Option 1: Nginx Load Balancer](#option-1-nginx-load-balancer)
      - [Option 2: Docker Compose](#option-2-docker-compose)
      - [Option 3: Kubernetes](#option-3-kubernetes)
    - [Complete Production Example](#complete-production-example)
    - [Key Takeaways](#key-takeaways)
    - [Frontend with React Query](#frontend-with-react-query)
      - [Install Dependencies](#install-dependencies-1)
      - [Setup React Query Provider](#setup-react-query-provider)
      - [Chat Component with React Query](#chat-component-with-react-query)
      - [Alternative: Plain Fetch with React Query](#alternative-plain-fetch-with-react-query)
  - [Option B: Manual SSE Streaming (Advanced)](#option-b-manual-sse-streaming-advanced)
  - [1. FastAPI Backend with Streaming](#1-fastapi-backend-with-streaming)
    - [Install Dependencies](#install-dependencies-2)
    - [FastAPI Server with SSE (Server-Sent Events)](#fastapi-server-with-sse-server-sent-events)
  - [2. Alternative: Simple LLM Streaming (No Agent)](#2-alternative-simple-llm-streaming-no-agent)
  - [3. React Frontend](#3-react-frontend)
    - [Install Dependencies](#install-dependencies-3)
    - [React Component with Streaming](#react-component-with-streaming)
    - [Simple CSS](#simple-css)
  - [4. Alternative: Using EventSource API](#4-alternative-using-eventsource-api)
  - [5. FastAPI with GET Endpoint (For EventSource)](#5-fastapi-with-get-endpoint-for-eventsource)
  - [6. Advanced: Streaming with Agent Tool Calls](#6-advanced-streaming-with-agent-tool-calls)
  - [Comparison Table](#comparison-table)
  - [7. Running the Full Stack](#7-running-the-full-stack)
    - [Option A: LangServe](#option-a-langserve)
    - [Option B: Manual SSE](#option-b-manual-sse)
  - [Key Points](#key-points)
    - [LangServe Approach](#langserve-approach)
    - [Manual SSE Approach](#manual-sse-approach)
  - [Debugging Tips](#debugging-tips)
    - [LangServe](#langserve)
    - [Manual SSE](#manual-sse)
  - [Production Considerations](#production-considerations)
    - [Both Approaches](#both-approaches)
    - [LangServe Specific](#langserve-specific)
    - [Manual SSE Specific](#manual-sse-specific)

---

## Overview

Enable real-time streaming of LangChain agent/LLM responses to a React frontend via FastAPI.

**Two approaches:**

- ✅ **Option A (Simple)**: Use LangServe - 95% less code, production-ready
- ⚙️ **Option B (Advanced)**: Manual SSE - full control, custom events

---

## Option A: Simple Way - Using LangServe (RECOMMENDED)

### Backend Setup (LangServe)

#### Install Dependencies

```bash
pip install langserve[all] langchain langchain-openai
```

#### FastAPI Server (20 lines!)

```python
# app.py
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langserve import add_routes

app = FastAPI(title="Banking Agent API")

# Your agent setup
llm = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful banking and investment assistant."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Add your tools here
from langchain_core.tools import tool

@tool
def calculate_roi(initial: float, final: float) -> str:
    """Calculate ROI percentage."""
    roi = ((final - initial) / initial) * 100
    return f"ROI: {roi:.2f}%"

tools = [calculate_roi]

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# THIS IS ALL YOU NEED - LangServe handles everything!
add_routes(
    app,
    agent_executor,
    path="/agent",
    enable_feedback_endpoint=True,
)

# Bonus: Get playground UI at http://localhost:8000/agent/playground

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**That's it!** LangServe automatically provides:

- ✅ Streaming endpoints
- ✅ CORS handling
- ✅ WebSocket support
- ✅ Interactive playground
- ✅ OpenAPI docs
- ❌ Load balancing (requires external infrastructure)

---

### User Authentication & Session Management

#### How to Distinguish Users in LangServe

**Option 1: Using `per_req_config_modifier` (Recommended)**

```python
# app.py with user authentication
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langserve import add_routes
import jwt

app = FastAPI(title="Banking Agent API")
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info."""
    try:
        payload = jwt.decode(credentials.credentials, "your-secret-key", algorithms=["HS256"])
        user_id = payload.get("user_id")
        return {"user_id": user_id, "name": payload.get("name")}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def inject_user_context(request: Request, user=Depends(verify_token)):
    """Inject user info into request config."""
    return {
        "configurable": {
            "user_id": user["user_id"],
            "user_name": user["name"],
        }
    }

# Add routes with user context
add_routes(
    app,
    agent_executor,
    path="/agent",
    per_req_config_modifier=inject_user_context,  # Inject user info!
)
```

Now access user info in your agent:

```python
# In your tools
@tool
def get_portfolio_performance(query: str, config: dict) -> str:
    """Get portfolio performance for the current user."""
    user_id = config.get("configurable", {}).get("user_id")
    # Query database with user-specific data
    return f"User {user_id} portfolio: +12.5%"
```

**Option 2: Simple API Key**

```python
def inject_api_key_context(request: Request):
    """Extract API key from headers."""
    api_key = request.headers.get("x-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    # Map API key to user
    user_id = {"key123": "user123", "key456": "user456"}.get(api_key)

    return {"configurable": {"user_id": user_id}}

add_routes(app, agent_executor, path="/agent", per_req_config_modifier=inject_api_key_context)
```

Frontend usage:

```typescript
const agent = new RemoteRunnable({
  url: "http://localhost:8000/agent",
  options: {
    headers: {
      Authorization: "Bearer YOUR_JWT_TOKEN",
      // or 'x-api-key': 'key123'
    },
  },
});
```

---

### Load Balancing Setup (Not Automatic!)

**LangServe doesn't provide automatic load balancing** - use standard infrastructure:

#### Option 1: Nginx Load Balancer

```nginx
# nginx.conf
upstream langserve_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    location /agent {
        proxy_pass http://langserve_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        # For streaming
        proxy_buffering off;
    }
}
```

Start multiple instances:

```bash
uvicorn app:app --port 8000 &
uvicorn app:app --port 8001 &
uvicorn app:app --port 8002 &
```

#### Option 2: Docker Compose

```yaml
# docker-compose.yml
version: "3.8"
services:
  langserve:
    build: .
    deploy:
      replicas: 3 # Run 3 instances
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

```bash
docker-compose up --scale langserve=5
```

#### Option 3: Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langserve-agent
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: langserve
          image: your-registry/langserve:latest
          ports:
            - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: langserve-lb
spec:
  type: LoadBalancer
  ports:
    - port: 80
      targetPort: 8000
```

```bash
kubectl scale deployment langserve-agent --replicas=10
```

---

### Complete Production Example

```python
# production_app.py
from fastapi import FastAPI, Request, Depends
from fastapi.security import HTTPBearer
from langserve import add_routes
import os

app = FastAPI()
security = HTTPBearer()

def verify_and_inject_user(request: Request, credentials = Depends(security)):
    """Verify JWT and inject user context."""
    # Verify token
    user = verify_jwt(credentials.credentials)

    return {
        "configurable": {
            "user_id": user["user_id"],
            "portfolio_id": user["portfolio_id"],
        }
    }

# Your agent setup (same as before)
agent_executor = create_your_agent()

add_routes(
    app,
    agent_executor,
    path="/agent",
    per_req_config_modifier=verify_and_inject_user,
)

if __name__ == "__main__":
    import uvicorn
    # Run with multiple workers for better performance
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

---

### Key Takeaways

**Load Balancing:**

- ❌ NOT automatic in LangServe
- ✅ Use Nginx, AWS ALB, or Kubernetes
- ✅ LangServe is stateless - easy to scale horizontally
- ✅ Run multiple instances with `--workers` or containers

**User Identification:**

- ✅ Use `per_req_config_modifier` to inject user context
- ✅ Support JWT, API keys, sessions, OAuth
- ✅ Access user info in prompts and tools via `config`
- ✅ User-specific caching and data isolation

---

### Frontend with React Query

#### Install Dependencies

```bash
npm install @tanstack/react-query @langchain/core axios
```

#### Setup React Query Provider

```typescript
// App.tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ChatComponent from './ChatComponent';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ChatComponent />
    </QueryClientProvider>
  );
}

export default App;
```

#### Chat Component with React Query

```typescript
// ChatComponent.tsx
import { useState, useRef } from 'react';
import { useMutation } from '@tanstack/react-query';
import { RemoteRunnable } from '@langchain/core/runnables/remote';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function ChatComponent() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [streamingContent, setStreamingContent] = useState('');
  const abortControllerRef = useRef<AbortController | null>(null);

  // Initialize LangServe agent
  const agent = new RemoteRunnable({
    url: 'http://localhost:8000/agent',
  });

  // React Query mutation for streaming
  const streamMutation = useMutation({
    mutationFn: async (query: string) => {
      abortControllerRef.current = new AbortController();

      // Add user message
      const userMsg: Message = { role: 'user', content: query };
      setMessages(prev => [...prev, userMsg]);

      // Add placeholder for assistant
      setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

      let fullResponse = '';

      try {
        // Stream from LangServe
        const stream = await agent.stream(
          { input: query },
          { signal: abortControllerRef.current.signal }
        );

        for await (const chunk of stream) {
          const content = chunk?.content || chunk?.output || '';
          fullResponse += content;
          setStreamingContent(fullResponse);

          // Update last message in real-time
          setMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              role: 'assistant',
              content: fullResponse
            };
            return updated;
          });
        }

        return fullResponse;
      } finally {
        setStreamingContent('');
      }
    },
    onError: (error) => {
      console.error('Streaming error:', error);
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: 'Sorry, an error occurred.'
        };
        return updated;
      });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || streamMutation.isPending) return;

    streamMutation.mutate(input);
    setInput('');
  };

  const stopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    streamMutation.reset();
  };

  return (
    <div className="chat-container">
      <h1>Banking Agent Chat</h1>

      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <strong>{msg.role === 'user' ? '👤 You' : '🤖 Assistant'}:</strong>
            <p>{msg.content}</p>
            {streamMutation.isPending && idx === messages.length - 1 && (
              <span className="cursor">▋</span>
            )}
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your investments..."
          disabled={streamMutation.isPending}
        />
        {streamMutation.isPending ? (
          <button type="button" onClick={stopStreaming} className="stop-btn">
            ⏹ Stop
          </button>
        ) : (
          <button type="submit">Send</button>
        )}
      </form>

      {streamMutation.isPending && (
        <div className="status">Streaming... {streamingContent.length} chars</div>
      )}
    </div>
  );
}
```

#### Alternative: Plain Fetch with React Query

```typescript
// ChatWithFetch.tsx
import { useMutation } from '@tanstack/react-query';

export default function ChatWithFetch() {
  const streamMutation = useMutation({
    mutationFn: async (query: string) => {
      const response = await fetch('http://localhost:8000/agent/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: { input: query },
          config: { configurable: {} }
        }),
      });

      if (!response.ok) throw new Error('Network error');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullText = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n').filter(line => line.trim());

          for (const line of lines) {
            try {
              const data = JSON.parse(line);
              if (data.content) {
                fullText += data.content;
                // Update UI here
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }

      return fullText;
    },
  });

  return (
    <button onClick={() => streamMutation.mutate('Calculate ROI')}>
      Ask Agent
    </button>
  );
}
```

---

## Option B: Manual SSE Streaming (Advanced)

## 1. FastAPI Backend with Streaming

### Install Dependencies

```bash
pip install fastapi uvicorn langchain langchain-openai sse-starlette
```

### FastAPI Server with SSE (Server-Sent Events)

```python
# streaming_server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel
import json
import asyncio
from queue import Queue
from threading import Thread

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StreamingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to capture streaming tokens."""

    def __init__(self, queue: Queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM generates a new token."""
        self.queue.put({"type": "token", "content": token})

    def on_llm_start(self, *args, **kwargs) -> None:
        """Called when LLM starts."""
        self.queue.put({"type": "start", "content": ""})

    def on_llm_end(self, *args, **kwargs) -> None:
        """Called when LLM ends."""
        self.queue.put({"type": "end", "content": ""})

    def on_tool_start(self, serialized, input_str: str, **kwargs) -> None:
        """Called when tool starts."""
        tool_name = serialized.get("name", "unknown")
        self.queue.put({
            "type": "tool_start",
            "content": f"Using tool: {tool_name}"
        })

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool ends."""
        self.queue.put({
            "type": "tool_end",
            "content": f"Tool result: {output[:100]}"
        })

    def on_agent_action(self, action, **kwargs) -> None:
        """Called when agent takes action."""
        self.queue.put({
            "type": "agent_action",
            "content": f"Action: {action.tool}"
        })


class QueryRequest(BaseModel):
    query: str


def run_agent_with_streaming(query: str, queue: Queue, tools: list):
    """Run agent in separate thread with streaming callback."""
    try:
        # Initialize LLM with streaming
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-4",
            streaming=True,
            callbacks=[StreamingCallbackHandler(queue)]
        )

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create agent
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            callbacks=[StreamingCallbackHandler(queue)]
        )

        # Run agent
        result = agent_executor.invoke({"input": query})

        # Send final result
        queue.put({"type": "complete", "content": result["output"]})

    except Exception as e:
        queue.put({"type": "error", "content": str(e)})
    finally:
        queue.put(None)  # Signal completion


async def event_generator(query: str, tools: list):
    """Generate SSE events from agent execution."""
    queue = Queue()

    # Run agent in background thread
    thread = Thread(target=run_agent_with_streaming, args=(query, queue, tools))
    thread.start()

    # Stream events
    while True:
        event = queue.get()

        if event is None:  # Completion signal
            break

        # Format as SSE
        yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(0)  # Allow other tasks to run

    thread.join()


@app.post("/api/chat/stream")
async def chat_stream(request: QueryRequest):
    """Stream chat responses."""
    # Add your tools here
    from base_chain_tools import calculator_tool
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        return calculator_tool(expression)

    tools = [calculator]

    return StreamingResponse(
        event_generator(request.query, tools),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/chat")
async def chat_no_stream(request: QueryRequest):
    """Non-streaming endpoint for comparison."""
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    response = llm.invoke(request.query)
    return {"response": response.content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 2. Alternative: Simple LLM Streaming (No Agent)

```python
# simple_streaming.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import json

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


@app.post("/api/stream")
async def stream_response(request: QueryRequest):
    """Simple streaming without agents."""

    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-4",
        streaming=True
    )

    async def generate():
        async for chunk in llm.astream(request.query):
            if chunk.content:
                yield f"data: {json.dumps({'token': chunk.content})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 3. React Frontend

### Install Dependencies

```bash
npm install axios
# or
npm install fetch-event-source
```

### React Component with Streaming

```typescript
// ChatComponent.tsx
import React, { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean;
}

export default function ChatComponent() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Add placeholder for assistant message
    const assistantMessageIndex = messages.length + 1;
    setMessages(prev => [...prev, { role: 'assistant', content: '', isStreaming: true }]);

    try {
      abortControllerRef.current = new AbortController();

      const response = await fetch('http://localhost:8000/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: input }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (reader) {
        let accumulatedContent = '';

        while (true) {
          const { done, value } = await reader.read();

          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6));

              if (data.type === 'token') {
                accumulatedContent += data.content;
                setMessages(prev => {
                  const updated = [...prev];
                  updated[assistantMessageIndex] = {
                    role: 'assistant',
                    content: accumulatedContent,
                    isStreaming: true
                  };
                  return updated;
                });
              } else if (data.type === 'tool_start') {
                accumulatedContent += `\n\n🔧 ${data.content}\n\n`;
                setMessages(prev => {
                  const updated = [...prev];
                  updated[assistantMessageIndex] = {
                    role: 'assistant',
                    content: accumulatedContent,
                    isStreaming: true
                  };
                  return updated;
                });
              } else if (data.type === 'complete') {
                setMessages(prev => {
                  const updated = [...prev];
                  updated[assistantMessageIndex] = {
                    role: 'assistant',
                    content: data.content,
                    isStreaming: false
                  };
                  return updated;
                });
              } else if (data.type === 'error') {
                setMessages(prev => {
                  const updated = [...prev];
                  updated[assistantMessageIndex] = {
                    role: 'assistant',
                    content: `Error: ${data.content}`,
                    isStreaming: false
                  };
                  return updated;
                });
              }
            }
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Stream aborted');
      } else {
        console.error('Streaming error:', error);
        setMessages(prev => {
          const updated = [...prev];
          updated[assistantMessageIndex] = {
            role: 'assistant',
            content: 'Sorry, an error occurred.',
            isStreaming: false
          };
          return updated;
        });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const stopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <strong>{msg.role === 'user' ? 'You' : 'Assistant'}:</strong>
            <p>{msg.content}</p>
            {msg.isStreaming && <span className="cursor">▋</span>}
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={isLoading}
        />
        {isLoading ? (
          <button type="button" onClick={stopStreaming}>Stop</button>
        ) : (
          <button type="submit">Send</button>
        )}
      </form>
    </div>
  );
}
```

### Simple CSS

```css
/* Chat.css */
.chat-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.messages {
  height: 500px;
  overflow-y: auto;
  border: 1px solid #ccc;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
  background: #f9f9f9;
}

.message {
  margin-bottom: 15px;
  padding: 10px;
  border-radius: 6px;
}

.message.user {
  background: #e3f2fd;
  margin-left: 20%;
}

.message.assistant {
  background: #f5f5f5;
  margin-right: 20%;
}

.cursor {
  animation: blink 1s infinite;
}

@keyframes blink {
  0%,
  50% {
    opacity: 1;
  }
  51%,
  100% {
    opacity: 0;
  }
}

.input-form {
  display: flex;
  gap: 10px;
}

.input-form input {
  flex: 1;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.input-form button {
  padding: 10px 20px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.input-form button:hover {
  background: #0056b3;
}
```

---

## 4. Alternative: Using EventSource API

```typescript
// ChatWithEventSource.tsx
import { useState } from 'react';

export default function ChatWithEventSource() {
  const [response, setResponse] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);

  const handleQuery = (query: string) => {
    setResponse('');
    setIsStreaming(true);

    const eventSource = new EventSource(
      `http://localhost:8000/api/stream?query=${encodeURIComponent(query)}`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.token) {
        setResponse(prev => prev + data.token);
      }

      if (data.done) {
        eventSource.close();
        setIsStreaming(false);
      }
    };

    eventSource.onerror = (error) => {
      console.error('EventSource error:', error);
      eventSource.close();
      setIsStreaming(false);
    };
  };

  return (
    <div>
      <button onClick={() => handleQuery('Tell me a story')}>
        Start Streaming
      </button>
      <div>{response}</div>
      {isStreaming && <span>▋</span>}
    </div>
  );
}
```

---

## 5. FastAPI with GET Endpoint (For EventSource)

```python
@app.get("/api/stream")
async def stream_get(query: str):
    """GET endpoint for EventSource API."""

    llm = ChatOpenAI(temperature=0.7, model="gpt-4", streaming=True)

    async def generate():
        async for chunk in llm.astream(query):
            if chunk.content:
                yield f"data: {json.dumps({'token': chunk.content})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )
```

---

## 6. Advanced: Streaming with Agent Tool Calls

```typescript
// AdvancedStreamingChat.tsx
interface StreamEvent {
  type: "token" | "tool_start" | "tool_end" | "complete" | "error";
  content: string;
}

const processStreamEvent = (event: StreamEvent) => {
  switch (event.type) {
    case "token":
      // Append token to message
      return { action: "append", text: event.content };

    case "tool_start":
      // Show tool usage
      return { action: "append", text: `\n\n🔧 ${event.content}\n\n` };

    case "tool_end":
      // Show tool result
      return { action: "append", text: `✓ ${event.content}\n\n` };

    case "complete":
      // Final message
      return { action: "complete", text: event.content };

    case "error":
      return { action: "error", text: event.content };
  }
};
```

---

## Comparison Table

| Feature              | LangServe (Option A) | Manual SSE (Option B)  |
| -------------------- | -------------------- | ---------------------- |
| **Backend Code**     | ~20 lines            | ~200 lines             |
| **Frontend Code**    | ~50 lines            | ~150 lines             |
| **Complexity**       | ⭐ Low               | ⭐⭐⭐⭐ High          |
| **Streaming**        | ✅ Automatic         | ⚙️ Manual setup        |
| **CORS**             | ✅ Built-in          | ⚙️ Manual config       |
| **Playground UI**    | ✅ Included          | ❌ Not included        |
| **Custom Events**    | ⚠️ Limited           | ✅ Full control        |
| **Production Ready** | ✅ Yes               | ⚠️ Needs work          |
| **Learning Curve**   | Easy                 | Steep                  |
| **Best For**         | Most use cases       | Advanced customization |

**Recommendation**: Use **LangServe (Option A)** for 95% of use cases. Only use Option B if you need custom event types or very specific control.

---

## 7. Running the Full Stack

### Option A: LangServe

**Terminal 1: Start FastAPI**

```bash
python app.py
# Server runs on http://localhost:8000
# Playground at http://localhost:8000/agent/playground
```

**Terminal 2: Start React**

```bash
npm start
# App runs on http://localhost:3000
```

---

### Option B: Manual SSE

**Terminal 1: Start FastAPI**

```bash
cd backend
python streaming_server.py
# Server runs on http://localhost:8000
```

**Terminal 2: Start React**

```bash
cd frontend
npm start
# App runs on http://localhost:3000
```

---

## Key Points

### LangServe Approach

- ✅ Minimal code (20 lines backend, 50 lines frontend)
- ✅ Production-ready out of the box
- ✅ Built-in playground for testing
- ✅ Works with React Query seamlessly
- ✅ Automatic error handling and retries

### Manual SSE Approach

1. **SSE (Server-Sent Events)** is simpler than WebSockets for one-way streaming
2. **FastAPI StreamingResponse** with `text/event-stream` media type
3. **Custom CallbackHandler** captures LangChain events
4. **React Fetch API** with ReadableStream for consuming SSE
5. **AbortController** for stopping streams
6. **CORS** must be configured for cross-origin requests

---

## Debugging Tips

### LangServe

- Test in playground first: `http://localhost:8000/agent/playground`
- Check OpenAPI docs: `http://localhost:8000/docs`
- Enable verbose mode in AgentExecutor
- Use React Query DevTools for frontend debugging

### Manual SSE

- Use browser DevTools → Network → EventStream to see live events
- Add verbose logging in callback handlers
- Test with Postman (supports SSE)
- Handle connection errors gracefully
- Implement reconnection logic for production

---

## Production Considerations

### Both Approaches

- Add authentication/authorization
- Rate limiting
- Connection pooling
- Error recovery
- Timeout handling
- Load balancing for multiple users

### LangServe Specific

- Configure `per_req_config_modifier` for user-specific settings
- Use `enable_feedback_endpoint` for user feedback
- Add monitoring with LangSmith

### Manual SSE Specific

- Implement reconnection logic
- Handle partial message recovery
- WebSocket alternative for bi-directional needs
