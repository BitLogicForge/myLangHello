# Streaming LangChain Responses: FastAPI → React

## Overview

Enable real-time streaming of LangChain agent/LLM responses to a React frontend via FastAPI.

---

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

## 7. Running the Full Stack

### Terminal 1: Start FastAPI

```bash
cd backend
python streaming_server.py
# Server runs on http://localhost:8000
```

### Terminal 2: Start React

```bash
cd frontend
npm start
# App runs on http://localhost:3000
```

---

## Key Points

1. **SSE (Server-Sent Events)** is simpler than WebSockets for one-way streaming
2. **FastAPI StreamingResponse** with `text/event-stream` media type
3. **Custom CallbackHandler** captures LangChain events
4. **React Fetch API** with ReadableStream for consuming SSE
5. **AbortController** for stopping streams
6. **CORS** must be configured for cross-origin requests

---

## Debugging Tips

- Use browser DevTools → Network → EventStream to see live events
- Add verbose logging in callback handlers
- Test with Postman (supports SSE)
- Handle connection errors gracefully
- Implement reconnection logic for production

---

## Production Considerations

- Add authentication/authorization
- Rate limiting
- Connection pooling
- Error recovery
- Timeout handling
- Load balancing for multiple users
- WebSocket alternative for bi-directional needs
