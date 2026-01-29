# LangServe Streaming with Configuration Parameters

## Overview

This guide explains how to use the `/agent/stream` endpoint with configurable parameters from React applications.

## LangServe Endpoints

When you use `add_routes` in your FastAPI app, LangServe automatically creates these endpoints:

| Endpoint            | Method | Purpose                         |
| ------------------- | ------ | ------------------------------- |
| `/agent/invoke`     | POST   | Non-streaming (single response) |
| `/agent/stream`     | POST   | Streaming (real-time chunks)    |
| `/agent/playground` | GET    | Interactive UI for testing      |

## Input Format

LangServe expects requests in this format:

```json
{
  "input": {
    "messages": [["user", "your question here"]]
  },
  "config": {
    "configurable": {
      "thread_id": "user-123",
      "temperature": 0.7,
      "model": "gpt-4",
      "max_tokens": 1000
    }
  }
}
```

### Key Parts:

- **`input.messages`**: Array of message tuples `[role, content]` for LangGraph agents
- **`config.configurable`**: Custom parameters including:
  - `thread_id`: For conversation memory/session tracking
  - `temperature`: Model temperature (0.0 - 1.0)
  - `model`: Model name override
  - `max_tokens`: Maximum response length
  - Any custom parameters your tools need

## React Implementation

### Method 1: Using LangChain's RemoteRunnable (Recommended)

```typescript
import { useState } from 'react';
import { RemoteRunnable } from '@langchain/core/runnables/remote';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface AgentConfig {
  temperature: number;
  model: string;
  sessionId: string;
}

export default function ChatWithAgent() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);

  // User-configurable parameters
  const [config, setConfig] = useState<AgentConfig>({
    temperature: 0.7,
    model: 'gpt-3.5-turbo',
    sessionId: 'user-' + Math.random().toString(36).substring(7)
  });

  // Initialize LangServe agent
  const agent = new RemoteRunnable({
    url: 'http://localhost:8000/agent',
    options: {
      headers: {
        'Authorization': 'Bearer YOUR_TOKEN', // Optional
        'Content-Type': 'application/json'
      }
    }
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    setIsStreaming(true);
    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');

    // Add placeholder for assistant response
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);
    let fullResponse = '';

    try {
      // Stream with custom configuration
      const stream = await agent.stream(
        {
          messages: [['user', input]] // LangGraph format
        },
        {
          configurable: {
            thread_id: config.sessionId,      // For memory
            temperature: config.temperature,  // Custom temp
            model: config.model               // Model selection
          }
        }
      );

      // Process streaming chunks
      for await (const chunk of stream) {
        // Extract content (format may vary)
        const content = chunk?.messages?.[0]?.content ||
                       chunk?.content ||
                       chunk?.output || '';

        if (content) {
          fullResponse += content;

          // Update UI in real-time
          setMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              role: 'assistant',
              content: fullResponse
            };
            return updated;
          });
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: '❌ Error: ' + (error as Error).message
        };
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="chat-container">
      {/* Configuration Panel */}
      <div className="config-panel">
        <h3>Configuration</h3>

        <label>
          Temperature: {config.temperature}
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={config.temperature}
            onChange={(e) => setConfig({
              ...config,
              temperature: parseFloat(e.target.value)
            })}
          />
        </label>

        <label>
          Model:
          <select
            value={config.model}
            onChange={(e) => setConfig({...config, model: e.target.value})}
          >
            <option value="gpt-3.5-turbo">GPT-3.5 Turbo (Fast)</option>
            <option value="gpt-4">GPT-4 (Powerful)</option>
          </select>
        </label>

        <label>
          Session ID:
          <input
            type="text"
            value={config.sessionId}
            onChange={(e) => setConfig({...config, sessionId: e.target.value})}
            placeholder="user-123"
          />
        </label>
      </div>

      {/* Messages Display */}
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <strong>{msg.role === 'user' ? '👤 You' : '🤖 Assistant'}</strong>
            <div className="content">{msg.content}</div>
            {isStreaming && idx === messages.length - 1 && (
              <span className="typing-cursor">▋</span>
            )}
          </div>
        ))}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={isStreaming}
        />
        <button type="submit" disabled={isStreaming}>
          {isStreaming ? '⏸ Streaming...' : '📤 Send'}
        </button>
      </form>

      {/* Status */}
      {isStreaming && (
        <div className="status">
          Streaming response... ({fullResponse.length} characters)
        </div>
      )}
    </div>
  );
}
```

### Method 2: Using Plain Fetch API

```typescript
interface StreamConfig {
  temperature?: number;
  model?: string;
  threadId?: string;
  maxTokens?: number;
}

async function streamAgentResponse(
  question: string,
  config: StreamConfig,
  onChunk: (content: string) => void,
  onError: (error: Error) => void
) {
  try {
    const response = await fetch('http://localhost:8000/agent/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer YOUR_TOKEN' // Optional
      },
      body: JSON.stringify({
        input: {
          messages: [['user', question]]
        },
        config: {
          configurable: {
            thread_id: config.threadId || 'default',
            temperature: config.temperature ?? 0.7,
            model: config.model || 'gpt-3.5-turbo',
            max_tokens: config.maxTokens || 1000
          }
        }
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error('Response body is not readable');
    }

    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      // Keep last incomplete line in buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;

        try {
          const data = JSON.parse(line);

          // Handle different response formats
          if (data.messages && data.messages.length > 0) {
            const lastMessage = data.messages[data.messages.length - 1];
            const content = lastMessage.content || '';
            if (content) {
              onChunk(content);
            }
          } else if (data.content) {
            onChunk(data.content);
          } else if (data.output) {
            onChunk(data.output);
          }
        } catch (parseError) {
          console.warn('Failed to parse chunk:', line);
        }
      }
    }
  } catch (error) {
    onError(error as Error);
  }
}

// Usage example
function ChatComponent() {
  const [response, setResponse] = useState('');

  const handleQuery = async () => {
    setResponse('');

    await streamAgentResponse(
      'What are my top investments?',
      {
        temperature: 0.7,
        model: 'gpt-4',
        threadId: 'user-123',
        maxTokens: 1000
      },
      (chunk) => {
        setResponse(prev => prev + chunk);
      },
      (error) => {
        console.error('Error:', error);
        setResponse('Error: ' + error.message);
      }
    );
  };

  return (
    <div>
      <button onClick={handleQuery}>Ask Agent</button>
      <div className="response">{response}</div>
    </div>
  );
}
```

## Common Configuration Patterns

### 1. Session-Based Memory

```typescript
// Each user gets their own conversation thread
const userSessionId = `user-${userId}-${Date.now()}`;

const stream = await agent.stream(
  { messages: [["user", question]] },
  {
    configurable: {
      thread_id: userSessionId, // Maintains conversation history
    },
  },
);
```

### 2. Dynamic Temperature Control

```typescript
// Allow users to adjust creativity
const [creativity, setCreativity] = useState<
  "factual" | "balanced" | "creative"
>("balanced");

const temperatureMap = {
  factual: 0.1, // Precise, deterministic
  balanced: 0.7, // Good balance
  creative: 0.9, // More varied responses
};

const stream = await agent.stream(
  { messages: [["user", question]] },
  {
    configurable: {
      temperature: temperatureMap[creativity],
    },
  },
);
```

### 3. Model Selection

```typescript
// Let users choose speed vs quality
const [modelTier, setModelTier] = useState<"fast" | "powerful">("fast");

const modelMap = {
  fast: "gpt-3.5-turbo", // Cheaper, faster
  powerful: "gpt-4", // Better quality
};

const stream = await agent.stream(
  { messages: [["user", question]] },
  {
    configurable: {
      model: modelMap[modelTier],
    },
  },
);
```

### 4. Custom Parameters for Tools

```typescript
// Pass custom config to your tools
const stream = await agent.stream(
  { messages: [["user", "Get my portfolio"]] },
  {
    configurable: {
      thread_id: "user-123",
      user_id: userId, // For DB queries
      include_archived: false, // Filter flag
      max_results: 10, // Limit results
      currency: "USD", // Formatting
    },
  },
);
```

## Handling Configuration in Backend

### Accessing Config in Tools

```python
from langchain_core.tools import tool

@tool
def get_portfolio(query: str, config: dict) -> str:
    """Get user's portfolio with custom filters."""
    # Extract config values
    user_id = config.get("configurable", {}).get("user_id")
    include_archived = config.get("configurable", {}).get("include_archived", False)
    max_results = config.get("configurable", {}).get("max_results", 10)

    # Use config in your logic
    portfolios = db.query_portfolios(
        user_id=user_id,
        include_archived=include_archived,
        limit=max_results
    )

    return f"Found {len(portfolios)} portfolios"
```

### Middleware for Config Injection

```python
from fastapi import Request, Depends

def inject_user_config(request: Request):
    """Extract user info from JWT and inject into config."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = verify_jwt(token)

    return {
        "configurable": {
            "user_id": user["id"],
            "role": user["role"],
            "permissions": user["permissions"]
        }
    }

# In api.py
add_routes(
    app,
    agent_executor,
    path="/agent",
    per_req_config_modifier=inject_user_config  # Auto-inject config
)
```

## Complete Working Example

```typescript
// AgentChat.tsx - Full implementation
import { useState, useEffect } from 'react';
import { RemoteRunnable } from '@langchain/core/runnables/remote';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface AgentSettings {
  temperature: number;
  model: 'gpt-3.5-turbo' | 'gpt-4';
  maxTokens: number;
  sessionId: string;
}

export default function AgentChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [settings, setSettings] = useState<AgentSettings>({
    temperature: 0.7,
    model: 'gpt-3.5-turbo',
    maxTokens: 1000,
    sessionId: `session-${Date.now()}`
  });

  const agent = new RemoteRunnable({
    url: 'http://localhost:8000/agent'
  });

  const sendMessage = async () => {
    if (!input.trim() || isStreaming) return;

    const userMsg: Message = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsStreaming(true);

    const assistantMsg: Message = {
      role: 'assistant',
      content: '',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, assistantMsg]);

    let accumulated = '';

    try {
      const stream = await agent.stream(
        { messages: [['user', userMsg.content]] },
        {
          configurable: {
            thread_id: settings.sessionId,
            temperature: settings.temperature,
            model: settings.model,
            max_tokens: settings.maxTokens
          }
        }
      );

      for await (const chunk of stream) {
        const content = chunk?.messages?.[0]?.content ||
                       chunk?.content ||
                       chunk?.output || '';

        accumulated += content;

        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1].content = accumulated;
          return updated;
        });
      }
    } catch (error) {
      console.error('Stream error:', error);
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1].content = '❌ ' + (error as Error).message;
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="agent-chat">
      <aside className="settings-panel">
        <h3>Settings</h3>

        <div className="setting">
          <label>Temperature: {settings.temperature}</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={settings.temperature}
            onChange={(e) => setSettings({
              ...settings,
              temperature: parseFloat(e.target.value)
            })}
          />
          <small>Lower = Focused, Higher = Creative</small>
        </div>

        <div className="setting">
          <label>Model</label>
          <select
            value={settings.model}
            onChange={(e) => setSettings({
              ...settings,
              model: e.target.value as AgentSettings['model']
            })}
          >
            <option value="gpt-3.5-turbo">GPT-3.5 (Fast)</option>
            <option value="gpt-4">GPT-4 (Powerful)</option>
          </select>
        </div>

        <div className="setting">
          <label>Max Tokens</label>
          <input
            type="number"
            value={settings.maxTokens}
            onChange={(e) => setSettings({
              ...settings,
              maxTokens: parseInt(e.target.value)
            })}
            min={100}
            max={4000}
          />
        </div>

        <button
          onClick={() => setSettings({
            ...settings,
            sessionId: `session-${Date.now()}`
          })}
        >
          🔄 New Session
        </button>
      </aside>

      <main className="chat-area">
        <div className="messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-header">
                <span className="role">
                  {msg.role === 'user' ? '👤 You' : '🤖 Assistant'}
                </span>
                <span className="time">
                  {msg.timestamp.toLocaleTimeString()}
                </span>
              </div>
              <div className="message-content">
                {msg.content}
                {isStreaming && idx === messages.length - 1 && (
                  <span className="cursor">▋</span>
                )}
              </div>
            </div>
          ))}
        </div>

        <form onSubmit={(e) => { e.preventDefault(); sendMessage(); }}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask anything..."
            disabled={isStreaming}
          />
          <button type="submit" disabled={isStreaming || !input.trim()}>
            {isStreaming ? '⏸ Streaming...' : '📤 Send'}
          </button>
        </form>
      </main>
    </div>
  );
}
```

## Best Practices

1. **Always provide a thread_id** for conversation memory
2. **Validate config values** on both frontend and backend
3. **Handle streaming errors gracefully** with try-catch
4. **Show loading indicators** during streaming
5. **Allow users to cancel** long-running streams
6. **Store session IDs** to resume conversations
7. **Rate limit** config changes to prevent abuse
8. **Log config usage** for debugging and analytics

## Troubleshooting

### Stream not working?

- Check if LangServe is installed: `pip list | grep langserve`
- Verify endpoint URL is correct
- Check CORS settings in FastAPI
- Ensure `streaming=True` in LLM initialization

### Config not applied?

- Verify config is in `configurable` key
- Check backend logs for config reception
- Ensure config values are correct types (number, string, etc.)
- Test in playground first: `http://localhost:8000/agent/playground`

### Memory not working?

- Ensure same `thread_id` is used across requests
- Check if checkpointer is configured in backend
- Verify database/memory persistence setup

## See Also

- [streaming_langchain_fastapi_react.md](streaming_langchain_fastapi_react.md) - Complete streaming guide
- [langserve_usage.md](langserve_usage.md) - LangServe setup and features
- [memory_and_conversation_history.md](memory_and_conversation_history.md) - Conversation persistence
