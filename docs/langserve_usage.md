# LangServe Endpoints Usage Guide

## Overview

LangServe automatically creates several endpoints for your agent with streaming support. While the schema looks complex, it's actually quite flexible.

## Available Endpoints

When LangServe is enabled, these endpoints are created under `/agent`:

1. **`POST /agent/invoke`** - Single request/response (no streaming)
2. **`POST /agent/stream`** - Streaming response (SSE - Server-Sent Events)
3. **`POST /agent/batch`** - Process multiple requests at once
4. **`POST /agent/stream_log`** - Stream detailed execution logs
5. **`GET /agent/playground`** - Interactive web UI for testing

## Simple Usage (Recommended)

### Option 1: Use Your Custom Endpoint (Simplest)

```bash
# Your clean, simple endpoint - NO CHANGES NEEDED
POST http://localhost:8000/query

{
  "question": "What is the weather today?",
  "session_id": "user-123",
  "user_id": "john",
  "history": []
}
```

**This is your endpoint - keep using it!** It's simpler and has a cleaner schema.

### Option 2: LangServe Invoke (Simple Request)

For **non-streaming** requests, LangServe needs this format:

```bash
POST http://localhost:8000/agent/invoke

{
  "input": {
    "messages": [
      {"type": "human", "content": "What is 2+2?"}
    ]
  },
  "config": {}
}
```

**Python example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/agent/invoke",
    json={
        "input": {
            "messages": [
                {"type": "human", "content": "What is the weather?"}
            ]
        },
        "config": {}
    }
)

result = response.json()
print(result["output"]["messages"][-1]["content"])
```

### Option 3: LangServe Stream (Real-time Streaming)

For **streaming** responses (see agent thinking in real-time):

```bash
POST http://localhost:8000/agent/stream

{
  "input": {
    "messages": [
      {"type": "human", "content": "Explain quantum physics"}
    ]
  },
  "config": {}
}
```

**Python streaming example:**

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/agent/stream",
    json={
        "input": {
            "messages": [
                {"type": "human", "content": "Tell me a story"}
            ]
        },
        "config": {}
    },
    stream=True  # Enable streaming
)

# Process streaming chunks
for line in response.iter_lines():
    if line:
        # Skip "data: " prefix
        if line.startswith(b"data: "):
            data = json.loads(line[6:])
            print(data)
```

**JavaScript/Fetch streaming:**

```javascript
const response = await fetch("http://localhost:8000/agent/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    input: {
      messages: [{ type: "human", content: "Hello!" }],
    },
    config: {},
  }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { value, done } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  console.log(chunk);
}
```

## With Conversation History

### Your Custom Endpoint (Simpler)

```json
POST /query

{
  "question": "What did I ask before?",
  "session_id": "user-123",
  "history": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4"}
  ]
}
```

### LangServe Format

```json
POST /agent/invoke

{
  "input": {
    "messages": [
      {"type": "human", "content": "What is 2+2?"},
      {"type": "ai", "content": "2+2 equals 4"},
      {"type": "human", "content": "What did I ask before?"}
    ]
  },
  "config": {}
}
```

## Advanced: Thread-based Memory (Optional)

LangServe supports persistent conversation threads using the `config` field:

```json
{
  "input": {
    "messages": [{ "type": "human", "content": "Remember my name is John" }]
  },
  "config": {
    "configurable": {
      "thread_id": "conversation-123"
    }
  }
}
```

**Note:** This requires setting up a checkpointer in your agent (not currently configured).

## Playground UI

The easiest way to test - just open in your browser:

```
http://localhost:8000/agent/playground
```

Interactive UI where you can:

- Type messages
- See streaming responses
- Test different configurations
- No code needed!

## Schema Comparison

### Your Endpoint vs LangServe

| Feature    | Your `/query`  | LangServe `/agent/invoke` |
| ---------- | -------------- | ------------------------- |
| Schema     | Simple, custom | LangChain standard        |
| Streaming  | No             | Yes (with `/stream`)      |
| History    | Built-in field | In messages array         |
| Playground | No             | Yes                       |
| Complexity | Low ⭐         | Medium ⭐⭐               |

## Recommendation

**For your use case:**

1. **Keep using `/query`** for simple integrations - it's cleaner
2. **Use `/agent/stream`** if you want real-time streaming responses
3. **Use `/agent/playground`** for quick testing and demos

## Complete cURL Examples

### Your Endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "session_id": "test-123",
    "user_id": "user1",
    "history": []
  }'
```

### LangServe Invoke

```bash
curl -X POST http://localhost:8000/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [
        {"type": "human", "content": "What is the capital of France?"}
      ]
    },
    "config": {}
  }'
```

### LangServe Stream

```bash
curl -X POST http://localhost:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [
        {"type": "human", "content": "Tell me a joke"}
      ]
    },
    "config": {}
  }' \
  --no-buffer
```

## Troubleshooting

### "Expected messages array"

Make sure your input has this structure:

```json
{
  "input": {
    "messages": [...]  // ← Must be "messages", not "message"
  }
}
```

### Empty messages error

Ensure you have at least one message:

```json
{
  "input": {
    "messages": [{ "type": "human", "content": "Your question here" }]
  }
}
```

### Message types

Valid message types:

- `"human"` or `"user"` - User messages
- `"ai"` or `"assistant"` - Agent responses
- `"system"` - System prompts (optional)

## Summary

**TLDR:** Your `/query` endpoint is fine - it's simpler! Only use LangServe endpoints if you need:

- ✅ Real-time streaming
- ✅ Playground UI
- ✅ Batch processing
- ✅ Standard LangChain compatibility

Otherwise, stick with your clean `/query` endpoint! 🚀
