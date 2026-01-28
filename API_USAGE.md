# FastAPI API Usage Examples

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
# Option 1: Using the api.py directly
python api.py

# Option 2: Using uvicorn
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Note:** Logs are written to both console and `api.log` file.

## Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "agent_loaded": true,
  "langserve_available": true
}
```

### 1.1 Database Health Check

```bash
curl http://localhost:8000/health/db
```

**Response:**

```json
{
  "database_healthy": true,
  "status": "connected"
}
```

### 2. Query Agent (Manual - No Streaming)

**Basic Query:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 5 + 7?"}'
```

**Query with Conversation History:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What about 20?",
    "history": [
      {"role": "user", "content": "What is 5 + 7?"},
      {"role": "assistant", "content": "5 + 7 equals 12."}
    ]
  }'
```

**Query with Session ID:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many users are in the database?",
    "session_id": "user-123",
    "user_id": "john_doe"
  }'
```

### 3. Agent with LangServe (Streaming Support)

**Invoke (Single Response):**

```bash
curl -X POST http://localhost:8000/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"input": "Calculate 5 + 7 and tell me the weather in Paris"}}'
```

**Stream (Token-by-Token):**

```bash
curl -X POST http://localhost:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"input": {"input": "How many users in the database?"}}' \
  --no-buffer
```

**Stream Events (Detailed Agent Steps):**

```bash
curl -X POST http://localhost:8000/agent/stream_log \
  -H "Content-Type: application/json" \
  -d '{"input": {"input": "Check database users"}}' \
  --no-buffer
```

### 4. Configuration

```bash
curl http://localhost:8000/config
```

## Python Client Examples

### Using requests (Simple)

```python
import requests

# Basic query
response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What is 5 + 7?"}
)
print(response.json())

# Query with conversation history
response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "What about multiplying those numbers?",
        "history": [
            {"role": "user", "content": "What is 5 + 7?"},
            {"role": "assistant", "content": "5 + 7 equals 12."}
        ],
        "session_id": "session-123"
    }
)
print(response.json())
```

### Using LangServe RemoteRunnable (Streaming)

```python
from langserve import RemoteRunnable

agent = RemoteRunnable("http://localhost:8000/agent")

# Invoke
result = agent.invoke({"input": "Calculate 5 + 7"})
print(result)

# Stream
for chunk in agent.stream({"input": "Tell me about Python"}):
    print(chunk, end="", flush=True)
```

### Using httpx with streaming

```python
import httpx
import json

async def stream_agent():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/agent/stream",
            json={"input": {"input": "What's the weather?"}},
            timeout=30.0
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    print(line)

# Run
import asyncio
asyncio.run(stream_agent())
```

## Interactive Playground

Visit http://localhost:8000/agent/playground in your browser for an interactive UI!

## JavaScript/TypeScript Client

```javascript
// Using fetch API - Basic query
async function queryAgent(question) {
  const response = await fetch("http://localhost:8000/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return await response.json();
}

// Using fetch API - With conversation history
async function queryWithHistory(question, history) {
  const response = await fetch("http://localhost:8000/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      history,
      session_id: "session-" + Date.now(),
    }),
  });
  return await response.json();
}

// Example usage
const result = await queryWithHistory("What about 20?", [
  { role: "user", content: "What is 5 + 7?" },
  { role: "assistant", content: "5 + 7 equals 12." },
]);

// Using EventSource for streaming
const eventSource = new EventSource(
  "http://localhost:8000/agent/stream?input=" +
    encodeURIComponent(JSON.stringify({ input: "Your question" })),
);

eventSource.onmessage = (event) => {
  console.log("Received:", event.data);
};
```

## Production Notes

1. **CORS**: Update `allow_origins` in api.py for production domains
2. **Authentication**: Add auth middleware (JWT, API keys, etc.)
3. **Rate Limiting**: Consider adding rate limiting
4. **Monitoring**: Add logging and monitoring (Langsmith, etc.)
5. **Load Balancing**: Use nginx or cloud load balancer for multiple instances
6. **Environment**: Use proper .env files for secrets

## Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "api.py"]
```

```bash
docker build -t langchain-agent-api .
docker run -p 8000:8000 --env-file .env langchain-agent-api
```
