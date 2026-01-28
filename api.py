"""FastAPI application with LangServe for LangChain agent streaming."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Any
import uvicorn

try:
    from langserve import add_routes

    LANGSERVE_AVAILABLE = True
except ImportError:
    LANGSERVE_AVAILABLE = False
    print("Warning: langserve not installed. Run: pip install langserve[all]")

from base_chain_agen_func_tooling import AgentApp


# Request/Response Models
class MessageHistory(BaseModel):
    """Individual message in conversation history."""

    role: str = Field(..., description="Role of the message sender (human, ai, system)")
    content: str = Field(..., description="Content of the message")


class QueryRequest(BaseModel):
    """Request model for agent queries."""

    question: str = Field(..., description="Question to ask the agent")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    history: Optional[list[MessageHistory]] = Field(
        None,
        description="Conversation history as a list of messages",
    )


class QueryResponse(BaseModel):
    """Response model for agent queries."""

    output: str = Field(..., description="Agent response")
    session_id: Optional[str] = Field(None, description="Session ID")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    agent_loaded: bool
    langserve_available: bool


# Initialize FastAPI app
app = FastAPI(
    title="LangChain Agent API",
    description="FastAPI backend for LangChain function calling agent with streaming support",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent_executor: Optional[Any] = None
try:
    agent_app = AgentApp()
    agent_executor = agent_app.agent_executor
    AGENT_LOADED = True
except Exception as e:
    print(f"Error loading agent: {e}")
    agent_executor = None
    AGENT_LOADED = False


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "LangChain Agent API",
        "docs": "/docs",
        "health": "/health",
        "agent_endpoint": "/agent" if LANGSERVE_AVAILABLE else "/query",
        "playground": "/agent/playground" if LANGSERVE_AVAILABLE else None,
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if AGENT_LOADED else "degraded",
        agent_loaded=AGENT_LOADED,
        langserve_available=LANGSERVE_AVAILABLE,
    )


# LangServe Routes (Recommended - with streaming support)
if LANGSERVE_AVAILABLE and agent_executor:
    add_routes(  # type: ignore
        app,
        agent_executor,
        path="/agent",
        enabled_endpoints=["invoke", "stream", "batch", "stream_log"],
        playground_type="default",  # Enables interactive playground UI
        enable_feedback_endpoint=True,
    )
    print("✅ LangServe routes added at /agent")
    print("📊 Playground available at http://localhost:8000/agent/playground")


# Fallback manual endpoint (if LangServe not available)
@app.post("/query", response_model=QueryResponse, tags=["Agent"])
async def query_agent(request: QueryRequest):
    """
    Manual query endpoint (fallback if LangServe not available).

    For streaming support, install langserve and use /agent/stream endpoint.
    """
    if not AGENT_LOADED or agent_executor is None:
        raise HTTPException(status_code=503, detail="Agent not loaded")

    try:
        # Prepare input with history if provided
        invoke_input: dict[str, Any] = {"input": request.question}
        if request.history:
            # Convert MessageHistory objects to tuples for LangChain
            chat_history: list[tuple[str, str]] = [
                (msg.role, msg.content) for msg in request.history
            ]
            invoke_input["chat_history"] = chat_history

        response = agent_executor.invoke(invoke_input)
        return QueryResponse(
            output=response["output"],
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


# Configuration endpoint
@app.get("/config", tags=["Configuration"])
async def get_config():
    """Get agent configuration details."""
    if not AGENT_LOADED:
        raise HTTPException(status_code=503, detail="Agent not loaded")

    try:
        return {
            "model": agent_app.llm.model_name,
            "temperature": agent_app.llm.temperature,
            "max_iterations": agent_app.agent_executor.max_iterations,
            "max_execution_time": agent_app.agent_executor.max_execution_time,
            "tools_count": len(agent_app.tools_manager.get_tools()),
            "database_tables": agent_app.db_manager.include_tables,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config error: {str(e)}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found. Check /docs for available endpoints."},
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error. Check logs for details."}
    )


def main():
    """Run the FastAPI application."""
    print("\n" + "=" * 60)
    print("🚀 Starting LangChain Agent API Server")
    print("=" * 60)
    print("📍 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔄 ReDoc: http://localhost:8000/redoc")

    if LANGSERVE_AVAILABLE:
        print("🎮 Playground: http://localhost:8000/agent/playground")
        print("📡 Streaming: POST http://localhost:8000/agent/stream")
    else:
        print("⚠️  LangServe not available - install with: pip install langserve[all]")
        print("📡 Query endpoint: POST http://localhost:8000/query")

    print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
