"""FastAPI application with LangServe for LangChain agent streaming."""

import logging
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import Config
from main import AgentApp
from routes import agent_routes, config_routes, health_routes
from services.telemetry import TelemetryManager, get_telemetry
from utils import setup_logging

# Configure logging
api_config = Config()
setup_logging(debug=bool(api_config.get("agent.debug", False)))
logger = logging.getLogger(__name__)


try:
    from langserve import add_routes

    LANGSERVE_AVAILABLE = True
except ImportError:
    LANGSERVE_AVAILABLE = False
    logger.warning("langserve not installed. Run: pip install langserve[all]")


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

# Initialize telemetry (self-hosted metrics)
telemetry: Optional[TelemetryManager] = None
try:
    telemetry = get_telemetry(
        service_name="chatbot-agent-api",
        metrics_port=9090,
        enable_metrics_server=True,
    )
    logger.info("✅ Telemetry initialized")
except Exception as e:
    logger.warning(f"⚠️  Telemetry initialization failed: {e}")
    telemetry = None

# Initialize agent
agent_executor: Optional[Any] = None
agent_app: Optional[AgentApp] = None
try:
    logger.info("Initializing agent application...")
    agent_app = AgentApp()
    agent_executor = agent_app.agent_executor
    AGENT_LOADED = True
    logger.info("✅ Agent loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading agent: {e}", exc_info=True)
    agent_executor = None
    agent_app = None
    AGENT_LOADED = False


# Configure route modules with agent state
health_routes.set_agent_state(agent_app, AGENT_LOADED, LANGSERVE_AVAILABLE)
agent_routes.set_agent_executor(agent_executor, agent_app, AGENT_LOADED, telemetry)
config_routes.set_agent_app(agent_app, AGENT_LOADED)

# Register routers
app.include_router(health_routes.router)
app.include_router(agent_routes.router)
app.include_router(config_routes.router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "LangChain Agent API",
        "docs": "/docs",
        "health": "/health",
        "agent_endpoint": "/agent" if LANGSERVE_AVAILABLE else "/query",
        "playground": "/agent/playground" if LANGSERVE_AVAILABLE else None,
        "metrics": "http://localhost:9090/metrics" if telemetry else None,
    }


# LangServe Routes (Recommended - with streaming support)
if LANGSERVE_AVAILABLE and agent_executor:
    add_routes(  # type: ignore
        app,
        agent_executor,
        path="/agent",
        # Let LangServe enable all endpoints by default for playground to work
        playground_type="default",  # Default playground works with LangGraph agents
        enable_feedback_endpoint=True,
    )
    print("✅ LangServe routes added at /agent")
    print("📊 Playground available at http://localhost:8000/agent/playground")


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
