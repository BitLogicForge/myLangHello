"""FastAPI application with LangServe for LangChain agent streaming."""

import logging
from typing import cast
from pydantic import BaseModel, Field

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import Config
from main import AgentApp
from routes import agent_routes, config_routes, health_routes
from services import SupportsAStream
from services.telemetry import TelemetryManager, get_telemetry
from utils import setup_logging, is_tuple

# Configure logging
api_config = Config()
setup_logging(debug=bool(api_config.get("agent.debug", False)))
logger = logging.getLogger(__name__)


try:
    from langserve import add_routes  # pyright: ignore[reportUnknownVariableType]

    is_langserve_available = True
except ImportError:
    add_routes = None
    is_langserve_available = False
    logger.warning("langserve not installed. Run: pip install langserve[all]")


# MARK: App Initialization
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
telemetry: TelemetryManager | None = None
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
agent_executor: SupportsAStream | None = None
agent_app: AgentApp | None = None
try:
    logger.info("Initializing agent application...")
    agent_app = AgentApp()
    agent_executor = agent_app.agent_executor
    is_agent_loaded = True
    logger.info("✅ Agent loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading agent: {e}", exc_info=True)
    agent_executor = None
    agent_app = None
    is_agent_loaded = False


# MARK: Routes Registration
# Configure route modules with agent state
health_routes.set_agent_state(agent_app, is_agent_loaded, is_langserve_available)
agent_routes.set_agent_executor(agent_executor, is_agent_loaded, telemetry)
config_routes.set_agent_app(agent_app, is_agent_loaded)

# Register routers
app.include_router(health_routes.router)
app.include_router(agent_routes.router)
app.include_router(config_routes.router)

# Mount Chainlit Chat UI
try:
    import os
    from chainlit.utils import mount_chainlit
    from fastapi.responses import RedirectResponse

    @app.get("/chat", include_in_schema=False)
    async def redirect_to_chat():
        return RedirectResponse(url="/chat/")

    target_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "chat_app.py"
    )
    mount_chainlit(app=app, target=target_path, path="/chat")
    print("✅ Chainlit Chat UI mounted at /chat")
except Exception as e:
    print(f"⚠️ Failed to mount Chainlit: {e}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "LangChain Agent API",
        "docs": "/docs",
        "health": "/health",
        "agent_endpoint": "/agent" if is_langserve_available else "/query",
        "playground": "/agent/playground" if is_langserve_available else None,
        "metrics": "http://localhost:9090/metrics" if telemetry else None,
    }


# LangServe Routes (Recommended - with streaming support)
if is_langserve_available and agent_executor and add_routes is not None:
    add_routes(
        app,
        agent_executor,  # pyright: ignore[reportArgumentType]
        path="/agent",
        # Let LangServe enable all endpoints by default for playground to work
        playground_type="default",  # Default playground works with LangGraph agents
        enable_feedback_endpoint=True,
    )
    print("✅ LangServe routes added at /agent")
    print("📊 Playground available at http://localhost:8000/agent/playground")


# MARK: Example Routes
class StructuredInputExample(BaseModel):
    """Example of a structured input payload.

    Includes a basket of vegetables, amount of money, desire to buy list of groceries, and a query.
    """

    basket: list[str] = Field(
        ...,
        description="List of vegetables currently in the basket",
        json_schema_extra={"example": ["carrot", "cucumber", "spinach"]},
    )
    amount_of_money: float = Field(
        ...,
        description="Amount of money available to spend",
        json_schema_extra={"example": 50.0},
    )
    desire_to_buy: list[str] = Field(
        ...,
        description="List of groceries that the user wants to buy",
        json_schema_extra={"example": ["milk", "bread", "butter", "cheese"]},
    )
    query: str = Field(
        ...,
        description="Specific question or query to run with this context",
        json_schema_extra={
            "example": "Can I afford all the items in my desire list? What recipe can I make with my basket?"
        },
    )


class StructuredResponse(BaseModel):
    """Response model for the structured query endpoint."""

    formatted_prompt: str = Field(
        ..., description="The formatted prompt sent to the agent"
    )
    output: str = Field(..., description="The agent's response")
    status: str = Field(..., description="Status of the query execution")


@app.post(
    "/structured-query",
    response_model=StructuredResponse,
    tags=["Examples"],
    summary="Example endpoint showing how to handle structured inputs (e.g. basket, budget, grocery desires)",
)
async def structured_query(request: StructuredInputExample):
    """
    Example endpoint showing how to pass a structured request
    (basket list, budget, grocery list, and a natural language query) to the agent.
    """
    if not is_agent_loaded or agent_app is None:
        raise HTTPException(status_code=503, detail="Agent application is not loaded")

    # Format the structured parameters into a cohesive prompt for the agent
    formatted_prompt = (
        f"Vegetables in my basket: {', '.join(request.basket)}\n"
        f"Amount of money I have: ${request.amount_of_money:.2f}\n"
        f"Groceries I desire to buy: {', '.join(request.desire_to_buy)}\n"
        f"Query: {request.query}"
    )

    try:
        logger.info("Executing agent query with structured input...")
        response = await agent_app.run(question=formatted_prompt)

        # Extract the final message content from the LangGraph response
        output_text = "No response generated"
        if response and "messages" in response:
            messages_list = response["messages"]
            if isinstance(messages_list, list) and messages_list:
                final_message = cast(object, messages_list[-1])
                content = cast(object, getattr(final_message, "content", None))
                if content is not None:
                    output_text = str(content)
                elif is_tuple(final_message) and len(final_message) > 1:
                    final_msg_tuple = cast(tuple[object, ...], final_message)
                    output_text = str(final_msg_tuple[1])
                else:
                    output_text = str(final_message)
        elif response is not None:
            output_text = str(response.get("output", response))

        return StructuredResponse(
            formatted_prompt=formatted_prompt, output=output_text, status="success"
        )
    except Exception as e:
        logger.error(f"Error executing structured query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


class RecipeAndBudgetAnalysis(BaseModel):
    """Structured response containing budget calculation and recipe recommendations."""

    can_afford_all: bool = Field(
        ...,
        description="True if the total estimated cost of all desired groceries is within the budget",
    )
    total_estimated_cost: float = Field(
        ..., description="The estimated total cost of the desired groceries"
    )
    remaining_budget: float = Field(
        ...,
        description="The remaining money after buying the groceries (budget - estimated cost)",
    )
    affordable_items: list[str] = Field(
        ..., description="List of desired items that CAN be bought within the budget"
    )
    missing_items: list[str] = Field(
        ..., description="List of desired items that CANNOT be bought within the budget"
    )
    suggested_recipes: list[str] = Field(
        ...,
        description="1-3 recipes we can cook using the vegetables in the basket and/or groceries",
    )
    explanation: str = Field(
        ...,
        description="A short explanation of the cost estimates, budget check, and recipe selections",
    )


@app.post(
    "/structured-output",
    response_model=RecipeAndBudgetAnalysis,
    tags=["Examples"],
    summary="Example endpoint showing how to get a structured JSON response directly from the LLM",
)
async def structured_output(request: StructuredInputExample):
    """
    Example endpoint demonstrating how to force the LLM to return
    a fully structured JSON response (matching a Pydantic model)
    instead of plain text.
    """
    # Format the input parameters into the prompt
    formatted_prompt = (
        f"Basket of vegetables: {', '.join(request.basket)}\n"
        f"Available budget: ${request.amount_of_money:.2f}\n"
        f"Groceries to buy: {', '.join(request.desire_to_buy)}\n"
        f"Question: {request.query}\n"
    )

    try:
        logger.info("Executing structured output query directly with LLM...")

        # Instantiate LLM from factory
        from services.llm_factory import LLMFactory

        llm = LLMFactory.create_llm()

        # Bind the schema to the LLM to enforce structured output
        structured_llm = llm.with_structured_output(RecipeAndBudgetAnalysis)

        response = cast(
            RecipeAndBudgetAnalysis, await structured_llm.ainvoke(formatted_prompt)
        )
        return response

    except Exception as e:
        logger.error(f"Error executing structured output: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Structured output error: {str(e)}. (Make sure your configured provider supports structured output).",
        )


# MARK: Error Handlers
# Error handlers
@app.exception_handler(404)
async def not_found_handler(_request: Request, _exc: Exception):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found. Check /docs for available endpoints."},
    )


@app.exception_handler(500)
async def internal_error_handler(_request: Request, _exc: Exception):
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."},
    )


# MARK: Main Entrypoint
def main():
    """Run the FastAPI application."""
    from config import settings

    port = settings.port

    print("\n" + "=" * 60)
    print("🚀 Starting LangChain Agent API Server")
    print("=" * 60)
    print(f"📍 Server: http://localhost:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"🔄 ReDoc: http://localhost:{port}/redoc")
    print(f"💬 Chat Interface: http://localhost:{port}/chat")

    if is_langserve_available:
        print(f"🎮 Playground: http://localhost:{port}/agent/playground")
        print(f"📡 Streaming: POST http://localhost:{port}/agent/stream")
    else:
        print("⚠️  LangServe not available - install with: pip install langserve[all]")
        print(f"📡 Query endpoint: POST http://localhost:{port}/query")

    print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
