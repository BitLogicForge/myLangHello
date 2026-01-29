"""Agent query route handlers."""

import logging
from fastapi import APIRouter, HTTPException
from typing import Optional, Any

from models.api_models import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="", tags=["Agent"])

# Module-level variables to be set by main app
agent_executor: Optional[Any] = None
AGENT_LOADED: bool = False


def set_agent_executor(executor: Optional[Any], loaded: bool) -> None:
    """Set the agent executor for query handling."""
    global agent_executor, AGENT_LOADED
    agent_executor = executor
    AGENT_LOADED = loaded


@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Manual query endpoint (fallback if LangServe not available).

    For streaming support, install langserve and use /agent/stream endpoint.
    """
    if not AGENT_LOADED or agent_executor is None:
        logger.warning("Query attempted but agent not loaded")
        raise HTTPException(status_code=503, detail="Agent not loaded")

    try:
        logger.info(f"Processing query (session: {request.session_id})")
        logger.debug(f"Question: {request.question[:100]}...")

        # Prepare input with history if provided
        invoke_input: dict[str, Any] = {"input": request.question}
        if request.history:
            # Convert MessageHistory objects to tuples for LangChain
            chat_history: list[tuple[str, str]] = [
                (msg.role, msg.content) for msg in request.history
            ]
            invoke_input["chat_history"] = chat_history
            logger.debug(f"Included {len(chat_history)} history messages")

        response = agent_executor.invoke(invoke_input)
        logger.info(f"Query completed successfully (session: {request.session_id})")
        return QueryResponse(
            output=response["output"],
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
