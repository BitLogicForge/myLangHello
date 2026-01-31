"""Agent query route handlers."""

import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from models.api_models import QueryRequest, QueryResponse
from services.agent_utils import prepare_messages_with_history

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="", tags=["Agent"])

# Module-level variables to be set by main app
agent_executor: Optional[Any] = None
AGENT_LOADED: bool = False
telemetry: Optional[Any] = None


def set_agent_executor(executor: Optional[Any], loaded: bool, telem: Optional[Any] = None) -> None:
    """Set the agent executor for query handling."""
    global agent_executor, AGENT_LOADED, telemetry
    agent_executor = executor
    AGENT_LOADED = loaded
    telemetry = telem


@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Manual query endpoint (fallback if LangServe not available).

    For streaming support, install langserve and use /agent/stream endpoint.
    """
    if not AGENT_LOADED or agent_executor is None:
        logger.warning("Query attempted but agent not loaded")
        raise HTTPException(status_code=503, detail="Agent not loaded")

    # Track metrics if telemetry is available
    if telemetry:
        with telemetry.track_request("query"):
            return await _process_query(request)
    else:
        return await _process_query(request)


async def _process_query(request: QueryRequest) -> QueryResponse:
    """Internal query processing with metrics tracking."""
    if agent_executor is None:
        raise HTTPException(status_code=503, detail="Agent executor not available")

    try:
        logger.info(f"Processing query (session: {request.session_id})")
        logger.debug(f"Question: {request.question[:100]}...")

        # Convert history to format expected by utility function
        history_tuples = None
        if request.history:
            history_tuples = [(msg.role, msg.content) for msg in request.history]
            logger.debug(f"Included {len(request.history)} history messages")

        # Prepare messages using shared utility
        messages = prepare_messages_with_history(request.question, history_tuples)

        start_time = time.time()
        # LangGraph agents expect messages format
        response = agent_executor.invoke({"messages": messages})
        duration = time.time() - start_time

        logger.info(
            f"Query completed successfully in {duration:.2f}s (session: {request.session_id})"
        )

        # Track basic metrics if available
        if telemetry and hasattr(response, "get"):
            # Try to extract iteration count from response metadata
            metadata = response.get("metadata", {})
            if "iterations" in metadata:
                telemetry.track_agent_iterations(metadata["iterations"])

        # Extract the final message from LangGraph response
        # LangGraph returns {"messages": [...]} where last message is the response
        if isinstance(response, dict) and "messages" in response:
            messages_list = response["messages"]
            if messages_list and len(messages_list) > 0:
                # Get the last message (agent's response)
                final_message = messages_list[-1]
                # Extract content from the message
                if hasattr(final_message, "content"):
                    output_text = final_message.content
                elif isinstance(final_message, tuple) and len(final_message) > 1:
                    output_text = final_message[1]
                else:
                    output_text = str(final_message)
            else:
                output_text = "No response generated"
        else:
            # Fallback for old format
            output_text = response.get("output", str(response))

        return QueryResponse(
            output=output_text,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
