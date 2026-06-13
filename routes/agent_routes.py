"""Agent query route handlers."""

import logging
import time

from fastapi import APIRouter, HTTPException

from config import Config
from models.api_models import QueryRequest, QueryResponse
from services import AgentExecutionSettings, AgentRunner, TelemetryManager, SupportsAStream
from utils import prepare_messages_with_history, is_str_dict, is_list, is_tuple

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="", tags=["Agent"])


# MARK: State Management
# Module-level variables to be set by main app
agent_executor: SupportsAStream | None = None
agent_loaded_state: bool = False
telemetry: TelemetryManager | None = None
config = Config()


def set_agent_executor(executor: SupportsAStream | None, loaded: bool, telem: TelemetryManager | None = None) -> None:
    """Set the agent executor for query handling."""
    global agent_executor, agent_loaded_state, telemetry
    agent_executor = executor
    agent_loaded_state = loaded
    telemetry = telem


# MARK: Query Endpoint
@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Manual query endpoint (fallback if LangServe not available).

    For streaming support, install langserve and use /agent/stream endpoint.
    """
    if not agent_loaded_state or agent_executor is None:
        logger.warning("Query attempted but agent not loaded")
        raise HTTPException(status_code=503, detail="Agent not loaded")

    # Track metrics if telemetry is available
    if telemetry:
        with telemetry.track_request("query"):
            return await _process_query(request)
    else:
        return await _process_query(request)


# MARK: Query Processor
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
        runner = AgentRunner(agent_executor, AgentExecutionSettings.from_config(config))
        response = await runner.run({"messages": messages})
        duration = time.time() - start_time

        logger.info(
            f"Query completed successfully in {duration:.2f}s (session: {request.session_id})"
        )

        # Track basic metrics if available
        if telemetry and isinstance(response, dict):
            # Try to extract iteration count from response metadata
            metadata = response.get("metadata")
            if is_str_dict(metadata):
                iterations = metadata.get("iterations")
                if isinstance(iterations, int):
                    telemetry.track_agent_iterations(iterations)

        # Extract the final message from LangGraph response
        # LangGraph returns {"messages": [...]} where last message is the response
        if isinstance(response, dict):
            messages_list = response.get("messages")
            if is_list(messages_list) and len(messages_list) > 0:
                # Get the last message (agent's response)
                final_message = messages_list[-1]
                # Extract content from the message
                content: object = getattr(final_message, "content", None)
                if content is not None:
                    output_text = str(content)
                elif is_tuple(final_message) and len(final_message) > 1:
                    output_text = str(final_message[1])
                else:
                    output_text = str(final_message)
            else:
                # Fallback if "messages" is missing or empty but response is a dict
                output_text = str(response.get("output", response))
        elif response is not None:
            # Fallback for non-dict response format
            output_text = str(response)
        else:
            output_text = "No response generated"

        return QueryResponse(
            output=output_text,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
