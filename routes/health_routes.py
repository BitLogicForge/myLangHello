"""Health check route handlers."""

import logging

from fastapi import APIRouter

from main import AgentApp
from models.api_models import HealthResponse

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="", tags=["Health"])

# Module-level variables to be set by main app
agent_app: AgentApp | None = None
agent_loaded_state: bool = False
langserve_available_state: bool = False


def set_agent_state(app: AgentApp | None, loaded: bool, langserve: bool) -> None:
    """Set the agent state for health checks."""
    global agent_app, agent_loaded_state, langserve_available_state
    agent_app = app
    agent_loaded_state = loaded
    langserve_available_state = langserve


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if agent_loaded_state else "degraded",
        agent_loaded=agent_loaded_state,
        langserve_available=langserve_available_state,
    )
