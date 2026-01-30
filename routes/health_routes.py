"""Health check route handlers."""

import logging
from fastapi import APIRouter
from typing import Optional

from models.api_models import HealthResponse
from main import AgentApp

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="", tags=["Health"])

# Module-level variables to be set by main app
agent_app: Optional[AgentApp] = None
AGENT_LOADED: bool = False
LANGSERVE_AVAILABLE: bool = False


def set_agent_state(app: Optional[AgentApp], loaded: bool, langserve: bool) -> None:
    """Set the agent state for health checks."""
    global agent_app, AGENT_LOADED, LANGSERVE_AVAILABLE
    agent_app = app
    AGENT_LOADED = loaded
    LANGSERVE_AVAILABLE = langserve


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if AGENT_LOADED else "degraded",
        agent_loaded=AGENT_LOADED,
        langserve_available=LANGSERVE_AVAILABLE,
    )
