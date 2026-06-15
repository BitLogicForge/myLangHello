"""Configuration route handlers."""

import logging

from fastapi import APIRouter, HTTPException

from utils import is_list
from main import AgentApp

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="", tags=["Configuration"])

# MARK: State Management
# Module-level variables to be set by main app
agent_app: AgentApp | None = None
agent_loaded_state: bool = False


def set_agent_app(app: AgentApp | None, loaded: bool) -> None:
    """Set the agent app for configuration access."""
    global agent_app, agent_loaded_state
    agent_app = app
    agent_loaded_state = loaded


# MARK: Configuration Endpoint
@router.get("/config")
async def get_config():
    """Get agent configuration details."""
    if not agent_loaded_state or not agent_app:
        raise HTTPException(status_code=503, detail="Agent not loaded")

    try:
        # Access configurator components safely
        tools: object = getattr(agent_app, "tools", None)
        llm: object = getattr(agent_app, "llm", None)

        model_name: object = getattr(llm, "model_name", None) if llm else None
        temperature: object = getattr(llm, "temperature", None) if llm else None

        return {
            "model": str(model_name) if isinstance(model_name, str) else "unknown",
            "temperature": float(temperature)
            if isinstance(temperature, (int, float))
            else 0.0,
            "tools_count": len(tools) if is_list(tools) else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config error: {str(e)}")
